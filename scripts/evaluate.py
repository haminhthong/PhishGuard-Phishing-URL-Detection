"""Đánh giá test độc lập và các edge case của PhishGuard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from phishguard.calibration import DecisionThresholds, ProbabilityCalibrator
from phishguard.features import FeatureExtractor
from phishguard.training.evaluation import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
REPORTS_DIR = PROJECT_ROOT / "reports"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_split(name: str) -> pd.DataFrame:
    """Đọc Parquet, hoặc CSV nén nếu môi trường thiếu engine Parquet."""
    parquet_path = SPLITS_DIR / f"{name}.parquet"
    csv_path = SPLITS_DIR / f"{name}.csv.gz"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def score_frame(
    frame: pd.DataFrame,
    model: XGBClassifier,
    calibrator: ProbabilityCalibrator,
    extractor: FeatureExtractor | None = None,
) -> np.ndarray:
    """Trích xuất 25 feature, dự đoán và hiệu chuẩn điểm rủi ro."""
    fe = extractor or FeatureExtractor()
    features = fe.extract_frame(frame)
    return np.asarray(calibrator.calibrate(model.predict_proba(features)[:, 1]))


def evaluate_frame(
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: DecisionThresholds,
) -> dict[str, object]:
    """Tạo báo cáo cho một tập có nhãn để phân tích kết quả."""
    predictions = (scores >= thresholds.block_threshold).astype(int)
    report: dict[str, object] = {
        "cases": int(len(frame)),
        "metrics": classification_metrics(frame["label"], predictions, scores),
        "block_threshold": thresholds.block_threshold,
    }
    if "slice" in frame.columns:
        report["slices"] = {
            str(slice_name): classification_metrics(
                group["label"],
                predictions[group.index.to_numpy()],
                scores[group.index.to_numpy()],
            )
            for slice_name, group in frame.groupby("slice")
        }
    return report


def main() -> None:
    """Đánh giá test và edge cases, chỉ ghi báo cáo phân tích."""
    model = XGBClassifier()
    model.load_model(ARTIFACTS_DIR / "model.json")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((ARTIFACTS_DIR / "calibration.json").read_text(encoding="utf-8"))
    thresholds = DecisionThresholds.from_dict(
        json.loads((ARTIFACTS_DIR / "thresholds.json").read_text(encoding="utf-8"))
    )
    calibrator = ProbabilityCalibrator(
        method=str(calibration["method"]),
        params=dict(calibration["calibrator_params"]),
    )
    extractor = FeatureExtractor()

    test = read_split("test")
    report: dict[str, object] = {
        "model_version": metadata["model_version"],
        "feature_contract": metadata["feature_contract"],
        "test": evaluate_frame(test, score_frame(test, model, calibrator, extractor), thresholds),
        "edge_cases": {},
    }
    edge_cases = report["edge_cases"]
    assert isinstance(edge_cases, dict)
    for name in ("hard_dev", "hard_locked_test"):
        path = EVALUATION_DIR / f"{name}.jsonl"
        if path.exists():
            frame = read_jsonl(path)
            edge_cases[name] = evaluate_frame(
                frame,
                score_frame(frame, model, calibrator, extractor),
                thresholds,
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Đã ghi báo cáo: {output}")


if __name__ == "__main__":
    main()
