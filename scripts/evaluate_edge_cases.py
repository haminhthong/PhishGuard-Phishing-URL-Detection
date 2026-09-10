"""Đánh giá các URL biên đã được kiểm duyệt, không tạo release gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from phishguard.calibration import ActionPolicy, ProbabilityCalibrator
from phishguard.features import FEATURE_COLUMNS, FeatureExtractor
from phishguard.training.evaluation import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CASES_PATH = PROJECT_ROOT / "evaluation" / "hard_locked_test.jsonl"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    if not CASES_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy tập edge case: {CASES_PATH}")
    model = XGBClassifier()
    model.load_model(ARTIFACTS_DIR / "model.json")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((ARTIFACTS_DIR / "calibration.json").read_text(encoding="utf-8"))
    policy = ActionPolicy.from_dict(
        json.loads((ARTIFACTS_DIR / "thresholds.json").read_text(encoding="utf-8"))
    )
    calibrator = ProbabilityCalibrator(
        method=str(calibration["method"]),
        params=dict(calibration["calibrator_params"]),
    )
    rows = [
        json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line
    ]
    frame = pd.DataFrame(rows)
    url_column = "raw_url" if "raw_url" in frame.columns else "url"
    features = pd.DataFrame(
        [FeatureExtractor().extract(url) for url in frame[url_column]],
        columns=FEATURE_COLUMNS,
    )
    scores = calibrator.calibrate(model.predict_proba(features)[:, 1])
    labels = frame["label"].to_numpy() if "label" in frame else None
    report: dict[str, object] = {
        "model_version": metadata["model_version"],
        "cases": len(frame),
        "block_threshold": policy.block_threshold,
        "block_count": int((scores >= policy.block_threshold).sum()),
    }
    if labels is not None:
        report["metrics"] = classification_metrics(
            labels,
            (scores >= policy.block_threshold).astype(int),
            scores,
        )
    output_path = ARTIFACTS_DIR / "edge_case_metrics.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Đã ghi báo cáo edge case: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
