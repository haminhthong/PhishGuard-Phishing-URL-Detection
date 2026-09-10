"""Đánh giá locked test và hard slices từ artifact production hiện tại."""

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
SPLITS_DIR = ARTIFACTS_DIR / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_split(name: str) -> pd.DataFrame:
    parquet_path = SPLITS_DIR / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(SPLITS_DIR / f"{name}.csv.gz")


def score_frame(
    frame: pd.DataFrame, model: XGBClassifier, calibrator: ProbabilityCalibrator
) -> tuple[pd.DataFrame, pd.Series]:
    url_column = "raw_url" if "raw_url" in frame.columns else "url"
    features = pd.DataFrame(
        [FeatureExtractor().extract(url) for url in frame[url_column]],
        columns=FEATURE_COLUMNS,
    )
    scores = pd.Series(calibrator.calibrate(model.predict_proba(features)[:, 1]))
    return features, scores


def main(release_dir: Path | None = None) -> None:
    artifact_dir = (release_dir or ARTIFACTS_DIR).resolve()
    model = XGBClassifier()
    model.load_model(artifact_dir / "model.json")
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((artifact_dir / "calibration.json").read_text(encoding="utf-8"))
    thresholds = json.loads((artifact_dir / "thresholds.json").read_text(encoding="utf-8"))
    calibrator = ProbabilityCalibrator(
        method=str(calibration["method"]),
        params=dict(calibration["calibrator_params"]),
    )
    policy = ActionPolicy.from_dict(thresholds)
    test = read_split("test")
    _, scores = score_frame(test, model, calibrator)
    predictions = (scores >= policy.block_threshold).astype(int)
    report = {
        "model_version": metadata["model_version"],
        "feature_contract": metadata["feature_contract"],
        "locked_test": classification_metrics(test["label"], predictions, scores),
    }
    for name in ("hard_dev", "hard_locked_test"):
        path = PROJECT_ROOT / "evaluation" / f"{name}.jsonl"
        if not path.exists():
            continue
        frame = pd.DataFrame(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
        _, slice_scores = score_frame(frame, model, calibrator)
        frame["prediction"] = (slice_scores >= policy.block_threshold).astype(int)
        report[name] = {
            "metrics": classification_metrics(frame["label"], frame["prediction"], slice_scores),
            "slices": {
                str(slice_name): classification_metrics(
                    group["label"], group["prediction"], slice_scores[group.index]
                )
                for slice_name, group in frame.groupby("slice")
            },
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Đã ghi báo cáo: {output}")


if __name__ == "__main__":
    main()
