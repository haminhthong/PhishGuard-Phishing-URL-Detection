"""Security stress gate cho artifact XGBoost đã đóng gói."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from phishguard.calibration import ActionPolicy, ProbabilityCalibrator
from phishguard.features import FEATURE_COLUMNS, FeatureExtractor
from phishguard.training.evaluation import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "hard_locked_test.jsonl"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(release_dir: Path | None = None, cases: Path = DEFAULT_CASES) -> None:
    """Đánh giá block recall và benign block-rate; không tự sửa artifact."""
    if release_dir is None:
        parser = argparse.ArgumentParser(description="Chạy security stress gate")
        parser.add_argument("--release-dir", type=Path, required=True)
        parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
        args = parser.parse_args()
        release_dir, cases = args.release_dir, args.cases
    metadata = json.loads((release_dir / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((release_dir / "calibration.json").read_text(encoding="utf-8"))
    policy = ActionPolicy.from_dict(
        json.loads((release_dir / "thresholds.json").read_text(encoding="utf-8"))
    )
    calibrator = ProbabilityCalibrator(
        method=str(calibration["method"]),
        params=dict(calibration["calibrator_params"]),
    )
    model = XGBClassifier()
    model.load_model(release_dir / "model.json")
    rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows)
    features = pd.DataFrame(
        [FeatureExtractor().extract(url) for url in frame["url"]],
        columns=FEATURE_COLUMNS,
    )
    scores = calibrator.calibrate(model.predict_proba(features)[:, 1])
    labels = frame["label"].to_numpy(dtype=int)
    predictions = np.asarray(
        [policy.evaluate(float(score))[1] == "block" for score in scores], dtype=int
    )
    metrics = classification_metrics(labels, predictions, scores)
    benign = labels == 0
    benign_block_rate = float(predictions[benign].mean()) if benign.any() else 0.0
    settings = (yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}).get(
        "security_stress", {}
    )
    min_recall = float(settings.get("min_block_recall", 0.80))
    max_benign_rate = float(settings.get("max_benign_block_rate", 0.20))
    passed = metrics["recall"] >= min_recall and benign_block_rate <= max_benign_rate
    report = {
        "model_version": metadata["model_version"],
        "passed": passed,
        "metrics": metrics,
        "gate_metrics": {"block_recall": metrics["recall"], "benign_block_rate": benign_block_rate},
        "gates": {"min_block_recall": min_recall, "max_benign_block_rate": max_benign_rate},
    }
    (release_dir / "security_stress_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("Security stress gate không đạt")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
