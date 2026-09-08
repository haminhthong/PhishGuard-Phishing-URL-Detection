"""Chạy regression benchmark curated cho candidate release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from phishguard.calibration import ActionPolicy, ProbabilityCalibrator
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    FEATURE_COLUMNS_V4,
    FeatureExtractor,
)
from phishguard.features.resources import load_resource_bundle
from phishguard.training.evaluation import classification_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "hard_locked_test.jsonl"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(release_dir: Path | None = None, cases: Path = DEFAULT_CASES) -> None:
    if release_dir is None:
        parser = argparse.ArgumentParser(
            description="Đánh giá security stress benchmark cho candidate."
        )
        parser.add_argument("--release-dir", type=Path, required=True)
        parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
        args = parser.parse_args()
        release_dir = args.release_dir
        cases = args.cases
    release_dir = release_dir.resolve()
    metadata = read_json(release_dir / "metadata.json")
    policy = ActionPolicy.from_dict(read_json(release_dir / "action_policy.json"))
    calibration_data = read_json(release_dir / "calibration.json")
    calibrator = ProbabilityCalibrator(
        method=calibration_data["method"],
        params=calibration_data["calibrator_params"],
    )
    model = XGBClassifier()
    model.load_model(release_dir / "model.json")

    columns = {
        "lexical-v1": FEATURE_COLUMNS_V1,
        "lexical-v2": FEATURE_COLUMNS_V2,
        "lexical-v3": FEATURE_COLUMNS_V3,
        "lexical-v4": FEATURE_COLUMNS_V4,
    }[metadata["feature_contract"]]
    resource_dir = release_dir / "resources"
    if not resource_dir.is_dir():
        raise FileNotFoundError(f"Candidate thiếu resource bundle: {resource_dir}")
    resources = load_resource_bundle(resource_dir)
    extractor = FeatureExtractor(metadata["feature_contract"], resources)
    rows = [
        json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"Security stress case file rỗng: {cases}")
    if any("url" not in row or "label" not in row for row in rows):
        raise ValueError("Mỗi security stress case phải có url và label")
    frame = pd.DataFrame([extractor.extract(row["url"]) for row in rows], columns=columns)
    scores = calibrator.calibrate(model.predict_proba(frame)[:, 1])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    actions = np.asarray([policy.evaluate(float(score))[1] for score in scores])
    block_predictions = (actions == "block").astype(int)
    metrics = classification_metrics(labels, block_predictions, scores)
    benign_mask = labels == 0
    benign_block_rate = float(block_predictions[benign_mask].mean()) if benign_mask.any() else 0.0
    with CONFIG_PATH.open(encoding="utf-8") as file:
        stress_config = (yaml.safe_load(file) or {}).get("security_stress", {})
    min_block_recall = float(stress_config.get("min_block_recall", 0.80))
    max_benign_block_rate = float(stress_config.get("max_benign_block_rate", 0.20))
    passed = bool(
        len(rows) > 0
        and metrics.get("recall", 0.0) >= min_block_recall
        and benign_block_rate <= max_benign_block_rate
    )
    by_slice = {}
    for slice_name in sorted({row.get("slice", "unknown") for row in rows}):
        mask = np.asarray([row.get("slice", "unknown") == slice_name for row in rows])
        by_slice[slice_name] = {
            "rows": int(mask.sum()),
            "metrics": classification_metrics(labels[mask], block_predictions[mask], scores[mask]),
        }

    report = {
        "release": metadata["release_id"],
        "passed": passed,
        "reason": (
            "Curated cases đạt block recall và benign block-rate gate."
            if passed
            else "Curated cases không đạt security stress gate."
        ),
        "model_sha256": sha256_file(release_dir / "model.json"),
        "calibration_sha256": sha256_file(release_dir / "calibration.json"),
        "action_policy_sha256": sha256_file(release_dir / "action_policy.json"),
        "metrics": metrics,
        "gate_metrics": {
            "block_recall": metrics.get("recall", 0.0),
            "benign_block_rate": benign_block_rate,
        },
        "gates": {
            "min_block_recall": min_block_recall,
            "max_benign_block_rate": max_benign_block_rate,
        },
        "by_slice": by_slice,
    }
    (release_dir / "security_stress_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
