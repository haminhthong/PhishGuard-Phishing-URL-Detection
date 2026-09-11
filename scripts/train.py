"""Huấn luyện XGBoost, hiệu chuẩn xác suất và ghi artifact runtime."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
import yaml
from xgboost import XGBClassifier

from phishguard.calibration import (
    CalibrationArtifact,
    DecisionThresholds,
    ProbabilityCalibrator,
    select_decision_thresholds,
)
from phishguard.features import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION, FeatureExtractor
from phishguard.training.evaluation import classification_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"


def load_split(name: str) -> pd.DataFrame:
    """Đọc một split Parquet hoặc CSV fallback."""
    parquet_path = SPLITS_DIR / f"{name}.parquet"
    csv_path = SPLITS_DIR / f"{name}.csv.gz"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Không tìm thấy split {name} trong {SPLITS_DIR}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_model(model_config: dict[str, object], seed: int) -> XGBClassifier:
    """Tạo cùng một cấu hình XGBoost cho bước chọn và bước refit cuối."""
    return XGBClassifier(
        n_estimators=int(model_config.get("n_estimators", 300)),
        max_depth=int(model_config.get("max_depth", 6)),
        learning_rate=float(model_config.get("learning_rate", 0.08)),
        eval_metric=str(model_config.get("eval_metric", "logloss")),
        random_state=seed,
        n_jobs=int(model_config.get("n_jobs", 4)),
    )


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if config.get("feature_contract", FEATURE_CONTRACT_VERSION) != FEATURE_CONTRACT_VERSION:
        raise ValueError(f"Chỉ hỗ trợ feature contract {FEATURE_CONTRACT_VERSION}")

    seed = int(config.get("random_seed", 42))
    model_version = str(config.get("model_version", "4.0.0"))
    extractor = FeatureExtractor()
    frames = {
        name: load_split(name)
        for name in ("train", "validation", "calibration", "threshold_validation")
    }
    features = {name: extractor.extract_frame(frame) for name, frame in frames.items()}
    labels = {name: frame["label"].to_numpy() for name, frame in frames.items()}

    model_config = config.get("model", {})
    validation_model = build_model(model_config, seed)
    validation_model.fit(features["train"], labels["train"])
    validation_scores = validation_model.predict_proba(features["validation"])[:, 1]
    validation_predictions = (validation_scores >= 0.5).astype(int)
    validation_metrics = classification_metrics(
        labels["validation"], validation_predictions, validation_scores
    )

    train_frame = pd.concat([features["train"], features["validation"]], ignore_index=True)
    train_labels = np.concatenate([labels["train"], labels["validation"]])
    model = build_model(model_config, seed)
    model.fit(train_frame, train_labels)

    calibration_scores = model.predict_proba(features["calibration"])[:, 1]
    calibrator = ProbabilityCalibrator(
        method=str(config.get("calibration", {}).get("method", "isotonic"))
    )
    calibrator.fit(calibration_scores, labels["calibration"])
    threshold_scores = calibrator.calibrate(
        model.predict_proba(features["threshold_validation"])[:, 1]
    )
    threshold_config = config.get("threshold_policy", {})
    selection = select_decision_thresholds(
        labels["threshold_validation"],
        threshold_scores,
        caution_max_fpr=float(threshold_config.get("caution_max_fpr", 0.02)),
        block_max_fpr=float(threshold_config.get("block_max_fpr", 0.005)),
        caution_min_recall=float(threshold_config.get("caution_min_recall", 0.90)),
        block_min_recall=float(threshold_config.get("block_min_recall", 0.80)),
    )
    thresholds = DecisionThresholds.from_dict(selection["thresholds"])
    calibration_eval = calibrator.evaluate_fit(calibration_scores, labels["calibration"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.json"
    model.save_model(model_path)
    calibration_payload = CalibrationArtifact(
        method=calibrator.method,
        ece_before=calibration_eval["ece_before"],
        ece_after=calibration_eval["ece_after"],
        brier_before=calibration_eval["brier_before"],
        brier_after=calibration_eval["brier_after"],
        calibrator_params=calibrator.params,
    ).to_dict()
    calibration_payload["model_version"] = model_version
    thresholds_payload = {**thresholds.to_dict(), "selection": selection}
    write_json(ARTIFACTS_DIR / "calibration.json", calibration_payload)
    write_json(ARTIFACTS_DIR / "thresholds.json", thresholds_payload)
    metadata = {
        "model_version": model_version,
        "model_type": "XGBClassifier",
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_count": len(FEATURE_COLUMNS),
        "validation_metrics": validation_metrics,
        "training_date": str(datetime.date.today()),
        "xgboost_version": xgboost.__version__,
    }
    write_json(ARTIFACTS_DIR / "metadata.json", metadata)
    write_json(
        ARTIFACTS_DIR / "training_report.json",
        {
            "model_version": model_version,
            "calibration": calibration_eval,
            "threshold_selection": selection,
            "validation_metrics": validation_metrics,
        },
    )
    print(f"[OK] Đã ghi artifact vào {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
