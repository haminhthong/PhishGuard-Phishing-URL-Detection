"""Huấn luyện XGBoost, hiệu chuẩn xác suất và ghi artifact runtime."""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
import yaml
from xgboost import XGBClassifier

from phishguard.calibration import (
    ActionPolicy,
    CalibrationArtifact,
    ProbabilityCalibrator,
    select_action_policy,
)
from phishguard.features import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION, FeatureExtractor
from phishguard.features.resources import RESOURCE_BUNDLE
from phishguard.training.data import compute_sha256
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


def feature_contract_hash() -> str:
    return hashlib.sha256(",".join(FEATURE_COLUMNS).encode("utf-8")).hexdigest()


def build_features(frame: pd.DataFrame, extractor: FeatureExtractor) -> pd.DataFrame:
    """Luôn lấy feature từ raw_url, không lấy từ canonical_url."""
    url_column = "raw_url" if "raw_url" in frame.columns else "url"
    return pd.DataFrame(
        [extractor.extract(url) for url in frame[url_column]],
        columns=FEATURE_COLUMNS,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    features = {name: build_features(frame, extractor) for name, frame in frames.items()}
    labels = {name: frame["label"].to_numpy() for name, frame in frames.items()}

    model_config = config.get("model", {})
    model = XGBClassifier(
        n_estimators=int(model_config.get("n_estimators", 300)),
        max_depth=int(model_config.get("max_depth", 6)),
        learning_rate=float(model_config.get("learning_rate", 0.08)),
        eval_metric=str(model_config.get("eval_metric", "logloss")),
        random_state=seed,
        n_jobs=-1,
    )
    train_frame = pd.concat([features["train"], features["validation"]], ignore_index=True)
    train_labels = np.concatenate([labels["train"], labels["validation"]])
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
    selection = select_action_policy(
        labels["threshold_validation"],
        threshold_scores,
        caution_max_fpr=float(threshold_config.get("caution_max_fpr", 0.02)),
        block_max_fpr=float(threshold_config.get("block_max_fpr", 0.005)),
        caution_min_recall=float(threshold_config.get("caution_min_recall", 0.90)),
        block_min_recall=float(threshold_config.get("block_min_recall", 0.80)),
        policy_version=str(threshold_config.get("policy_version", "browser-risk-v1")),
    )
    policy = ActionPolicy.from_dict(selection["policy"])
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
    policy_payload = {**policy.to_dict(), "selection": selection}
    write_json(ARTIFACTS_DIR / "calibration.json", calibration_payload)
    write_json(ARTIFACTS_DIR / "thresholds.json", policy_payload)
    write_json(
        ARTIFACTS_DIR / "feature_contract.json",
        {
            "contract": FEATURE_CONTRACT_VERSION,
            "feature_count": len(FEATURE_COLUMNS),
            "features": list(FEATURE_COLUMNS),
            "contract_hash": feature_contract_hash(),
        },
    )
    validation_scores = calibrator.calibrate(model.predict_proba(features["validation"])[:, 1])
    validation_predictions = (validation_scores >= policy.caution_threshold).astype(int)
    validation_metrics = classification_metrics(
        labels["validation"], validation_predictions, validation_scores
    )
    metadata = {
        "artifact_schema_version": "5.0.0",
        "model_version": model_version,
        "model_type": "XGBClassifier",
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_contract_hash": feature_contract_hash(),
        "policy_version": policy.policy_version,
        "validation_metrics": validation_metrics,
        "training_date": str(datetime.date.today()),
        "model_sha256": compute_sha256(model_path),
        "calibration_sha256": compute_sha256(ARTIFACTS_DIR / "calibration.json"),
        "thresholds_sha256": compute_sha256(ARTIFACTS_DIR / "thresholds.json"),
        "resource_hashes": RESOURCE_BUNDLE.hashes,
        "resource_versions": RESOURCE_BUNDLE.versions,
        "tld_library_version": RESOURCE_BUNDLE.tld_library_version,
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
    # Giữ một bundle candidate riêng cho các gate cuối; API chỉ đọc artifacts/.
    candidate_dir = PROJECT_ROOT / "releases" / "candidates" / f"phishguard-{model_version}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.json", "metadata.json", "calibration.json", "thresholds.json"):
        shutil.copy2(ARTIFACTS_DIR / name, candidate_dir / name)
    print(f"[OK] Đã ghi artifact vào {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
