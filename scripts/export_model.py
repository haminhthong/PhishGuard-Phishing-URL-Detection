"""Script đóng gói và xuất mô hình JSON kèm metadata an toàn cho API backend."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

from xgboost import XGBClassifier

from phishguard.features import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION
from phishguard.training.data import compute_sha256

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
API_DIR = PROJECT_ROOT / "API"
SOURCE_XGB_JSON = ARTIFACTS_DIR / "XGB.json"
TARGET_API_XGB_JSON = API_DIR / "XGB.json"
TARGET_API_METADATA_JSON = API_DIR / "model_metadata.json"
DATA_REPORT_JSON = ARTIFACTS_DIR / "data_quality_report.json"
TEST_REPORT_JSON = ARTIFACTS_DIR / "test_evaluation_report.json"
VAL_SUMMARY_JSON = ARTIFACTS_DIR / "validation_summary.json"


def main() -> None:
    print("=" * 65)
    print(" 📦 Bắt đầu Đóng gói & Xuất Mô hình XGBoost Native JSON...")
    print("=" * 65)

    if not SOURCE_XGB_JSON.exists():
        raise FileNotFoundError(f"Không tìm thấy file {SOURCE_XGB_JSON}. Vui lòng chạy scripts/train.py trước.")

    # Load model to verify JSON integrity
    model = XGBClassifier()
    model.load_model(SOURCE_XGB_JSON)

    API_DIR.mkdir(parents=True, exist_ok=True)

    # Copy / Save JSON model to API directory
    model.save_model(TARGET_API_XGB_JSON)
    model_sha256 = compute_sha256(TARGET_API_XGB_JSON)
    print(f" [OK] Exported XGBoost JSON model: {TARGET_API_XGB_JSON} (SHA256: {model_sha256[:16]}...)")

    # Read data quality checksums
    legit_checksum = None
    phish_checksum = None
    if DATA_REPORT_JSON.exists():
        with open(DATA_REPORT_JSON, encoding="utf-8") as f:
            data_report = json.load(f)
            legit_checksum = data_report.get("legitimate_sha256")
            phish_checksum = data_report.get("phishing_sha256")

    # Read threshold and test metrics
    threshold = 0.5
    if VAL_SUMMARY_JSON.exists():
        with open(VAL_SUMMARY_JSON, encoding="utf-8") as f:
            val_summary = json.load(f)
            threshold = val_summary.get("optimal_threshold", 0.5)

    test_metrics = {}
    if TEST_REPORT_JSON.exists():
        with open(TEST_REPORT_JSON, encoding="utf-8") as f:
            test_report = json.load(f)
            test_metrics = test_report.get("metrics", {})

    metadata = {
        "model_version": "3.0.0",
        "model_type": "XGBClassifier",
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "feature_count": len(FEATURE_COLUMNS),
        "threshold": threshold,
        "training_seed": 42,
        "training_date": str(datetime.date.today()),
        "model_sha256": model_sha256,
        "dataset_checksums": {
            "legitimate": legit_checksum,
            "phishing": phish_checksum,
        },
        "test_metrics": {
            "pr_auc": test_metrics.get("pr_auc", 0.0),
            "recall": test_metrics.get("recall", 0.0),
            "precision": test_metrics.get("precision", 0.0),
            "f1": test_metrics.get("f1", 0.0),
            "false_positive_rate": test_metrics.get("false_positive_rate", 0.0),
            "false_negative_rate": test_metrics.get("false_negative_rate", 0.0),
            "p95_latency_ms": test_metrics.get("p95_latency_ms", 0.0),
        },
    }

    with open(TARGET_API_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save copy to artifacts as well
    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f" [OK] Exported Model Metadata JSON: {TARGET_API_METADATA_JSON}")
    print("=" * 65)
    print(" 🎉 XUẤT MÔ HÌNH VÀ METADATA THÀNH CÔNG!")
    print("=" * 65)


if __name__ == "__main__":
    main()
