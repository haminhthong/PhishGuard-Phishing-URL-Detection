"""Script đóng gói và xuất mô hình JSON kèm metadata an toàn và checksum cho API backend."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import xgboost
from xgboost import XGBClassifier

from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_CONTRACT_V2,
)
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


def compute_feature_contract_hash(features: tuple[str, ...]) -> str:
    """Tính mã băm tính toán từ danh sách các tên đặc trưng theo đúng thứ tự."""
    raw = ",".join(features).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    print("=" * 65)
    print(" 📦 Bắt đầu Đóng gói & Xuất Mô hình Native XGBoost JSON v2...")
    print("=" * 65)

    if not SOURCE_XGB_JSON.exists():
        raise FileNotFoundError(f"Không tìm thấy file {SOURCE_XGB_JSON}. Vui lòng chạy scripts/train.py trước.")

    # Nạp mô hình để kiểm tra tính toàn vẹn
    model = XGBClassifier()
    model.load_model(SOURCE_XGB_JSON)

    API_DIR.mkdir(parents=True, exist_ok=True)

    # Xuất file mô hình JSON sang thư mục API
    model.save_model(TARGET_API_XGB_JSON)
    model_sha256 = compute_sha256(TARGET_API_XGB_JSON)
    print(f" [OK] Exported XGBoost JSON model: {TARGET_API_XGB_JSON} (SHA256: {model_sha256[:16]}...)")

    # Đọc checksum dữ liệu nguồn
    legit_checksum = None
    phish_checksum = None
    if DATA_REPORT_JSON.exists():
        with open(DATA_REPORT_JSON, encoding="utf-8") as f:
            data_report = json.load(f)
            legit_checksum = data_report.get("legitimate_sha256")
            phish_checksum = data_report.get("phishing_sha256")

    # Đọc thông tin từ validation summary
    feature_contract = FEATURE_CONTRACT_V2
    threshold = 0.5
    th_policy_info = {}
    if VAL_SUMMARY_JSON.exists():
        with open(VAL_SUMMARY_JSON, encoding="utf-8") as f:
            val_summary = json.load(f)
            feature_contract = val_summary.get("feature_contract", FEATURE_CONTRACT_V2)
            threshold = val_summary.get("operating_threshold", val_summary.get("optimal_threshold", 0.5))
            th_policy_info = val_summary.get("threshold_policy", {})

    cols = FEATURE_COLUMNS_V2 if feature_contract == FEATURE_CONTRACT_V2 else FEATURE_COLUMNS_V1
    contract_hash = compute_feature_contract_hash(cols)

    test_metrics = {}
    if TEST_REPORT_JSON.exists():
        with open(TEST_REPORT_JSON, encoding="utf-8") as f:
            test_report = json.load(f)
            test_metrics = test_report.get("metrics", {})

    # Quality Gate verification
    pr_auc = test_metrics.get("pr_auc", 0.0)
    fpr = test_metrics.get("false_positive_rate", 1.0)
    p95_lat = test_metrics.get("p95_latency_ms", 999.0)
    quality_gate_passed = bool(pr_auc >= 0.85 and fpr <= 0.01 and p95_lat <= 5.0)

    metadata = {
        "artifact_schema_version": "2.0.0",
        "model_version": "3.1.0",
        "model_type": "XGBClassifier",
        "feature_contract": feature_contract,
        "feature_count": len(cols),
        "feature_names": list(cols),
        "feature_contract_hash": contract_hash,
        "threshold": threshold,
        "risk_thresholds": {
            "high": 0.75,
            "medium": 0.45,
        },
        "operating_policy": {
            "mode": th_policy_info.get("operating_mode", "constrained_recall"),
            "target_fpr": th_policy_info.get("constrained_max_fpr", 0.005),
            "cost_assumptions": th_policy_info.get("cost_assumptions", {"cost_fn": 10.0, "cost_fp": 1.0}),
        },
        "training_date": str(datetime.date.today()),
        "model_sha256": model_sha256,
        "xgboost_version": xgboost.__version__,
        "dataset_checksums": {
            "legitimate": legit_checksum,
            "phishing": phish_checksum,
        },
        "calibration": {
            "brier_score": test_metrics.get("brier_score", 0.0),
            "expected_calibration_error": test_metrics.get("expected_calibration_error", 0.0),
        },
        "test_metrics": {
            "pr_auc": test_metrics.get("pr_auc", 0.0),
            "roc_auc": test_metrics.get("roc_auc", 0.0),
            "recall": test_metrics.get("recall", 0.0),
            "precision": test_metrics.get("precision", 0.0),
            "f1": test_metrics.get("f1", 0.0),
            "false_positive_rate": test_metrics.get("false_positive_rate", 0.0),
            "false_negative_rate": test_metrics.get("false_negative_rate", 0.0),
            "p50_latency_ms": test_metrics.get("p50_latency_ms", 0.0),
            "p95_latency_ms": test_metrics.get("p95_latency_ms", 0.0),
        },
        "quality_gate": {
            "passed": quality_gate_passed,
            "rules": [
                "pr_auc >= 0.85",
                "false_positive_rate <= 0.01",
                "p95_latency_ms <= 5.0ms",
            ],
        },
    }

    with open(TARGET_API_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f" [OK] Exported Metadata JSON: {TARGET_API_METADATA_JSON}")
    print(f" [OK] Quality Gate: {'PASSED ✅' if quality_gate_passed else 'FAILED ❌'}")


if __name__ == "__main__":
    main()
