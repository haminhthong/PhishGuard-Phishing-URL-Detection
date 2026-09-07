"""Script đóng gói và xuất mô hình JSON kèm metadata an toàn và checksum cho API backend."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml
from xgboost import XGBClassifier

from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_V3,
)
from phishguard.training.data import compute_sha256

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PROD_JSON = MODELS_DIR / "production.json"
API_DIR = PROJECT_ROOT / "API"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"


def compute_feature_contract_hash(features: tuple[str, ...]) -> str:
    """Tính mã băm từ danh sách các tên đặc trưng theo đúng thứ tự."""
    raw = ",".join(features).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    print("=" * 65)
    print(" 📦 Bắt đầu Đóng gói & Xác thực Model Registry Artifacts...")
    print("=" * 65)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    promotion_policy = config.get("promotion_policy", {})
    val_pol = promotion_policy.get("validation", {})
    serv_pol = promotion_policy.get("serving", {})

    # 1. Định vị model directory từ Model Registry
    if PROD_JSON.exists():
        with open(PROD_JSON, encoding="utf-8") as f:
            pointer = json.load(f)
        active_version = pointer.get("active_version", "phishguard-3.2.0")
        model_dir = PROJECT_ROOT / pointer.get("model_dir", f"artifacts/models/{active_version}")
    else:
        active_version = "phishguard-3.2.0"
        model_dir = MODELS_DIR / active_version
        if not model_dir.exists():
            model_dir = ARTIFACTS_DIR

    model_file = model_dir / "model.json"
    if not model_file.exists():
        model_file = model_dir / "XGB.json"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại {model_file}. Vui lòng chạy scripts/train.py trước."
        )

    # 2. Nạp mô hình kiểm tra tính toàn vẹn
    model = XGBClassifier()
    model.load_model(model_file)
    model_sha256 = compute_sha256(model_file)
    print(f" [OK] Loaded Native XGBoost JSON model: {model_file} (SHA256: {model_sha256[:16]}...)")

    # 3. Nạp thông tin metadata và contract
    meta_path = model_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    feature_contract = metadata.get("feature_contract", FEATURE_CONTRACT_V2)
    cols = {
        "lexical-v1": FEATURE_COLUMNS_V1,
        FEATURE_CONTRACT_V2: FEATURE_COLUMNS_V2,
        FEATURE_CONTRACT_V3: FEATURE_COLUMNS_V3,
    }[feature_contract]
    contract_hash = compute_feature_contract_hash(cols)
    metadata["feature_contract_hash"] = contract_hash

    # 4. Kiểm tra Quality Gate từ centralized promotion policy
    test_report_path = ARTIFACTS_DIR / "test_evaluation_report.json"
    test_metrics = {}
    if test_report_path.exists():
        with open(test_report_path, encoding="utf-8") as f:
            test_report = json.load(f)
            test_metrics = test_report.get("metrics", {})

    pr_auc = test_metrics.get("pr_auc", metadata.get("validation_metrics", {}).get("pr_auc", 0.0))
    fpr = test_metrics.get(
        "false_positive_rate",
        metadata.get("validation_metrics", {}).get("false_positive_rate", 1.0),
    )
    p95_lat = test_metrics.get(
        "p95_latency_ms", metadata.get("validation_metrics", {}).get("p95_latency_ms", 999.0)
    )

    min_pr = val_pol.get("min_pr_auc", 0.85)
    max_fpr = val_pol.get("max_fpr", 0.01)
    max_lat = serv_pol.get("max_model_p95_ms", 5.0)

    quality_gate_passed = bool(pr_auc >= min_pr and fpr <= max_fpr and p95_lat <= max_lat)

    print(f" [OK] Quality Gate: {'PASSED ✅' if quality_gate_passed else 'FAILED ❌'}")
    print(f"   • PR-AUC:  {pr_auc:.4f} (Target >= {min_pr})")
    print(f"   • FPR:     {fpr * 100:.2f}% (Target <= {max_fpr * 100:.1f}%)")
    print(f"   • p95 Lat: {p95_lat:.2f}ms (Target <= {max_lat}ms)")

    # 5. Xuất bản sao tương thích ngược sang API/
    API_DIR.mkdir(parents=True, exist_ok=True)
    target_api_xgb = API_DIR / "XGB.json"
    target_api_meta = API_DIR / "model_metadata.json"

    shutil.copy(model_file, target_api_xgb)
    metadata["model_sha256"] = model_sha256
    metadata["quality_gate"] = {
        "passed": quality_gate_passed,
        "rules": [
            f"pr_auc >= {min_pr}",
            f"false_positive_rate <= {max_fpr}",
            f"p95_latency_ms <= {max_lat}ms",
        ],
    }

    with open(target_api_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f" [OK] Đã xuất bản sao tương thích sang: {target_api_xgb}")
    print(f" [OK] Đã xuất metadata tương thích sang: {target_api_meta}")


if __name__ == "__main__":
    main()
