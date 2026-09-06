"""Dịch vụ tải mô hình XGBoost JSON an toàn, hỗ trợ Model Registry versioned và kiểm tra hợp đồng Metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from API.errors import PhishGuardAPIException
from phishguard.calibration import ProbabilityCalibrator
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_VERSION,
)


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class LoadedModel:
    model: XGBClassifier
    metadata: dict[str, Any]
    model_version: str
    feature_contract: str
    feature_count: int
    threshold: float
    risk_thresholds: dict[str, float]
    calibrator: ProbabilityCalibrator | None = None


def resolve_model_paths(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """
    Xác định đường dẫn mô hình, metadata và file calibration:
    1. Nếu người dùng / unit test truyền explicit model_path (khác default), sử dụng ngay.
    2. Ưu tiên Model Registry versioned (artifacts/models/production.json).
    3. Fallback sang file mặc định (API/XGB.json).
    """
    project_root = Path(__file__).resolve().parents[2]
    production_registry = project_root / "artifacts" / "models" / "production.json"
    default_model = project_root / "API" / "XGB.json"
    default_meta = project_root / "API" / "model_metadata.json"

    # Nếu người dùng hoặc unit test truyền một đường dẫn cụ thể (khác default API/XGB.json)
    if model_path is not None and model_path != default_model:
        meta_p = metadata_path if metadata_path else model_path.parent / "model_metadata.json"
        cal_p = model_path.parent / "calibration.json"
        return model_path, meta_p, (cal_p if cal_p.is_file() else None)

    # Kiểm tra Model Registry active pointer
    if production_registry.is_file():
        try:
            with open(production_registry, encoding="utf-8") as f:
                reg_data = json.load(f)
            model_dir = project_root / reg_data.get("model_dir", "")
            target_model = model_dir / "model.json"
            target_meta = model_dir / "metadata.json"
            target_cal = model_dir / "calibration.json"
            if target_model.is_file():
                return target_model, target_meta, (target_cal if target_cal.is_file() else None)
        except Exception:
            pass

    # Fallback mặc định
    cal_default = default_model.parent / "calibration.json"
    return default_model, default_meta, (cal_default if cal_default.is_file() else None)


def load_phishguard_model(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> LoadedModel:
    """Tải mô hình XGBoost Native JSON, xác minh metadata, checksum và calibrator trước khi khởi chạy API."""
    resolved_model_path, resolved_meta_path, resolved_cal_path = resolve_model_paths(
        model_path, metadata_path
    )

    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost JSON tại {resolved_model_path}")

    if resolved_model_path.suffix.lower() != ".json":
        raise PhishGuardAPIException(
            code="MODEL_CONTRACT_MISMATCH",
            message=f"API chỉ hỗ trợ định dạng mô hình XGBoost JSON chính thức; từ chối file {resolved_model_path.name}",
            status_code=500,
        )

    # 1. Load native XGBoost model
    model = XGBClassifier()
    try:
        model.load_model(resolved_model_path)
    except Exception as error:
        raise PhishGuardAPIException(
            code="MODEL_NOT_FOUND",
            message=f"Không thể đọc file mô hình JSON: {error!s}",
            status_code=500,
        ) from error

    # 2. Load metadata if available
    metadata = {}
    if resolved_meta_path and resolved_meta_path.is_file():
        try:
            with open(resolved_meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as error:
            raise PhishGuardAPIException(
                code="METADATA_READ_ERROR",
                message=f"Lỗi đọc file metadata JSON: {error!s}",
                status_code=500,
            ) from error

    # 3. Checksum verification if metadata specifies model_sha256
    expected_sha256 = metadata.get("model_sha256")
    if expected_sha256:
        actual_sha256 = compute_sha256(resolved_model_path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise PhishGuardAPIException(
                code="MODEL_INTEGRITY_ERROR",
                message="Mã băm SHA-256 của file mô hình không khớp với metadata bảo mật",
                status_code=500,
            )

    model_version = metadata.get("model_version", "3.2.0")
    feature_contract = metadata.get("feature_contract", FEATURE_CONTRACT_VERSION)
    threshold = float(metadata.get("threshold", 0.5))

    if not (0.0 <= threshold <= 1.0):
        raise RuntimeError(f"Threshold mô hình ({threshold}) không nằm trong khoảng hợp lệ [0.0, 1.0]")

    # 4. Contract & Feature Count validation
    if feature_contract not in {FEATURE_CONTRACT_V1, FEATURE_CONTRACT_V2}:
        raise RuntimeError(
            f"Hợp đồng đặc trưng mô hình ({feature_contract}) không nằm trong danh sách hỗ trợ: {[FEATURE_CONTRACT_V1, FEATURE_CONTRACT_V2]}"
        )

    expected_cols = FEATURE_COLUMNS_V2 if feature_contract == FEATURE_CONTRACT_V2 else FEATURE_COLUMNS_V1
    feature_count = getattr(model, "n_features_in_", len(expected_cols))
    if len(expected_cols) != feature_count:
        raise RuntimeError(
            f"Mô hình yêu cầu {feature_count} đặc trưng, hợp đồng {feature_contract} cung cấp {len(expected_cols)}"
        )

    # 5. Risk Policy Thresholds
    risk_thresholds = metadata.get("risk_thresholds", {"high": 0.75, "medium": 0.45})

    # 6. Load Calibrator
    calibrator = None
    if resolved_cal_path and resolved_cal_path.is_file():
        try:
            with open(resolved_cal_path, encoding="utf-8") as f:
                cal_data = json.load(f)
            calibrator = ProbabilityCalibrator(
                method=cal_data.get("method", "isotonic"),
                params=cal_data.get("calibrator_params", {}),
            )
        except Exception:
            pass

    return LoadedModel(
        model=model,
        metadata=metadata,
        model_version=model_version,
        feature_contract=feature_contract,
        feature_count=feature_count,
        threshold=threshold,
        risk_thresholds=risk_thresholds,
        calibrator=calibrator,
    )
