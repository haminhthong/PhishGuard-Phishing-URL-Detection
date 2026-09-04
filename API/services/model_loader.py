"""Dịch vụ tải mô hình XGBoost JSON an toàn và kiểm tra hợp đồng Metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from API.errors import PhishGuardAPIException
from phishguard.features import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION


@dataclass
class LoadedModel:
    model: XGBClassifier
    metadata: dict[str, Any]
    model_version: str
    feature_contract: str
    feature_count: int
    threshold: float


def load_phishguard_model(model_path: Path, metadata_path: Path) -> LoadedModel:
    """Tải mô hình XGBoost Native JSON và xác minh metadata trước khi khởi chạy API."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost JSON tại {model_path}")

    if model_path.suffix.lower() != ".json":
        raise PhishGuardAPIException(
            code="MODEL_CONTRACT_MISMATCH",
            message=f"API chỉ hỗ trợ định dạng mô hình XGBoost JSON chính thức; từ chối file {model_path.name}",
            status_code=500,
        )

    # 1. Load native XGBoost model
    model = XGBClassifier()
    try:
        model.load_model(model_path)
    except Exception as error:
        raise PhishGuardAPIException(
            code="MODEL_NOT_FOUND",
            message=f"Không thể đọc file mô hình JSON: {error!s}",
            status_code=500,
        ) from error

    # 2. Load metadata if available
    metadata = {}
    if metadata_path.is_file():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    model_version = metadata.get("model_version", "3.0.0")
    feature_contract = metadata.get("feature_contract", FEATURE_CONTRACT_VERSION)
    threshold = float(metadata.get("threshold", 0.5))

    # 3. Contract & Feature Count validation
    if feature_contract != FEATURE_CONTRACT_VERSION:
        raise RuntimeError(
            f"Hợp đồng đặc trưng mô hình ({feature_contract}) không khớp với API ({FEATURE_CONTRACT_VERSION})"
        )

    feature_count = getattr(model, "n_features_in_", len(FEATURE_COLUMNS))
    if len(FEATURE_COLUMNS) != feature_count:
        raise RuntimeError(
            f"Mô hình yêu cầu {feature_count} đặc trưng, API cung cấp {len(FEATURE_COLUMNS)}"
        )

    return LoadedModel(
        model=model,
        metadata=metadata,
        model_version=model_version,
        feature_contract=feature_contract,
        feature_count=feature_count,
        threshold=threshold,
    )
