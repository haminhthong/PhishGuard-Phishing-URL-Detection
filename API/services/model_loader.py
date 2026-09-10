"""Nạp các artifact cần thiết cho API chạy local."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from API.errors import PhishGuardAPIException
from phishguard.calibration import DecisionThresholds, ProbabilityCalibrator
from phishguard.features import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION, FeatureExtractor
from phishguard.features.resources import RESOURCES_DIR, ResourceBundle, load_resource_bundle


@dataclass
class LoadedModel:
    """Model và các thành phần dùng chung giữa mọi request."""

    model: XGBClassifier
    metadata: dict[str, Any]
    model_version: str
    feature_contract: str
    feature_count: int
    calibrator: ProbabilityCalibrator
    thresholds: DecisionThresholds
    resource_bundle: ResourceBundle
    feature_extractor: FeatureExtractor


def resolve_model_paths(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Xác định model, metadata, calibration và thresholds trong artifacts/."""
    artifact_dir = Path(__file__).resolve().parents[2] / "artifacts"
    resolved_model = model_path or artifact_dir / "model.json"
    resolved_metadata = metadata_path or resolved_model.parent / "metadata.json"
    return (
        resolved_model,
        resolved_metadata,
        resolved_model.parent / "calibration.json",
        resolved_model.parent / "thresholds.json",
    )


def _read_json(path: Path, code: str) -> dict[str, Any]:
    if not path.is_file():
        raise PhishGuardAPIException(
            code="ARTIFACT_MISSING",
            message=f"Thiếu artifact bắt buộc: {path.name}",
            status_code=503,
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhishGuardAPIException(
            code=code,
            message=f"Không thể đọc {path.name}: {error}",
            status_code=503,
        ) from error
    if not isinstance(data, dict):
        raise PhishGuardAPIException(
            code=code,
            message=f"{path.name} phải là JSON object",
            status_code=503,
        )
    return data


def load_phishguard_model(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> LoadedModel:
    """Nạp model và kiểm tra các điều kiện tối thiểu trước khi phục vụ."""
    resolved_model, resolved_meta, resolved_calibration, resolved_thresholds = resolve_model_paths(
        model_path, metadata_path
    )
    if not resolved_model.is_file():
        raise PhishGuardAPIException(
            code="ARTIFACT_MISSING",
            message=f"Không tìm thấy model tại {resolved_model}",
            status_code=503,
        )
    if resolved_model.suffix.lower() != ".json":
        raise PhishGuardAPIException(
            code="MODEL_CONTRACT_MISMATCH",
            message="API chỉ hỗ trợ artifact XGBoost Native JSON",
            status_code=503,
        )

    metadata = _read_json(resolved_meta, "METADATA_READ_ERROR")
    try:
        model = XGBClassifier()
        model.load_model(resolved_model)
    except Exception as error:
        raise PhishGuardAPIException(
            code="MODEL_LOAD_ERROR",
            message=f"Không thể nạp model JSON: {error}",
            status_code=503,
        ) from error

    model_version = str(metadata.get("model_version", "")).strip()
    feature_contract = str(metadata.get("feature_contract", ""))
    feature_count = int(getattr(model, "n_features_in_", 0))
    if not model_version:
        raise PhishGuardAPIException(
            code="ARTIFACT_METADATA_ERROR",
            message="Metadata thiếu model_version",
            status_code=503,
        )
    if feature_contract != FEATURE_CONTRACT_VERSION:
        raise PhishGuardAPIException(
            code="FEATURE_CONTRACT_ERROR",
            message=f"Chỉ hỗ trợ feature contract {FEATURE_CONTRACT_VERSION}",
            status_code=503,
        )
    try:
        metadata_feature_count = int(metadata.get("feature_count", -1))
    except (TypeError, ValueError) as error:
        raise PhishGuardAPIException(
            code="ARTIFACT_METADATA_ERROR",
            message=f"Metadata có feature_count không hợp lệ: {error}",
            status_code=503,
        ) from error
    if feature_count != len(FEATURE_COLUMNS) or metadata_feature_count != len(FEATURE_COLUMNS):
        raise PhishGuardAPIException(
            code="FEATURE_CONTRACT_ERROR",
            message="Số lượng feature của model và metadata không khớp extractor",
            status_code=503,
        )

    try:
        resource_bundle = load_resource_bundle(RESOURCES_DIR)
    except (FileNotFoundError, RuntimeError) as error:
        raise PhishGuardAPIException(
            code="RESOURCE_LOAD_ERROR",
            message=f"Không thể nạp lexical resources: {error}",
            status_code=503,
        ) from error
    calibration = _read_json(resolved_calibration, "CALIBRATION_READ_ERROR")
    if calibration.get("model_version", model_version) != model_version:
        raise PhishGuardAPIException(
            code="ARTIFACT_VERSION_MISMATCH",
            message="Calibration không cùng model_version với model",
            status_code=503,
        )
    try:
        calibrator = ProbabilityCalibrator(
            method=str(calibration["method"]),
            params=dict(calibration["calibrator_params"]),
        )
        if calibration["method"] != "none" and not calibrator.is_fitted:
            raise ValueError("calibrator chưa có params")
    except (KeyError, TypeError, ValueError) as error:
        raise PhishGuardAPIException(
            code="CALIBRATION_CONTRACT_ERROR",
            message=f"Calibration không hợp lệ: {error}",
            status_code=503,
        ) from error

    policy_data = _read_json(resolved_thresholds, "THRESHOLDS_READ_ERROR")
    try:
        thresholds = DecisionThresholds.from_dict(policy_data)
    except (TypeError, ValueError) as error:
        raise PhishGuardAPIException(
            code="THRESHOLDS_CONTRACT_ERROR",
            message=f"Thresholds không hợp lệ: {error}",
            status_code=503,
        ) from error

    return LoadedModel(
        model=model,
        metadata=metadata,
        model_version=model_version,
        feature_contract=feature_contract,
        feature_count=feature_count,
        calibrator=calibrator,
        thresholds=thresholds,
        resource_bundle=resource_bundle,
        feature_extractor=FeatureExtractor(feature_contract, resource_bundle),
    )
