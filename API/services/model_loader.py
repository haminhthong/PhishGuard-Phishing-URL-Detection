"""Tải release bundle và fail-closed khi artifact không toàn vẹn."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from API.errors import PhishGuardAPIException
from phishguard.calibration import ActionPolicy, ProbabilityCalibrator
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_V3,
)
from phishguard.features.resources import RESOURCE_BUNDLE


def compute_sha256(file_path: Path) -> str:
    """Tính SHA-256 theo byte của artifact."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_feature_contract_hash(columns: tuple[str, ...]) -> str:
    return hashlib.sha256(",".join(columns).encode("utf-8")).hexdigest()


@dataclass
class LoadedModel:
    model: XGBClassifier
    metadata: dict[str, Any]
    model_version: str
    feature_contract: str
    feature_count: int
    threshold: float
    risk_thresholds: dict[str, float]
    calibrator: ProbabilityCalibrator
    action_policy: ActionPolicy

    @property
    def policy_version(self) -> str:
        return self.action_policy.policy_version


def resolve_model_paths(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Xác định model, metadata và calibration từ active release pointer."""
    project_root = Path(__file__).resolve().parents[2]
    production_registry = project_root / "artifacts" / "models" / "production.json"
    default_model = project_root / "API" / "XGB.json"
    default_meta = project_root / "API" / "model_metadata.json"

    if model_path is not None and model_path != default_model:
        meta_path = metadata_path or model_path.parent / "metadata.json"
        return model_path, meta_path, model_path.parent / "calibration.json"

    if production_registry.is_file():
        try:
            registry = json.loads(production_registry.read_text(encoding="utf-8"))
            model_dir_value = registry.get("model_dir")
            if not isinstance(model_dir_value, str) or not model_dir_value:
                raise ValueError("production.json thiếu model_dir")
            model_dir = project_root / model_dir_value
            return (
                model_dir / "model.json",
                model_dir / "metadata.json",
                model_dir / "calibration.json",
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise PhishGuardAPIException(
                code="REGISTRY_READ_ERROR",
                message=f"Active release pointer không hợp lệ: {error}",
                status_code=503,
            ) from error

    return default_model, default_meta, default_model.parent / "calibration.json"


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
            code=code, message=f"{path.name} phải là JSON object", status_code=503
        )
    return data


def _verify_checksum(path: Path, expected: str | None, name: str) -> None:
    if not expected:
        raise PhishGuardAPIException(
            code="ARTIFACT_METADATA_ERROR",
            message=f"Metadata thiếu checksum bắt buộc cho {name}",
            status_code=503,
        )
    if compute_sha256(path).lower() != str(expected).lower():
        raise PhishGuardAPIException(
            code="MODEL_INTEGRITY_ERROR",
            message=f"Checksum SHA-256 của {name} không khớp metadata",
            status_code=503,
        )


def _verify_bundled_resources(release_dir: Path, expected_hashes: dict[str, str]) -> None:
    """Kiểm tra bản sao resource nếu release bundle có đóng gói kèm."""
    resource_dir = release_dir / "resources"
    if not resource_dir.is_dir():
        # Artifact 3.2 cũ dùng resources ở project root; v4 phải tự chứa.
        return

    resource_files = {
        "brand_terms": "brand_terms.json",
        "shorteners": "shortener_domains.json",
        "suspicious_tlds": "suspicious_tlds.json",
    }
    for resource_name, filename in resource_files.items():
        path = resource_dir / filename
        if (
            not path.is_file()
            or compute_sha256(path).lower() != str(expected_hashes[resource_name]).lower()
        ):
            raise PhishGuardAPIException(
                code="RESOURCE_CONTRACT_ERROR",
                message=f"Resource đóng gói không khớp checksum: {resource_name}",
                status_code=503,
            )


def load_phishguard_model(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> LoadedModel:
    """Nạp model + calibrator + policy + resource contract theo nguyên tắc fail-closed."""
    resolved_model, resolved_meta, resolved_calibration = resolve_model_paths(
        model_path, metadata_path
    )
    if not resolved_model.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost JSON tại {resolved_model}")
    if resolved_model.suffix.lower() != ".json":
        raise PhishGuardAPIException(
            code="MODEL_CONTRACT_MISMATCH",
            message="API chỉ hỗ trợ artifact XGBoost Native JSON",
            status_code=503,
        )

    metadata = _read_json(resolved_meta, "METADATA_READ_ERROR")
    _verify_checksum(resolved_model, metadata.get("model_sha256"), "model")

    try:
        model = XGBClassifier()
        model.load_model(resolved_model)
    except Exception as error:
        raise PhishGuardAPIException(
            code="MODEL_LOAD_ERROR", message=f"Không thể nạp model JSON: {error}", status_code=503
        ) from error

    model_version = str(metadata.get("model_version", ""))
    feature_contract = str(metadata.get("feature_contract", ""))
    contract_columns = {
        FEATURE_CONTRACT_V1: FEATURE_COLUMNS_V1,
        FEATURE_CONTRACT_V2: FEATURE_COLUMNS_V2,
        FEATURE_CONTRACT_V3: FEATURE_COLUMNS_V3,
    }
    if feature_contract not in contract_columns:
        raise PhishGuardAPIException(
            code="FEATURE_CONTRACT_ERROR",
            message=f"Feature contract không được hỗ trợ: {feature_contract}",
            status_code=503,
        )
    columns = contract_columns[feature_contract]
    expected_hash = compute_feature_contract_hash(columns)
    if metadata.get("feature_contract_hash") != expected_hash:
        raise PhishGuardAPIException(
            code="FEATURE_CONTRACT_ERROR",
            message="Feature order/hash trong metadata không khớp runtime",
            status_code=503,
        )
    feature_count = int(getattr(model, "n_features_in_", len(columns)))
    if feature_count != len(columns) or int(metadata.get("feature_count", -1)) != len(columns):
        raise PhishGuardAPIException(
            code="FEATURE_CONTRACT_ERROR",
            message="Số lượng feature của model, metadata và extractor không khớp",
            status_code=503,
        )

    resource_hashes = metadata.get("resource_hashes")
    if not isinstance(resource_hashes, dict):
        raise PhishGuardAPIException(
            code="RESOURCE_CONTRACT_ERROR",
            message="Release metadata thiếu resource_hashes",
            status_code=503,
        )
    for resource_name, expected_hash in RESOURCE_BUNDLE.hashes.items():
        if str(resource_hashes.get(resource_name, "")).lower() != expected_hash.lower():
            raise PhishGuardAPIException(
                code="RESOURCE_CONTRACT_ERROR",
                message=f"Checksum resource không khớp: {resource_name}",
                status_code=503,
            )
    _verify_bundled_resources(resolved_calibration.parent, RESOURCE_BUNDLE.hashes)
    if metadata.get("tld_library_version") != RESOURCE_BUNDLE.tld_library_version:
        raise PhishGuardAPIException(
            code="RESOURCE_CONTRACT_ERROR",
            message="Phiên bản thư viện PSL/TLD không khớp release metadata",
            status_code=503,
        )

    calibration = _read_json(resolved_calibration, "CALIBRATION_READ_ERROR")
    _verify_checksum(
        resolved_calibration,
        metadata.get("calibration_sha256"),
        "calibration",
    )
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

    policy_path = resolved_calibration.parent / "action_policy.json"
    policy_data = _read_json(policy_path, "POLICY_READ_ERROR")
    _verify_checksum(policy_path, metadata.get("action_policy_sha256"), "action_policy")
    try:
        action_policy = ActionPolicy.from_dict(policy_data)
    except (TypeError, ValueError) as error:
        raise PhishGuardAPIException(
            code="POLICY_CONTRACT_ERROR",
            message=f"Action policy không hợp lệ: {error}",
            status_code=503,
        ) from error

    risk_thresholds = {
        "caution": action_policy.caution_threshold,
        "block": action_policy.block_threshold,
    }
    return LoadedModel(
        model=model,
        metadata=metadata,
        model_version=model_version,
        feature_contract=feature_contract,
        feature_count=feature_count,
        # Giữ trường legacy để client cũ không crash; product decision không dùng nó.
        threshold=action_policy.block_threshold,
        risk_thresholds=risk_thresholds,
        calibrator=calibrator,
        action_policy=action_policy,
    )
