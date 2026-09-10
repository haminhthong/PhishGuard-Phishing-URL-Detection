"""Endpoints hệ thống, health check, model info và thống kê cho PhishGuard ML API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from API.dependencies import get_loaded_model
from API.services.model_loader import LoadedModel

router = APIRouter(tags=["Hệ thống"])


@router.get("/health/live", summary="Liveness probe kiểm tra tiến trình dịch vụ đang hoạt động")
def liveness_probe() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready", summary="Readiness probe kiểm tra mô hình đã nạp và sẵn sàng phục vụ")
def readiness_probe(
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> dict[str, Any]:
    return {
        "status": "ready",
        "model_version": loaded_model.model_version,
        "feature_contract": loaded_model.feature_contract,
        "feature_count": loaded_model.feature_count,
        "action_policy": loaded_model.action_policy.to_dict(),
        "calibration_loaded": loaded_model.calibrator.is_fitted,
    }


@router.get("/health", summary="Kiểm tra sức khỏe hệ thống API")
def health_check(
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": loaded_model.model_version,
        "feature_count": loaded_model.feature_count,
        "feature_contract": loaded_model.feature_contract,
        "action_policy": loaded_model.action_policy.to_dict(),
    }


@router.get("/model-info", summary="Thông tin mô hình và hợp đồng đặc trưng")
def model_info(
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> dict[str, Any]:
    from phishguard.features import FEATURE_COLUMNS

    return {
        "model_type": loaded_model.metadata.get("model_type", type(loaded_model.model).__name__),
        "model_version": loaded_model.model_version,
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "feature_contract": loaded_model.feature_contract,
        "policy_version": loaded_model.policy_version,
        "action_policy": loaded_model.action_policy.to_dict(),
        "training_date": loaded_model.metadata.get("training_date"),
        "validation_metrics": loaded_model.metadata.get("validation_metrics", {}),
    }
