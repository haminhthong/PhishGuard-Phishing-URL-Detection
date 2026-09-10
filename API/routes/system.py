"""Health check tối thiểu cho PhishGuard ML API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from API.dependencies import get_loaded_model
from API.services.model_loader import LoadedModel

router = APIRouter(tags=["Hệ thống"])


@router.get("/health", summary="Kiểm tra sức khỏe hệ thống API")
def health_check(
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": loaded_model.model_version,
        "feature_count": loaded_model.feature_count,
        "feature_contract": loaded_model.feature_contract,
        "thresholds": loaded_model.thresholds.to_dict(),
        "calibration_loaded": loaded_model.calibrator.is_fitted,
    }
