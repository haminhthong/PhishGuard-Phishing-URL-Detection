"""Endpoints hệ thống, health check, model info và thống kê cho PhishGuard ML API."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends

from API.dependencies import get_cache, get_loaded_model
from API.services.cache import PredictionCache
from API.services.model_loader import LoadedModel
from phishguard import __version__

router = APIRouter(tags=["Hệ thống"])
START_TIME = time.monotonic()


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"name": "PhishGuard ML API", "version": __version__, "documentation": "/docs"}


@router.get("/health/live", summary="Liveness probe kiểm tra tiến trình dịch vụ đang hoạt động")
def liveness_probe() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready", summary="Readiness probe kiểm tra mô hình đã nạp và sẵn sàng phục vụ")
def readiness_probe(
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> dict[str, Any]:
    return {
        "status": "ready",
        "release": loaded_model.release_id,
        "model_version": loaded_model.model_version,
        "feature_contract": loaded_model.feature_contract,
        "feature_count": loaded_model.feature_count,
        "action_policy": loaded_model.action_policy.to_dict(),
        "calibration_loaded": loaded_model.calibrator.is_fitted,
    }


@router.get("/health", summary="Kiểm tra sức khỏe hệ thống API")
def health_check(
    loaded_model: LoadedModel = Depends(get_loaded_model),
    cache: PredictionCache = Depends(get_cache),
) -> dict[str, Any]:
    return {
        "status": "ok",
        "release": loaded_model.release_id,
        "model_version": loaded_model.model_version,
        "feature_count": loaded_model.feature_count,
        "feature_contract": loaded_model.feature_contract,
        "action_policy": loaded_model.action_policy.to_dict(),
        "cache_entries": cache.stats()["used"],
    }


@router.get("/model-info", summary="Thông tin mô hình và hợp đồng đặc trưng")
def model_info(
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> dict[str, Any]:
    from phishguard.features import (
        FEATURE_COLUMNS_V1,
        FEATURE_COLUMNS_V2,
        FEATURE_COLUMNS_V3,
        FEATURE_COLUMNS_V4,
        FEATURE_CONTRACT_V2,
        FEATURE_CONTRACT_V3,
        FEATURE_CONTRACT_V4,
    )

    cols = {
        "lexical-v1": FEATURE_COLUMNS_V1,
        FEATURE_CONTRACT_V2: FEATURE_COLUMNS_V2,
        FEATURE_CONTRACT_V3: FEATURE_COLUMNS_V3,
        FEATURE_CONTRACT_V4: FEATURE_COLUMNS_V4,
    }[loaded_model.feature_contract]
    return {
        "model_type": loaded_model.metadata.get("model_type", type(loaded_model.model).__name__),
        "model_version": loaded_model.model_version,
        "release": loaded_model.release_id,
        "feature_count": len(cols),
        "features": list(cols),
        "feature_contract": loaded_model.feature_contract,
        "policy_version": loaded_model.policy_version,
        "action_policy": loaded_model.action_policy.to_dict(),
        "artifact_hashes": {
            "model": loaded_model.model_sha256,
            "calibration": loaded_model.calibration_sha256,
            "action_policy": loaded_model.action_policy_sha256,
            "feature_contract": loaded_model.feature_contract_hash,
            "resources": loaded_model.resource_hashes,
        },
        "training_date": loaded_model.metadata.get("training_date"),
        "test_metrics": loaded_model.metadata.get("test_metrics", {}),
    }


@router.get("/stats", summary="Thống kê lưu lượng request và bộ nhớ cache")
def system_stats(
    cache: PredictionCache = Depends(get_cache),
) -> dict[str, Any]:
    c_stats = cache.stats()
    return {
        "uptime_seconds": int(time.monotonic() - START_TIME),
        "cache_hit_rate_percent": c_stats["hit_rate_percent"],
        "cache_hits": c_stats["hits"],
        "cache_misses": c_stats["misses"],
        "total_requests": c_stats["total_requests"],
        "cache_capacity": c_stats["capacity"],
        "cache_used": c_stats["used"],
    }


@router.delete("/cache", summary="Xóa bộ nhớ đệm dự đoán")
def clear_cache(
    cache: PredictionCache = Depends(get_cache),
) -> dict[str, Any]:
    cleared_count = cache.clear()
    return {"status": "ok", "cleared_entries": cleared_count}
