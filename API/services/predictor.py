"""Dịch vụ suy luận URL với một ActionPolicy duy nhất."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from API.errors import PhishGuardAPIException
from API.services.cache import PredictionCache
from API.services.model_loader import LoadedModel
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    extract_features,
)

LOGGER = logging.getLogger("phishguard.api.predictor")
LABEL_NAMES = {0: "Low lexical phishing risk", 1: "High lexical phishing risk"}


def safe_log_host(url: str) -> str:
    """Chỉ lấy hostname để log, không ghi path/query/fragment nhạy cảm."""
    try:
        return urlparse(url).hostname or "unknown-host"
    except (TypeError, ValueError):
        return "unknown-host"


class PredictorService:
    def __init__(self, loaded_model: LoadedModel, cache: PredictionCache) -> None:
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.calibrator = loaded_model.calibrator
        self.action_policy = loaded_model.action_policy
        self.cache = cache
        self.model_version = loaded_model.model_version
        self.policy_version = loaded_model.policy_version
        self.feature_contract = loaded_model.feature_contract
        self.feature_columns = {
            "lexical-v1": FEATURE_COLUMNS_V1,
            "lexical-v2": FEATURE_COLUMNS_V2,
            "lexical-v3": FEATURE_COLUMNS_V3,
        }[self.feature_contract]

    def evaluate_action_policy(self, score: float) -> tuple[str, str]:
        """Ủy quyền hoàn toàn quyết định cho ActionPolicy đã được checksum."""
        return self.action_policy.evaluate(score)

    def predict_url(self, url: str) -> dict[str, Any]:
        """Trả risk score và action; không dùng binary threshold thứ hai."""
        cached_result = self.cache.get(url, self.model_version, self.feature_contract)
        if cached_result is not None:
            result = dict(cached_result)
            result["cached"] = True
            return result

        try:
            features = extract_features(url, contract=self.feature_contract)
            input_frame = pd.DataFrame([features], columns=self.feature_columns)
        except (TypeError, ValueError, KeyError) as error:
            LOGGER.exception("Trích xuất đặc trưng thất bại cho hostname=%s", safe_log_host(url))
            raise PhishGuardAPIException(
                code="INVALID_URL",
                message="Không thể trích xuất đặc trưng cho URL này",
                status_code=400,
            ) from error

        try:
            raw_score = float(self.model.predict_proba(input_frame)[0, 1])
            risk_score = round(float(self.calibrator.calibrate(raw_score)), 4)
            risk_level, action = self.evaluate_action_policy(risk_score)
        except Exception as error:
            LOGGER.exception("Dự đoán thất bại cho hostname=%s", safe_log_host(url))
            raise PhishGuardAPIException(
                code="PREDICTION_FAILED",
                message="Lỗi khi mô hình thực hiện dự đoán",
                status_code=500,
            ) from error

        # Các trường label/prediction chỉ giữ để client cũ không crash. Chúng
        # được suy ra từ action canonical, không tham gia quyết định sản phẩm.
        legacy_label = int(action == "block")
        result = {
            "url": url,
            "phishing_risk_score": risk_score,
            "action": action,
            "policy_version": self.policy_version,
            "risk_level": risk_level,
            "risk": {"level": risk_level, "action": action},
            "model": {
                "score": risk_score,
                "threshold": self.action_policy.block_threshold,
                "label": legacy_label,
                "version": self.model_version,
            },
            "feature_contract": self.feature_contract,
            "cached": False,
            "label": legacy_label,
            "prediction": LABEL_NAMES[legacy_label],
            "model_score": risk_score,
            "model_version": self.model_version,
        }
        self.cache.put(url, result, self.model_version, self.feature_contract)
        LOGGER.info(
            "Predicted hostname=%s score=%.4f action=%s policy=%s",
            safe_log_host(url),
            risk_score,
            action,
            self.policy_version,
        )
        return result

    def predict_batch(self, urls: list[str]) -> dict[str, Any]:
        """Dự đoán theo lô và giữ nguyên thứ tự input."""
        results = [self.predict_url(url) for url in urls]
        return {"total": len(results), "results": results}
