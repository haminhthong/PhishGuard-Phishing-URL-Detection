"""Dịch vụ suy luận URL và áp dụng ngưỡng quyết định."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd

from API.errors import PhishGuardAPIException
from API.services.model_loader import LoadedModel
from phishguard.features import FEATURE_COLUMNS

LOGGER = logging.getLogger("phishguard.api.predictor")
ACTION_REASONS = {
    "allow": "LEXICAL_RISK_BELOW_CAUTION_THRESHOLD",
    "caution": "LEXICAL_RISK_ABOVE_CAUTION_THRESHOLD",
    "block": "LEXICAL_RISK_ABOVE_BLOCK_THRESHOLD",
}


def safe_log_host(url: str) -> str:
    """Chỉ lấy hostname để log, không ghi path/query/fragment nhạy cảm."""
    try:
        return urlparse(url).hostname or "unknown-host"
    except (TypeError, ValueError):
        return "unknown-host"


class PredictorService:
    def __init__(self, loaded_model: LoadedModel) -> None:
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.calibrator = loaded_model.calibrator
        self.thresholds = loaded_model.thresholds
        self.feature_extractor = loaded_model.feature_extractor
        self.model_version = loaded_model.model_version
        self.feature_contract = loaded_model.feature_contract
        self.feature_columns = FEATURE_COLUMNS

    def evaluate_thresholds(self, score: float) -> tuple[str, str]:
        """Áp dụng đúng cặp ngưỡng đã chọn trên threshold validation."""
        return self.thresholds.evaluate(score)

    def predict_url(self, url: str) -> dict[str, Any]:
        """Trả risk score và browser decision từ cùng một cặp thresholds."""
        try:
            features = self.feature_extractor.extract(url)
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
            risk_level, action = self.evaluate_thresholds(risk_score)
        except Exception as error:
            LOGGER.exception("Dự đoán thất bại cho hostname=%s", safe_log_host(url))
            raise PhishGuardAPIException(
                code="PREDICTION_FAILED",
                message="Lỗi khi mô hình thực hiện dự đoán",
                status_code=500,
            ) from error

        result = {
            "request_id": str(uuid4()),
            "url": url,
            "risk_score": risk_score,
            "action": action.upper(),
            "risk_level": risk_level.upper(),
            "reason": ACTION_REASONS[action],
            "signals": {
                "punycode": bool(features.get("has_punycode", 0)),
                "brand_mismatch": bool(features.get("brand_not_registered_domain", 0)),
                "shortener": bool(features.get("uses_shortening_service", 0)),
                "suspicious_tld": bool(features.get("is_suspicious_tld", 0)),
            },
            "model_version": self.model_version,
        }
        LOGGER.info(
            "Predicted hostname=%s score=%.4f action=%s",
            safe_log_host(url),
            risk_score,
            action,
        )
        return result

    def predict_batch(self, urls: list[str]) -> dict[str, Any]:
        """Dự đoán theo lô và giữ nguyên thứ tự input."""
        results = [self.predict_url(url) for url in urls]
        return {"total": len(results), "results": results}
