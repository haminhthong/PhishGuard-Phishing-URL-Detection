"""Dịch vụ dự đoán nhãn URL và xác định mức độ rủi ro PhishGuard ML."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from API.errors import PhishGuardAPIException
from API.services.cache import PredictionCache
from API.services.model_loader import LoadedModel
from phishguard.features import FEATURE_COLUMNS, extract_features

LOGGER = logging.getLogger("phishguard.api.predictor")
LABEL_NAMES = {0: "Legitimate URL", 1: "Phishing URL"}


def calculate_risk_level(score: float) -> str:
    """Xác định mức độ rủi ro dựa trên ngưỡng điểm mô hình."""
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def safe_log_host(url: str) -> str:
    """Chỉ lấy hostname để log nhằm ngăn ngừa làm lộ query string nhạy cảm."""
    try:
        return urlparse(url).hostname or "unknown-host"
    except Exception:
        return "unknown-host"


class PredictorService:
    def __init__(self, loaded_model: LoadedModel, cache: PredictionCache) -> None:
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.cache = cache
        self.threshold = loaded_model.threshold
        self.model_version = loaded_model.model_version
        self.feature_contract = loaded_model.feature_contract

    def predict_url(self, url: str) -> dict[str, Any]:
        """Dự đoán cho 1 URL duy nhất, tích hợp LRU cache và risk level."""
        # 1. Check LRU Cache
        cached_result = self.cache.get(url)
        if cached_result is not None:
            # Reconstruct response with cached=True
            res = dict(cached_result)
            res["cached"] = True
            return res

        # 2. Extract features
        try:
            features = extract_features(url)
            input_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        except Exception as error:
            LOGGER.exception("Trích xuất đặc trưng thất bại cho hostname=%s", safe_log_host(url))
            raise PhishGuardAPIException(
                code="INVALID_URL",
                message="Không thể trích xuất đặc trưng cho URL này",
                status_code=400,
            ) from error

        # 3. Model Inference
        try:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(input_frame)[0]
                # Index 1 corresponds to phishing probability
                model_score = round(float(probabilities[1]), 4)
            else:
                raw_label = int(self.model.predict(input_frame)[0])
                model_score = 1.0 if raw_label == 1 else 0.0

            label = int(model_score >= self.threshold)
            prediction_str = LABEL_NAMES.get(label, "Unknown")
            risk = calculate_risk_level(model_score)

        except Exception as error:
            LOGGER.exception("Dự đoán thất bại cho hostname=%s", safe_log_host(url))
            raise PhishGuardAPIException(
                code="PREDICTION_FAILED",
                message="Lỗi khi mô hình thực hiện dự đoán",
                status_code=500,
            ) from error

        result = {
            "url": url,
            "label": label,
            "prediction": prediction_str,
            "model_score": model_score,
            "risk_level": risk,
            "model_version": self.model_version,
            "feature_contract": self.feature_contract,
            "cached": False,
        }

        # Save to cache
        self.cache.put(url, result)
        LOGGER.info("Predicted hostname=%s label=%d score=%.4f risk=%s", safe_log_host(url), label, model_score, risk)
        return result

    def predict_batch(self, urls: list[str]) -> dict[str, Any]:
        """Dự đoán theo lô danh sách URL, bảo toàn đúng thứ tự input."""
        results = [self.predict_url(url) for url in urls]
        return {
            "total": len(results),
            "results": results,
        }
