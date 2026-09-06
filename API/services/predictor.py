"""Dịch vụ dự đoán nhãn URL và tách bạch chính sách rủi ro (Risk Policy) PhishGuard ML."""

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
    FEATURE_CONTRACT_V2,
    extract_features,
)

LOGGER = logging.getLogger("phishguard.api.predictor")
LABEL_NAMES = {0: "Legitimate URL", 1: "Phishing URL"}


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
        self.calibrator = loaded_model.calibrator
        self.cache = cache
        self.threshold = loaded_model.threshold
        self.model_version = loaded_model.model_version
        self.feature_contract = loaded_model.feature_contract
        self.risk_thresholds = loaded_model.risk_thresholds
        self.feature_columns = (
            FEATURE_COLUMNS_V2 if self.feature_contract == FEATURE_CONTRACT_V2 else FEATURE_COLUMNS_V1
        )

    def evaluate_risk_policy(self, score: float) -> tuple[str, str]:
        """
        Tách bạch quyết định của mô hình (probability/score) khỏi chính sách rủi ro sản phẩm:
        - LOW (< medium_threshold): An toàn -> Hành động: 'allow'
        - MEDIUM (medium_threshold <= score < high_threshold): Nghi ngờ -> Hành động: 'caution'
        - HIGH (>= high_threshold): Rủi ro cao -> Hành động: 'warn'
        """
        high_th = self.risk_thresholds.get("high", 0.75)
        medium_th = self.risk_thresholds.get("medium", 0.45)

        if score >= high_th:
            return "high", "warn"
        if score >= medium_th:
            return "medium", "caution"
        return "low", "allow"

    def predict_url(self, url: str) -> dict[str, Any]:
        """Dự đoán cho 1 URL duy nhất, tích hợp LRU cache và tách biệt model decision với risk policy."""
        # 1. Kiểm tra LRU Cache với composite key (model_version + feature_contract + url)
        cached_result = self.cache.get(url, self.model_version, self.feature_contract)
        if cached_result is not None:
            res = dict(cached_result)
            res["cached"] = True
            return res

        # 2. Trích xuất đặc trưng in-memory theo hợp đồng đang hoạt động (trực tiếp từ raw url)
        try:
            features = extract_features(url, contract=self.feature_contract)
            input_frame = pd.DataFrame([features], columns=self.feature_columns)
        except Exception as error:
            LOGGER.exception("Trích xuất đặc trưng thất bại cho hostname=%s", safe_log_host(url))
            raise PhishGuardAPIException(
                code="INVALID_URL",
                message="Không thể trích xuất đặc trưng cho URL này",
                status_code=400,
            ) from error

        # 3. Model Inference & Calibration
        try:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(input_frame)[0]
                raw_score = float(probabilities[1])
                if self.calibrator is not None:
                    model_score = round(float(self.calibrator.calibrate(raw_score)), 4)
                else:
                    model_score = round(raw_score, 4)
            else:
                raw_label = int(self.model.predict(input_frame)[0])
                model_score = 1.0 if raw_label == 1 else 0.0

            # Phân loại nhị phân dựa trên operating threshold
            label = int(model_score >= self.threshold)
            prediction_str = LABEL_NAMES.get(label, "Unknown")
            risk_level, risk_action = self.evaluate_risk_policy(model_score)

        except Exception as error:
            LOGGER.exception("Dự đoán thất bại cho hostname=%s", safe_log_host(url))
            raise PhishGuardAPIException(
                code="PREDICTION_FAILED",
                message="Lỗi khi mô hình thực hiện dự đoán",
                status_code=500,
            ) from error

        result = {
            "url": url,
            "model": {
                "score": model_score,
                "threshold": self.threshold,
                "label": label,
                "version": self.model_version,
            },
            "risk": {
                "level": risk_level,
                "action": risk_action,
            },
            "feature_contract": self.feature_contract,
            "cached": False,
            # Tương thích ngược với các trường phẳng của client / test hiện có
            "label": label,
            "prediction": prediction_str,
            "model_score": model_score,
            "risk_level": risk_level,
            "model_version": self.model_version,
        }

        # Lưu vào cache
        self.cache.put(url, result, self.model_version, self.feature_contract)
        LOGGER.info(
            "Predicted hostname=%s label=%d score=%.4f risk=%s action=%s",
            safe_log_host(url),
            label,
            model_score,
            risk_level,
            risk_action,
        )
        return result

    def predict_batch(self, urls: list[str]) -> dict[str, Any]:
        """Dự đoán theo lô danh sách URL, bảo toàn đúng thứ tự input."""
        results = [self.predict_url(url) for url in urls]
        return {
            "total": len(results),
            "results": results,
        }
