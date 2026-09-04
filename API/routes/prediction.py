"""Endpoints dự đoán URL và batch URL cho PhishGuard ML API."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, field_validator

from API.dependencies import get_predictor
from API.services.predictor import PredictorService


def normalize_web_url(value: str) -> str:
    """Chuẩn hóa và xác thực URL HTTP/HTTPS dùng chung cho mọi request."""
    if not isinstance(value, str):
        raise ValueError("URL phải là chuỗi ký tự")
    normalized = value.strip()
    if len(normalized) > 2048:
        raise ValueError("Độ dài URL vượt quá giới hạn tối đa 2.048 ký tự")
    try:
        parsed = urlparse(normalized)
        hostname = parsed.hostname
    except ValueError as error:
        raise ValueError("URL có hostname không hợp lệ") from error
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError("URL phải sử dụng giao thức HTTP hoặc HTTPS và có hostname hợp lệ")
    return normalized


class URLRequest(BaseModel):
    """Yêu cầu dự đoán một URL."""

    url: str = Field(min_length=8, max_length=2048, examples=["https://example.com/login"])

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return normalize_web_url(value)


class BatchURLRequest(BaseModel):
    """Yêu cầu dự đoán danh sách tối đa 50 URL."""

    urls: list[str] = Field(min_length=1, max_length=50)

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, values: list[str]) -> list[str]:
        return [normalize_web_url(val) for val in values]


class PredictionResponse(BaseModel):
    """Kết quả phân loại một URL theo hợp đồng mới."""

    url: str
    label: int
    prediction: str
    model_score: float
    risk_level: str
    model_version: str
    feature_contract: str
    cached: bool = False


class BatchPredictionResponse(BaseModel):
    """Kết quả phân loại theo lô."""

    total: int
    results: list[PredictionResponse]


router = APIRouter(tags=["Dự đoán URL"])


@router.post(
    "/phish-url-prediction",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Dự đoán nhãn rủi ro cho một URL",
)
def predict_single_url(
    request: URLRequest,
    predictor: PredictorService = Depends(get_predictor),
) -> PredictionResponse:
    result = predictor.predict_url(request.url)
    return PredictionResponse(**result)


@router.post(
    "/phish-url-prediction/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Dự đoán hàng loạt (tối đa 50 URL)",
)
def predict_batch_urls(
    request: BatchURLRequest,
    predictor: PredictorService = Depends(get_predictor),
) -> BatchPredictionResponse:
    batch_result = predictor.predict_batch(request.urls)
    return BatchPredictionResponse(
        total=batch_result["total"],
        results=[PredictionResponse(**item) for item in batch_result["results"]],
    )
