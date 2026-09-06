"""Chính sách rủi ro (Risk Policy) và Artifact hiệu chuẩn (Calibration Artifact)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CalibrationArtifact:
    """Artifact lưu trữ toàn bộ tham số hiệu chuẩn và kết quả kiểm toán ECE/Brier."""

    method: str
    threshold: float
    target_fpr: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    calibrator_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationArtifact:
        return cls(
            method=str(data.get("method", "isotonic")),
            threshold=float(data.get("threshold", 0.5)),
            target_fpr=float(data.get("target_fpr", 0.005)),
            ece_before=float(data.get("ece_before", 0.0)),
            ece_after=float(data.get("ece_after", 0.0)),
            brier_before=float(data.get("brier_before", 0.0)),
            brier_after=float(data.get("brier_after", 0.0)),
            calibrator_params=dict(data.get("calibrator_params", {})),
        )


@dataclass
class RiskPolicyConfig:
    """
    Tách bạch rõ rệt giữa:
    - ML Binary Threshold (dùng để ra quyết định phân loại nội bộ và đo đạc benchmark)
    - Product Risk Policy (ALLOW, CAUTION, WARN/BLOCK dùng cho trải nghiệm người dùng cuối)
    """

    high_threshold: float = 0.75
    medium_threshold: float = 0.45

    def __post_init__(self) -> None:
        if not (0.0 <= self.medium_threshold <= self.high_threshold <= 1.0):
            raise ValueError(
                f"Thứ tự ngưỡng rủi ro không hợp lệ: 0.0 <= {self.medium_threshold} <= {self.high_threshold} <= 1.0"
            )

    def evaluate(self, score: float) -> tuple[str, str]:
        """
        Đánh giá mức độ rủi ro dựa trên xác suất đã hiệu chuẩn:
        - HIGH (>= high_threshold): Rủi ro cao -> Hành động: 'warn' (chặn và hiển thị warning page)
        - MEDIUM (>= medium_threshold): Đáng ngờ -> Hành động: 'caution' (hiển thị soft badge vàng)
        - LOW (< medium_threshold): Bình thường -> Hành động: 'allow' (cho phép điều hướng mượt)
        """
        if score >= self.high_threshold:
            return "high", "warn"
        if score >= self.medium_threshold:
            return "medium", "caution"
        return "low", "allow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "high": self.high_threshold,
            "medium": self.medium_threshold,
            "actions": {
                "high": "warn",
                "medium": "caution",
                "low": "allow",
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskPolicyConfig:
        return cls(
            high_threshold=float(data.get("high", 0.75)),
            medium_threshold=float(data.get("medium", 0.45)),
        )
