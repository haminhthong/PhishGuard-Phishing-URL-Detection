"""Artifact hiệu chuẩn và chính sách hành động browser của PhishGuard."""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class ActionPolicy:
    """Nguồn sự thật duy nhất cho quyết định ALLOW/CAUTION/BLOCK.

    `caution_threshold` và `block_threshold` phải được chọn trên tập Policy
    Validation độc lập. Điểm thấp chỉ có nghĩa là rủi ro lexical thấp, không
    phải cam kết website an toàn tuyệt đối.
    """

    caution_threshold: float
    block_threshold: float
    policy_version: str = "browser-risk-v1"

    def __post_init__(self) -> None:
        if not (0.0 <= self.caution_threshold < self.block_threshold <= 1.0):
            raise ValueError("Ngưỡng ActionPolicy phải thỏa 0.0 <= caution < block <= 1.0")
        if not self.policy_version.strip():
            raise ValueError("policy_version không được để trống")

    def evaluate(self, score: float) -> tuple[str, str]:
        """Trả về mức rủi ro và đúng một hành động sản phẩm."""
        bounded_score = float(score)
        if not math.isfinite(bounded_score) or not 0.0 <= bounded_score <= 1.0:
            raise ValueError("risk score phải là số hữu hạn trong khoảng [0.0, 1.0]")
        if bounded_score >= self.block_threshold:
            return "high", "block"
        if bounded_score >= self.caution_threshold:
            return "medium", "caution"
        return "low", "allow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "caution_threshold": self.caution_threshold,
            "block_threshold": self.block_threshold,
            "actions": {"low": "allow", "medium": "caution", "high": "block"},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionPolicy:
        """Nạp policy mới; chỉ chấp nhận key legacy để migrate có kiểm soát."""
        caution = data.get("caution_threshold", data.get("medium"))
        block = data.get("block_threshold", data.get("high"))
        if caution is None or block is None:
            raise ValueError("Action policy thiếu caution_threshold hoặc block_threshold")
        actions = data.get("actions", {})
        expected_actions = {"low": "allow", "medium": "caution", "high": "block"}
        if actions and any(actions.get(level, action) != action for level, action in expected_actions.items()):
            raise ValueError("Action policy phải ánh xạ low/medium/high thành allow/caution/block")
        return cls(
            caution_threshold=float(caution),
            block_threshold=float(block),
            policy_version=str(data.get("policy_version", "browser-risk-v1")),
        )


@dataclass
class RiskPolicyConfig:
    """
    Policy legacy để đọc artifact v3.2 cũ.

    Production không dùng class này; API dùng `ActionPolicy` để tránh hành động
    `warn` mâu thuẫn với nhãn nhị phân hoặc ngưỡng cũ.
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
