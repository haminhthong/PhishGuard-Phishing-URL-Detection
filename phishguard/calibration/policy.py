"""Artifact hiệu chuẩn và chính sách hành động browser của PhishGuard."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CalibrationArtifact:
    """Artifact chỉ lưu calibrator và metric; threshold lưu riêng."""

    method: str
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
            ece_before=float(data.get("ece_before", 0.0)),
            ece_after=float(data.get("ece_after", 0.0)),
            brier_before=float(data.get("brier_before", 0.0)),
            brier_after=float(data.get("brier_after", 0.0)),
            calibrator_params=dict(data.get("calibrator_params", {})),
        )


@dataclass(frozen=True)
class DecisionThresholds:
    """Ngưỡng duy nhất để ánh xạ điểm rủi ro thành hành động trình duyệt.

    `caution_threshold` và `block_threshold` phải được chọn trên tập
    Validation độc lập. Điểm thấp chỉ có nghĩa là rủi ro lexical thấp, không
    phải cam kết website an toàn tuyệt đối.
    """

    caution_threshold: float
    block_threshold: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.caution_threshold < self.block_threshold <= 1.0):
            raise ValueError("Ngưỡng phải thỏa 0.0 <= caution < block <= 1.0")

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
            "caution_threshold": self.caution_threshold,
            "block_threshold": self.block_threshold,
            "actions": {"low": "allow", "medium": "caution", "high": "block"},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionThresholds:
        """Nạp ngưỡng với schema cố định."""
        caution = data.get("caution_threshold")
        block = data.get("block_threshold")
        if caution is None or block is None:
            raise ValueError("Thresholds thiếu caution_threshold hoặc block_threshold")
        actions = data.get("actions", {})
        expected_actions = {"low": "allow", "medium": "caution", "high": "block"}
        if actions and any(
            actions.get(level, action) != action for level, action in expected_actions.items()
        ):
            raise ValueError("Thresholds phải ánh xạ low/medium/high thành allow/caution/block")
        return cls(
            caution_threshold=float(caution),
            block_threshold=float(block),
        )
