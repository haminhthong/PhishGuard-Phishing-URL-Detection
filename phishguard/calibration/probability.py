"""Module hiệu chuẩn xác suất (Probability Calibration) độc lập cho PhishGuard ML."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from phishguard.training.evaluation import compute_ece


class ProbabilityCalibrator:
    """
    Bộ hiệu chuẩn xác suất post-hoc (Isotonic / Sigmoid / None).
    Hỗ trợ serialize/deserialize thuần túy bằng JSON (không cần pickle).
    """

    def __init__(
        self,
        method: str = "isotonic",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.method = method.lower()
        self.params = params or {}
        self.is_fitted = bool(self.params)

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        """Huấn luyện calibrator trên điểm số raw và nhãn ground-truth của tập Calibration."""
        raw_scores_arr = np.asarray(raw_scores, dtype=float).ravel()
        y_true_arr = np.asarray(y_true, dtype=int).ravel()

        if self.method == "isotonic":
            reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            reg.fit(raw_scores_arr, y_true_arr)
            # Serialize breakpoints để có thể suy luận thuần JSON không phụ thuộc pickle
            # Dùng các điểm x và y nội suy
            x_eval = np.linspace(0.0, 1.0, 201)
            y_eval = reg.predict(x_eval)
            self.params = {
                "x_thresholds": [round(float(x), 6) for x in x_eval],
                "y_thresholds": [round(float(y), 6) for y in y_eval],
            }

        elif self.method == "sigmoid":
            # Platt Scaling qua Logistic Regression
            lr = LogisticRegression(solver="lbfgs", max_iter=200)
            lr.fit(raw_scores_arr.reshape(-1, 1), y_true_arr)
            self.params = {
                "coef": float(lr.coef_[0][0]),
                "intercept": float(lr.intercept_[0]),
            }

        elif self.method == "none":
            self.params = {}

        else:
            raise ValueError(f"Phương pháp calibration không được hỗ trợ: {self.method}")

        self.is_fitted = True
        return self

    def calibrate(self, raw_scores: np.ndarray | list[float] | float) -> np.ndarray:
        """Chuyển đổi điểm số thô thành xác suất đã hiệu chuẩn trong khoảng [0.0, 1.0]."""
        scores_arr = np.asarray(raw_scores, dtype=float)
        scalar_input = scores_arr.ndim == 0
        scores_flat = scores_arr.reshape(-1)

        if not self.is_fitted or self.method == "none":
            calibrated = np.clip(scores_flat, 0.0, 1.0)

        elif self.method == "isotonic":
            x_pts = np.asarray(self.params.get("x_thresholds", [0.0, 1.0]), dtype=float)
            y_pts = np.asarray(self.params.get("y_thresholds", [0.0, 1.0]), dtype=float)
            calibrated = np.interp(scores_flat, x_pts, y_pts, left=y_pts[0], right=y_pts[-1])

        elif self.method == "sigmoid":
            coef = self.params.get("coef", 1.0)
            intercept = self.params.get("intercept", 0.0)
            z = coef * scores_flat + intercept
            # Sigmoid an toàn chống tràn số
            calibrated = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

        else:
            calibrated = np.clip(scores_flat, 0.0, 1.0)

        calibrated = np.clip(calibrated, 0.0, 1.0)
        return float(calibrated[0]) if scalar_input else calibrated

    def evaluate_fit(
        self,
        raw_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Đo lường ECE và Brier Score trước và sau khi hiệu chuẩn."""
        raw_arr = np.asarray(raw_scores, dtype=float).ravel()
        y_arr = np.asarray(y_true, dtype=int).ravel()
        cal_arr = self.calibrate(raw_arr)

        return {
            "ece_before": compute_ece(y_arr, raw_arr),
            "ece_after": compute_ece(y_arr, cal_arr),
            "brier_before": float(round(brier_score_loss(y_arr, np.clip(raw_arr, 0.0, 1.0)), 4)),
            "brier_after": float(round(brier_score_loss(y_arr, cal_arr), 4)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbabilityCalibrator:
        method = str(data.get("method", "isotonic"))
        params = dict(data.get("params", {}))
        return cls(method=method, params=params)
