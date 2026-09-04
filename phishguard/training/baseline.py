"""Các mô hình baseline để so sánh benchmark với Random Forest và XGBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


class RuleBasedPhishingClassifier(BaseEstimator, ClassifierMixin):
    """
    Rule-based baseline classifier đơn giản dựa trên quy tắc cộng điểm đặc trưng lexical.
    Dùng làm mốc so sánh tối thiểu cho các mô hình Machine Learning.
    """

    def __init__(self, score_threshold: float = 3.0) -> None:
        self.score_threshold = score_threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> RuleBasedPhishingClassifier:
        return self

    def _compute_risk_score(self, row: pd.Series | np.ndarray) -> float:
        # Giả định thứ tự hoặc tên cột theo FEATURE_COLUMNS:
        # Having_IP, Tiny_URL, TLD_Length, Digit_Count, Dot_Count, At_Count,
        # Hyphen_Count, Per_Count, Equal_Count, Redirection, Depth, FD_Length
        if isinstance(row, pd.Series):
            having_ip = row.get("Having_IP", 0)
            tiny_url = row.get("Tiny_URL", 0)
            at_count = row.get("At_Count", 0)
            dot_count = row.get("Dot_Count", 0)
            hyphen_count = row.get("Hyphen_Count", 0)
            per_count = row.get("Per_Count", 0)
            redirection = row.get("Redirection", 0)
            depth = row.get("Depth", 0)
        else:
            having_ip = row[0]
            tiny_url = row[1]
            dot_count = row[4]
            at_count = row[5]
            hyphen_count = row[6]
            per_count = row[7]
            redirection = row[9]
            depth = row[10]

        score = 0.0
        if having_ip > 0:
            score += 2.5
        if tiny_url > 0:
            score += 2.0
        if at_count > 0:
            score += 2.0
        if redirection > 0:
            score += 2.0
        if dot_count > 3:
            score += 1.0
        if hyphen_count > 1:
            score += 1.0
        if per_count > 0:
            score += 1.0
        if depth > 3:
            score += 1.0

        return score

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            scores = np.array([self._compute_risk_score(row) for _, row in X.iterrows()])
        else:
            scores = np.array([self._compute_risk_score(row) for row in X])

        # Chuẩn hóa score về dải [0, 1] để giả định xác suất
        prob_phishing = np.clip(scores / 8.0, 0.0, 1.0)
        prob_legit = 1.0 - prob_phishing
        return np.column_stack([prob_legit, prob_phishing])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] >= (self.score_threshold / 8.0)).astype(int)


def create_baseline_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Tạo tập hợp các mô hình baseline để đánh giá so sánh."""
    return {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Dummy Stratified": DummyClassifier(strategy="stratified", random_state=random_state),
        "Rule-based": RuleBasedPhishingClassifier(score_threshold=3.0),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
    }
