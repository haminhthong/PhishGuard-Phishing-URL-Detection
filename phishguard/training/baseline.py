"""Các baseline sanity và Logistic Regression của PhishGuard."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from phishguard.features import FEATURE_COLUMNS


class RuleBasedPhishingClassifier(BaseEstimator, ClassifierMixin):
    """
    Rule-based baseline classifier đơn giản dựa trên quy tắc cộng điểm đặc trưng lexical.
    Dùng làm mốc so sánh tối thiểu cho các mô hình Machine Learning.
    """

    def __init__(self, score_threshold: float = 3.0) -> None:
        self.score_threshold = score_threshold
        self.classes_ = np.array([0, 1])

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None
    ) -> RuleBasedPhishingClassifier:
        return self

    def _compute_risk_score(self, row: pd.Series | np.ndarray) -> float:
        score = 0.0
        if isinstance(row, pd.Series):
            having_ip = row.get("has_ip_address", 0)
            tiny_url = row.get("uses_shortening_service", 0)
            at_count = row.get("at_count", 0)
            dot_count = row.get("dot_count", 0)
            hyphen_count = row.get("hyphen_count", 0)
            per_count = row.get("query_length", 0)
            redirection = row.get("has_redirection_pattern", 0)
            depth = row.get("path_depth", 0)
            brand_abuse = row.get("brand_not_registered_domain", 0)
            punycode = row.get("has_punycode", 0)
            subdomains = row.get("subdomain_count", 0)
            suspicious_tld = row.get("is_suspicious_tld", 0)
            if brand_abuse > 0:
                score += 3.0
            if punycode > 0:
                score += 2.5
            if subdomains > 2:
                score += 1.5
            if suspicious_tld > 0:
                score += 1.5
        else:
            values = dict(zip(FEATURE_COLUMNS, row, strict=False))
            having_ip = values.get("has_ip_address", 0)
            tiny_url = values.get("uses_shortening_service", 0)
            dot_count = values.get("dot_count", 0)
            at_count = values.get("at_count", 0)
            hyphen_count = values.get("hyphen_count", 0)
            per_count = values.get("query_length", 0)
            redirection = values.get("has_redirection_pattern", 0)
            depth = values.get("path_depth", 0)

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
