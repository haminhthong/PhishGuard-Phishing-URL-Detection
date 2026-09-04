"""Metric phù hợp cho bài toán phát hiện phishing mất cân bằng dữ liệu."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    latency_ms: float | None = None,
) -> dict[str, float]:
    """Tính metric báo cáo đầy đủ, bao gồm PR-AUC, ROC-AUC, FPR, FNR, Brier score và Latency."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    if latency_ms is not None:
        metrics["latency_ms"] = float(latency_ms)
    return metrics


def measure_inference_latency(
    model: Any,
    X_sample: Any,
    num_runs: int = 100,
) -> tuple[float, float]:
    """Đo thời gian dự đoán p50 và p95 (ms)."""
    latencies = []
    # Warmup
    for _ in range(5):
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_sample[:10])
        else:
            _ = model.predict(X_sample[:10])

    for _ in range(num_runs):
        start = time.perf_counter()
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_sample[:10])
        else:
            _ = model.predict(X_sample[:10])
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 10.0
        latencies.append(elapsed_ms)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    return p50, p95
