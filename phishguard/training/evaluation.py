"""Metric phù hợp cho bài toán phát hiện phishing mất cân bằng dữ liệu."""

from __future__ import annotations

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


def compute_ece(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """Tính Expected Calibration Error (ECE) chia làm n_bins."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true_arr)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob_arr >= bin_lower) & (
            y_prob_arr < bin_upper if i < n_bins - 1 else y_prob_arr <= bin_upper
        )
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true_arr[mask]))
            bin_conf = float(np.mean(y_prob_arr[mask]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
) -> dict[str, Any]:
    """Tính metric cho test và edge-case report."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ece = compute_ece(y_true, y_score)
    has_both_classes = np.unique(np.asarray(y_true)).size > 1
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # AUC không xác định trên slice chỉ có một nhãn; trả null thay vì hỏng report.
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both_classes else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "expected_calibration_error": ece,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics
