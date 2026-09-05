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
        mask = (y_prob_arr >= bin_lower) & (y_prob_arr < bin_upper if i < n_bins - 1 else y_prob_arr <= bin_upper)
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
    latency_ms: float | None = None,
) -> dict[str, float]:
    """Tính metric báo cáo đầy đủ, bao gồm PR-AUC, ROC-AUC, FPR, FNR, Brier score, ECE và Latency."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ece = compute_ece(y_true, y_score)
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
        "expected_calibration_error": ece,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    if latency_ms is not None:
        metrics["latency_ms"] = float(latency_ms)
    return metrics


def threshold_sweep(
    y_true: Any,
    y_scores: Any,
    *,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    max_fpr: float = 0.005,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Quét threshold toàn diện từ 0.01 đến 0.99 để chọn ngưỡng vận hành:
    1. Max F1 Threshold
    2. Constrained Threshold: Maximize Recall s.t. FPR <= max_fpr (ví dụ <= 0.5%)
    3. Cost-Optimal Threshold: Minimize Expected Cost = cost_fn * FN + cost_fp * FP
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    sweep_records = []
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    best_f1 = -1.0
    best_f1_th = 0.5

    min_cost = float("inf")
    min_cost_th = 0.5

    best_constrained_recall = -1.0
    best_constrained_th = 0.5

    for th in thresholds:
        th_val = round(float(th), 4)
        preds = (y_scores_arr >= th_val).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, preds, labels=[0, 1]).ravel()

        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
        expected_cost = float(cost_fn * fn + cost_fp * fp)

        record = {
            "threshold": th_val,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "expected_cost": round(expected_cost, 2),
        }
        sweep_records.append(record)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = th_val

        if expected_cost < min_cost:
            min_cost = expected_cost
            min_cost_th = th_val

        if fpr <= max_fpr and recall > best_constrained_recall:
            best_constrained_recall = recall
            best_constrained_th = th_val

    if best_constrained_recall < 0:
        best_constrained_th = min_cost_th

    return {
        "max_f1_threshold": best_f1_th,
        "max_f1_value": round(best_f1, 4),
        "min_cost_threshold": min_cost_th,
        "min_cost_value": round(min_cost, 2),
        "constrained_threshold": best_constrained_th,
        "constrained_max_fpr": max_fpr,
        "constrained_recall": round(best_constrained_recall, 4),
        "cost_assumptions": {"cost_fn": cost_fn, "cost_fp": cost_fp},
        "sweep_table": sweep_records,
    }


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
