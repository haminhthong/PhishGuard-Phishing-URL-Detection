"""Tối ưu hóa ngưỡng vận hành (Operating Threshold Optimization) trên tập Calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


def sweep_operating_threshold(
    y_true: Any,
    y_scores: Any,
    *,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    max_fpr: float = 0.005,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Quét threshold toàn diện từ 0.01 đến 0.99 trên tập Calibration:
    1. Max F1 Threshold
    2. Constrained Threshold: Maximize Recall s.t. FPR <= max_fpr (ví dụ <= 0.5%)
    3. Cost-Optimal Threshold: Minimize Expected Cost = cost_fn * FN + cost_fp * FP
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    sweep_records = []
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_scores_arr = np.asarray(y_scores, dtype=float).ravel()

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
