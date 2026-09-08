"""Tối ưu hóa ngưỡng vận hành (Operating Threshold Optimization) trên tập Calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from .policy import ActionPolicy


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


def select_action_policy(
    y_true: Any,
    calibrated_scores: Any,
    *,
    caution_max_fpr: float = 0.02,
    block_max_fpr: float = 0.005,
    caution_min_recall: float = 0.90,
    block_min_recall: float = 0.80,
    policy_version: str = "browser-risk-v1",
) -> dict[str, Any]:
    """Chọn hai ngưỡng trên Policy Validation độc lập với Calibration.

    Caution được phép nhạy hơn vì chỉ hiển thị cảnh báo mềm. Block phải giữ FPR
    thấp hơn để hạn chế chặn nhầm. Không dùng cost giả định để quyết định hành
    động production; cost chỉ là thông tin nhạy cảm trong benchmark.
    """
    if not (
        0.0 <= block_max_fpr <= caution_max_fpr <= 1.0
        and 0.0 <= block_min_recall <= 1.0
        and 0.0 <= caution_min_recall <= 1.0
    ):
        raise ValueError("FPR policy phải thỏa 0 <= block_max_fpr <= caution_max_fpr <= 1")

    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    scores_arr = np.clip(np.asarray(calibrated_scores, dtype=float).ravel(), 0.0, 1.0)
    if len(y_true_arr) == 0 or len(y_true_arr) != len(scores_arr):
        raise ValueError("y_true và calibrated_scores phải có cùng số phần tử và không rỗng")

    candidates = np.unique(np.concatenate(([0.0, 1.0], scores_arr)))

    def metrics_at(threshold: float) -> dict[str, float]:
        predictions = (scores_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, predictions, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if fp + tn else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        return {"threshold": float(threshold), "fpr": fpr, "recall": recall}

    candidate_metrics = [metrics_at(float(th)) for th in candidates]
    caution_candidates = [
        item
        for item in candidate_metrics
        if item["fpr"] <= caution_max_fpr and item["recall"] >= caution_min_recall
    ]
    block_candidates = [
        item
        for item in candidate_metrics
        if item["fpr"] <= block_max_fpr and item["recall"] >= block_min_recall
    ]
    pairs = [
        (caution, block)
        for caution in caution_candidates
        for block in block_candidates
        if caution["threshold"] < block["threshold"]
    ]
    if not pairs:
        raise ValueError(
            "POLICY_NOT_RELEASABLE: không đạt đồng thời FPR, minimum recall và thứ tự ngưỡng"
        )

    # Chọn cặp đồng thời: ưu tiên caution recall, sau đó block recall; cuối cùng
    # ưu tiên threshold cao hơn để giảm cảnh báo/chặn nhầm trong các trường hợp hòa.
    caution, block = max(
        pairs,
        key=lambda pair: (
            pair[0]["recall"],
            pair[1]["recall"],
            pair[1]["threshold"],
            pair[0]["threshold"],
        ),
    )

    policy = ActionPolicy(
        caution_threshold=round(caution["threshold"], 6),
        block_threshold=round(block["threshold"], 6),
        policy_version=policy_version,
    )
    return {
        "policy": policy.to_dict(),
        "caution_metrics": caution,
        "block_metrics": block,
        "constraints": {
            "caution_max_fpr": caution_max_fpr,
            "block_max_fpr": block_max_fpr,
            "caution_min_recall": caution_min_recall,
            "block_min_recall": block_min_recall,
        },
    }
