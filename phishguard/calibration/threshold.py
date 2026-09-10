"""Tối ưu hóa ngưỡng vận hành trên tập threshold validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from .policy import DecisionThresholds


def select_decision_thresholds(
    y_true: Any,
    calibrated_scores: Any,
    *,
    caution_max_fpr: float = 0.02,
    block_max_fpr: float = 0.005,
    caution_min_recall: float = 0.90,
    block_min_recall: float = 0.80,
) -> dict[str, Any]:
    """Chọn hai ngưỡng trên threshold validation độc lập với calibration.

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
        raise ValueError("Không tìm được cặp threshold thỏa FPR, recall và thứ tự ngưỡng")

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

    thresholds = DecisionThresholds(
        caution_threshold=round(caution["threshold"], 6),
        block_threshold=round(block["threshold"], 6),
    )
    return {
        "thresholds": thresholds.to_dict(),
        "caution_metrics": caution,
        "block_metrics": block,
        "constraints": {
            "caution_max_fpr": caution_max_fpr,
            "block_max_fpr": block_max_fpr,
            "caution_min_recall": caution_min_recall,
            "block_min_recall": block_min_recall,
        },
    }
