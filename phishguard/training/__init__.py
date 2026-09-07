"""Pipeline dữ liệu và đánh giá mô hình có thể tái lập PhishGuard ML."""

from .baseline import RuleBasedPhishingClassifier, create_baseline_models
from .data import (
    DatasetSplits,
    FiveWayDatasetSplits,
    FourWayDatasetSplits,
    TemporalSplits,
    audit_and_clean_data,
    clean_dataset,
    registered_domain,
    split_by_domain,
    split_by_domain_4way,
    split_by_domain_5way,
    temporal_split_future_unseen_domains,
    temporal_split_protocol_b,
)
from .evaluation import (
    classification_metrics,
    compute_ece,
    measure_inference_latency,
    threshold_sweep,
)

__all__ = [
    "DatasetSplits",
    "FourWayDatasetSplits",
    "FiveWayDatasetSplits",
    "RuleBasedPhishingClassifier",
    "TemporalSplits",
    "audit_and_clean_data",
    "classification_metrics",
    "clean_dataset",
    "compute_ece",
    "create_baseline_models",
    "measure_inference_latency",
    "registered_domain",
    "split_by_domain",
    "split_by_domain_4way",
    "split_by_domain_5way",
    "temporal_split_protocol_b",
    "temporal_split_future_unseen_domains",
    "threshold_sweep",
]
