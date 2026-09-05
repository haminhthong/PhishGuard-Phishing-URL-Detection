"""Hợp đồng đầu vào giữa pipeline huấn luyện và dịch vụ dự đoán PhishGuard ML."""

from __future__ import annotations

# Legacy v1 contract (12 features)
FEATURE_CONTRACT_V1 = "lexical-v1"
FEATURE_COLUMNS_V1: tuple[str, ...] = (
    "Having_IP",
    "Tiny_URL",
    "TLD_Length",
    "Digit_Count",
    "Dot_Count",
    "At_Count",
    "Hyphen_Count",
    "Per_Count",
    "Equal_Count",
    "Redirection",
    "Depth",
    "FD_Length",
)

# Canonical v2 contract (25 features in 4 groups)
FEATURE_CONTRACT_V2 = "lexical-v2"
FEATURE_COLUMNS_V2: tuple[str, ...] = (
    # Group A: Lexical Structure
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "url_entropy",
    "digit_ratio",
    "special_char_ratio",
    "dot_count",
    "hyphen_count",
    "at_count",
    "path_depth",
    "first_directory_length",
    # Group B: Host & Domain
    "subdomain_count",
    "hostname_label_count",
    "max_label_length",
    "has_ip_address",
    "tld_length",
    "domain_length",
    "is_suspicious_tld",
    "has_punycode",
    # Group C: Brand Abuse
    "brand_in_subdomain",
    "brand_in_path",
    "brand_not_registered_domain",
    # Group D: Heuristics & Redirection
    "uses_shortening_service",
    "has_redirection_pattern",
)

# Active contract for baseline (v1) / upgradeable to v2
FEATURE_CONTRACT_VERSION = FEATURE_CONTRACT_V1
FEATURE_COLUMNS = FEATURE_COLUMNS_V1
