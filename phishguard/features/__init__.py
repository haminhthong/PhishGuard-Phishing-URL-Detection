"""Hợp đồng và hàm trích xuất đặc trưng URL dùng chung PhishGuard ML."""

from .contract import (
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
)
from .extractor import (
    FeatureExtractor,
    calculate_entropy,
    detect_brand_abuse,
    extract_features,
    first_directory_length,
    get_subdomain_and_labels,
    has_ip_address,
    has_punycode,
    has_redirection_pattern,
    parse_url,
    path_depth,
    tld_info,
    tld_length,
    uses_shortening_service,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_CONTRACT_VERSION",
    "calculate_entropy",
    "detect_brand_abuse",
    "extract_features",
    "first_directory_length",
    "get_subdomain_and_labels",
    "has_ip_address",
    "has_punycode",
    "has_redirection_pattern",
    "parse_url",
    "path_depth",
    "tld_info",
    "tld_length",
    "uses_shortening_service",
    "FeatureExtractor",
]
