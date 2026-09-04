"""Hợp đồng và hàm trích xuất đặc trưng URL dùng chung."""

from .contract import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION
from .extractor import (
    extract_features,
    first_directory_length,
    has_ip_address,
    has_redirection_pattern,
    parse_url,
    path_depth,
    tld_length,
    uses_shortening_service,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_CONTRACT_VERSION",
    "extract_features",
    "first_directory_length",
    "has_ip_address",
    "has_redirection_pattern",
    "parse_url",
    "path_depth",
    "tld_length",
    "uses_shortening_service",
]
