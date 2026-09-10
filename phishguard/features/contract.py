"""Hợp đồng đặc trưng duy nhất dùng chung cho train và serving."""

from __future__ import annotations

FEATURE_CONTRACT_VERSION = "lexical-v4"
FEATURE_COLUMNS: tuple[str, ...] = (
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
