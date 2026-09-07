"""Hợp đồng đặc trưng dùng chung giữa training và serving.

`lexical-v3` giữ nguyên 25 cột của v2 để không làm thay đổi semantics mô hình,
nhưng đóng băng thêm phiên bản của các tài nguyên bên ngoài. Vì vậy việc nâng
version không tự ý tạo thêm feature hoặc làm thay đổi thứ tự cột.
"""

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

# V3 đóng băng semantics 25 feature của v2 và bổ sung resource contract.
# Không sao chép tuple để tránh hai danh sách cột bị lệch theo thời gian.
FEATURE_CONTRACT_V3 = "lexical-v3"
FEATURE_COLUMNS_V3: tuple[str, ...] = FEATURE_COLUMNS_V2

# Hợp đồng mặc định mới cho pipeline huấn luyện. Artifact lexical-v2 cũ vẫn
# được loader hỗ trợ để có thể rollback an toàn.
FEATURE_CONTRACT_VERSION = FEATURE_CONTRACT_V3
FEATURE_COLUMNS = FEATURE_COLUMNS_V3
