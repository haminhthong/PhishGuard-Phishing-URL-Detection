"""Trích xuất đặc trưng lexical dùng chung cho training và inference."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import ParseResult, urlparse

from tld import get_tld

from .contract import FEATURE_COLUMNS

SHORTENING_SERVICES = re.compile(
    r"^(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tiny\.cc|is\.gd|"
    r"ow\.ly|buff\.ly|rebrand\.ly|cutt\.ly|shorturl\.at|rb\.gy|v\.gd)$",
    re.IGNORECASE,
)


def parse_url(url: str) -> ParseResult:
    """Phân tích URL và không làm sập pipeline nếu hostname sai cú pháp."""
    try:
        return urlparse(url)
    except ValueError:
        return urlparse("")


def has_ip_address(url: str) -> int:
    """Trả 1 khi hostname là địa chỉ IPv4 hoặc IPv6."""
    hostname = parse_url(url).hostname
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return 0
    return 1


def uses_shortening_service(url: str) -> int:
    """Trả 1 khi toàn bộ hostname khớp dịch vụ rút gọn đã biết."""
    hostname = (parse_url(url).hostname or "").lower().rstrip(".")
    return int(bool(SHORTENING_SERVICES.fullmatch(hostname)))


def tld_length(url: str) -> int:
    """Tính độ dài TLD; trả 0 nếu không xác định được."""
    try:
        top_level_domain = get_tld(url, fail_silently=True)
    except (TypeError, ValueError):
        return 0
    return len(top_level_domain) if top_level_domain else 0


def first_directory_length(url: str) -> int:
    """Tính độ dài thành phần đầu tiên trong đường dẫn."""
    parts = [part for part in parse_url(url).path.split("/") if part]
    return len(parts[0]) if parts else 0


def path_depth(url: str) -> int:
    """Đếm số thành phần không rỗng trong đường dẫn."""
    return sum(bool(part) for part in parse_url(url).path.split("/"))


def has_redirection_pattern(url: str) -> int:
    """Phát hiện dấu // nằm ngoài dấu phân cách giao thức."""
    scheme_separator = url.find("://")
    search_start = scheme_separator + 3 if scheme_separator >= 0 else 0
    return int("//" in url[search_start:])


def extract_features(url: str) -> dict[str, int]:
    """Tạo đúng 12 đặc trưng theo hợp đồng lexical-v1."""
    features = {
        "Having_IP": has_ip_address(url),
        "Tiny_URL": uses_shortening_service(url),
        "TLD_Length": tld_length(url),
        "Digit_Count": sum(character.isdigit() for character in url),
        "Dot_Count": url.count("."),
        "At_Count": url.count("@"),
        "Hyphen_Count": url.count("-"),
        "Per_Count": url.count("%"),
        "Equal_Count": url.count("="),
        "Redirection": has_redirection_pattern(url),
        "Depth": path_depth(url),
        "FD_Length": first_directory_length(url),
    }
    assert tuple(features) == FEATURE_COLUMNS, "Sai thứ tự hợp đồng đặc trưng"
    return features
