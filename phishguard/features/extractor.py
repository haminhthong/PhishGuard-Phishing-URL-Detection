"""Trích xuất đặc trưng lexical v1 và v2 dùng chung cho training và inference."""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import ParseResult, urlparse

from tld import get_fld, get_tld

from .contract import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_VERSION,
)

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
BRAND_TERMS_PATH = RESOURCES_DIR / "brand_terms.json"
SHORTENER_DOMAINS_PATH = RESOURCES_DIR / "shortener_domains.json"

SUSPICIOUS_TLDS = {
    "xyz", "top", "icu", "buzz", "cc", "tk", "ml", "ga", "cf", "gq",
    "work", "fit", "surf", "click", "link", "club", "rest", "cam", "vip",
}


def _load_shortener_domains() -> set[str]:
    """Nạp danh sách tên miền rút gọn có version từ file tài nguyên."""
    if SHORTENER_DOMAINS_PATH.is_file():
        try:
            with open(SHORTENER_DOMAINS_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return set(domain.lower() for domain in data.get("shorteners", []))
        except Exception:
            pass
    # Fallback mặc định an toàn
    return {
        "bit.ly", "goo.gl", "t.co", "tinyurl.com", "tiny.cc", "is.gd",
        "ow.ly", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "v.gd",
    }


def _load_brand_terms() -> list[dict[str, Any]]:
    """Nạp danh mục thương hiệu và tên miền chính thức có version."""
    if BRAND_TERMS_PATH.is_file():
        try:
            with open(BRAND_TERMS_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("brands", [])
        except Exception:
            pass
    # Fallback mặc định
    return [
        {"name": "paypal", "official_domains": ["paypal.com", "paypal.me"]},
        {"name": "google", "official_domains": ["google.com", "google.com.vn", "google.co.in", "gmail.com", "youtube.com"]},
        {"name": "microsoft", "official_domains": ["microsoft.com", "live.com", "office.com", "outlook.com"]},
        {"name": "apple", "official_domains": ["apple.com", "icloud.com"]},
        {"name": "amazon", "official_domains": ["amazon.com", "aws.amazon.com"]},
        {"name": "facebook", "official_domains": ["facebook.com", "fb.com", "instagram.com"]},
        {"name": "netflix", "official_domains": ["netflix.com"]},
        {"name": "chase", "official_domains": ["chase.com"]},
        {"name": "wellsfargo", "official_domains": ["wellsfargo.com"]},
        {"name": "bankofamerica", "official_domains": ["bankofamerica.com", "bofa.com"]},
        {"name": "binance", "official_domains": ["binance.com"]},
        {"name": "coinbase", "official_domains": ["coinbase.com"]},
        {"name": "steam", "official_domains": ["steampowered.com", "steamcommunity.com"]},
        {"name": "dhl", "official_domains": ["dhl.com"]},
        {"name": "allegro", "official_domains": ["allegro.pl", "allegrolokalnie.pl"]},
    ]


KNOWN_SHORTENERS = _load_shortener_domains()
KNOWN_BRANDS = _load_brand_terms()


def parse_url(url: str) -> ParseResult:
    """Phân tích URL an toàn, không làm crash pipeline nếu URL lỗi."""
    try:
        return urlparse(url)
    except ValueError:
        return urlparse("")


def calculate_entropy(text: str) -> float:
    """Tính Shannon entropy đo mức độ ngẫu nhiên của chuỗi ký tự."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())
    return round(float(entropy), 4)


def has_ip_address(url: str) -> int:
    """Trả 1 khi hostname là địa chỉ IPv4 hoặc IPv6."""
    hostname = parse_url(url).hostname
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def uses_shortening_service(url: str) -> int:
    """Trả 1 khi hostname thuộc danh sách dịch vụ rút gọn đã biết."""
    hostname = (parse_url(url).hostname or "").lower().rstrip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return int(hostname in KNOWN_SHORTENERS)


def tld_info(url: str) -> tuple[int, int]:
    """Tính độ dài TLD và kiểm tra TLD có nằm trong nhóm có rủi ro cao (suspicious)."""
    try:
        tld_str = get_tld(url, fail_silently=True)
    except (TypeError, ValueError):
        tld_str = None
    if not tld_str:
        return 0, 0
    tld_clean = str(tld_str).lower().strip(".")
    length = len(tld_clean)
    is_suspicious = int(tld_clean in SUSPICIOUS_TLDS)
    return length, is_suspicious


def tld_length(url: str) -> int:
    """Tính độ dài TLD; trả 0 nếu không xác định được."""
    return tld_info(url)[0]


def first_directory_length(url: str) -> int:
    """Tính độ dài thành phần thư mục đầu tiên trong đường dẫn."""
    parts = [part for part in parse_url(url).path.split("/") if part]
    return len(parts[0]) if parts else 0


def path_depth(url: str) -> int:
    """Đếm số thành phần không rỗng trong đường dẫn."""
    return sum(bool(part) for part in parse_url(url).path.split("/"))


def has_redirection_pattern(url: str) -> int:
    """Phát hiện dấu // nằm ngoài dấu phân cách giao thức chuẩn."""
    scheme_separator = url.find("://")
    search_start = scheme_separator + 3 if scheme_separator >= 0 else 0
    return int("//" in url[search_start:])


def has_punycode(url: str) -> int:
    """Phát hiện URL sử dụng mã hóa ký tự Punycode (xn--)."""
    parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower()
    netloc = (parsed.netloc or "").lower()
    return int("xn--" in hostname or "xn--" in netloc)


def get_subdomain_and_labels(url: str) -> tuple[int, int, int, str]:
    """
    Phân tích cấu trúc hostname:
    - Số lượng subdomain
    - Tổng số nhãn (dot-separated labels)
    - Độ dài nhãn lớn nhất
    - Chuỗi subdomain
    """
    parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        return 0, 0, 0, ""

    labels = [label for label in hostname.split(".") if label]
    label_count = len(labels)
    max_label_len = max((len(lbl) for lbl in labels), default=0)

    try:
        fld = get_fld(url, fail_silently=True)
    except (TypeError, ValueError):
        fld = None

    if fld and hostname.endswith(fld):
        subdomain_part = hostname[: -len(fld)].rstrip(".")
        subdomain_labels = [s for s in subdomain_part.split(".") if s]
        subdomain_count = len(subdomain_labels)
    else:
        # Fallback nếu không parse được fld
        subdomain_count = max(0, label_count - 2)
        subdomain_part = ".".join(labels[:-2]) if label_count > 2 else ""

    return subdomain_count, label_count, max_label_len, subdomain_part


def detect_brand_abuse(url: str) -> tuple[int, int, int]:
    """
    Phát hiện tín hiệu mạo danh thương hiệu (Brand Impersonation):
    - brand_in_subdomain: Tên thương hiệu xuất hiện trong subdomain
    - brand_in_path: Tên thương hiệu xuất hiện trong path
    - brand_not_registered_domain: Thương hiệu xuất hiện trong URL nhưng không thuộc domain chính thức
    """
    parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    path_lower = (parsed.path or "").lower()

    try:
        registered_domain = get_fld(url, fail_silently=True)
    except (TypeError, ValueError):
        registered_domain = None
    if not registered_domain:
        registered_domain = hostname

    reg_domain_clean = (registered_domain or "").lower()

    _, _, _, subdomain = get_subdomain_and_labels(url)
    subdomain_lower = subdomain.lower()

    brand_in_sub = 0
    brand_in_path = 0
    brand_not_reg = 0

    for brand_info in KNOWN_BRANDS:
        brand_name = brand_info["name"].lower()
        official_domains = [d.lower() for d in brand_info["official_domains"]]

        in_sub = brand_name in subdomain_lower
        in_path = bool(re.search(rf"(?:^|/|-|_){re.escape(brand_name)}(?:$|/|-|_|\.)", path_lower))

        if in_sub or in_path:
            is_official = any(reg_domain_clean == off or reg_domain_clean.endswith("." + off) for off in official_domains)
            if in_sub:
                brand_in_sub = 1
            if in_path:
                brand_in_path = 1
            if not is_official:
                brand_not_reg = 1

    return brand_in_sub, brand_in_path, brand_not_reg


def extract_features_v1(url: str) -> dict[str, int]:
    """Tạo đúng 12 đặc trưng theo hợp đồng lexical-v1 (Legacy)."""
    tld_len, _ = tld_info(url)
    features = {
        "Having_IP": has_ip_address(url),
        "Tiny_URL": uses_shortening_service(url),
        "TLD_Length": tld_len,
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
    assert tuple(features) == FEATURE_COLUMNS_V1, "Sai thứ tự hợp đồng đặc trưng v1"
    return features


def extract_features_v2(url: str) -> dict[str, int | float]:
    """
    Tạo 25 đặc trưng theo hợp đồng lexical-v2 phân thành 4 nhóm:
    A. Cấu trúc Lexical & Ratios
    B. Host & Domain
    C. Brand Impersonation
    D. Heuristics & Redirection
    """
    parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or ""
    query = parsed.query or ""

    url_len = len(url)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    # Ratios
    digits = sum(c.isdigit() for c in url)
    digit_ratio = round(digits / max(url_len, 1), 4)

    special_chars = sum(url.count(ch) for ch in (".", "-", "@", "%", "=", "/", "?", "_", "&"))
    special_char_ratio = round(special_chars / max(url_len, 1), 4)

    # Domain & Host
    subdomain_count, label_count, max_label_len, _ = get_subdomain_and_labels(url)
    tld_len, is_suspicious = tld_info(url)

    try:
        reg_domain = get_fld(url, fail_silently=True) or hostname
    except Exception:
        reg_domain = hostname
    domain_len = len(reg_domain)

    # Brand signals
    brand_sub, brand_path, brand_not_reg = detect_brand_abuse(url)

    features = {
        # Group A: Lexical Structure
        "url_length": url_len,
        "hostname_length": host_len,
        "path_length": path_len,
        "query_length": query_len,
        "url_entropy": calculate_entropy(url),
        "digit_ratio": digit_ratio,
        "special_char_ratio": special_char_ratio,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "path_depth": path_depth(url),
        "first_directory_length": first_directory_length(url),
        # Group B: Host & Domain
        "subdomain_count": subdomain_count,
        "hostname_label_count": label_count,
        "max_label_length": max_label_len,
        "has_ip_address": has_ip_address(url),
        "tld_length": tld_len,
        "domain_length": domain_len,
        "is_suspicious_tld": is_suspicious,
        "has_punycode": has_punycode(url),
        # Group C: Brand Abuse
        "brand_in_subdomain": brand_sub,
        "brand_in_path": brand_path,
        "brand_not_registered_domain": brand_not_reg,
        # Group D: Heuristics & Redirection
        "uses_shortening_service": uses_shortening_service(url),
        "has_redirection_pattern": has_redirection_pattern(url),
    }

    assert tuple(features) == FEATURE_COLUMNS_V2, "Sai thứ tự hợp đồng đặc trưng v2"
    return features


def extract_features(url: str, contract: str = FEATURE_CONTRACT_VERSION) -> dict[str, Any]:
    """Hàm trích xuất đặc trưng chính dựa trên phiên bản hợp đồng được chỉ định."""
    if contract == FEATURE_CONTRACT_V1:
        return extract_features_v1(url)
    return extract_features_v2(url)
