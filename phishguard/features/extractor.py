"""Trích xuất đặc trưng lexical dùng chung cho training và inference."""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import ParseResult, urlparse

import pandas as pd
from tld import get_fld, get_tld

from .contract import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION
from .resources import RESOURCE_BUNDLE, ResourceBundle


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


def uses_shortening_service(url: str, resources: ResourceBundle = RESOURCE_BUNDLE) -> int:
    """Trả 1 khi hostname thuộc danh sách dịch vụ rút gọn đã biết."""
    hostname = (parse_url(url).hostname or "").lower().rstrip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return int(hostname in resources.shorteners)


def tld_info(url: str, resources: ResourceBundle = RESOURCE_BUNDLE) -> tuple[int, int]:
    """Tính độ dài TLD và kiểm tra TLD có nằm trong nhóm có rủi ro cao (suspicious)."""
    try:
        tld_str = get_tld(url, fail_silently=True)
    except (TypeError, ValueError):
        tld_str = None
    if not tld_str:
        return 0, 0
    tld_clean = str(tld_str).lower().strip(".")
    length = len(tld_clean)
    is_suspicious = int(tld_clean in resources.suspicious_tlds)
    return length, is_suspicious


def tld_length(url: str, resources: ResourceBundle = RESOURCE_BUNDLE) -> int:
    """Tính độ dài TLD; trả 0 nếu không xác định được."""
    return tld_info(url, resources)[0]


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


def detect_brand_abuse(
    url: str,
    resources: ResourceBundle = RESOURCE_BUNDLE,
    *,
    strict_brand_matching: bool = False,
) -> tuple[int, int, int]:
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

    for brand_info in resources.brands:
        brand_name = brand_info["name"].lower()
        official_domains = [d.lower() for d in brand_info["official_domains"]]

        if strict_brand_matching:
            subdomain_tokens = set(re.split(r"[^a-z0-9]+", subdomain_lower))
            path_tokens = set(re.split(r"[^a-z0-9]+", path_lower))
            in_sub = brand_name in subdomain_tokens
            in_path = brand_name in path_tokens
        else:
            in_sub = brand_name in subdomain_lower
            in_path = bool(
                re.search(rf"(?:^|/|-|_|\.){re.escape(brand_name)}(?:$|/|-|_|\.)", path_lower)
            )

        if in_sub or in_path:
            is_official = any(
                reg_domain_clean == off or reg_domain_clean.endswith("." + off)
                for off in official_domains
            )
            if in_sub:
                brand_in_sub = 1
            if in_path:
                brand_in_path = 1
            if not is_official:
                brand_not_reg = 1

    return brand_in_sub, brand_in_path, brand_not_reg


def extract_features(
    url: str,
    resources: ResourceBundle = RESOURCE_BUNDLE,
    *,
    contract: str = FEATURE_CONTRACT_VERSION,
) -> dict[str, int | float]:
    """
    Tạo 25 đặc trưng lexical-v4 phân thành 4 nhóm:
    A. Cấu trúc Lexical & Ratios
    B. Host & Domain
    C. Brand Impersonation
    D. Heuristics & Redirection
    """
    if contract != FEATURE_CONTRACT_VERSION:
        raise ValueError(f"Chỉ hỗ trợ feature contract {FEATURE_CONTRACT_VERSION}")
    parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or ""
    query = parsed.query or ""

    url_len = len(url)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    # Các tỷ lệ chuẩn hóa
    digits = sum(c.isdigit() for c in url)
    digit_ratio = round(digits / max(url_len, 1), 4)

    special_chars = sum(url.count(ch) for ch in (".", "-", "@", "%", "=", "/", "?", "_", "&"))
    special_char_ratio = round(special_chars / max(url_len, 1), 4)

    # Tên miền và máy chủ
    subdomain_count, label_count, max_label_len, _ = get_subdomain_and_labels(url)
    tld_len, is_suspicious = tld_info(url, resources)

    try:
        reg_domain = get_fld(url, fail_silently=True) or hostname
    except (TypeError, ValueError):
        reg_domain = hostname
    domain_len = len(reg_domain)

    # Tín hiệu mạo danh thương hiệu
    brand_sub, brand_path, brand_not_reg = detect_brand_abuse(
        url, resources, strict_brand_matching=True
    )

    features = {
        # Nhóm A: cấu trúc lexical
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
        # Nhóm B: máy chủ và tên miền
        "subdomain_count": subdomain_count,
        "hostname_label_count": label_count,
        "max_label_length": max_label_len,
        "has_ip_address": has_ip_address(url),
        "tld_length": tld_len,
        "domain_length": domain_len,
        "is_suspicious_tld": is_suspicious,
        "has_punycode": has_punycode(url),
        # Nhóm C: mạo danh thương hiệu
        "brand_in_subdomain": brand_sub,
        "brand_in_path": brand_path,
        "brand_not_registered_domain": brand_not_reg,
        # Nhóm D: heuristic và chuyển hướng
        "uses_shortening_service": uses_shortening_service(url, resources),
        "has_redirection_pattern": has_redirection_pattern(url),
    }

    assert tuple(features) == FEATURE_COLUMNS, "Sai thứ tự hợp đồng lexical-v4"
    return features


@dataclass(frozen=True)
class FeatureExtractor:
    """Extractor gắn với contract và resource runtime đã kiểm tra."""

    contract: str = FEATURE_CONTRACT_VERSION
    resources: ResourceBundle = RESOURCE_BUNDLE

    def extract(self, url: str) -> dict[str, Any]:
        """Trích xuất feature bằng resource đã đóng băng, không đọc global runtime."""
        if self.contract != FEATURE_CONTRACT_VERSION:
            raise ValueError(f"Chỉ hỗ trợ feature contract {FEATURE_CONTRACT_VERSION}")
        return extract_features(url, resources=self.resources)

    def extract_frame(
        self,
        frame: pd.DataFrame,
        url_column: str | None = None,
    ) -> pd.DataFrame:
        """Trích xuất 25 đặc trưng cho DataFrame URLs theo đúng thứ tự FEATURE_COLUMNS."""
        target_col = url_column or ("raw_url" if "raw_url" in frame.columns else "url")
        if target_col not in frame.columns:
            raise KeyError(f"DataFrame thiếu cột URL ('{target_col}' hoặc 'raw_url'/'url')")
        return pd.DataFrame(
            [self.extract(url) for url in frame[target_col]],
            columns=FEATURE_COLUMNS,
        )


def extract_features_dataframe(
    frame: pd.DataFrame,
    url_column: str | None = None,
    resources: ResourceBundle = RESOURCE_BUNDLE,
    contract: str = FEATURE_CONTRACT_VERSION,
) -> pd.DataFrame:
    """Trích xuất DataFrame 25 đặc trưng tiện ích dùng chung."""
    extractor = FeatureExtractor(contract=contract, resources=resources)
    return extractor.extract_frame(frame, url_column=url_column)
