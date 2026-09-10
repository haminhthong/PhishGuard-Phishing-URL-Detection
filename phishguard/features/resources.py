"""Nạp và kiểm tra các tài nguyên lexical của PhishGuard.

Tài nguyên là một phần của model contract. Không dùng fallback im lặng vì
fallback có thể tạo train-serving skew mà người vận hành không nhận ra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
BRAND_TERMS_FILENAME = "brand_terms.json"
SHORTENER_DOMAINS_FILENAME = "shortener_domains.json"
SUSPICIOUS_TLDS_FILENAME = "suspicious_tlds.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu tài nguyên bắt buộc: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Không thể đọc tài nguyên {path.name}: {error}") from error
    if not isinstance(value, dict) or not value.get("version"):
        raise RuntimeError(f"Tài nguyên {path.name} thiếu trường version hợp lệ")
    return value


@dataclass(frozen=True)
class ResourceBundle:
    """Bộ tài nguyên lexical được kiểm tra schema khi khởi động."""

    brands: tuple[dict[str, Any], ...]
    shorteners: frozenset[str]
    suspicious_tlds: frozenset[str]


def load_resource_bundle(resources_dir: Path = RESOURCES_DIR) -> ResourceBundle:
    """Nạp resource và fail-fast nếu thiếu, sai JSON hoặc sai schema."""
    brand_path = resources_dir / BRAND_TERMS_FILENAME
    shortener_path = resources_dir / SHORTENER_DOMAINS_FILENAME
    tld_path = resources_dir / SUSPICIOUS_TLDS_FILENAME
    brand_data = _load_json(brand_path)
    shortener_data = _load_json(shortener_path)
    tld_data = _load_json(tld_path)

    brands = brand_data.get("brands")
    shorteners = shortener_data.get("shorteners")
    suspicious_tlds = tld_data.get("tlds")
    if not isinstance(brands, list) or not brands:
        raise RuntimeError("brand_terms.json phải chứa danh sách brands không rỗng")
    if not isinstance(shorteners, list) or not shorteners:
        raise RuntimeError("shortener_domains.json phải chứa danh sách shorteners không rỗng")
    if not isinstance(suspicious_tlds, list) or not suspicious_tlds:
        raise RuntimeError("suspicious_tlds.json phải chứa danh sách tlds không rỗng")

    for brand in brands:
        if (
            not isinstance(brand, dict)
            or not brand.get("name")
            or not isinstance(brand.get("official_domains"), list)
        ):
            raise RuntimeError("Mỗi brand phải có name và official_domains")

    return ResourceBundle(
        brands=tuple(brands),
        shorteners=frozenset(str(item).lower().strip() for item in shorteners),
        suspicious_tlds=frozenset(str(item).lower().strip(".") for item in suspicious_tlds),
    )


RESOURCE_BUNDLE = load_resource_bundle()
