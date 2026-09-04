"""Làm sạch, audit dữ liệu và chia tập theo registered domain để chống data leakage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tld import get_fld


@dataclass(frozen=True)
class DatasetSplits:
    """Ba tập độc lập; test chỉ dùng một lần sau khi chọn mô hình."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def compute_sha256(file_path: str | Path) -> str:
    """Tính mã băm SHA-256 của file dữ liệu."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_url(url: str) -> str:
    """Chuẩn hóa an toàn các phần không làm đổi ngữ nghĩa URL."""
    if not isinstance(url, str):
        return ""
    value = url.strip()
    if not value or len(value) > 2048:
        return ""
    try:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""

    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        return ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port}"
    return urlunparse((parsed.scheme.lower(), netloc, parsed.path or "/", "", parsed.query, ""))


def registered_domain(url: str) -> str:
    """Lấy miền đăng ký (registered domain) làm group; fallback sang hostname khi cần."""
    try:
        domain = get_fld(url, fail_silently=True)
    except (TypeError, ValueError):
        domain = None
    if not domain:
        try:
            domain = urlparse(url).hostname
        except ValueError:
            domain = None
    return (domain or "").lower().rstrip(".")


def audit_and_clean_data(
    legit_df: pd.DataFrame,
    phishing_df: pd.DataFrame,
    legit_path: str | Path | None = None,
    phishing_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Audit chất lượng dữ liệu nguồn, loại bỏ URL không hợp lệ, URL trùng lặp,
    và các registered domain có nhãn mâu thuẫn xuất hiện ở cả 2 nguồn.
    """
    legit_raw_count = len(legit_df)
    phishing_raw_count = len(phishing_df)

    legit_clean = legit_df[["url"]].dropna().copy()
    legit_clean["label"] = 0

    phishing_clean = phishing_df[["url"]].dropna().copy()
    phishing_clean["label"] = 1

    if "submission_time" in phishing_df.columns:
        phishing_clean["submission_time"] = phishing_df["submission_time"]

    combined = pd.concat([legit_clean, phishing_clean], ignore_index=True)

    combined["normalized_url"] = combined["url"].map(normalize_url)
    invalid_mask = combined["normalized_url"] == ""
    invalid_count = int(invalid_mask.sum())
    valid_df = combined[~invalid_mask].copy()

    valid_df["url"] = valid_df["normalized_url"]
    valid_df.drop(columns=["normalized_url"], inplace=True)

    before_dedup = len(valid_df)
    valid_df = valid_df.drop_duplicates(subset=["url"]).copy()
    duplicate_count = before_dedup - len(valid_df)

    valid_df["domain"] = valid_df["url"].map(registered_domain)
    valid_df = valid_df[valid_df["domain"] != ""].copy()

    domain_labels = valid_df.groupby("domain")["label"].nunique()
    conflicting_domains = domain_labels[domain_labels > 1].index.tolist()

    cleaned_df = valid_df[~valid_df["domain"].isin(conflicting_domains)].reset_index(drop=True)
    conflicting_rows_removed = len(valid_df) - len(cleaned_df)

    report = {
        "legitimate_rows": legit_raw_count,
        "phishing_rows": phishing_raw_count,
        "invalid_urls_removed": invalid_count,
        "duplicate_urls_removed": duplicate_count,
        "conflicting_domains_count": len(conflicting_domains),
        "conflicting_rows_removed": conflicting_rows_removed,
        "conflicting_domains_list": sorted(conflicting_domains),
        "cleaned_total_rows": len(cleaned_df),
        "cleaned_label_distribution": cleaned_df["label"].value_counts().to_dict(),
        "cleaned_unique_domains": cleaned_df["domain"].nunique(),
        "legitimate_sha256": compute_sha256(legit_path) if legit_path and Path(legit_path).exists() else None,
        "phishing_sha256": compute_sha256(phishing_path) if phishing_path and Path(phishing_path).exists() else None,
    }

    return cleaned_df, report


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Hàm tương thích ngược làm sạch dataframe đơn lẻ."""
    required = {"url", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dữ liệu phải có các cột: {sorted(required)}")

    cleaned = frame.loc[:, ["url", "label"]].dropna().copy()
    cleaned["url"] = cleaned["url"].map(normalize_url)
    cleaned = cleaned[cleaned["url"] != ""].drop_duplicates("url")
    cleaned["domain"] = cleaned["url"].map(registered_domain)

    conflicting = cleaned.groupby("domain")["label"].nunique()
    conflicting_domains = conflicting[conflicting > 1].index
    return cleaned[~cleaned["domain"].isin(conflicting_domains)].reset_index(drop=True)


def split_by_domain(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """Chia 70/15/15 theo domain và kiểm tra không có domain hay URL giao nhau giữa các tập."""
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("Tỷ lệ validation/test không hợp lệ")

    cleaned = frame.copy()
    if "domain" not in cleaned.columns:
        cleaned = clean_dataset(cleaned)

    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(first_split.split(cleaned, groups=cleaned["domain"]))
    train_val = cleaned.iloc[train_val_idx].reset_index(drop=True)
    test = cleaned.iloc[test_idx].reset_index(drop=True)

    relative_validation_size = validation_size / (1 - test_size)
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=random_state,
    )
    train_idx, validation_idx = next(
        second_split.split(train_val, groups=train_val["domain"])
    )
    splits = DatasetSplits(
        train=train_val.iloc[train_idx].reset_index(drop=True),
        validation=train_val.iloc[validation_idx].reset_index(drop=True),
        test=test,
    )
    _assert_disjoint_splits(splits)
    return splits


def _assert_disjoint_splits(splits: DatasetSplits) -> None:
    """Kiểm tra nghiêm ngặt không có domain hoặc URL trùng lắp giữa Train/Validation/Test."""
    train_domains = set(splits.train["domain"])
    validation_domains = set(splits.validation["domain"])
    test_domains = set(splits.test["domain"])

    assert train_domains.isdisjoint(validation_domains), "Leakage phát hiện: Domain giao giữa Train và Validation!"
    assert train_domains.isdisjoint(test_domains), "Leakage phát hiện: Domain giao giữa Train và Test!"
    assert validation_domains.isdisjoint(test_domains), "Leakage phát hiện: Domain giao giữa Validation và Test!"

    train_urls = set(splits.train["url"])
    val_urls = set(splits.validation["url"])
    test_urls = set(splits.test["url"])

    assert train_urls.isdisjoint(val_urls), "Leakage phát hiện: URL giao giữa Train và Validation!"
    assert train_urls.isdisjoint(test_urls), "Leakage phát hiện: URL giao giữa Train và Test!"
    assert val_urls.isdisjoint(test_urls), "Leakage phát hiện: URL giao giữa Validation và Test!"
