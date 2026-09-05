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


@dataclass(frozen=True)
class FourWayDatasetSplits:
    """Bốn tập độc lập: Train (65%) / Validation (15%) / Calibration (10%) / Test (10%)."""

    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def split_by_domain_4way(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.10,
    calibration_size: float = 0.10,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> FourWayDatasetSplits:
    """
    Chia 4 tập theo domain với zero overlap:
    Train: Model training
    Validation: Model selection & hyperparameter tuning
    Calibration: Probability calibration (Isotonic/Sigmoid) & threshold selection
    Test: Final untouched evaluation once
    """
    total_holdout = test_size + calibration_size + validation_size
    if total_holdout >= 1.0 or any(s <= 0 for s in (test_size, calibration_size, validation_size)):
        raise ValueError("Tỷ lệ chia tập 4-way không hợp lệ")

    cleaned = frame.copy()
    if "domain" not in cleaned.columns:
        cleaned = clean_dataset(cleaned)

    # 1. Tách Test set
    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    remain1_idx, test_idx = next(first_split.split(cleaned, groups=cleaned["domain"]))
    remain1 = cleaned.iloc[remain1_idx].reset_index(drop=True)
    test = cleaned.iloc[test_idx].reset_index(drop=True)

    # 2. Tách Calibration set
    rel_cal_size = calibration_size / (1.0 - test_size)
    second_split = GroupShuffleSplit(n_splits=1, test_size=rel_cal_size, random_state=random_state)
    remain2_idx, cal_idx = next(second_split.split(remain1, groups=remain1["domain"]))
    remain2 = remain1.iloc[remain2_idx].reset_index(drop=True)
    calibration = remain1.iloc[cal_idx].reset_index(drop=True)

    # 3. Tách Validation set
    rel_val_size = validation_size / (1.0 - test_size - calibration_size)
    third_split = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state)
    train_idx, val_idx = next(third_split.split(remain2, groups=remain2["domain"]))
    train = remain2.iloc[train_idx].reset_index(drop=True)
    validation = remain2.iloc[val_idx].reset_index(drop=True)

    splits = FourWayDatasetSplits(
        train=train,
        validation=validation,
        calibration=calibration,
        test=test,
    )
    _assert_disjoint_splits_4way(splits)
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


def _assert_disjoint_splits_4way(splits: FourWayDatasetSplits) -> None:
    """Kiểm tra nghiêm ngặt không có domain hoặc URL trùng lắp giữa 4 tập."""
    domain_sets = {
        "train": set(splits.train["domain"]),
        "validation": set(splits.validation["domain"]),
        "calibration": set(splits.calibration["domain"]),
        "test": set(splits.test["domain"]),
    }
    url_sets = {
        "train": set(splits.train["url"]),
        "validation": set(splits.validation["url"]),
        "calibration": set(splits.calibration["url"]),
        "test": set(splits.test["url"]),
    }

    names = list(domain_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            assert domain_sets[n1].isdisjoint(domain_sets[n2]), f"Leakage: Domain giao giữa {n1} và {n2}!"
            assert url_sets[n1].isdisjoint(url_sets[n2]), f"Leakage: URL giao giữa {n1} và {n2}!"


@dataclass(frozen=True)
class TemporalSplits:
    """Tập train (quá khứ) và test (tương lai) theo thời gian ghi nhận (Protocol B)."""

    train: pd.DataFrame
    test: pd.DataFrame


def temporal_split_protocol_b(
    frame: pd.DataFrame,
    *,
    test_ratio: float = 0.20,
    time_col: str = "submission_time",
) -> TemporalSplits:
    """
    Protocol B - Đánh giá Temporal Robustness (đo concept drift):
    Huấn luyện trên các chiến dịch cũ (quá khứ) và kiểm thử trên các chiến dịch mới hơn (tương lai).
    """
    if time_col not in frame.columns:
        raise ValueError(f"Dữ liệu không chứa cột thời gian {time_col}")

    # Chuyển đổi thời gian an toàn
    frame_with_time = frame.copy()
    frame_with_time["_parsed_time"] = pd.to_datetime(frame_with_time[time_col], errors="coerce")
    valid_time_df = frame_with_time.dropna(subset=["_parsed_time"]).sort_values("_parsed_time").reset_index(drop=True)

    split_idx = int(len(valid_time_df) * (1.0 - test_ratio))
    train = valid_time_df.iloc[:split_idx].drop(columns=["_parsed_time"]).reset_index(drop=True)
    test = valid_time_df.iloc[split_idx:].drop(columns=["_parsed_time"]).reset_index(drop=True)

    return TemporalSplits(train=train, test=test)
