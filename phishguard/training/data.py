"""Làm sạch, audit dữ liệu và chia tập theo registered domain để chống data leakage."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tld import get_fld


@dataclass(frozen=True)
class URLRecord:
    """Bản ghi URL với định danh rõ ràng giữa raw, canonical và domain."""

    raw_url: str
    canonical_url: str
    registered_domain: str
    label: int
    submission_time: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    """Thông tin kiểm toán và nguồn gốc dữ liệu (Provenance & Integrity)."""

    dataset_version: str
    source_checksums: dict[str, str | None]
    rows_raw: int
    rows_clean: int
    unique_canonical_urls: int
    unique_domains: int
    label_distribution: dict[str, int]
    exact_conflicts_removed: int
    multi_label_domains_preserved: int


@dataclass(frozen=True)
class SplitManifest:
    """Tóm tắt phân bố dữ liệu và nhãn giữa các tập split (Grouped Split Verification)."""

    seed: int
    train_rows: int
    validation_rows: int
    calibration_rows: int
    test_rows: int
    train_domains: int
    validation_domains: int
    calibration_domains: int
    test_domains: int
    train_positive_rate: float
    val_positive_rate: float
    cal_positive_rate: float
    test_positive_rate: float


@dataclass(frozen=True)
class DatasetSplits:
    """Ba tập độc lập; test chỉ dùng một lần sau khi chọn mô hình."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FourWayDatasetSplits:
    """Bốn tập độc lập: Train (65%) / Validation (15%) / Calibration (10%) / Test (10%)."""

    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class TemporalSplits:
    """Tập train (quá khứ) và test (tương lai) theo thời gian ghi nhận (Protocol B)."""

    train: pd.DataFrame
    test: pd.DataFrame


def compute_sha256(file_path: str | Path) -> str:
    """Tính mã băm SHA-256 của file dữ liệu."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_url(url: str) -> str:
    """
    Chuẩn hóa URL thành dạng canonical phục vụ deduplication, cache key và conflict audit.
    LƯU Ý: Không dùng URL này để trích xuất đặc trưng bảo mật; đặc trưng luôn lấy từ raw_url.
    """
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
    """Lấy miền đăng ký (registered domain) qua Mozilla PSL làm group key; fallback sang hostname."""
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
    Audit chất lượng dữ liệu nguồn:
    1. Giữ nguyên raw_url gốc cho feature extraction.
    2. Tạo canonical_url phục vụ loại bỏ URL không hợp lệ và deduplication.
    3. Loại bỏ EXACT URL conflict (cùng canonical_url nhưng mang cả 2 nhãn mâu thuẫn).
    4. GIỮ NGUYÊN các registered domain có cả legitimate và phishing URLs khác nhau (như google.com, dropbox.com).
    5. Đảm bảo registered domain được dùng làm group key cho split để zero leakage.
    """
    legit_raw_count = len(legit_df)
    phishing_raw_count = len(phishing_df)

    legit_clean = legit_df[["url"]].dropna().copy()
    legit_clean.rename(columns={"url": "raw_url"}, inplace=True)
    legit_clean["label"] = 0

    phishing_clean = phishing_df[["url"]].dropna().copy()
    phishing_clean.rename(columns={"url": "raw_url"}, inplace=True)
    phishing_clean["label"] = 1

    if "submission_time" in phishing_df.columns:
        phishing_clean["submission_time"] = phishing_df["submission_time"]

    combined = pd.concat([legit_clean, phishing_clean], ignore_index=True)

    # 1. Canonicalization
    combined["canonical_url"] = combined["raw_url"].map(normalize_url)
    invalid_mask = combined["canonical_url"] == ""
    invalid_count = int(invalid_mask.sum())
    valid_df = combined[~invalid_mask].copy()

    # 2. Xử lý Exact Canonical URL Label Conflicts:
    # Chỉ loại bỏ khi CÙNG 1 CANONICAL URL mà mang nhãn mâu thuẫn (0 và 1)
    url_label_counts = valid_df.groupby("canonical_url")["label"].nunique()
    conflicting_canonical_urls = set(url_label_counts[url_label_counts > 1].index)
    exact_conflicts_removed = int(valid_df["canonical_url"].isin(conflicting_canonical_urls).sum())
    valid_df = valid_df[~valid_df["canonical_url"].isin(conflicting_canonical_urls)].copy()

    # 3. Deduplication trên canonical_url
    before_dedup = len(valid_df)
    valid_df = valid_df.drop_duplicates(subset=["canonical_url"]).copy()
    duplicate_count = before_dedup - len(valid_df)

    # 4. Trích xuất registered domain
    valid_df["domain"] = valid_df["canonical_url"].map(registered_domain)
    valid_df = valid_df[valid_df["domain"] != ""].copy()

    # 5. Kiểm tra các domain đa nhãn (Multi-label domains) được BẢO TỒN (google.com, wix.com, dropbox.com, etc.)
    domain_labels = valid_df.groupby("domain")["label"].nunique()
    multi_label_domains = domain_labels[domain_labels > 1].index.tolist()
    multi_label_rows = int(valid_df["domain"].isin(multi_label_domains).sum())

    # Đồng bộ trường 'url' trỏ vào raw_url để tương thích ngược 100% với các script và test
    valid_df["url"] = valid_df["raw_url"]
    cleaned_df = valid_df.reset_index(drop=True)

    legit_hash = compute_sha256(legit_path) if legit_path and Path(legit_path).exists() else None
    phish_hash = compute_sha256(phishing_path) if phishing_path and Path(phishing_path).exists() else None

    report = {
        "legitimate_rows": legit_raw_count,
        "phishing_rows": phishing_raw_count,
        "invalid_urls_removed": invalid_count,
        "exact_conflicts_removed": exact_conflicts_removed,
        "duplicate_urls_removed": duplicate_count,
        "multi_label_domains_preserved_count": len(multi_label_domains),
        "multi_label_rows_preserved": multi_label_rows,
        "multi_label_domains_sample": sorted(multi_label_domains)[:20],
        "cleaned_total_rows": len(cleaned_df),
        "cleaned_label_distribution": cleaned_df["label"].value_counts().to_dict(),
        "cleaned_unique_canonical_urls": cleaned_df["canonical_url"].nunique(),
        "cleaned_unique_domains": cleaned_df["domain"].nunique(),
        "legitimate_sha256": legit_hash,
        "phishing_sha256": phish_hash,
    }

    return cleaned_df, report


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Hàm làm sạch dataframe đơn lẻ, tuân thủ chính sách bảo tồn domain đa nhãn và loại exact url conflict."""
    required = {"url", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dữ liệu phải có các cột: {sorted(required)}")

    df = frame.loc[:, ["url", "label"]].dropna().copy()
    df.rename(columns={"url": "raw_url"}, inplace=True)
    df["canonical_url"] = df["raw_url"].map(normalize_url)
    df = df[df["canonical_url"] != ""].copy()

    # Loại bỏ exact URL conflicts
    url_label_counts = df.groupby("canonical_url")["label"].nunique()
    conflicts = set(url_label_counts[url_label_counts > 1].index)
    df = df[~df["canonical_url"].isin(conflicts)].copy()

    df = df.drop_duplicates(subset=["canonical_url"]).copy()
    df["domain"] = df["canonical_url"].map(registered_domain)
    df = df[df["domain"] != ""].copy()

    df["url"] = df["raw_url"]
    return df.reset_index(drop=True)


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
    Train: Model training (65%)
    Validation: Model selection & hyperparameter tuning (15%)
    Calibration: Probability calibration (Isotonic/Sigmoid) & threshold selection (10%)
    Test: Final untouched evaluation once (10%)
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


def create_split_manifest(splits: FourWayDatasetSplits, seed: int = 42) -> SplitManifest:
    """Tạo SplitManifest định lượng phân bố số dòng, số domain và positive rate trên từng split."""
    return SplitManifest(
        seed=seed,
        train_rows=len(splits.train),
        validation_rows=len(splits.validation),
        calibration_rows=len(splits.calibration),
        test_rows=len(splits.test),
        train_domains=splits.train["domain"].nunique(),
        validation_domains=splits.validation["domain"].nunique(),
        calibration_domains=splits.calibration["domain"].nunique(),
        test_domains=splits.test["domain"].nunique(),
        train_positive_rate=round(float(splits.train["label"].mean()), 4),
        val_positive_rate=round(float(splits.validation["label"].mean()), 4),
        cal_positive_rate=round(float(splits.calibration["label"].mean()), 4),
        test_positive_rate=round(float(splits.test["label"].mean()), 4),
    )


def _assert_disjoint_splits(splits: DatasetSplits) -> None:
    """Kiểm tra nghiêm ngặt không có domain hoặc URL trùng lắp giữa Train/Validation/Test."""
    domain_sets = {
        "train": set(splits.train["domain"]),
        "validation": set(splits.validation["domain"]),
        "test": set(splits.test["domain"]),
    }
    url_sets = {
        "train": set(splits.train["raw_url"] if "raw_url" in splits.train.columns else splits.train["url"]),
        "validation": set(splits.validation["raw_url"] if "raw_url" in splits.validation.columns else splits.validation["url"]),
        "test": set(splits.test["raw_url"] if "raw_url" in splits.test.columns else splits.test["url"]),
    }
    for n1 in ["train", "validation"]:
        for n2 in ["validation", "test"]:
            if n1 != n2:
                assert domain_sets[n1].isdisjoint(domain_sets[n2]), f"Leakage: Domain giao giữa {n1} và {n2}!"
                assert url_sets[n1].isdisjoint(url_sets[n2]), f"Leakage: URL giao giữa {n1} và {n2}!"


def _assert_disjoint_splits_4way(splits: FourWayDatasetSplits) -> None:
    """Kiểm tra nghiêm ngặt không có domain hoặc URL trùng lắp giữa 4 tập."""
    domain_sets = {
        "train": set(splits.train["domain"]),
        "validation": set(splits.validation["domain"]),
        "calibration": set(splits.calibration["domain"]),
        "test": set(splits.test["domain"]),
    }
    url_sets = {
        "train": set(splits.train["raw_url"] if "raw_url" in splits.train.columns else splits.train["url"]),
        "validation": set(splits.validation["raw_url"] if "raw_url" in splits.validation.columns else splits.validation["url"]),
        "calibration": set(splits.calibration["raw_url"] if "raw_url" in splits.calibration.columns else splits.calibration["url"]),
        "test": set(splits.test["raw_url"] if "raw_url" in splits.test.columns else splits.test["url"]),
    }

    names = list(domain_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            assert domain_sets[n1].isdisjoint(domain_sets[n2]), f"Leakage: Domain giao giữa {n1} và {n2}!"
            assert url_sets[n1].isdisjoint(url_sets[n2]), f"Leakage: URL giao giữa {n1} và {n2}!"


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

    frame_with_time = frame.copy()
    frame_with_time["_parsed_time"] = pd.to_datetime(frame_with_time[time_col], errors="coerce")
    valid_time_df = frame_with_time.dropna(subset=["_parsed_time"]).sort_values("_parsed_time").reset_index(drop=True)

    split_idx = int(len(valid_time_df) * (1.0 - test_ratio))
    train = valid_time_df.iloc[:split_idx].drop(columns=["_parsed_time"]).reset_index(drop=True)
    test = valid_time_df.iloc[split_idx:].drop(columns=["_parsed_time"]).reset_index(drop=True)

    return TemporalSplits(train=train, test=test)
