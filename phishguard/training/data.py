"""Làm sạch, audit dữ liệu và chia tập theo registered domain để chống data leakage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import pandas as pd
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
    threshold_validation_rows: int
    threshold_validation_domains: int
    threshold_positive_rate: float
    strategy: str = "stratified-registered-domain-5way"


@dataclass(frozen=True)
class DatasetSplits:
    """Năm tập domain-disjoint cho một pipeline train/evaluate duy nhất."""

    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    threshold_validation: pd.DataFrame
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
    Chuẩn hóa URL thành dạng canonical phục vụ deduplication và conflict audit.
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
    phish_hash = (
        compute_sha256(phishing_path) if phishing_path and Path(phishing_path).exists() else None
    )

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
    train_size: float = 0.60,
    validation_size: float = 0.15,
    calibration_size: float = 0.10,
    threshold_validation_size: float = 0.05,
    test_size: float = 0.10,
    random_state: int = 42,
) -> DatasetSplits:
    """Chia train/validation/calibration/threshold-validation/test theo domain."""
    sizes = {
        "train": train_size,
        "validation": validation_size,
        "calibration": calibration_size,
        "threshold_validation": threshold_validation_size,
        "test": test_size,
    }
    if any(value <= 0 for value in sizes.values()) or abs(sum(sizes.values()) - 1.0) > 1e-9:
        raise ValueError("Tỷ lệ chia 5-way phải dương và tổng đúng bằng 1.0")

    cleaned = frame.copy()
    if "domain" not in cleaned.columns:
        cleaned = clean_dataset(cleaned)
    allocations = _stratified_group_allocation(cleaned, sizes, random_state=random_state)
    splits = DatasetSplits(**allocations)
    _assert_disjoint_splits(splits)
    return splits


def _stratified_group_allocation(
    frame: pd.DataFrame,
    proportions: dict[str, float],
    *,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Phân bổ group nguyên vẹn, cân bằng row count và positive rate."""
    if frame.empty:
        raise ValueError("Không thể chia một dataframe rỗng")

    names = list(proportions)
    group_stats = (
        frame.groupby("domain", sort=False)["label"].agg(rows="size", positives="sum").reset_index()
    )
    rng = pd.Series(range(len(group_stats))).sample(frac=1.0, random_state=random_state)
    group_stats["tie_break"] = 0
    group_stats.loc[rng.index, "tie_break"] = range(len(group_stats))
    group_stats["positive_rate"] = group_stats["positives"] / group_stats["rows"]
    # Xen kẽ nhóm thiên positive và nhóm thiên negative. Nếu sắp toàn bộ theo
    # positive_rate, các split đầu sẽ bị nhồi phishing còn split cuối gần như
    # toàn legitimate dù hàm mục tiêu có phạt lệch prior.
    overall_positive_rate = float(frame["label"].mean())
    positive_groups = group_stats[
        group_stats["positive_rate"] >= overall_positive_rate
    ].sort_values(["rows", "tie_break"], ascending=[False, True])
    negative_groups = group_stats[group_stats["positive_rate"] < overall_positive_rate].sort_values(
        ["rows", "tie_break"], ascending=[False, True]
    )
    ordered_groups = []
    for index in range(max(len(positive_groups), len(negative_groups))):
        if index < len(positive_groups):
            ordered_groups.append(positive_groups.iloc[index])
        if index < len(negative_groups):
            ordered_groups.append(negative_groups.iloc[index])
    group_stats = pd.DataFrame(ordered_groups).reset_index(drop=True)

    total_rows = float(len(frame))
    total_positives = float(frame["label"].sum())
    target_rows = {name: total_rows * proportions[name] for name in names}
    target_positives = {name: total_positives * proportions[name] for name in names}
    total_negatives = total_rows - total_positives
    target_negatives = {name: total_negatives * proportions[name] for name in names}
    current_rows = {name: 0.0 for name in names}
    current_positives = {name: 0.0 for name in names}
    group_to_split: dict[str, str] = {}

    for row in group_stats.itertuples(index=False):
        candidates: list[tuple[float, str]] = []
        group_negatives = row.rows - row.positives
        for name in names:
            new_rows = current_rows[name] + row.rows
            new_positives = current_positives[name] + row.positives
            current_negatives = current_rows[name] - current_positives[name]
            new_negatives = current_negatives + group_negatives
            positive_need = max(0.0, target_positives[name] - current_positives[name])
            negative_need = max(0.0, target_negatives[name] - current_negatives)
            row_need = max(0.0, target_rows[name] - current_rows[name])
            row_overrun = max(0.0, new_rows - target_rows[name])
            positive_overrun = max(0.0, new_positives - target_positives[name])
            negative_overrun = max(0.0, new_negatives - target_negatives[name])
            score = (
                row_need * 0.01
                + row.positives * positive_need
                + group_negatives * negative_need
                - 100.0 * (row_overrun + positive_overrun + negative_overrun)
            )
            candidates.append((score, name))
        _, selected = max(candidates, key=lambda item: (item[0], -names.index(item[1])))
        group_to_split[row.domain] = selected
        current_rows[selected] += row.rows
        current_positives[selected] += row.positives

    result: dict[str, pd.DataFrame] = {}
    for name in names:
        mask = frame["domain"].map(group_to_split.__getitem__) == name
        result[name] = frame.loc[mask].reset_index(drop=True)
    if any(value.empty for value in result.values()):
        raise ValueError("Không thể tạo split không rỗng với số registered domain hiện tại")
    return result


def create_split_manifest(
    splits: DatasetSplits,
    seed: int = 42,
) -> SplitManifest:
    """Tạo manifest kiểm tra tỷ lệ, domain và nhãn của năm tập."""
    return SplitManifest(
        seed=seed,
        train_rows=len(splits.train),
        validation_rows=len(splits.validation),
        calibration_rows=len(splits.calibration),
        threshold_validation_rows=len(splits.threshold_validation),
        test_rows=len(splits.test),
        train_domains=splits.train["domain"].nunique(),
        validation_domains=splits.validation["domain"].nunique(),
        calibration_domains=splits.calibration["domain"].nunique(),
        threshold_validation_domains=splits.threshold_validation["domain"].nunique(),
        test_domains=splits.test["domain"].nunique(),
        train_positive_rate=round(float(splits.train["label"].mean()), 4),
        val_positive_rate=round(float(splits.validation["label"].mean()), 4),
        cal_positive_rate=round(float(splits.calibration["label"].mean()), 4),
        threshold_positive_rate=round(float(splits.threshold_validation["label"].mean()), 4),
        test_positive_rate=round(float(splits.test["label"].mean()), 4),
        strategy="stratified-registered-domain-5way",
    )


def _assert_disjoint_splits(splits: DatasetSplits) -> None:
    """Bảo đảm registered domain và canonical URL không rò rỉ giữa các tập."""
    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "calibration": splits.calibration,
        "threshold_validation": splits.threshold_validation,
        "test": splits.test,
    }
    domains = {name: set(frame["domain"]) for name, frame in frames.items()}
    urls = {name: set(frame["canonical_url"]) for name, frame in frames.items()}
    names = list(frames)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if not domains[first].isdisjoint(domains[second]):
                raise AssertionError(f"Leakage: domain giao giữa {first} và {second}")
            if not urls[first].isdisjoint(urls[second]):
                raise AssertionError(f"Leakage: URL giao giữa {first} và {second}")
