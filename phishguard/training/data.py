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
    policy_validation_rows: int = 0
    policy_validation_domains: int = 0
    policy_positive_rate: float = 0.0
    strategy: str = "stratified-group-disjoint"


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
class FiveWayDatasetSplits:
    """Train/Validation/Calibration/Policy Validation/Locked Test."""

    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    policy_validation: pd.DataFrame
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

    allocations = _stratified_group_allocation(
        cleaned,
        {
            "train": 1.0 - test_size - validation_size,
            "validation": validation_size,
            "test": test_size,
        },
        random_state=random_state,
    )
    splits = DatasetSplits(**allocations)
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

    allocations = _stratified_group_allocation(
        cleaned,
        {
            "train": 1.0 - test_size - calibration_size - validation_size,
            "validation": validation_size,
            "calibration": calibration_size,
            "test": test_size,
        },
        random_state=random_state,
    )
    splits = FourWayDatasetSplits(**allocations)
    _assert_disjoint_splits_4way(splits)
    return splits


def split_by_domain_5way(
    frame: pd.DataFrame,
    *,
    train_size: float = 0.60,
    validation_size: float = 0.15,
    calibration_size: float = 0.10,
    policy_validation_size: float = 0.05,
    test_size: float = 0.10,
    random_state: int = 42,
) -> FiveWayDatasetSplits:
    """Chia domain-disjoint và gần stratified theo lifecycle 5 tập.

    Mỗi registered domain được gán nguyên vẹn vào đúng một split. Bộ điều phối
    nhóm tối ưu đồng thời tỷ lệ số dòng và tỷ lệ phishing; không dùng
    `GroupShuffleSplit` ngẫu nhiên vì nó làm prior giữa các tập lệch mạnh.
    """
    sizes = {
        "train": train_size,
        "validation": validation_size,
        "calibration": calibration_size,
        "policy_validation": policy_validation_size,
        "test": test_size,
    }
    if any(value <= 0 for value in sizes.values()) or abs(sum(sizes.values()) - 1.0) > 1e-9:
        raise ValueError("Tỷ lệ chia 5-way phải dương và tổng đúng bằng 1.0")

    cleaned = frame.copy()
    if "domain" not in cleaned.columns:
        cleaned = clean_dataset(cleaned)
    allocations = _stratified_group_allocation(cleaned, sizes, random_state=random_state)
    splits = FiveWayDatasetSplits(**allocations)
    _assert_disjoint_splits_5way(splits)
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
    splits: FourWayDatasetSplits | FiveWayDatasetSplits,
    seed: int = 42,
) -> SplitManifest:
    """Tạo manifest thống nhất cho lifecycle 4-way cũ hoặc 5-way mới."""
    policy_validation = getattr(splits, "policy_validation", None)
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
        policy_validation_rows=len(policy_validation) if policy_validation is not None else 0,
        policy_validation_domains=(
            policy_validation["domain"].nunique() if policy_validation is not None else 0
        ),
        policy_positive_rate=(
            round(float(policy_validation["label"].mean()), 4)
            if policy_validation is not None
            else 0.0
        ),
        strategy="stratified-group-disjoint-5way"
        if policy_validation is not None
        else "stratified-group-disjoint-4way",
    )


def _assert_disjoint_splits(splits: DatasetSplits) -> None:
    """Kiểm tra nghiêm ngặt không có domain hoặc URL trùng lắp giữa Train/Validation/Test."""
    domain_sets = {
        "train": set(splits.train["domain"]),
        "validation": set(splits.validation["domain"]),
        "test": set(splits.test["domain"]),
    }
    url_sets = {
        "train": set(
            splits.train["raw_url"] if "raw_url" in splits.train.columns else splits.train["url"]
        ),
        "validation": set(
            splits.validation["raw_url"]
            if "raw_url" in splits.validation.columns
            else splits.validation["url"]
        ),
        "test": set(
            splits.test["raw_url"] if "raw_url" in splits.test.columns else splits.test["url"]
        ),
    }
    for n1 in ["train", "validation"]:
        for n2 in ["validation", "test"]:
            if n1 != n2:
                assert domain_sets[n1].isdisjoint(domain_sets[n2]), (
                    f"Leakage: Domain giao giữa {n1} và {n2}!"
                )
                assert url_sets[n1].isdisjoint(url_sets[n2]), (
                    f"Leakage: URL giao giữa {n1} và {n2}!"
                )


def _assert_disjoint_splits_4way(splits: FourWayDatasetSplits) -> None:
    """Kiểm tra nghiêm ngặt không có domain hoặc URL trùng lắp giữa 4 tập."""
    domain_sets = {
        "train": set(splits.train["domain"]),
        "validation": set(splits.validation["domain"]),
        "calibration": set(splits.calibration["domain"]),
        "test": set(splits.test["domain"]),
    }
    url_sets = {
        "train": set(
            splits.train["raw_url"] if "raw_url" in splits.train.columns else splits.train["url"]
        ),
        "validation": set(
            splits.validation["raw_url"]
            if "raw_url" in splits.validation.columns
            else splits.validation["url"]
        ),
        "calibration": set(
            splits.calibration["raw_url"]
            if "raw_url" in splits.calibration.columns
            else splits.calibration["url"]
        ),
        "test": set(
            splits.test["raw_url"] if "raw_url" in splits.test.columns else splits.test["url"]
        ),
    }

    names = list(domain_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            assert domain_sets[n1].isdisjoint(domain_sets[n2]), (
                f"Leakage: Domain giao giữa {n1} và {n2}!"
            )
            assert url_sets[n1].isdisjoint(url_sets[n2]), f"Leakage: URL giao giữa {n1} và {n2}!"


def _assert_disjoint_splits_5way(splits: FiveWayDatasetSplits) -> None:
    """Kiểm tra domain và canonical/raw URL không giao giữa 5 tập."""
    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "calibration": splits.calibration,
        "policy_validation": splits.policy_validation,
        "test": splits.test,
    }
    domains = {name: set(df["domain"]) for name, df in frames.items()}
    urls = {
        name: set(df["raw_url"] if "raw_url" in df.columns else df["url"])
        for name, df in frames.items()
    }
    names = list(frames)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if not domains[first].isdisjoint(domains[second]):
                raise AssertionError(f"Leakage: Domain giao giữa {first} và {second}!")
            if not urls[first].isdisjoint(urls[second]):
                raise AssertionError(f"Leakage: URL giao giữa {first} và {second}!")


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
    valid_time_df = (
        frame_with_time.dropna(subset=["_parsed_time"])
        .sort_values("_parsed_time")
        .reset_index(drop=True)
    )

    split_idx = int(len(valid_time_df) * (1.0 - test_ratio))
    train = valid_time_df.iloc[:split_idx].drop(columns=["_parsed_time"]).reset_index(drop=True)
    test = valid_time_df.iloc[split_idx:].drop(columns=["_parsed_time"]).reset_index(drop=True)

    return TemporalSplits(train=train, test=test)


def temporal_split_future_unseen_domains(
    frame: pd.DataFrame,
    *,
    test_ratio: float = 0.20,
    time_col: str = "submission_time",
) -> TemporalSplits:
    """Tách future benchmark theo first-seen domain, không để domain overlap.

    Protocol này ưu tiên tính chất unseen-domain hơn tỷ lệ dòng chính xác: các
    domain có thời điểm xuất hiện đầu tiên muộn nhất được đưa trọn vào future.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio phải nằm trong khoảng (0, 1)")
    if time_col not in frame.columns:
        raise ValueError(f"Dữ liệu không chứa cột thời gian {time_col}")
    working = frame.copy()
    working["_parsed_time"] = pd.to_datetime(working[time_col], errors="coerce", utc=True)
    working = working.dropna(subset=["_parsed_time"]).copy()
    if "domain" not in working.columns:
        working["domain"] = working["url"].map(registered_domain)
    first_seen = working.groupby("domain")["_parsed_time"].min().sort_values(kind="stable")
    test_domain_count = max(1, int(round(len(first_seen) * test_ratio)))
    future_domains = set(first_seen.tail(test_domain_count).index)
    test_mask = working["domain"].isin(future_domains)
    train = working[~test_mask].sort_values("_parsed_time")
    test = working[test_mask].sort_values("_parsed_time")
    train = train.drop(columns=["_parsed_time"]).reset_index(drop=True)
    test = test.drop(columns=["_parsed_time"]).reset_index(drop=True)
    if set(train["domain"]).intersection(test["domain"]):
        raise AssertionError("Future benchmark bị overlap registered domain")
    return TemporalSplits(train=train, test=test)
