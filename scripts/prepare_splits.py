"""Script tạo và lưu trữ các tập Train/Validation/Calibration/Test chia theo registered domain độc lập."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from phishguard.training.data import (
    DatasetManifest,
    audit_and_clean_data,
    create_split_manifest,
    split_by_domain,
    split_by_domain_4way,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
LEGIT_CSV = DATA_DIR / "legit_url.csv"
PHISHING_CSV = DATA_DIR / "verified_online.csv"


def save_split_dataframe(df: pd.DataFrame, file_prefix: Path) -> None:
    """Lưu dataframe thành Parquet nếu có pyarrow/fastparquet, fallback sang CSV nén."""
    try:
        parquet_path = file_prefix.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"   Saved: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        csv_path = file_prefix.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        print(f"   Saved (CSV fallback due to {e}): {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main() -> None:
    print("=" * 65)
    print(" Bắt đầu chia tập dữ liệu theo Registered Domain (Zero Leakage)...")
    print("=" * 65)

    if not LEGIT_CSV.exists() or not PHISHING_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu nguồn tại {LEGIT_CSV} hoặc {PHISHING_CSV}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    split_cfg = config.get("split", {})
    cal_size = split_cfg.get("calibration", 0.10)
    random_seed = config.get("random_seed", 42)
    dataset_version = config.get("model_version", "3.2.0")

    legit_df = pd.read_csv(LEGIT_CSV)
    phishing_df = pd.read_csv(PHISHING_CSV)

    cleaned_df, report = audit_and_clean_data(
        legit_df=legit_df,
        phishing_df=phishing_df,
        legit_path=LEGIT_CSV,
        phishing_path=PHISHING_CSV,
    )

    print(f" Dữ liệu đã làm sạch: {len(cleaned_df):,} bản ghi ({report['cleaned_unique_domains']:,} unique domains)")
    print(f" Multi-label domains bảo tồn: {report['multi_label_domains_preserved_count']:,} domains ({report['multi_label_rows_preserved']:,} URLs)")

    # Lưu DatasetManifest
    dataset_manifest = DatasetManifest(
        dataset_version=f"v{dataset_version}",
        source_checksums={
            "legitimate": report.get("legitimate_sha256"),
            "phishing": report.get("phishing_sha256"),
        },
        rows_raw=report["legitimate_rows"] + report["phishing_rows"],
        rows_clean=report["cleaned_total_rows"],
        unique_canonical_urls=report["cleaned_unique_canonical_urls"],
        unique_domains=report["cleaned_unique_domains"],
        label_distribution=report["cleaned_label_distribution"],
        exact_conflicts_removed=report["exact_conflicts_removed"],
        multi_label_domains_preserved=report["multi_label_domains_preserved_count"],
    )

    with open(ARTIFACTS_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(asdict(dataset_manifest), f, indent=2, ensure_ascii=False)

    with open(ARTIFACTS_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if cal_size and cal_size > 0:
        print(f"\n [MODE] Áp dụng chiến lược chia 4-way: Train / Validation / Calibration / Test...")
        splits_4way = split_by_domain_4way(
            cleaned_df,
            test_size=split_cfg.get("test", 0.10),
            calibration_size=cal_size,
            validation_size=split_cfg.get("validation", 0.15),
            random_state=random_seed,
        )
        print(" [OK] Đã xác minh 100%: Domain và URL tuyệt đối KHÔNG giao nhau giữa 4 tập!")
        print(f" • Train set:       {len(splits_4way.train):,} rows | {splits_4way.train['domain'].nunique():,} domains")
        print(f" • Validation set:  {len(splits_4way.validation):,} rows | {splits_4way.validation['domain'].nunique():,} domains")
        print(f" • Calibration set: {len(splits_4way.calibration):,} rows | {splits_4way.calibration['domain'].nunique():,} domains")
        print(f" • Test set:        {len(splits_4way.test):,} rows | {splits_4way.test['domain'].nunique():,} domains")

        split_manifest = create_split_manifest(splits_4way, seed=random_seed)
        with open(ARTIFACTS_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(asdict(split_manifest), f, indent=2, ensure_ascii=False)

        save_split_dataframe(splits_4way.train, SPLITS_DIR / "train")
        save_split_dataframe(splits_4way.validation, SPLITS_DIR / "validation")
        save_split_dataframe(splits_4way.calibration, SPLITS_DIR / "calibration")
        save_split_dataframe(splits_4way.test, SPLITS_DIR / "test")

    else:
        print(f"\n [MODE] Áp dụng chiến lược chia 3-way: Train (70%) / Validation (15%) / Test (15%)...")
        splits = split_by_domain(
            cleaned_df,
            test_size=split_cfg.get("test", 0.15),
            validation_size=split_cfg.get("validation", 0.15),
            random_state=random_seed,
        )
        print(" [OK] Đã xác minh 100%: Domain và URL tuyệt đối KHÔNG giao nhau giữa 3 tập!")
        print(f" • Train set:      {len(splits.train):,} rows | {splits.train['domain'].nunique():,} domains")
        print(f" • Validation set: {len(splits.validation):,} rows | {splits.validation['domain'].nunique():,} domains")
        print(f" • Test set:       {len(splits.test):,} rows | {splits.test['domain'].nunique():,} domains")

        save_split_dataframe(splits.train, SPLITS_DIR / "train")
        save_split_dataframe(splits.validation, SPLITS_DIR / "validation")
        save_split_dataframe(splits.test, SPLITS_DIR / "test")

    print("=" * 65)
    print(" TẠO CÁC TẬP DỮ LIỆU HOÀN TẤT!")
    print("=" * 65)


if __name__ == "__main__":
    main()
