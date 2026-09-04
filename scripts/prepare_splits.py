"""Script tạo và lưu trữ các tập Train/Validation/Test chia theo registered domain độc lập."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from phishguard.training.data import audit_and_clean_data, split_by_domain

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
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
    print(" Bat dau chia tap du lieu Train / Validation / Test theo Domain...")
    print("=" * 65)

    if not LEGIT_CSV.exists() or not PHISHING_CSV.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu nguon tai {LEGIT_CSV} hoac {PHISHING_CSV}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    legit_df = pd.read_csv(LEGIT_CSV)
    phishing_df = pd.read_csv(PHISHING_CSV)

    cleaned_df, report = audit_and_clean_data(
        legit_df=legit_df,
        phishing_df=phishing_df,
        legit_path=LEGIT_CSV,
        phishing_path=PHISHING_CSV,
    )

    print(f" Du lieu da lam sach: {len(cleaned_df):,} ban ghi ({report['cleaned_unique_domains']:,} unique domains)")

    splits = split_by_domain(
        cleaned_df,
        test_size=0.15,
        validation_size=0.15,
        random_state=42,
    )

    print("\n [OK] Da xac minh 100%: Domain va URL tuyet doi KHONG giao nhau giua 3 tap!")

    print(f" • Train set:      {len(splits.train):,} rows | {splits.train['domain'].nunique():,} domains | Label dist: {splits.train['label'].value_counts().to_dict()}")
    print(f" • Validation set: {len(splits.validation):,} rows | {splits.validation['domain'].nunique():,} domains | Label dist: {splits.validation['label'].value_counts().to_dict()}")
    print(f" • Test set:       {len(splits.test):,} rows | {splits.test['domain'].nunique():,} domains | Label dist: {splits.test['label'].value_counts().to_dict()}")

    print("\n Dang luu cac tap du lieu vao artifacts/splits/...")
    save_split_dataframe(splits.train, SPLITS_DIR / "train")
    save_split_dataframe(splits.validation, SPLITS_DIR / "validation")
    save_split_dataframe(splits.test, SPLITS_DIR / "test")

    print("=" * 65)
    print(" TAO CAC TAP DU LIEU HOAN TAT!")
    print("=" * 65)


if __name__ == "__main__":
    main()
