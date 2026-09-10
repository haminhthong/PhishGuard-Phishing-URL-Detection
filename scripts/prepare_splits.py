"""Tạo snapshot dữ liệu và 5 split domain-disjoint có stratification."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from phishguard.training.data import DatasetManifest, audit_and_clean_data, split_by_domain

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
LEGIT_CSV = DATA_DIR / "legit_url.csv"
PHISHING_CSV = DATA_DIR / "verified_online.csv"
SPLIT_NAMES = ("train", "validation", "calibration", "threshold_validation", "test")


def save_split_dataframe(df: pd.DataFrame, file_prefix: Path) -> None:
    """Lưu split dạng Parquet; fallback CSV khi engine Parquet không có."""
    try:
        parquet_path = file_prefix.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"   Đã lưu: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.2f} MB)")
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        csv_path = file_prefix.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        print(f"   Đã lưu CSV fallback ({error}): {csv_path}")


def main() -> None:
    print("=" * 65)
    print(" Bắt đầu snapshot và chia dữ liệu 5-way stratified group-disjoint...")
    print("=" * 65)

    if not LEGIT_CSV.exists() or not PHISHING_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {LEGIT_CSV} hoặc {PHISHING_CSV}")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    split_config = config.get("split", {})
    random_seed = int(config.get("random_seed", 42))
    dataset_version = str(config.get("model_version", "4.0.0"))
    proportions = {
        "train": float(split_config.get("train", 0.60)),
        "validation": float(split_config.get("validation", 0.15)),
        "calibration": float(split_config.get("calibration", 0.10)),
        "threshold_validation": float(split_config.get("threshold_validation", 0.05)),
        "test": float(split_config.get("test", 0.10)),
    }

    legit_df = pd.read_csv(LEGIT_CSV)
    phishing_df = pd.read_csv(PHISHING_CSV)
    cleaned_df, report = audit_and_clean_data(
        legit_df=legit_df,
        phishing_df=phishing_df,
        legit_path=LEGIT_CSV,
        phishing_path=PHISHING_CSV,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
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
    (ARTIFACTS_DIR / "dataset_manifest.json").write_text(
        json.dumps(asdict(dataset_manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    splits = split_by_domain(
        cleaned_df,
        train_size=proportions["train"],
        validation_size=proportions["validation"],
        calibration_size=proportions["calibration"],
        threshold_validation_size=proportions["threshold_validation"],
        test_size=proportions["test"],
        random_state=random_seed,
    )
    split_manifest = {
        "split_version": "domain-5way-v3",
        "seed": random_seed,
        "strategy": "stratified-registered-domain-5way",
        "source_dataset_sha256": hashlib.sha256(
            json.dumps(dataset_manifest.source_checksums, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "target_proportions": proportions,
        "splits": {},
    }
    for name in SPLIT_NAMES:
        split = getattr(splits, name)
        stats = {
            "rows": int(len(split)),
            "domains": int(split["domain"].nunique()),
            "positive_rate": round(float(split["label"].mean()), 6),
        }
        canonical_name = {
            "validation": "development",
            "test": "locked_test",
        }.get(name, name)
        split_manifest["splits"][canonical_name] = stats
        print(
            f" • {name:18s}: {len(split):,} rows | "
            f"{split['domain'].nunique():,} domains | positive={split['label'].mean():.2%}"
        )
        save_split_dataframe(split, SPLITS_DIR / name)

    domain_sets = [set(getattr(splits, name)["domain"]) for name in SPLIT_NAMES]
    url_sets = [set(getattr(splits, name)["canonical_url"]) for name in SPLIT_NAMES]
    split_manifest["domain_overlap"] = int(
        sum(
            len(left.intersection(right))
            for i, left in enumerate(domain_sets)
            for right in domain_sets[i + 1 :]
        )
    )
    split_manifest["canonical_url_overlap"] = int(
        sum(
            len(left.intersection(right))
            for i, left in enumerate(url_sets)
            for right in url_sets[i + 1 :]
        )
    )
    if split_manifest["domain_overlap"] or split_manifest["canonical_url_overlap"]:
        raise AssertionError("Split manifest phát hiện overlap domain hoặc canonical URL")

    (ARTIFACTS_DIR / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(" [OK] Hoàn tất. Mỗi registered domain chỉ xuất hiện trong một split.")


if __name__ == "__main__":
    main()
