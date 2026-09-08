"""Script audit dữ liệu tự động cho PhishGuard ML."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from phishguard.features import FeatureExtractor
from phishguard.training.data import DatasetManifest, audit_and_clean_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
LEGIT_CSV = DATA_DIR / "legit_url.csv"
PHISHING_CSV = DATA_DIR / "verified_online.csv"
REPORT_JSON = ARTIFACTS_DIR / "data_quality_report.json"
DATASET_MANIFEST_JSON = ARTIFACTS_DIR / "dataset_manifest.json"
SOURCE_BIAS_JSON = ARTIFACTS_DIR / "source_bias_report.json"


def build_source_bias_report(frame: pd.DataFrame) -> dict[str, object]:
    """Đo các thuộc tính lexical dễ phân biệt source và label."""
    url_column = "raw_url" if "raw_url" in frame.columns else "url"
    features = pd.DataFrame([FeatureExtractor().extract(url) for url in frame[url_column]])
    report: dict[str, object] = {
        "status": "computed",
        "warning": "Source statistics are diagnostic, not evidence of causal phishing signals.",
        "by_label": {},
    }
    for label, source_name in ((0, "legitimate_source"), (1, "phishing_source")):
        mask = frame["label"].to_numpy() == label
        subset = features.loc[mask]
        report["by_label"][source_name] = {
            "rows": int(mask.sum()),
            "mean_url_length": round(float(subset["url_length"].mean()), 4),
            "mean_path_length": round(float(subset["path_length"].mean()), 4),
            "mean_query_length": round(float(subset["query_length"].mean()), 4),
            "root_page_rate": round(float((subset["path_length"] == 0).mean()), 4),
            "shortener_rate": round(float(subset["uses_shortening_service"].mean()), 4),
            "punycode_rate": round(float(subset["has_punycode"].mean()), 4),
            "suspicious_tld_rate": round(float(subset["is_suspicious_tld"].mean()), 4),
            "brand_term_rate": round(float(subset["brand_not_registered_domain"].mean()), 4),
        }
    return report


def main() -> None:
    print("=" * 65)
    print(" Bắt đầu audit chất lượng dữ liệu nguồn PhishGuard ML...")
    print("=" * 65)

    if not LEGIT_CSV.exists() or not PHISHING_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {LEGIT_CSV} hoặc {PHISHING_CSV}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f" Reading legitimate dataset: {LEGIT_CSV}")
    legit_df = pd.read_csv(LEGIT_CSV)

    print(f" Reading phishing dataset:   {PHISHING_CSV}")
    phishing_df = pd.read_csv(PHISHING_CSV)

    cleaned_df, report = audit_and_clean_data(
        legit_df=legit_df,
        phishing_df=phishing_df,
        legit_path=LEGIT_CSV,
        phishing_path=PHISHING_CSV,
    )

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with CONFIG_PATH.open(encoding="utf-8") as file:
        model_version = str((yaml.safe_load(file) or {}).get("model_version", "4.0.0"))

    manifest = DatasetManifest(
        dataset_version=f"v{model_version}",
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

    with open(DATASET_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)

    with open(SOURCE_BIAS_JSON, "w", encoding="utf-8") as f:
        json.dump(build_source_bias_report(cleaned_df), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(" BÁO CÁO AUDIT CHẤT LƯỢNG DỮ LIỆU NGUỒN")
    print("=" * 65)
    print(f" • Raw Legitimate rows:           {report['legitimate_rows']:,}")
    print(f" • Raw Phishing rows:             {report['phishing_rows']:,}")
    print(f" • Invalid URLs removed:          {report['invalid_urls_removed']:,}")
    print(f" • Exact conflict URLs removed:   {report['exact_conflicts_removed']:,}")
    print(f" • Duplicate URLs removed:        {report['duplicate_urls_removed']:,}")
    print(
        f" • Multi-label domains preserved: {report['multi_label_domains_preserved_count']:,} domains ({report['multi_label_rows_preserved']:,} URLs)"
    )
    print(f" • Final Cleaned total rows:      {report['cleaned_total_rows']:,}")
    print(f" • Cleaned Label Dist:            {report['cleaned_label_distribution']}")
    print(f" • Cleaned Unique Domains:        {report['cleaned_unique_domains']:,}")
    print(f" • Legitimate SHA-256:            {report['legitimate_sha256']}")
    print(f" • Phishing SHA-256:              {report['phishing_sha256']}")
    print(f" • Report location:               {REPORT_JSON}")
    print(f" • Manifest location:             {DATASET_MANIFEST_JSON}")
    print(f" • Source-bias report:             {SOURCE_BIAS_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    main()
