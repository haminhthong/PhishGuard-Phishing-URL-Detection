"""Script audit dữ liệu tự động cho PhishGuard ML."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from phishguard.training.data import audit_and_clean_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LEGIT_CSV = DATA_DIR / "legit_url.csv"
PHISHING_CSV = DATA_DIR / "verified_online.csv"
REPORT_JSON = ARTIFACTS_DIR / "data_quality_report.json"


def main() -> None:
    print("=" * 65)
    print(" Bat dau audit chat luong du lieu nguon PhishGuard ML...")
    print("=" * 65)

    if not LEGIT_CSV.exists() or not PHISHING_CSV.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu tai {LEGIT_CSV} hoac {PHISHING_CSV}")

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

    print("\n" + "=" * 65)
    print(" BAO CAO AUDIT CHAT LUONG DU LIEU NGUON")
    print("=" * 65)
    print(f" • Raw Legitimate rows:      {report['legitimate_rows']:,}")
    print(f" • Raw Phishing rows:        {report['phishing_rows']:,}")
    print(f" • Invalid URLs removed:     {report['invalid_urls_removed']:,}")
    print(f" • Duplicate URLs removed:   {report['duplicate_urls_removed']:,}")
    print(f" • Conflicting domains count:{report['conflicting_domains_count']:,}")
    print(f" • Conflicting rows removed: {report['conflicting_rows_removed']:,}")
    print(f" • Final Cleaned total rows: {report['cleaned_total_rows']:,}")
    print(f" • Cleaned Label Dist:       {report['cleaned_label_distribution']}")
    print(f" • Cleaned Unique Domains:   {report['cleaned_unique_domains']:,}")
    print(f" • Legitimate SHA-256:       {report['legitimate_sha256']}")
    print(f" • Phishing SHA-256:         {report['phishing_sha256']}")
    print(f" • Save location:            {REPORT_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    main()
