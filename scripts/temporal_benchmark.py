"""Future stress test dùng frozen release và phishing domain chưa từng gặp."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml

from API.services.model_loader import load_phishguard_model
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    FEATURE_COLUMNS_V4,
    FeatureExtractor,
)
from phishguard.training.data import (
    audit_and_clean_data,
    split_by_domain,
    temporal_split_future_unseen_domains,
)
from phishguard.training.evaluation import classification_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
LEGIT_CSV = DATA_DIR / "legit_url.csv"
PHISHING_CSV = DATA_DIR / "verified_online.csv"
RELEASES_DIR = PROJECT_ROOT / "releases" / "candidates"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _features(frame: pd.DataFrame, extractor: FeatureExtractor) -> pd.DataFrame:
    columns = {
        "lexical-v1": FEATURE_COLUMNS_V1,
        "lexical-v2": FEATURE_COLUMNS_V2,
        "lexical-v3": FEATURE_COLUMNS_V3,
        "lexical-v4": FEATURE_COLUMNS_V4,
    }[extractor.contract]
    url_col = "raw_url" if "raw_url" in frame.columns else "url"
    return pd.DataFrame(
        [extractor.extract(url) for url in frame[url_col]],
        columns=columns,
    )


def resolve_release_dir(release_dir: Path | None = None) -> Path:
    """Resolve candidate rõ ràng, không phụ thuộc production pointer."""
    if release_dir is not None:
        return release_dir.resolve()
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    with config_path.open(encoding="utf-8") as file:
        version = str((yaml.safe_load(file) or {}).get("model_version", "4.0.0"))
    return (RELEASES_DIR / f"phishguard-{version}").resolve()


def main(release_dir: Path | None = None) -> None:
    if release_dir is None:
        parser = ArgumentParser(description="Future Phishing Unseen-Domain Stress Test")
        parser.add_argument("--release-dir", type=Path, help="Candidate cần đánh giá")
        args = parser.parse_args()
        release_dir = args.release_dir
    release_dir = resolve_release_dir(release_dir)
    if not LEGIT_CSV.exists() or not PHISHING_CSV.exists():
        raise FileNotFoundError("Không tìm thấy snapshot dữ liệu nguồn")

    if not release_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy candidate: {release_dir}")
    loaded = load_phishguard_model(
        model_path=release_dir / "model.json",
        metadata_path=release_dir / "metadata.json",
    )
    legit_df = pd.read_csv(LEGIT_CSV)
    phishing_df = pd.read_csv(PHISHING_CSV)
    cleaned, _ = audit_and_clean_data(legit_df, phishing_df, LEGIT_CSV, PHISHING_CSV)

    phishing = cleaned[cleaned["label"] == 1].copy()
    if "submission_time" not in phishing.columns:
        raise ValueError("Phishing snapshot thiếu submission_time")
    future_phishing = temporal_split_future_unseen_domains(phishing, test_ratio=0.20).test

    legitimate = cleaned[cleaned["label"] == 0].copy()
    legit_holdout = split_by_domain(legitimate, test_size=0.20, validation_size=0.01).test
    future = pd.concat([future_phishing, legit_holdout], ignore_index=True).sample(
        frac=1.0, random_state=42
    )
    X_future = _features(future, loaded.feature_extractor)
    y_future = future["label"].to_numpy()
    raw_scores = loaded.model.predict_proba(X_future)[:, 1]
    scores = loaded.calibrator.calibrate(raw_scores)

    caution_predictions = (scores >= loaded.action_policy.caution_threshold).astype(int)
    block_predictions = (scores >= loaded.action_policy.block_threshold).astype(int)
    report = {
        "protocol": "future-phishing-unseen-domain-stress-test",
        "release": loaded.release_id,
        "model_version": loaded.model_version,
        "policy_version": loaded.policy_version,
        "future_phishing_domains": int(future_phishing["domain"].nunique()),
        "future_rows": int(len(future)),
        "domain_overlap_with_legitimate_holdout": int(
            len(set(future_phishing["domain"]).intersection(legit_holdout["domain"]))
        ),
        "recall_at_caution": classification_metrics(y_future, caution_predictions, scores),
        "recall_at_block": classification_metrics(y_future, block_predictions, scores),
    }
    locked_report = REPORTS_DIR / loaded.model_version / "locked_test_metrics.json"
    if locked_report.exists():
        locked_metrics = json.loads(locked_report.read_text(encoding="utf-8")).get("metrics", {})
        report["delta_pr_auc_vs_locked_test"] = report["recall_at_block"]["pr_auc"] - float(
            locked_metrics.get("pr_auc", 0.0)
        )
        report["delta_recall_vs_locked_test"] = report["recall_at_block"]["recall"] - float(
            locked_metrics.get("recall", 0.0)
        )

    report_path = REPORTS_DIR / loaded.model_version / "future_phishing_unseen_domain_stress.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
