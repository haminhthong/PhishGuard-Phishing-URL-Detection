"""Script đánh giá độc lập duy nhất một lần (Evaluate Once) trên tập Test kèm Slices & Error Analysis."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from phishguard.calibration import ActionPolicy, ProbabilityCalibrator
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    FEATURE_COLUMNS_V4,
    FEATURE_CONTRACT_V2,
    FeatureExtractor,
)
from phishguard.features.resources import load_resource_bundle
from phishguard.training.evaluation import classification_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
RELEASES_DIR = PROJECT_ROOT / "releases"
CANDIDATES_DIR = RELEASES_DIR / "candidates"
REPORTS_DIR = PROJECT_ROOT / "reports"


def sha256_file(path: Path) -> str:
    """Tính checksum artifact để gắn lineage cho evaluation report."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str | None, name: str) -> str:
    """Fail-closed nếu artifact evaluation không khớp metadata."""
    actual = sha256_file(path)
    if not expected or actual.lower() != str(expected).lower():
        raise ValueError(f"Checksum {name} không khớp metadata")
    return actual


def load_split(name: str) -> pd.DataFrame:
    """Tải tập dữ liệu đã split từ Parquet hoặc CSV fallback."""
    parquet_file = SPLITS_DIR / f"{name}.parquet"
    if parquet_file.exists():
        print(f" Loading {name} split from Parquet: {parquet_file}")
        return pd.read_parquet(parquet_file)
    csv_file = SPLITS_DIR / f"{name}.csv.gz"
    if csv_file.exists():
        print(f" Loading {name} split from CSV.gz: {csv_file}")
        return pd.read_csv(csv_file)
    raise FileNotFoundError(f"Không tìm thấy dữ liệu split {name} tại {SPLITS_DIR}")


def measure_detailed_latencies(
    model: XGBClassifier,
    calibrator: ProbabilityCalibrator,
    action_policy: ActionPolicy,
    sample_urls: list[str],
    contract: str,
    feature_cols: tuple[str, ...],
    extractor: FeatureExtractor,
    num_runs: int = 50,
) -> dict[str, float]:
    """Đo tách bạch độ trễ từng công đoạn: feature extraction, model inference, risk policy, E2E."""
    sample_slice = sample_urls[:20] if len(sample_urls) >= 20 else sample_urls
    feat_lats, model_lats, policy_lats, e2e_lats = [], [], [], []

    for _ in range(num_runs):
        for u in sample_slice:
            t0 = time.perf_counter()

            # 1. Trích xuất đặc trưng
            t_feat_start = time.perf_counter()
            features = extractor.extract(u)
            frame = pd.DataFrame([features], columns=feature_cols)
            t_feat_end = time.perf_counter()

            # 2. Suy luận mô hình
            t_mod_start = time.perf_counter()
            raw_prob = float(model.predict_proba(frame)[0, 1])
            cal_prob = float(calibrator.calibrate(raw_prob))
            t_mod_end = time.perf_counter()

            # 3. Đánh giá policy rủi ro
            t_pol_start = time.perf_counter()
            _, _ = action_policy.evaluate(cal_prob)
            t_pol_end = time.perf_counter()

            t1 = time.perf_counter()

            feat_lats.append((t_feat_end - t_feat_start) * 1000.0)
            model_lats.append((t_mod_end - t_mod_start) * 1000.0)
            policy_lats.append((t_pol_end - t_pol_start) * 1000.0)
            e2e_lats.append((t1 - t0) * 1000.0)

    return {
        "feature_extraction_p50_ms": round(float(np.percentile(feat_lats, 50)), 4),
        "feature_extraction_p95_ms": round(float(np.percentile(feat_lats, 95)), 4),
        "model_inference_p50_ms": round(float(np.percentile(model_lats, 50)), 4),
        "model_inference_p95_ms": round(float(np.percentile(model_lats, 95)), 4),
        "risk_policy_p50_ms": round(float(np.percentile(policy_lats, 50)), 4),
        "risk_policy_p95_ms": round(float(np.percentile(policy_lats, 95)), 4),
        "total_e2e_p50_ms": round(float(np.percentile(e2e_lats, 50)), 4),
        "total_e2e_p95_ms": round(float(np.percentile(e2e_lats, 95)), 4),
    }


def evaluate_hard_slices(
    test_df: pd.DataFrame,
    features_df: pd.DataFrame,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    block_threshold: float,
) -> dict[str, Any]:
    """Đánh giá chi tiết trên các lát cắt hard cases (Hard Slices Evaluation)."""
    y_preds = (y_scores >= block_threshold).astype(int)

    shared_domains = {
        "google.com",
        "dropbox.com",
        "live.com",
        "office.com",
        "wix.com",
        "wordpress.com",
        "t.co",
        "telegram.org",
        "github.io",
        "pages.dev",
        "firebaseapp.com",
        "web.app",
        "amazonaws.com",
    }

    slices = {
        "brand_impersonation": (
            (features_df["brand_in_subdomain"] == 1)
            | (features_df["brand_in_path"] == 1)
            | (features_df["brand_not_registered_domain"] == 1)
        ).values
        if "brand_in_subdomain" in features_df.columns
        else np.zeros(len(test_df), dtype=bool),
        "shared_hosting_cloud": (test_df["domain"].isin(shared_domains)).values
        if "domain" in test_df.columns
        else np.zeros(len(test_df), dtype=bool),
        "url_shortener": (features_df["uses_shortening_service"] == 1).values
        if "uses_shortening_service" in features_df.columns
        else np.zeros(len(test_df), dtype=bool),
        "punycode": (features_df["has_punycode"] == 1).values
        if "has_punycode" in features_df.columns
        else np.zeros(len(test_df), dtype=bool),
        "long_url": (features_df["url_length"] > 75).values
        if "url_length" in features_df.columns
        else np.zeros(len(test_df), dtype=bool),
    }

    slice_reports = {}
    for slice_name, mask in slices.items():
        count = int(np.sum(mask))
        if count == 0:
            slice_reports[slice_name] = {"count": 0, "status": "no_samples"}
            continue

        slice_true = y_true[mask]
        slice_pred = y_preds[mask]
        slice_scores = y_scores[mask]

        pos_count = int(np.sum(slice_true == 1))
        neg_count = int(np.sum(slice_true == 0))

        metrics = classification_metrics(slice_true, slice_pred, slice_scores)
        slice_reports[slice_name] = {
            "sample_count": count,
            "phishing_count": pos_count,
            "legitimate_count": neg_count,
            "recall": round(metrics.get("recall", 0.0), 4),
            "precision": round(metrics.get("precision", 0.0), 4),
            "false_positive_rate": round(metrics.get("false_positive_rate", 0.0), 4),
            "f1": round(metrics.get("f1", 0.0), 4),
            "pr_auc": round(metrics.get("pr_auc", 0.0), 4)
            if pos_count > 0 and neg_count > 0
            else None,
        }

    return slice_reports


def analyze_errors(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    y_true: Any,
    y_preds: Any,
    y_scores: Any,
) -> dict[str, Any]:
    """Phân tích lỗi chuyên sâu cho False Positives và False Negatives."""
    error_df = df.copy()
    error_df["y_true"] = y_true
    error_df["y_pred"] = y_preds
    error_df["y_score"] = y_scores

    url_col = "raw_url" if "raw_url" in error_df.columns else "url"

    for col in [
        "url_length",
        "subdomain_count",
        "brand_not_registered_domain",
        "has_ip_address",
        "has_punycode",
        "is_suspicious_tld",
    ]:
        if col in features_df.columns:
            error_df[col] = features_df[col].values

    fps = error_df[(error_df["y_true"] == 0) & (error_df["y_pred"] == 1)]
    fns = error_df[(error_df["y_true"] == 1) & (error_df["y_pred"] == 0)]

    fp_analysis = {
        "total_false_positives": len(fps),
        "by_url_length": {
            "short_under_35_chars": int((fps[url_col].str.len() < 35).sum()),
            "medium_35_to_75_chars": int(
                ((fps[url_col].str.len() >= 35) & (fps[url_col].str.len() <= 75)).sum()
            ),
            "long_over_75_chars": int((fps[url_col].str.len() > 75).sum()),
        },
        "sample_false_positives": fps[[url_col, "y_score"]].head(10).to_dict(orient="records"),
    }

    fn_analysis = {
        "total_false_negatives": len(fns),
        "by_url_length": {
            "short_under_35_chars": int((fns[url_col].str.len() < 35).sum()),
            "medium_35_to_75_chars": int(
                ((fns[url_col].str.len() >= 35) & (fns[url_col].str.len() <= 75)).sum()
            ),
            "long_over_75_chars": int((fns[url_col].str.len() > 75).sum()),
        },
        "sample_false_negatives": fns[[url_col, "y_score"]].head(10).to_dict(orient="records"),
    }

    return {
        "false_positives_analysis": fp_analysis,
        "false_negatives_analysis": fn_analysis,
    }


def resolve_candidate_dir(release_dir: Path | None = None) -> Path:
    """Chỉ resolve candidate; không đánh giá active production pointer."""
    if release_dir is not None:
        return release_dir
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    version = str(config.get("model_version", "4.0.0"))
    return CANDIDATES_DIR / f"phishguard-{version}"


def main(release_dir: Path | None = None) -> None:
    print("=" * 70)
    print(" 🧪 Đánh giá Locked Test cho release candidate (không tự promote)...")
    print("=" * 70)

    # 1. Chỉ nạp candidate đã freeze; production pointer không được dùng để tìm test score.
    model_dir = resolve_candidate_dir(release_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy release candidate tại {model_dir}. Hãy chạy scripts/train.py trước."
        )

    print(f" Nạp mô hình từ: {model_dir}")
    model_json_path = model_dir / "model.json"

    if not model_json_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại {model_json_path}. Vui lòng chạy scripts/train.py trước."
        )

    # 2. Nạp model và metadata
    model = XGBClassifier()
    model.load_model(model_json_path)

    metadata = {}
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Thiếu metadata.json trong release: {model_dir}")
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    if not metadata.get("release_id"):
        raise ValueError("Candidate metadata thiếu release_id")

    policy_path = model_dir / "action_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"Thiếu action_policy.json trong release: {model_dir}")
    with policy_path.open(encoding="utf-8") as f:
        action_policy = ActionPolicy.from_dict(json.load(f))
    block_threshold = action_policy.block_threshold
    feature_contract = metadata.get("feature_contract", FEATURE_CONTRACT_V2)
    feature_cols = {
        "lexical-v1": FEATURE_COLUMNS_V1,
        FEATURE_CONTRACT_V2: FEATURE_COLUMNS_V2,
        "lexical-v3": FEATURE_COLUMNS_V3,
        "lexical-v4": FEATURE_COLUMNS_V4,
    }[feature_contract]

    resource_dir = model_dir / "resources"
    if not resource_dir.is_dir():
        raise FileNotFoundError(f"Candidate thiếu resource bundle: {resource_dir}")
    resource_bundle = load_resource_bundle(resource_dir)
    extractor = FeatureExtractor(feature_contract, resource_bundle)

    # Calibration là artifact bắt buộc; không chạy raw score khi thiếu file.
    calib_path = model_dir / "calibration.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Thiếu calibration.json trong release: {model_dir}")
    with calib_path.open(encoding="utf-8") as f:
        calib_data = json.load(f)
    model_hash = verify_hash(model_json_path, metadata.get("model_sha256"), "model")
    calibration_hash = verify_hash(calib_path, metadata.get("calibration_sha256"), "calibration")
    action_policy_hash = verify_hash(
        policy_path, metadata.get("action_policy_sha256"), "action_policy"
    )
    calibrator = ProbabilityCalibrator(
        method=calib_data["method"],
        params=calib_data["calibrator_params"],
    )

    print(f" Loaded Block Threshold:      {block_threshold}")
    print(f" Loaded Feature Contract:    {feature_contract} ({len(feature_cols)} features)")
    print(f" Loaded Action Policy:       {action_policy.to_dict()}")
    print(f" Loaded Calibrator:          {calibrator.method} (is_fitted={calibrator.is_fitted})")

    # 3. Tải tập Locked Test
    test_df = load_split("test")
    url_col = "raw_url" if "raw_url" in test_df.columns else "url"
    print(
        f" Test dataset size: {len(test_df):,} rows ({test_df['domain'].nunique():,} unique domains)"
    )

    # 4. Trích xuất đặc trưng cho Locked Test
    print(f" Extracting {len(feature_cols)} features ({feature_contract}) from {url_col}...")
    features_list = [extractor.extract(u) for u in test_df[url_col]]
    X_test = pd.DataFrame(features_list, columns=feature_cols)
    y_test = test_df["label"].values

    # 5. Dự đoán và Hiệu chuẩn
    print(" Computing calibrated inference predictions...")
    raw_scores = model.predict_proba(X_test)[:, 1]
    calibrated_scores = calibrator.calibrate(raw_scores)
    y_preds = (calibrated_scores >= block_threshold).astype(int)

    metrics = classification_metrics(y_test, y_preds, calibrated_scores)
    policy_metrics = {
        "caution": classification_metrics(
            y_test,
            (calibrated_scores >= action_policy.caution_threshold).astype(int),
            calibrated_scores,
        ),
        "block": metrics,
    }

    # 6. Đo tách bạch độ trễ
    print(" Measuring fine-grained latency breakdown (Feature extraction, Model, Policy, E2E)...")
    lat_breakdown = measure_detailed_latencies(
        model=model,
        calibrator=calibrator,
        action_policy=action_policy,
        sample_urls=test_df[url_col].tolist(),
        contract=feature_contract,
        feature_cols=feature_cols,
        extractor=extractor,
        num_runs=30,
    )
    metrics.update(lat_breakdown)

    # 7. Đánh giá các hard slice chi tiết
    print(" Evaluating granular performance on critical hard slices...")
    slice_evaluation = evaluate_hard_slices(
        test_df=test_df,
        features_df=X_test,
        y_true=y_test,
        y_scores=calibrated_scores,
        block_threshold=block_threshold,
    )

    # 8. Phân tích lỗi
    print(" Running granular error analysis on False Positives and False Negatives...")
    error_analysis = analyze_errors(test_df, X_test, y_test, y_preds, calibrated_scores)

    split_manifest_path = model_dir / "split_manifest.json"
    split_hash = sha256_file(split_manifest_path) if split_manifest_path.is_file() else None
    evaluation_run_id = (
        f"eval-{metadata.get('model_version', model_dir.name)}-{sha256_file(model_json_path)[:12]}"
    )
    test_report = {
        "evaluation_run_id": evaluation_run_id,
        "release": metadata.get("release_id", metadata.get("model_tag", model_dir.name)),
        "dataset_name": "Test Set (Domain-Grouped Split, Zero Leakage)",
        "test_size_rows": len(test_df),
        "test_unique_domains": test_df["domain"].nunique(),
        "model_type": type(model).__name__,
        "model_version": metadata.get("model_version", "4.0.0"),
        "feature_contract": feature_contract,
        "locked_test": True,
        "metrics": metrics,
        "policy_metrics": policy_metrics,
        "action_policy": action_policy.to_dict(),
        "artifact_hashes": {
            "model": model_hash,
            "calibration": calibration_hash,
            "action_policy": action_policy_hash,
            "feature_contract": metadata.get("feature_contract_hash"),
            "resources": resource_bundle.hashes,
        },
        "split_manifest_sha256": split_hash,
        "hard_slices": slice_evaluation,
        "latency_breakdown": lat_breakdown,
    }

    release_report_dir = REPORTS_DIR / str(metadata.get("model_version", model_dir.name))
    release_report_dir.mkdir(parents=True, exist_ok=True)
    with (release_report_dir / "locked_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)

    with (release_report_dir / "error_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)

    # Lưu bản sao evaluation.json vào thư mục candidate theo version
    if model_dir.is_dir():
        with open(model_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP TẬP TEST (FINAL UNTOUCHED TEST METRICS)")
    print("=" * 70)
    print(f" • PR-AUC:                     {metrics['pr_auc']:.4f}")
    print(f" • ROC-AUC:                    {metrics['roc_auc']:.4f}")
    print(f" • Recall:                     {metrics['recall'] * 100:.2f}%")
    print(f" • Precision:                  {metrics['precision'] * 100:.2f}%")
    print(
        f" • False Positive Rate:        {metrics['false_positive_rate'] * 100:.2f}% ({metrics['false_positives']:,} FPs)"
    )
    print(
        f" • False Negative Rate:        {metrics['false_negative_rate'] * 100:.2f}% ({metrics['false_negatives']:,} FNs)"
    )
    print(f" • F1 Score:                   {metrics['f1']:.4f}")
    print(f" • Brier Score:                {metrics['brier_score']:.4f}")
    print(f" • ECE (Calibration):          {metrics['expected_calibration_error']:.4f}")
    print(f" • Accuracy:                   {metrics['accuracy'] * 100:.2f}%")
    print(
        f" • Model Inference Latency:    p50={lat_breakdown['model_inference_p50_ms']} ms | p95={lat_breakdown['model_inference_p95_ms']} ms"
    )
    print(
        f" • Full End-to-End Latency:    p50={lat_breakdown['total_e2e_p50_ms']} ms | p95={lat_breakdown['total_e2e_p95_ms']} ms"
    )
    print("=" * 70)
    print(f" [OK] Báo cáo locked test đã lưu: {release_report_dir / 'locked_test_metrics.json'}")
    print(f" [OK] Phân tích lỗi đã lưu:        {release_report_dir / 'error_analysis.json'}")
    if (model_dir / "evaluation.json").exists():
        print(f" [OK] Báo cáo versioned đã lưu: {model_dir / 'evaluation.json'}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Đánh giá locked test cho một release candidate.")
    parser.add_argument("--release-dir", type=Path, help="Thư mục candidate cần đánh giá.")
    args = parser.parse_args()
    main(args.release_dir)
