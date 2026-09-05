"""Script đánh giá duy nhất một lần (Evaluate Once) trên tập Test độc lập kèm Error Analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_CONTRACT_V2,
    extract_features,
)
from phishguard.training.evaluation import classification_metrics, measure_inference_latency

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VAL_SUMMARY_JSON = ARTIFACTS_DIR / "validation_summary.json"
XGB_JSON = ARTIFACTS_DIR / "XGB.json"
TEST_REPORT_JSON = ARTIFACTS_DIR / "test_evaluation_report.json"
ERROR_ANALYSIS_JSON = ARTIFACTS_DIR / "error_analysis_report.json"


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


def analyze_errors(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    y_true: Any,
    y_preds: Any,
    y_scores: Any,
) -> dict[str, Any]:
    """Phân tích lỗi (Error Analysis) chuyên sâu cho False Positives và False Negatives theo pattern."""
    error_df = df.copy()
    error_df["y_true"] = y_true
    error_df["y_pred"] = y_preds
    error_df["y_score"] = y_scores

    # Ghép thêm các cột đặc trưng quan trọng
    for col in ["url_length", "subdomain_count", "brand_not_registered_domain", "has_ip_address", "has_punycode", "is_suspicious_tld"]:
        if col in features_df.columns:
            error_df[col] = features_df[col].values

    fps = error_df[(error_df["y_true"] == 0) & (error_df["y_pred"] == 1)]
    fns = error_df[(error_df["y_true"] == 1) & (error_df["y_pred"] == 0)]

    fp_analysis = {
        "total_false_positives": len(fps),
        "by_url_length": {
            "short_under_35_chars": int((fps["url"].str.len() < 35).sum()),
            "medium_35_to_75_chars": int(((fps["url"].str.len() >= 35) & (fps["url"].str.len() <= 75)).sum()),
            "long_over_75_chars": int((fps["url"].str.len() > 75).sum()),
        },
        "sample_false_positives": fps[["url", "y_score"]].head(10).to_dict(orient="records"),
    }

    fn_analysis = {
        "total_false_negatives": len(fns),
        "by_url_length": {
            "short_under_35_chars": int((fns["url"].str.len() < 35).sum()),
            "medium_35_to_75_chars": int(((fns["url"].str.len() >= 35) & (fns["url"].str.len() <= 75)).sum()),
            "long_over_75_chars": int((fns["url"].str.len() > 75).sum()),
        },
        "sample_false_negatives": fns[["url", "y_score"]].head(10).to_dict(orient="records"),
    }

    return {
        "false_positives_analysis": fp_analysis,
        "false_negatives_analysis": fn_analysis,
    }


def main() -> None:
    print("=" * 65)
    print(" 🧪 Bắt đầu Đánh giá Độc lập Duy nhất 1 Lần trên Tập Test...")
    print("=" * 65)

    if not XGB_JSON.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình XGBoost tại {XGB_JSON}. Vui lòng chạy scripts/train.py trước.")
    if not VAL_SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Không tìm thấy báo cáo validation tại {VAL_SUMMARY_JSON}.")

    with open(VAL_SUMMARY_JSON, encoding="utf-8") as f:
        val_summary = json.load(f)

    operating_threshold = val_summary.get("operating_threshold", val_summary.get("optimal_threshold", 0.5))
    feature_contract = val_summary.get("feature_contract", FEATURE_CONTRACT_V2)
    feature_cols = FEATURE_COLUMNS_V2 if feature_contract == FEATURE_CONTRACT_V2 else FEATURE_COLUMNS_V1

    print(f" Loaded Operating Threshold from Validation: {operating_threshold}")
    print(f" Loaded Feature Contract: {feature_contract} ({len(feature_cols)} features)")

    test_df = load_split("test")
    print(f" Test dataset size: {len(test_df):,} rows ({test_df['domain'].nunique():,} unique domains)")

    # Extract features
    print(f" Extracting {len(feature_cols)} features ({feature_contract}) for Test set...")
    features_list = [extract_features(u, contract=feature_contract) for u in test_df["url"]]
    X_test = pd.DataFrame(features_list, columns=feature_cols)
    y_test = test_df["label"].values

    # Load model
    print(f" Loading XGBoost model: {XGB_JSON}")
    model = XGBClassifier()
    model.load_model(XGB_JSON)

    print(" Computing inference predictions...")
    y_scores = model.predict_proba(X_test)[:, 1]
    y_preds = (y_scores >= operating_threshold).astype(int)

    metrics = classification_metrics(y_test, y_preds, y_scores)
    p50, p95 = measure_inference_latency(model, X_test, num_runs=50)
    metrics["p50_latency_ms"] = round(p50, 4)
    metrics["p95_latency_ms"] = round(p95, 4)

    test_report = {
        "dataset_name": "Test Set (Domain-Grouped Split)",
        "test_size_rows": len(test_df),
        "test_unique_domains": test_df["domain"].nunique(),
        "model_type": type(model).__name__,
        "feature_contract": feature_contract,
        "operating_threshold": operating_threshold,
        "metrics": metrics,
    }

    with open(TEST_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)

    # Perform Error Analysis
    print(" Running granular error analysis on False Positives and False Negatives...")
    error_analysis = analyze_errors(test_df, X_test, y_test, y_preds, y_scores)
    with open(ERROR_ANALYSIS_JSON, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP TẬP TEST (FINAL TEST METRICS)")
    print("=" * 65)
    print(f" • PR-AUC:              {metrics['pr_auc']:.4f}")
    print(f" • ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f" • Recall:              {metrics['recall'] * 100:.2f}%")
    print(f" • Precision:           {metrics['precision'] * 100:.2f}%")
    print(f" • False Positive Rate: {metrics['false_positive_rate'] * 100:.2f}% ({metrics['false_positives']:,} FPs)")
    print(f" • False Negative Rate: {metrics['false_negative_rate'] * 100:.2f}% ({metrics['false_negatives']:,} FNs)")
    print(f" • F1 Score:            {metrics['f1']:.4f}")
    print(f" • Brier Score:         {metrics['brier_score']:.4f}")
    print(f" • ECE (Calibration):   {metrics['expected_calibration_error']:.4f}")
    print(f" • Accuracy:            {metrics['accuracy'] * 100:.2f}%")
    print(f" • Latency:             p50 = {metrics['p50_latency_ms']} ms | p95 = {metrics['p95_latency_ms']} ms")
    print("=" * 65)
    print(f" [OK] Báo cáo đánh giá đã lưu vào: {TEST_REPORT_JSON}")
    print(f" [OK] Báo cáo phân tích lỗi đã lưu vào: {ERROR_ANALYSIS_JSON}")


if __name__ == "__main__":
    main()
