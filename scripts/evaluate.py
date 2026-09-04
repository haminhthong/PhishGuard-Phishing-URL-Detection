"""Script đánh giá duy nhất một lần (Evaluate Once) trên tập Test độc lập."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from phishguard.features import FEATURE_COLUMNS, extract_features
from phishguard.training.evaluation import classification_metrics, measure_inference_latency

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VAL_SUMMARY_JSON = ARTIFACTS_DIR / "validation_summary.json"
XGB_JSON = ARTIFACTS_DIR / "XGB.json"
TEST_REPORT_JSON = ARTIFACTS_DIR / "test_evaluation_report.json"


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

    optimal_threshold = val_summary.get("optimal_threshold", 0.5)
    print(f" Loaded optimal threshold from Validation: {optimal_threshold}")

    test_df = load_split("test")

    print(f" Test dataset size: {len(test_df):,} rows ({test_df['domain'].nunique():,} unique domains)")

    # Extract features
    print(" Extracting 12 lexical features for Test set...")
    features_list = test_df["url"].map(extract_features).tolist()
    X_test = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)
    y_test = test_df["label"].values

    # Load model
    print(f" Loading XGBoost model: {XGB_JSON}")
    model = XGBClassifier()
    model.load_model(XGB_JSON)

    print(" Computing inference predictions...")
    y_scores = model.predict_proba(X_test)[:, 1]
    y_preds = (y_scores >= optimal_threshold).astype(int)

    metrics = classification_metrics(y_test, y_preds, y_scores)
    p50, p95 = measure_inference_latency(model, X_test, num_runs=50)
    metrics["p50_latency_ms"] = round(p50, 4)
    metrics["p95_latency_ms"] = round(p95, 4)

    test_report = {
        "dataset_name": "Test Set (Domain-Grouped Split)",
        "test_size_rows": len(test_df),
        "test_unique_domains": test_df["domain"].nunique(),
        "model_type": "XGBClassifier",
        "threshold": optimal_threshold,
        "metrics": metrics,
    }

    with open(TEST_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP TẬP TEST (FINAL TEST METRICS)")
    print("=" * 65)
    print(f" • Accuracy:            {metrics['accuracy'] * 100:.2f}%")
    print(f" • Precision:           {metrics['precision'] * 100:.2f}%")
    print(f" • Recall:              {metrics['recall'] * 100:.2f}%")
    print(f" • F1 Score:            {metrics['f1']:.4f}")
    print(f" • PR-AUC:              {metrics['pr_auc']:.4f}")
    print(f" • ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f" • False Positive Rate: {metrics['false_positive_rate'] * 100:.2f}%")
    print(f" • False Negative Rate: {metrics['false_negative_rate'] * 100:.2f}%")
    print(f" • Brier Score:         {metrics['brier_score']:.4f}")
    print(f" • Confusion Matrix:    TN={metrics['true_negatives']:,}, FP={metrics['false_positives']:,}, FN={metrics['false_negatives']:,}, TP={metrics['true_positives']:,}")
    print(f" • Latency p50:         {metrics['p50_latency_ms']} ms")
    print(f" • Latency p95:         {metrics['p95_latency_ms']} ms")
    print(f" • Saved report to:     {TEST_REPORT_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    main()
