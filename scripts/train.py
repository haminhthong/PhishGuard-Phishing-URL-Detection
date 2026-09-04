"""Script huấn luyện và so sánh mô hình PhishGuard ML theo quy chuẩn 15 giai đoạn."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from phishguard.features import FEATURE_COLUMNS, extract_features
from phishguard.training.baseline import create_baseline_models
from phishguard.training.evaluation import classification_metrics, measure_inference_latency

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


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


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất 12 đặc trưng lexical thống nhất qua phishguard.features."""
    start = time.perf_counter()
    extracted_features = df["url"].map(extract_features).tolist()
    feature_df = pd.DataFrame(extracted_features, columns=FEATURE_COLUMNS)
    elapsed = time.perf_counter() - start
    print(f"   Extracted {len(feature_df):,} rows x {len(FEATURE_COLUMNS)} features in {elapsed:.2f}s")
    return feature_df


def main() -> None:
    print("=" * 65)
    print(" 🚀 Bắt đầu Pipeline Huấn luyện & Benchmark PhishGuard ML...")
    print("=" * 65)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random_seed = config.get("random_seed", 42)

    # 1. Load splits
    train_raw = load_split("train")
    val_raw = load_split("validation")

    print("\n 🔍 Trích xuất đặc trưng cho tập Train và Validation...")
    X_train = build_feature_dataframe(train_raw)
    y_train = train_raw["label"].values

    X_val = build_feature_dataframe(val_raw)
    y_val = val_raw["label"].values

    results_table = []
    models_dict = {}

    # 2. Benchmark Baseline Models
    print("\n 📊 1. Huấn luyện và Đánh giá Baseline Models trên Validation Set...")
    baselines = create_baseline_models(random_state=random_seed)

    for name, model in baselines.items():
        print(f"   • Training {name}...")
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_val)[:, 1]
        else:
            y_scores = model.predict(X_val).astype(float)
        y_preds = (y_scores >= 0.5).astype(int)

        metrics = classification_metrics(y_val, y_preds, y_scores)
        p50, p95 = measure_inference_latency(model, X_val, num_runs=50)
        metrics["p50_latency_ms"] = round(p50, 4)
        metrics["p95_latency_ms"] = round(p95, 4)

        results_table.append({"model": name, **metrics})
        models_dict[name] = model

    # 3. Candidate Model 1: Random Forest
    print("\n 🌲 2. Huấn luyện Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=random_seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_scores = rf_model.predict_proba(X_val)[:, 1]
    rf_preds = (rf_scores >= 0.5).astype(int)
    rf_metrics = classification_metrics(y_val, rf_preds, rf_scores)
    p50, p95 = measure_inference_latency(rf_model, X_val, num_runs=50)
    rf_metrics["p50_latency_ms"] = round(p50, 4)
    rf_metrics["p95_latency_ms"] = round(p95, 4)
    results_table.append({"model": "Random Forest", **rf_metrics})
    models_dict["Random Forest"] = rf_model

    # 4. Candidate Model 2: XGBoost
    print("\n ⚡ 3. Huấn luyện XGBoost Classifier...")
    xgb_config = config.get("model", {})
    xgb_model = XGBClassifier(
        n_estimators=xgb_config.get("n_estimators", 300),
        max_depth=xgb_config.get("max_depth", 5),
        learning_rate=xgb_config.get("learning_rate", 0.1),
        eval_metric=xgb_config.get("eval_metric", "logloss"),
        random_state=random_seed,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_scores = xgb_model.predict_proba(X_val)[:, 1]
    xgb_preds = (xgb_scores >= 0.5).astype(int)
    xgb_metrics = classification_metrics(y_val, xgb_preds, xgb_scores)
    p50, p95 = measure_inference_latency(xgb_model, X_val, num_runs=50)
    xgb_metrics["p50_latency_ms"] = round(p50, 4)
    xgb_metrics["p95_latency_ms"] = round(p95, 4)
    results_table.append({"model": "XGBoost", **xgb_metrics})
    models_dict["XGBoost"] = xgb_model

    # 5. Threshold Selection on Validation Set for XGBoost
    print("\n 🎯 4. Lựa chọn Threshold tối ưu trên Tập Validation...")
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds_to_test = np.linspace(0.1, 0.9, 81)
    for th in thresholds_to_test:
        preds = (xgb_scores >= th).astype(int)
        metrics_th = classification_metrics(y_val, preds, xgb_scores)
        if metrics_th["f1"] > best_f1:
            best_f1 = metrics_th["f1"]
            best_threshold = round(float(th), 4)

    print(f"   Selected Optimal Threshold (F1 Max): {best_threshold} (Validation F1={best_f1:.4f})")

    # Display comparison summary
    results_df = pd.DataFrame(results_table)
    print("\n" + "=" * 65)
    print(" 📋 BẢNG SO SÁNH KẾT QUẢ BENCHMARK TRÊN TẬP VALIDATION")
    print("=" * 65)
    cols_display = ["model", "pr_auc", "recall", "precision", "false_positive_rate", "false_negative_rate", "p95_latency_ms"]
    print(results_df[cols_display].to_string(index=False))
    print("=" * 65)

    # Save trained XGBoost model and summary to artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = ARTIFACTS_DIR / "validation_summary.json"
    val_summary = {
        "benchmark_table": results_table,
        "selected_model": "XGBoost",
        "optimal_threshold": best_threshold,
        "validation_metrics_at_optimal_threshold": classification_metrics(
            y_val, (xgb_scores >= best_threshold).astype(int), xgb_scores
        ),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    # Export intermediate model
    xgb_json_path = ARTIFACTS_DIR / "XGB.json"
    xgb_model.save_model(xgb_json_path)
    print(f"\n [OK] Saved XGBoost model to JSON format: {xgb_json_path}")
    print(f" [OK] Saved Validation summary to: {summary_path}")


if __name__ == "__main__":
    main()
