"""Script huấn luyện, so sánh benchmark và tự động lựa chọn mô hình PhishGuard ML."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    extract_features,
)
from phishguard.training.baseline import create_baseline_models
from phishguard.training.evaluation import (
    classification_metrics,
    compute_ece,
    measure_inference_latency,
    threshold_sweep,
)

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


def build_feature_dataframe(df: pd.DataFrame, contract: str) -> pd.DataFrame:
    """Trích xuất đặc trưng thống nhất qua phishguard.features theo hợp đồng v1 hoặc v2."""
    cols = FEATURE_COLUMNS_V2 if contract == FEATURE_CONTRACT_V2 else FEATURE_COLUMNS_V1
    start = time.perf_counter()
    extracted = [extract_features(u, contract=contract) for u in df["url"]]
    feature_df = pd.DataFrame(extracted, columns=cols)
    elapsed = time.perf_counter() - start
    print(f"   Trích xuất {len(feature_df):,} bản ghi x {len(cols)} đặc trưng ({contract}) trong {elapsed:.2f}s")
    return feature_df


def select_best_candidate_model(
    benchmark_results: list[dict[str, Any]],
    guardrails: dict[str, Any],
) -> tuple[str, str]:
    """
    Tự động lựa chọn mô hình tối ưu dựa trên PR-AUC và các guardrails:
    - Recall >= min_recall (e.g. 0.80)
    - FPR <= max_fpr (e.g. 0.01)
    - p95_latency_ms <= max_latency (e.g. 5.0 ms)
    """
    min_recall = guardrails.get("min_recall", 0.80)
    max_fpr = guardrails.get("max_fpr", 0.01)
    max_latency = guardrails.get("max_latency_p95_ms", 5.0)

    # Lọc các mô hình thỏa mãn guardrails
    qualified = []
    for row in benchmark_results:
        rec = row.get("recall", 0.0)
        fpr = row.get("false_positive_rate", 1.0)
        lat = row.get("p95_latency_ms", 999.0)
        if rec >= min_recall and fpr <= max_fpr and lat <= max_latency:
            qualified.append(row)

    if not qualified:
        # Nếu không mô hình nào thỏa toàn bộ guardrail, chọn theo PR-AUC cao nhất
        sorted_by_pr = sorted(benchmark_results, key=lambda x: x.get("pr_auc", 0.0), reverse=True)
        chosen = sorted_by_pr[0]["model"]
        rationale = f"Không mô hình nào thỏa toàn bộ guardrails. Chọn {chosen} do có PR-AUC cao nhất ({sorted_by_pr[0]['pr_auc']:.4f})."
        return chosen, rationale

    # Sắp xếp các mô hình đủ điều kiện theo PR-AUC
    qualified.sort(key=lambda x: x.get("pr_auc", 0.0), reverse=True)
    top_model = qualified[0]

    # Kiểm tra tie-breaking: Nếu top 2 mô hình chênh lệch PR-AUC <= 0.005, ưu tiên độ trễ và artifact format
    if len(qualified) > 1:
        second_model = qualified[1]
        diff_pr = abs(top_model["pr_auc"] - second_model["pr_auc"])
        if diff_pr <= 0.005:
            # So sánh latency
            if second_model["p95_latency_ms"] < top_model["p95_latency_ms"]:
                chosen = second_model["model"]
                rationale = (
                    f"Chọn {chosen} vì PR-AUC tương đương ({second_model['pr_auc']:.4f} vs {top_model['pr_auc']:.4f}, diff={diff_pr:.4f}) "
                    f"nhưng có p95 latency tốt hơn rõ rệt ({second_model['p95_latency_ms']}ms vs {top_model['p95_latency_ms']}ms) "
                    f"và định dạng Native JSON không phụ thuộc pickle."
                )
                return chosen, rationale

    chosen = top_model["model"]
    rationale = (
        f"Chọn {chosen} vì thỏa mãn toàn bộ guardrails (Recall={top_model['recall']*100:.1f}%, FPR={top_model['false_positive_rate']*100:.2f}%) "
        f"và đạt PR-AUC cao nhất ({top_model['pr_auc']:.4f}) với độ trễ p95={top_model['p95_latency_ms']}ms."
    )
    return chosen, rationale


def main() -> None:
    print("=" * 65)
    print(" 🚀 Bắt đầu Pipeline Huấn luyện & Benchmark PhishGuard ML...")
    print("=" * 65)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random_seed = config.get("random_seed", 42)
    feature_contract = config.get("feature_contract", FEATURE_CONTRACT_V2)
    guardrails = config.get("model_selection", {}).get("guardrails", {})

    print(f" Cấu hình: feature_contract={feature_contract}, random_seed={random_seed}")

    # 1. Load splits
    train_raw = load_split("train")
    val_raw = load_split("validation")
    cal_raw = None
    try:
        cal_raw = load_split("calibration")
    except Exception:
        print(" Không có calibration split riêng biệt; sử dụng validation split cho hiệu chuẩn.")

    print("\n 🔍 Trích xuất đặc trưng...")
    X_train = build_feature_dataframe(train_raw, contract=feature_contract)
    y_train = train_raw["label"].values

    X_val = build_feature_dataframe(val_raw, contract=feature_contract)
    y_val = val_raw["label"].values

    if cal_raw is not None:
        X_cal = build_feature_dataframe(cal_raw, contract=feature_contract)
        y_cal = cal_raw["label"].values
    else:
        X_cal, y_cal = X_val, y_val

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
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=random_seed, n_jobs=-1)
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
        max_depth=xgb_config.get("max_depth", 6),
        learning_rate=xgb_config.get("learning_rate", 0.08),
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

    # 5. Automated Model Selection
    print("\n 🏆 4. Tự động lựa chọn mô hình tối ưu theo PR-AUC và Guardrails...")
    selected_model_name, selection_rationale = select_best_candidate_model(results_table, guardrails)
    selected_model = models_dict[selected_model_name]
    print(f"   Selected Model: {selected_model_name}")
    print(f"   Rationale:      {selection_rationale}")

    # 6. Probability Calibration Benchmark
    print("\n 🎯 5. Kiểm thử Hiệu chuẩn Xác suất (Probability Calibration Benchmark)...")
    raw_val_scores = selected_model.predict_proba(X_val)[:, 1] if hasattr(selected_model, "predict_proba") else xgb_scores
    raw_ece = compute_ece(y_val, raw_val_scores)
    print(f"   • Raw {selected_model_name} ECE: {raw_ece:.4f}")

    # 7. Constrained & Cost-Aware Threshold Optimization
    print("\n ⚖️ 6. Quét Threshold toàn diện (Constrained & Cost-Aware Threshold Sweep)...")
    th_policy_cfg = config.get("threshold_policy", {})
    cost_cfg = th_policy_cfg.get("asymmetric_cost", {})
    cost_fn = cost_cfg.get("cost_fn", 10.0)
    cost_fp = cost_cfg.get("cost_fp", 1.0)
    target_fpr = th_policy_cfg.get("target_fpr", 0.005)

    sweep_results = threshold_sweep(
        y_val,
        raw_val_scores,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        max_fpr=target_fpr,
    )

    print(f"   • Max F1 Threshold:         {sweep_results['max_f1_threshold']} (F1={sweep_results['max_f1_value']})")
    print(f"   • Min Cost Threshold:       {sweep_results['min_cost_threshold']} (Expected Cost={sweep_results['min_cost_value']})")
    print(f"   • Constrained Threshold:    {sweep_results['constrained_threshold']} (Recall={sweep_results['constrained_recall']}, FPR<={target_fpr})")

    # Quyết định operating threshold
    operating_mode = th_policy_cfg.get("operating_mode", "constrained_recall")
    if operating_mode == "constrained_recall":
        final_operating_threshold = sweep_results["constrained_threshold"]
    elif operating_mode == "min_cost":
        final_operating_threshold = sweep_results["min_cost_threshold"]
    else:
        final_operating_threshold = sweep_results["max_f1_threshold"]

    print(f"   => Selected Operating Threshold ({operating_mode}): {final_operating_threshold}")

    # Display comparison table
    results_df = pd.DataFrame(results_table)
    print("\n" + "=" * 70)
    print(" 📋 BẢNG SO SÁNH BENCHMARK CÁC MÔ HÌNH TRÊN TẬP VALIDATION")
    print("=" * 70)
    cols_display = ["model", "pr_auc", "recall", "precision", "false_positive_rate", "expected_calibration_error", "p95_latency_ms"]
    print(results_df[cols_display].to_string(index=False))
    print("=" * 70)

    # Save summary and intermediate artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = ARTIFACTS_DIR / "validation_summary.json"
    val_summary = {
        "benchmark_table": results_table,
        "selected_model": selected_model_name,
        "selection_rationale": selection_rationale,
        "feature_contract": feature_contract,
        "operating_threshold": final_operating_threshold,
        "threshold_policy": {
            "operating_mode": operating_mode,
            "max_f1_threshold": sweep_results["max_f1_threshold"],
            "min_cost_threshold": sweep_results["min_cost_threshold"],
            "constrained_threshold": sweep_results["constrained_threshold"],
            "cost_assumptions": sweep_results["cost_assumptions"],
        },
        "validation_metrics_at_operating_threshold": classification_metrics(
            y_val, (raw_val_scores >= final_operating_threshold).astype(int), raw_val_scores
        ),
        "calibration": {
            "raw_ece": raw_ece,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    # Save model artifact
    if hasattr(selected_model, "save_model"):
        xgb_json_path = ARTIFACTS_DIR / "XGB.json"
        selected_model.save_model(xgb_json_path)
        print(f"\n [OK] Đã lưu mô hình {selected_model_name} Native JSON: {xgb_json_path}")

    print(f" [OK] Đã lưu tóm tắt validation vào: {summary_path}")


if __name__ == "__main__":
    main()
