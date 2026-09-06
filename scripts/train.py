"""Script huấn luyện, so sánh benchmark, refit champion và hiệu chuẩn mô hình PhishGuard ML."""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
import yaml
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from phishguard.calibration import (
    CalibrationArtifact,
    ProbabilityCalibrator,
    RiskPolicyConfig,
    sweep_operating_threshold,
)
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    extract_features,
)
from phishguard.training.baseline import create_baseline_models
from phishguard.training.data import compute_sha256
from phishguard.training.evaluation import (
    classification_metrics,
    measure_inference_latency,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
API_DIR = PROJECT_ROOT / "API"


def compute_feature_contract_hash(features: tuple[str, ...]) -> str:
    """Tính mã băm từ danh sách tên đặc trưng theo đúng thứ tự."""
    raw = ",".join(features).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
    """Trích xuất đặc trưng từ raw_url (hoặc url) theo hợp đồng chỉ định."""
    cols = FEATURE_COLUMNS_V2 if contract == FEATURE_CONTRACT_V2 else FEATURE_COLUMNS_V1
    url_col = "raw_url" if "raw_url" in df.columns else "url"
    start = time.perf_counter()
    extracted = [extract_features(u, contract=contract) for u in df[url_col]]
    feature_df = pd.DataFrame(extracted, columns=cols)
    elapsed = time.perf_counter() - start
    print(f"   Trích xuất {len(feature_df):,} bản ghi x {len(cols)} đặc trưng ({contract}) trong {elapsed:.2f}s")
    return feature_df


def select_best_candidate_model(
    benchmark_results: list[dict[str, Any]],
    guardrails: dict[str, Any],
) -> tuple[str, str, bool]:
    """
    Lựa chọn mô hình tối ưu dựa trên PR-AUC và các guardrails:
    - Recall >= min_recall (e.g. 0.80)
    - FPR <= max_fpr (e.g. 0.01)
    - p95_latency_ms <= max_latency (e.g. 5.0 ms)
    """
    min_recall = guardrails.get("min_recall", 0.80)
    max_fpr = guardrails.get("max_fpr", 0.01)
    max_latency = guardrails.get("max_latency_p95_ms", 5.0)

    qualified = []
    for row in benchmark_results:
        rec = row.get("recall", 0.0)
        fpr = row.get("false_positive_rate", 1.0)
        lat = row.get("p95_latency_ms", 999.0)
        if rec >= min_recall and fpr <= max_fpr and lat <= max_latency:
            qualified.append(row)

    if not qualified:
        sorted_by_pr = sorted(benchmark_results, key=lambda x: x.get("pr_auc", 0.0), reverse=True)
        chosen = sorted_by_pr[0]["model"]
        rationale = f"Không mô hình nào thỏa toàn bộ guardrails. Chọn {chosen} do PR-AUC cao nhất ({sorted_by_pr[0]['pr_auc']:.4f}), nhưng KHÔNG đủ điều kiện promotion."
        return chosen, rationale, False

    qualified.sort(key=lambda x: x.get("pr_auc", 0.0), reverse=True)
    top_model = qualified[0]

    # Tie-breaking logic
    if len(qualified) > 1:
        second_model = qualified[1]
        diff_pr = abs(top_model["pr_auc"] - second_model["pr_auc"])
        if diff_pr <= 0.005:
            if second_model["p95_latency_ms"] < top_model["p95_latency_ms"]:
                chosen = second_model["model"]
                rationale = (
                    f"Chọn {chosen} vì PR-AUC tương đương ({second_model['pr_auc']:.4f} vs {top_model['pr_auc']:.4f}) "
                    f"nhưng có latency p95 tốt hơn ({second_model['p95_latency_ms']}ms vs {top_model['p95_latency_ms']}ms) "
                    f"và định dạng Native JSON an toàn."
                )
                return chosen, rationale, True

    chosen = top_model["model"]
    rationale = (
        f"Chọn {chosen} vì thỏa mãn toàn bộ guardrails (Recall={top_model['recall']*100:.1f}%, FPR={top_model['false_positive_rate']*100:.2f}%) "
        f"và đạt PR-AUC cao nhất ({top_model['pr_auc']:.4f}) với độ trễ p95={top_model['p95_latency_ms']}ms."
    )
    return chosen, rationale, True


def main() -> None:
    print("=" * 70)
    print(" 🚀 Bắt đầu Canonical Lifecycle: Train -> Select -> Refit -> Calibrate -> Freeze")
    print("=" * 70)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_ver = config.get("model_version", "3.2.0")
    random_seed = config.get("random_seed", 42)
    feature_contract = config.get("feature_contract", FEATURE_CONTRACT_V2)
    guardrails = config.get("model_selection", {}).get("guardrails", {})
    promotion_policy = config.get("promotion_policy", {})
    cal_method = config.get("calibration", {}).get("method", "isotonic")

    print(f" Cấu hình: version={model_ver}, feature_contract={feature_contract}, seed={random_seed}")

    # 1. Tải các tập split
    train_raw = load_split("train")
    val_raw = load_split("validation")
    cal_raw = load_split("calibration")

    print("\n 🔍 1. Trích xuất đặc trưng thống nhất từ raw_url...")
    X_train = build_feature_dataframe(train_raw, contract=feature_contract)
    y_train = train_raw["label"].values

    X_val = build_feature_dataframe(val_raw, contract=feature_contract)
    y_val = val_raw["label"].values

    X_cal = build_feature_dataframe(cal_raw, contract=feature_contract)
    y_cal = cal_raw["label"].values

    results_table = []
    models_dict = {}

    # 2. Benchmark Baseline Models trên Validation Set
    print("\n 📊 2. Huấn luyện và Đánh giá Baseline Models trên Validation Set...")
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
        p50, p95 = measure_inference_latency(model, X_val, num_runs=30)
        metrics["p50_latency_ms"] = round(p50, 4)
        metrics["p95_latency_ms"] = round(p95, 4)

        results_table.append({"model": name, **metrics})
        models_dict[name] = model

    # 3. Candidate 1: Random Forest
    print("\n 🌲 3. Huấn luyện Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=random_seed, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_scores = rf_model.predict_proba(X_val)[:, 1]
    rf_preds = (rf_scores >= 0.5).astype(int)
    rf_metrics = classification_metrics(y_val, rf_preds, rf_scores)
    p50, p95 = measure_inference_latency(rf_model, X_val, num_runs=30)
    rf_metrics["p50_latency_ms"] = round(p50, 4)
    rf_metrics["p95_latency_ms"] = round(p95, 4)
    results_table.append({"model": "Random Forest", **rf_metrics})
    models_dict["Random Forest"] = rf_model

    # 4. Candidate 2: XGBoost (Deployable Champion Family)
    print("\n ⚡ 4. Huấn luyện XGBoost Classifier...")
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
    p50, p95 = measure_inference_latency(xgb_model, X_val, num_runs=30)
    xgb_metrics["p50_latency_ms"] = round(p50, 4)
    xgb_metrics["p95_latency_ms"] = round(p95, 4)
    results_table.append({"model": "XGBoost", **xgb_metrics})
    models_dict["XGBoost"] = xgb_model

    # 5. Model Architecture Selection trên Validation Set
    print("\n 🏆 5. Lựa chọn kiến trúc mô hình tối ưu theo PR-AUC và Guardrails...")
    selected_name, rationale, satisfies_guardrails = select_best_candidate_model(results_table, guardrails)
    print(f"   Selected Model Architecture: {selected_name}")
    print(f"   Rationale:                   {rationale}")

    # Kiểm tra deployability (Deployable Champion Family)
    is_deployable = selected_name in {"XGBoost"}
    if not is_deployable:
        print(f"   [CẢNH BÁO] Mô hình {selected_name} không thuộc deployable model family (XGBoost JSON). Sẽ từ chối auto-promotion.")

    # 6. REFIT CHAMPION TRÊN TRAIN + VALIDATION (P0.5)
    print("\n 🔄 6. Tái huấn luyện (Refit) Champion Model trên tập gộp [Train + Validation]...")
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])

    champion_model = XGBClassifier(
        n_estimators=xgb_config.get("n_estimators", 300),
        max_depth=xgb_config.get("max_depth", 6),
        learning_rate=xgb_config.get("learning_rate", 0.08),
        eval_metric=xgb_config.get("eval_metric", "logloss"),
        random_state=random_seed,
        n_jobs=-1,
    )
    champion_model.fit(X_train_val, y_train_val)
    print(f"   [OK] Champion model refitted trên {len(X_train_val):,} bản ghi Train+Val.")

    # 7. PROBABILITY CALIBRATION TRÊN TẬP CALIBRATION (P0.5)
    print(f"\n 🎯 7. Hiệu chuẩn xác suất ({cal_method}) độc lập trên tập Calibration ({len(X_cal):,} rows)...")
    raw_cal_scores = champion_model.predict_proba(X_cal)[:, 1]

    calibrator = ProbabilityCalibrator(method=cal_method)
    calibrator.fit(raw_cal_scores, y_cal)
    cal_eval = calibrator.evaluate_fit(raw_cal_scores, y_cal)

    print(f"   • ECE trước hiệu chuẩn: {cal_eval['ece_before']:.4f} ──► Sau hiệu chuẩn: {cal_eval['ece_after']:.4f}")
    print(f"   • Brier score trước:    {cal_eval['brier_before']:.4f} ──► Sau hiệu chuẩn: {cal_eval['brier_after']:.4f}")

    calibrated_cal_scores = calibrator.calibrate(raw_cal_scores)

    # 8. CONSTRAINED & COST-AWARE THRESHOLD SWEEP TRÊN TẬP CALIBRATION (P0.5)
    print("\n ⚖️ 8. Quét ngưỡng vận hành trên tập Calibration đã hiệu chuẩn...")
    th_policy_cfg = config.get("threshold_policy", {})
    cost_cfg = th_policy_cfg.get("asymmetric_cost", {})
    cost_fn = cost_cfg.get("cost_fn", 10.0)
    cost_fp = cost_cfg.get("cost_fp", 1.0)
    target_fpr = th_policy_cfg.get("target_fpr", 0.005)

    sweep_results = sweep_operating_threshold(
        y_cal,
        calibrated_cal_scores,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        max_fpr=target_fpr,
    )

    operating_mode = th_policy_cfg.get("operating_mode", "constrained_recall")
    if operating_mode == "constrained_recall":
        final_operating_threshold = sweep_results["constrained_threshold"]
    elif operating_mode == "min_cost":
        final_operating_threshold = sweep_results["min_cost_threshold"]
    else:
        final_operating_threshold = sweep_results["max_f1_threshold"]

    print(f"   • Max F1 Threshold:         {sweep_results['max_f1_threshold']} (F1={sweep_results['max_f1_value']})")
    print(f"   • Min Cost Threshold:       {sweep_results['min_cost_threshold']} (Cost={sweep_results['min_cost_value']})")
    print(f"   • Constrained Threshold:    {sweep_results['constrained_threshold']} (Recall={sweep_results['constrained_recall']}, FPR<={target_fpr})")
    print(f"   => Selected Operating Threshold ({operating_mode}): {final_operating_threshold}")

    # 9. KIỂM TRA QUALITY GATE VÀ PROMOTION ELIGIBILITY (P0.7)
    print("\n 🛡️ 9. Kiểm toán Quality Gate & Promotion Policy...")
    val_pr_auc = xgb_metrics["pr_auc"]
    val_recall = xgb_metrics["recall"]
    val_fpr = xgb_metrics["false_positive_rate"]
    val_lat = xgb_metrics["p95_latency_ms"]

    val_pol = promotion_policy.get("validation", {})
    cal_pol = promotion_policy.get("calibration", {})
    serv_pol = promotion_policy.get("serving", {})

    passed_val = bool(
        val_pr_auc >= val_pol.get("min_pr_auc", 0.85)
        and val_recall >= val_pol.get("min_recall", 0.80)
        and val_fpr <= val_pol.get("max_fpr", 0.01)
    )
    passed_cal = bool(cal_eval["ece_after"] <= cal_pol.get("max_ece", 0.05))
    passed_lat = bool(val_lat <= serv_pol.get("max_model_p95_ms", 5.0))

    promotion_eligible = bool(passed_val and passed_cal and passed_lat and is_deployable)
    print(f"   • Validation Guardrails:  {'PASSED ✅' if passed_val else 'FAILED ❌'}")
    print(f"   • Calibration ECE Target: {'PASSED ✅' if passed_cal else 'FAILED ❌'}")
    print(f"   • Serving Latency Target: {'PASSED ✅' if passed_lat else 'FAILED ❌'}")
    print(f"   • Deployable Artifact:    {'PASSED ✅' if is_deployable else 'FAILED ❌'}")
    print(f"   => PROMOTION ELIGIBILITY: {'ELIGIBLE ✅' if promotion_eligible else 'BLOCKED ❌'}")

    # 10. ĐÓNG GÓI VERSIONED FROZEN ARTIFACT (P0.8)
    print(f"\n 📦 10. Đóng gói Frozen Model Artifacts: phishguard-{model_ver}...")
    version_tag = f"phishguard-{model_ver}"
    model_version_dir = MODELS_DIR / version_tag
    model_version_dir.mkdir(parents=True, exist_ok=True)

    # A. Model Native JSON
    model_json_path = model_version_dir / "model.json"
    champion_model.save_model(model_json_path)
    model_sha256 = compute_sha256(model_json_path)

    # B. Feature Contract JSON
    cols = FEATURE_COLUMNS_V2 if feature_contract == FEATURE_CONTRACT_V2 else FEATURE_COLUMNS_V1
    contract_hash = compute_feature_contract_hash(cols)
    feature_contract_data = {
        "contract": feature_contract,
        "feature_count": len(cols),
        "features": list(cols),
        "contract_hash": contract_hash,
    }
    with open(model_version_dir / "feature_contract.json", "w", encoding="utf-8") as f:
        json.dump(feature_contract_data, f, indent=2)

    # C. Calibration JSON
    calibration_artifact = CalibrationArtifact(
        method=cal_method,
        threshold=final_operating_threshold,
        target_fpr=target_fpr,
        ece_before=cal_eval["ece_before"],
        ece_after=cal_eval["ece_after"],
        brier_before=cal_eval["brier_before"],
        brier_after=cal_eval["brier_after"],
        calibrator_params=calibrator.params,
    )
    with open(model_version_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration_artifact.to_dict(), f, indent=2)

    # D. Threshold Policy JSON
    risk_policy = RiskPolicyConfig(
        high_threshold=th_policy_cfg.get("risk_levels", {}).get("high", 0.75),
        medium_threshold=th_policy_cfg.get("risk_levels", {}).get("medium", 0.45),
    )
    threshold_policy_data = {
        "operating_mode": operating_mode,
        "operating_threshold": final_operating_threshold,
        "target_fpr": target_fpr,
        "asymmetric_cost": cost_cfg,
        "risk_levels": risk_policy.to_dict(),
        "sweep_summary": {
            "max_f1_threshold": sweep_results["max_f1_threshold"],
            "min_cost_threshold": sweep_results["min_cost_threshold"],
            "constrained_threshold": sweep_results["constrained_threshold"],
        },
    }
    with open(model_version_dir / "threshold_policy.json", "w", encoding="utf-8") as f:
        json.dump(threshold_policy_data, f, indent=2)

    # E. Metadata JSON
    metadata = {
        "artifact_schema_version": "3.2.0",
        "model_version": model_ver,
        "model_tag": version_tag,
        "model_type": "XGBClassifier",
        "feature_contract": feature_contract,
        "feature_count": len(cols),
        "feature_contract_hash": contract_hash,
        "threshold": final_operating_threshold,
        "risk_thresholds": {
            "high": risk_policy.high_threshold,
            "medium": risk_policy.medium_threshold,
        },
        "operating_policy": {
            "mode": operating_mode,
            "target_fpr": target_fpr,
            "cost_assumptions": cost_cfg,
        },
        "calibration": {
            "method": cal_method,
            "ece_before": cal_eval["ece_before"],
            "ece_after": cal_eval["ece_after"],
            "brier_before": cal_eval["brier_before"],
            "brier_after": cal_eval["brier_after"],
        },
        "validation_metrics": xgb_metrics,
        "training_date": str(datetime.date.today()),
        "model_sha256": model_sha256,
        "xgboost_version": xgboost.__version__,
        "promotion_eligible": promotion_eligible,
        "promotion_rules": [
            f"min_pr_auc >= {val_pol.get('min_pr_auc', 0.85)}",
            f"min_recall >= {val_pol.get('min_recall', 0.80)}",
            f"max_fpr <= {val_pol.get('max_fpr', 0.01)}",
            f"max_ece <= {cal_pol.get('max_ece', 0.05)}",
            f"max_p95_latency <= {serv_pol.get('max_model_p95_ms', 5.0)}ms",
        ],
    }

    with open(model_version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Copy split_manifest and data_manifest into version directory
    if (ARTIFACTS_DIR / "split_manifest.json").exists():
        shutil.copy(ARTIFACTS_DIR / "split_manifest.json", model_version_dir / "split_manifest.json")
    if (ARTIFACTS_DIR / "dataset_manifest.json").exists():
        shutil.copy(ARTIFACTS_DIR / "dataset_manifest.json", model_version_dir / "data_manifest.json")

    # F. Registry Pointer production.json & Backward Compatibility Exports
    if promotion_eligible:
        registry_pointer = {
            "active_version": version_tag,
            "model_version": model_ver,
            "model_dir": f"artifacts/models/{version_tag}",
            "promoted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(MODELS_DIR / "production.json", "w", encoding="utf-8") as f:
            json.dump(registry_pointer, f, indent=2)
        print(f"   [OK] Đã cập nhật Model Registry Active Pointer: {MODELS_DIR / 'production.json'}")

        # Tương thích ngược: xuất bản sao vào artifacts/ và API/
        API_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_json_path, ARTIFACTS_DIR / "XGB.json")
        shutil.copy(model_json_path, API_DIR / "XGB.json")
        with open(API_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print("   [OK] Đã đồng bộ bản sao tương thích ngược sang API/XGB.json và API/model_metadata.json")
    else:
        print("   [CẢNH BÁO] Mô hình không đạt Promotion Policy. KHÔNG cập nhật production.json!")

    # Summary table output
    results_df = pd.DataFrame(results_table)
    print("\n" + "=" * 70)
    print(" 📋 BẢNG SO SÁNH BENCHMARK CÁC MÔ HÌNH TRÊN TẬP VALIDATION")
    print("=" * 70)
    cols_display = ["model", "pr_auc", "recall", "precision", "false_positive_rate", "expected_calibration_error", "p95_latency_ms"]
    print(results_df[cols_display].to_string(index=False))
    print("=" * 70)

    val_summary = {
        "benchmark_table": results_table,
        "selected_model": selected_name,
        "selection_rationale": rationale,
        "feature_contract": feature_contract,
        "operating_threshold": final_operating_threshold,
        "promotion_eligible": promotion_eligible,
        "calibration": cal_eval,
    }
    with open(ARTIFACTS_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2, ensure_ascii=False)

    print(f"\n [OK] Hoàn tất quá trình huấn luyện và đóng gói: {model_version_dir}")


if __name__ == "__main__":
    main()
