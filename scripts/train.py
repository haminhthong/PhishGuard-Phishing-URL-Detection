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
from xgboost import XGBClassifier

from phishguard.calibration import (
    ActionPolicy,
    CalibrationArtifact,
    ProbabilityCalibrator,
    select_action_policy,
)
from phishguard.features import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    FEATURE_COLUMNS_V4,
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_V3,
    FEATURE_CONTRACT_V4,
    FeatureExtractor,
)
from phishguard.features.resources import RESOURCE_BUNDLE, RESOURCES_DIR
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
CANDIDATES_DIR = PROJECT_ROOT / "releases" / "candidates"


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


def build_feature_dataframe(
    df: pd.DataFrame, contract: str, extractor: FeatureExtractor | None = None
) -> pd.DataFrame:
    """Trích xuất đặc trưng từ raw_url (hoặc url) theo hợp đồng chỉ định."""
    cols = {
        FEATURE_CONTRACT_V1: FEATURE_COLUMNS_V1,
        FEATURE_CONTRACT_V2: FEATURE_COLUMNS_V2,
        FEATURE_CONTRACT_V3: FEATURE_COLUMNS_V3,
        FEATURE_CONTRACT_V4: FEATURE_COLUMNS_V4,
    }[contract]
    url_col = "raw_url" if "raw_url" in df.columns else "url"
    start = time.perf_counter()
    active_extractor = extractor or FeatureExtractor(contract=contract)
    extracted = [active_extractor.extract(u) for u in df[url_col]]
    feature_df = pd.DataFrame(extracted, columns=cols)
    elapsed = time.perf_counter() - start
    print(
        f"   Trích xuất {len(feature_df):,} bản ghi x {len(cols)} đặc trưng ({contract}) trong {elapsed:.2f}s"
    )
    return feature_df


def select_best_candidate_model(
    benchmark_results: list[dict[str, Any]],
    guardrails: dict[str, Any],
) -> tuple[str, str, bool]:
    """Chọn kiến trúc bằng ranking score, latency chỉ là guardrail serving.

    Recall/FPR ở threshold 0.5 không được dùng để chọn kiến trúc vì browser
    thực tế dùng calibrated ActionPolicy ở lifecycle sau.
    """
    max_latency = guardrails.get("max_latency_p95_ms", 5.0)
    qualified = [
        row for row in benchmark_results if row.get("p95_latency_ms", 999.0) <= max_latency
    ]

    if not qualified:
        sorted_by_pr = sorted(benchmark_results, key=lambda x: x.get("pr_auc", 0.0), reverse=True)
        chosen = sorted_by_pr[0]["model"]
        rationale = f"Không mô hình nào đạt latency guardrail. Chọn {chosen} do PR-AUC cao nhất, nhưng không đủ điều kiện promotion."
        return chosen, rationale, False

    qualified.sort(key=lambda x: x.get("pr_auc", 0.0), reverse=True)
    top_model = qualified[0]

    # Quy tắc phá hòa khi PR-AUC gần tương đương
    if len(qualified) > 1:
        second_model = qualified[1]
        diff_pr = abs(top_model["pr_auc"] - second_model["pr_auc"])
        if diff_pr <= 0.005 and second_model["p95_latency_ms"] < top_model["p95_latency_ms"]:
            chosen = second_model["model"]
            rationale = (
                f"Chọn {chosen} vì PR-AUC tương đương ({second_model['pr_auc']:.4f} vs {top_model['pr_auc']:.4f}) "
                f"nhưng có latency p95 tốt hơn ({second_model['p95_latency_ms']}ms vs {top_model['p95_latency_ms']}ms) "
                f"và định dạng Native JSON an toàn."
            )
            return chosen, rationale, True

    chosen = top_model["model"]
    rationale = (
        f"Chọn {chosen} vì đạt PR-AUC cao nhất ({top_model['pr_auc']:.4f}) "
        f"và latency p95={top_model['p95_latency_ms']}ms."
    )
    return chosen, rationale, True


def main() -> None:
    print("=" * 70)
    print(" 🚀 Bắt đầu Canonical Lifecycle: Train -> Select -> Refit -> Calibrate -> Freeze")
    print("=" * 70)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_ver = config.get("model_version", "4.0.0")
    random_seed = config.get("random_seed", 42)
    feature_contract = config.get("feature_contract", FEATURE_CONTRACT_V2)
    guardrails = config.get("model_selection", {}).get("guardrails", {})
    promotion_policy = config.get("promotion_policy", {})
    cal_method = config.get("calibration", {}).get("method", "isotonic")
    feature_extractor = FeatureExtractor(contract=feature_contract)

    print(
        f" Cấu hình: version={model_ver}, feature_contract={feature_contract}, seed={random_seed}"
    )

    # 1. Tải các tập split
    train_raw = load_split("train")
    val_raw = load_split("validation")
    cal_raw = load_split("calibration")
    policy_raw = load_split("policy_validation")

    print("\n 🔍 1. Trích xuất đặc trưng thống nhất từ raw_url...")
    X_train = build_feature_dataframe(
        train_raw, contract=feature_contract, extractor=feature_extractor
    )
    y_train = train_raw["label"].values

    X_val = build_feature_dataframe(val_raw, contract=feature_contract, extractor=feature_extractor)
    y_val = val_raw["label"].values

    X_cal = build_feature_dataframe(cal_raw, contract=feature_contract, extractor=feature_extractor)
    y_cal = cal_raw["label"].values

    X_policy = build_feature_dataframe(
        policy_raw, contract=feature_contract, extractor=feature_extractor
    )
    y_policy = policy_raw["label"].values

    results_table = []

    # 2. So sánh các mô hình baseline trên Validation
    print("\n 📊 2. Huấn luyện và Đánh giá Baseline Models trên Validation Set...")
    baselines = create_baseline_models(random_state=random_seed)

    for name in ("Rule-based", "Logistic Regression"):
        model = baselines[name]
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

    # 3. Ứng viên production duy nhất: XGBoost Native JSON
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
    p50, p95 = measure_inference_latency(xgb_model, X_val, num_runs=30)
    xgb_metrics["p50_latency_ms"] = round(p50, 4)
    xgb_metrics["p95_latency_ms"] = round(p95, 4)
    results_table.append({"model": "XGBoost", **xgb_metrics})

    # 4. Chọn kiến trúc trên Validation bằng PR-AUC và độ trễ, chưa chọn policy
    print("\n 🏆 5. Lựa chọn kiến trúc mô hình tối ưu theo PR-AUC và Guardrails...")
    selected_name, rationale, _satisfies_latency_guardrail = select_best_candidate_model(
        results_table, guardrails
    )
    print(f"   Selected Model Architecture: {selected_name}")
    print(f"   Rationale:                   {rationale}")

    # Kiểm tra mô hình có thuộc nhóm được phép đóng gói hay không
    is_deployable = selected_name in {"XGBoost"}
    if not is_deployable:
        print(
            f"   [CẢNH BÁO] Mô hình {selected_name} không thuộc deployable model family (XGBoost JSON). Sẽ từ chối auto-promotion."
        )

    # 5. Huấn luyện lại champion trên Train + Validation
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

    # 6. Chỉ fit calibrator trên Calibration
    print(
        f"\n 🎯 7. Hiệu chuẩn xác suất ({cal_method}) độc lập trên tập Calibration ({len(X_cal):,} rows)..."
    )
    raw_cal_scores = champion_model.predict_proba(X_cal)[:, 1]

    calibrator = ProbabilityCalibrator(method=cal_method)
    calibrator.fit(raw_cal_scores, y_cal)
    raw_policy_scores = champion_model.predict_proba(X_policy)[:, 1]
    calibrated_policy_scores = calibrator.calibrate(raw_policy_scores)
    raw_cal_scores = np.asarray(raw_cal_scores)
    cal_eval = calibrator.evaluate_fit(raw_cal_scores, y_cal)
    print(f"   • ECE Calibration trước hiệu chuẩn: {cal_eval['ece_before']:.4f}")
    print(f"   • ECE Calibration sau hiệu chuẩn:   {cal_eval['ece_after']:.4f}")

    # 7. Chọn ActionPolicy trên Policy Validation, tuyệt đối không dùng Calibration
    print("\n ⚖️ 7. Chọn caution/block threshold trên Policy Validation...")
    th_policy_cfg = config.get("threshold_policy", {})
    policy_selection = select_action_policy(
        y_policy,
        calibrated_policy_scores,
        caution_max_fpr=float(th_policy_cfg.get("caution_max_fpr", 0.02)),
        block_max_fpr=float(th_policy_cfg.get("block_max_fpr", 0.005)),
        caution_min_recall=float(th_policy_cfg.get("caution_min_recall", 0.90)),
        block_min_recall=float(th_policy_cfg.get("block_min_recall", 0.80)),
        policy_version=str(th_policy_cfg.get("policy_version", "browser-risk-v1")),
    )
    action_policy = ActionPolicy.from_dict(policy_selection["policy"])
    policy_split_path = SPLITS_DIR / "policy_validation.parquet"
    if not policy_split_path.exists():
        policy_split_path = SPLITS_DIR / "policy_validation.csv.gz"
    policy_dataset_hash = compute_sha256(policy_split_path)
    policy_eval = {
        "caution": classification_metrics(
            y_policy,
            (calibrated_policy_scores >= action_policy.caution_threshold).astype(int),
            calibrated_policy_scores,
        ),
        "block": classification_metrics(
            y_policy,
            (calibrated_policy_scores >= action_policy.block_threshold).astype(int),
            calibrated_policy_scores,
        ),
    }
    print(f"   • Caution threshold: {action_policy.caution_threshold}")
    print(f"   • Block threshold:   {action_policy.block_threshold}")

    # 9. Kiểm tra gate chất lượng để quyết định candidate có đủ điều kiện
    print("\n 🛡️ 9. Kiểm toán Quality Gate & Promotion Policy...")
    val_pr_auc = xgb_metrics["pr_auc"]
    val_lat = xgb_metrics["p95_latency_ms"]

    release_gates = config.get("release_gates", {})
    val_pol = release_gates.get("ranking", promotion_policy.get("validation", {}))
    cal_pol = release_gates.get("calibration", promotion_policy.get("calibration", {}))
    serv_pol = release_gates.get("latency", promotion_policy.get("serving", {}))
    block_gate = release_gates.get("block", {})
    caution_gate = release_gates.get("caution", {})

    passed_val = bool(val_pr_auc >= val_pol.get("min_pr_auc", 0.85))
    passed_cal = bool(cal_eval["ece_after"] <= cal_pol.get("max_ece", 0.05))
    passed_lat = bool(val_lat <= serv_pol.get("max_model_p95_ms", 5.0))
    passed_block = bool(
        policy_eval["block"]["false_positive_rate"] <= block_gate.get("max_fpr", 0.005)
        and policy_eval["block"]["recall"] >= block_gate.get("min_recall", 0.80)
    )
    passed_caution = bool(
        policy_eval["caution"]["false_positive_rate"] <= caution_gate.get("max_fpr", 0.02)
        and policy_eval["caution"]["recall"] >= caution_gate.get("min_recall", 0.90)
    )

    candidate_eligible = bool(
        passed_val
        and passed_cal
        and passed_lat
        and passed_block
        and passed_caution
        and is_deployable
    )
    print(f"   • Validation Guardrails:  {'PASSED ✅' if passed_val else 'FAILED ❌'}")
    print(f"   • Calibration ECE Target: {'PASSED ✅' if passed_cal else 'FAILED ❌'}")
    print(f"   • Serving Latency Target: {'PASSED ✅' if passed_lat else 'FAILED ❌'}")
    print(f"   • Block Policy Gate:       {'PASSED ✅' if passed_block else 'FAILED ❌'}")
    print(f"   • Caution Policy Gate:     {'PASSED ✅' if passed_caution else 'FAILED ❌'}")
    print(f"   • Deployable Artifact:    {'PASSED ✅' if is_deployable else 'FAILED ❌'}")
    print(f"   => CANDIDATE ELIGIBILITY: {'ELIGIBLE ✅' if candidate_eligible else 'BLOCKED ❌'}")

    # 10. Đóng gói artifact bất biến theo version
    print(f"\n 📦 10. Đóng gói Frozen Model Artifacts: phishguard-{model_ver}...")
    version_tag = f"phishguard-{model_ver}"
    model_version_dir = CANDIDATES_DIR / version_tag
    model_version_dir.mkdir(parents=True, exist_ok=True)

    # A. Mô hình JSON native
    model_json_path = model_version_dir / "model.json"
    champion_model.save_model(model_json_path)
    model_sha256 = compute_sha256(model_json_path)

    # B. JSON hợp đồng đặc trưng
    cols = {
        FEATURE_CONTRACT_V1: FEATURE_COLUMNS_V1,
        FEATURE_CONTRACT_V2: FEATURE_COLUMNS_V2,
        FEATURE_CONTRACT_V3: FEATURE_COLUMNS_V3,
        FEATURE_CONTRACT_V4: FEATURE_COLUMNS_V4,
    }[feature_contract]
    contract_hash = compute_feature_contract_hash(cols)
    feature_contract_data = {
        "contract": feature_contract,
        "feature_count": len(cols),
        "features": list(cols),
        "contract_hash": contract_hash,
    }
    with open(model_version_dir / "feature_contract.json", "w", encoding="utf-8") as f:
        json.dump(feature_contract_data, f, indent=2)

    # C. JSON hiệu chuẩn
    calibration_artifact = CalibrationArtifact(
        method=cal_method,
        ece_before=cal_eval["ece_before"],
        ece_after=cal_eval["ece_after"],
        brier_before=cal_eval["brier_before"],
        brier_after=cal_eval["brier_after"],
        calibrator_params=calibrator.params,
    )
    calibration_data = calibration_artifact.to_dict()
    calibration_data["model_version"] = model_ver
    calibration_path = model_version_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_data, indent=2), encoding="utf-8")
    calibration_sha256 = compute_sha256(calibration_path)

    # D. Action Policy JSON: artifact duy nhất cho quyết định browser.
    policy_path = model_version_dir / "action_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                **action_policy.to_dict(),
                "selection_dataset": "policy-validation-v1",
                "selection_dataset_hash": policy_dataset_hash,
                "constraints": policy_selection["constraints"],
                "selection_metrics": policy_selection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    action_policy_sha256 = compute_sha256(policy_path)

    # Đóng gói bản sao resource cùng release để artifact tự chứa và tái lập.
    release_resources_dir = model_version_dir / "resources"
    release_resources_dir.mkdir(exist_ok=True)
    for resource_name in ("brand_terms.json", "shortener_domains.json", "suspicious_tlds.json"):
        shutil.copy(RESOURCES_DIR / resource_name, release_resources_dir / resource_name)

    # E. JSON metadata
    metadata = {
        "artifact_schema_version": "4.0.0",
        "model_version": model_ver,
        "model_tag": version_tag,
        "model_type": "XGBClassifier",
        "feature_contract": feature_contract,
        "feature_count": len(cols),
        "feature_contract_hash": contract_hash,
        "policy_version": action_policy.policy_version,
        "operating_policy": action_policy.to_dict(),
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
        "calibration_sha256": calibration_sha256,
        "action_policy_sha256": action_policy_sha256,
        "resource_hashes": RESOURCE_BUNDLE.hashes,
        "resource_versions": RESOURCE_BUNDLE.versions,
        "tld_library_version": RESOURCE_BUNDLE.tld_library_version,
        "xgboost_version": xgboost.__version__,
        "release_id": version_tag,
        "candidate_eligible": candidate_eligible,
        "release_gates": {
            "ranking": {"passed": passed_val, **val_pol},
            "calibration": {"passed": passed_cal, **cal_pol},
            "latency": {"passed": passed_lat, **serv_pol},
            "block": {"passed": passed_block, **block_gate},
            "caution": {"passed": passed_caution, **caution_gate},
        },
        "candidate_rules": [
            f"min_pr_auc >= {val_pol.get('min_pr_auc', 0.85)}",
            "caution/block thresholds selected only on Policy Validation",
            f"max_ece <= {cal_pol.get('max_ece', 0.05)}",
            f"max_p95_latency <= {serv_pol.get('max_model_p95_ms', 5.0)}ms",
        ],
    }

    with open(model_version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Sao chép manifest split và dữ liệu vào thư mục version
    if (ARTIFACTS_DIR / "split_manifest.json").exists():
        shutil.copy(
            ARTIFACTS_DIR / "split_manifest.json", model_version_dir / "split_manifest.json"
        )
    if (ARTIFACTS_DIR / "dataset_manifest.json").exists():
        shutil.copy(
            ARTIFACTS_DIR / "dataset_manifest.json", model_version_dir / "data_manifest.json"
        )

    candidate_manifest = {
        "release_id": version_tag,
        "candidate_dir": str(model_version_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "candidate_eligible": candidate_eligible,
        "locked_test_required": True,
        "model_sha256": model_sha256,
        "calibration_sha256": calibration_sha256,
        "action_policy_sha256": action_policy_sha256,
        "feature_contract": feature_contract,
        "feature_contract_hash": contract_hash,
        "resource_hashes": RESOURCE_BUNDLE.hashes,
    }
    (model_version_dir / "candidate_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"   [OK] Candidate immutable đã tạo tại: {model_version_dir}")
    print("   [OK] Chưa cập nhật current_release.json; cần chạy Locked Test và promote_release.py.")

    # In bảng tóm tắt kết quả benchmark
    results_df = pd.DataFrame(results_table)
    print("\n" + "=" * 70)
    print(" 📋 BẢNG SO SÁNH BENCHMARK CÁC MÔ HÌNH TRÊN TẬP VALIDATION")
    print("=" * 70)
    cols_display = [
        "model",
        "pr_auc",
        "recall",
        "precision",
        "false_positive_rate",
        "expected_calibration_error",
        "p95_latency_ms",
    ]
    print(results_df[cols_display].to_string(index=False))
    print("=" * 70)

    val_summary = {
        "benchmark_table": results_table,
        "selected_model": selected_name,
        "selection_rationale": rationale,
        "feature_contract": feature_contract,
        "caution_threshold": action_policy.caution_threshold,
        "block_threshold": action_policy.block_threshold,
        "candidate_eligible": candidate_eligible,
        "calibration": cal_eval,
        "policy_validation": policy_eval,
        "action_policy": policy_selection,
    }
    with open(ARTIFACTS_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2, ensure_ascii=False)

    print(f"\n [OK] Hoàn tất quá trình huấn luyện và đóng gói: {model_version_dir}")


if __name__ == "__main__":
    main()
