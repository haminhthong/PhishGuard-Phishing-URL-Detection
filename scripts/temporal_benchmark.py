"""
Script thực nghiệm Protocol B — Phishing Temporal-Shift Experiment.
Đánh giá độ suy giảm hiệu năng (Concept Drift) khi huấn luyện trên các chiến dịch phishing quá khứ
và kiểm thử trên các chiến dịch phishing trong tương lai, kết hợp với holdout legitimate domain-disjoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from phishguard.features import FEATURE_COLUMNS_V2, extract_features_v2
from phishguard.training.data import audit_and_clean_data, temporal_split_protocol_b
from phishguard.training.evaluation import classification_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LEGIT_CSV = DATA_DIR / "legit_url.csv"
PHISHING_CSV = DATA_DIR / "verified_online.csv"
DRIFT_REPORT_JSON = ARTIFACTS_DIR / "temporal_drift_report.json"


def main() -> None:
    print("=" * 65)
    print(" ⏳ Bắt đầu Protocol B — Phishing Temporal-Shift Experiment (Concept Drift)...")
    print("=" * 65)

    if not LEGIT_CSV.exists() or not PHISHING_CSV.exists():
        raise FileNotFoundError("Không tìm thấy file dữ liệu nguồn legit_url.csv hoặc verified_online.csv")

    phishing_df = pd.read_csv(PHISHING_CSV)
    if "submission_time" not in phishing_df.columns:
        raise ValueError("verified_online.csv không chứa cột submission_time để chia tập thời gian")

    legit_df = pd.read_csv(LEGIT_CSV)

    cleaned_df, _ = audit_and_clean_data(
        legit_df=legit_df,
        phishing_df=phishing_df,
        legit_path=LEGIT_CSV,
        phishing_path=PHISHING_CSV,
    )

    # 1. Chia phishing theo thời gian (Past campaigns <= T0 vs Future campaigns > T0)
    phish_cleaned = cleaned_df[cleaned_df["label"] == 1].copy()
    print(f" Tổng số phishing URLs có thông tin thời gian: {len(phish_cleaned):,}")

    temporal_splits = temporal_split_protocol_b(phish_cleaned, test_ratio=0.20)
    train_phish = temporal_splits.train
    test_phish = temporal_splits.test

    print(f" • Past campaigns (Train):   {len(train_phish):,} phishing URLs")
    print(f" • Future campaigns (Test):  {len(test_phish):,} phishing URLs")

    # 2. Chia legitimate URLs theo GroupShuffleSplit (bảo đảm zero domain overlap giữa train và test)
    legit_cleaned = cleaned_df[cleaned_df["label"] == 0].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_l_idx, test_l_idx = next(gss.split(legit_cleaned, groups=legit_cleaned["domain"]))
    train_legit = legit_cleaned.iloc[train_l_idx].reset_index(drop=True)
    test_legit = legit_cleaned.iloc[test_l_idx].reset_index(drop=True)

    # Đảm bảo tuyệt đối không có domain overlap giữa train và test của legitimate
    assert set(train_legit["domain"]).isdisjoint(set(test_legit["domain"])), "Leakage phát hiện trong temporal legit split!"

    train_combined = pd.concat([train_legit, train_phish]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_combined = pd.concat([test_legit, test_phish]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Trích xuất đặc trưng mẫu (lấy 15.000 mẫu train, 5.000 mẫu test để benchmark nhanh)
    print(" Trích xuất đặc trưng lexical-v2 cho thực nghiệm Temporal...")
    train_sample = train_combined.sample(n=min(15000, len(train_combined)), random_state=42)
    test_sample = test_combined.sample(n=min(5000, len(test_combined)), random_state=42)

    url_col = "raw_url" if "raw_url" in train_sample.columns else "url"
    X_train = pd.DataFrame([extract_features_v2(u) for u in train_sample[url_col]], columns=FEATURE_COLUMNS_V2)
    y_train = train_sample["label"].values

    X_test = pd.DataFrame([extract_features_v2(u) for u in test_sample[url_col]], columns=FEATURE_COLUMNS_V2)
    y_test = test_sample["label"].values

    print(" Huấn luyện mô hình XGBoost trên dữ liệu quá khứ...")
    model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]
    y_preds = (y_scores >= 0.5).astype(int)

    metrics = classification_metrics(y_test, y_preds, y_scores)

    report = {
        "protocol": "Protocol B — Phishing Temporal-Shift Experiment",
        "train_phishing_count": len(train_phish),
        "test_future_phishing_count": len(test_phish),
        "train_legit_count": len(train_legit),
        "test_legit_count": len(test_legit),
        "legit_domain_disjoint": True,
        "metrics_on_future_campaigns": metrics,
        "observations": (
            "Hiệu năng trên các chiến dịch lừa đảo trong tương lai phản ánh mức độ concept drift của các "
            "kỹ thuật phishing mới so với các chiến dịch cũ trong quá khứ, kết hợp với legitimate holdout "
            "được chia domain-disjoint nhằm chống data leakage triệt để."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(" 📊 KẾT QUẢ PROTOCOL B (TEMPORAL DRIFT)")
    print("=" * 65)
    print(f" • PR-AUC trên Future Campaigns: {metrics['pr_auc']:.4f}")
    print(f" • ROC-AUC:                      {metrics['roc_auc']:.4f}")
    print(f" • Recall trên Future Phishing:  {metrics['recall'] * 100:.2f}%")
    print(f" • False Positive Rate:          {metrics['false_positive_rate'] * 100:.2f}%")
    print(f" [OK] Báo cáo đã lưu vào: {DRIFT_REPORT_JSON}")


if __name__ == "__main__":
    main()
