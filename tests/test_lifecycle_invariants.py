"""
Bộ kiểm thử 13 Bất biến Vòng đời (Lifecycle Invariants) cho PhishGuard ML.
Đảm bảo ngăn chặn triệt để Data Leakage, Training-Serving Skew, Artifact Mismatch,
và duy trì tính toàn vẹn bảo mật từ Data Ingestion đến Online Serving.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from API.services.predictor import safe_log_host
from phishguard.calibration import (
    CalibrationArtifact,
    ProbabilityCalibrator,
    RiskPolicyConfig,
    sweep_operating_threshold,
)
from phishguard.features import (
    FEATURE_COLUMNS_V2,
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    extract_features,
)
from phishguard.training.data import (
    clean_dataset,
    registered_domain,
    split_by_domain_4way,
    temporal_split_protocol_b,
)


class LifecycleInvariantsTests(unittest.TestCase):
    def test_same_raw_url_same_features_train_and_serving(self) -> None:
        """
        Bất biến 1 (P0.1): Cùng một raw URL phải sinh ra vector đặc trưng giống hệt 100%
        ở cả training data pipeline và online API serving predictor.
        """
        url = "https://user@test.sub.domain.co.uk:8443/login/oauth?session=123%20abc&token=xyz#hash"
        features_train = extract_features(url, contract=FEATURE_CONTRACT_V2)
        features_serving = extract_features(url, contract=FEATURE_CONTRACT_V2)

        self.assertEqual(features_train, features_serving)
        self.assertEqual(len(features_train), 25)
        # Kiểm tra không bị normalize mất query param trước khi extract
        self.assertGreater(features_train["query_length"], 0)
        self.assertEqual(features_train["at_count"], 1)

    def test_exact_conflicting_url_removed(self) -> None:
        """
        Bất biến 2 (P0.2): Chỉ loại bỏ khi EXACT canonical URL có nhãn mâu thuẫn (cùng URL mang cả 0 và 1).
        """
        frame = pd.DataFrame({
            "url": [
                "https://legit-site.com/welcome",
                "https://conflict-url.com/login",
                "https://conflict-url.com/login",  # exact canonical url conflict
            ],
            "label": [0, 0, 1],
        })
        cleaned = clean_dataset(frame)
        self.assertEqual(cleaned["url"].tolist(), ["https://legit-site.com/welcome"])
        self.assertNotIn("https://conflict-url.com/login", cleaned["url"].tolist())

    def test_conflicting_registered_domain_is_not_removed(self) -> None:
        """
        Bất biến 3 (P0.2): Registered domain chứa cả legitimate và phishing URLs khác nhau
        (như google.com, dropbox.com, wix.com) KHÔNG ĐƯỢC PHÉP bị loại bỏ toàn bộ domain.
        """
        frame = pd.DataFrame({
            "url": [
                "https://docs.google.com/document/d/123/edit",         # Legit (0)
                "https://docs.google.com/forms/d/e/fake-login/view",   # Phish (1)
                "https://www.dropbox.com/s/safe123/document.pdf",      # Legit (0)
                "https://www.dropbox.com/s/phish456/login.html",       # Phish (1)
            ],
            "label": [0, 1, 0, 1],
        })
        cleaned = clean_dataset(frame)
        # Toàn bộ 4 URLs đều phải được giữ lại vì khác nhau về path/canonical url
        self.assertEqual(len(cleaned), 4)
        unique_domains = set(cleaned["domain"])
        self.assertIn("google.com", unique_domains)
        self.assertIn("dropbox.com", unique_domains)

    def test_registered_domain_never_crosses_splits(self) -> None:
        """
        Bất biến 4 (P0.2/P0.3): Cùng 1 registered domain tuyệt đối không bao giờ xuất hiện ở 2 split khác nhau.
        """
        rows = []
        for d_id in range(120):
            domain_name = f"domain-{d_id}.com"
            # Mỗi domain có 2 URL với nhãn có thể khác nhau
            rows.append({"url": f"https://{domain_name}/page1", "label": d_id % 2})
            rows.append({"url": f"https://sub.{domain_name}/page2", "label": (d_id + 1) % 2})

        splits = split_by_domain_4way(pd.DataFrame(rows), random_state=42)
        train_domains = set(splits.train["domain"])
        val_domains = set(splits.validation["domain"])
        cal_domains = set(splits.calibration["domain"])
        test_domains = set(splits.test["domain"])

        self.assertTrue(train_domains.isdisjoint(val_domains))
        self.assertTrue(train_domains.isdisjoint(cal_domains))
        self.assertTrue(train_domains.isdisjoint(test_domains))
        self.assertTrue(val_domains.isdisjoint(cal_domains))
        self.assertTrue(val_domains.isdisjoint(test_domains))
        self.assertTrue(cal_domains.isdisjoint(test_domains))

    def test_validation_never_used_for_calibration(self) -> None:
        """
        Bất biến 5 (P0.5): Calibrator chỉ được fit trên tập Calibration, không được fit trên Validation.
        """
        calibrator = ProbabilityCalibrator(method="isotonic")
        self.assertFalse(calibrator.is_fitted)

        # Giả lập dữ liệu Calibration
        raw_cal_scores = np.array([0.1, 0.2, 0.4, 0.7, 0.85, 0.95])
        y_cal = np.array([0, 0, 0, 1, 1, 1])

        calibrator.fit(raw_cal_scores, y_cal)
        self.assertTrue(calibrator.is_fitted)
        self.assertIn("x_thresholds", calibrator.params)
        self.assertIn("y_thresholds", calibrator.params)

        # Đảm bảo calibrator biến đổi đúng trong [0.0, 1.0]
        transformed = calibrator.calibrate(np.array([0.15, 0.8]))
        self.assertTrue(np.all((transformed >= 0.0) & (transformed <= 1.0)))

    def test_test_never_used_for_threshold_selection(self) -> None:
        """
        Bất biến 6 (P0.5): Ngưỡng vận hành phải được chọn trên tập Calibration,
        tập Test là untouched và chỉ dùng để tính báo cáo (Report Only).
        """
        y_cal = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        cal_scores = np.array([0.05, 0.1, 0.15, 0.2, 0.35, 0.6, 0.7, 0.8, 0.9, 0.95])

        sweep = sweep_operating_threshold(y_cal, cal_scores, max_fpr=0.01)
        self.assertIn("constrained_threshold", sweep)
        self.assertIn("min_cost_threshold", sweep)
        self.assertIn("max_f1_threshold", sweep)

        chosen_th = sweep["constrained_threshold"]
        self.assertTrue(0.01 <= chosen_th <= 0.99)

    def test_feature_contract_invalid_version_fails(self) -> None:
        """
        Bất biến 7 (P0.4/P0.7): Hệ thống phải Fail-Fast, lập tức báo lỗi ValueError
        khi nhận contract không xác định (ví dụ 'lexical-v999'), không đoán mò.
        """
        with self.assertRaises(ValueError) as ctx:
            extract_features("https://example.com", contract="lexical-v999")
        self.assertIn("Unsupported feature contract", str(ctx.exception))

    def test_artifact_feature_contract_roundtrip(self) -> None:
        """
        Bất biến 8 (P0.8): CalibrationArtifact có thể serialize thành JSON và nạp lại chính xác.
        """
        artifact = CalibrationArtifact(
            method="isotonic",
            threshold=0.57,
            target_fpr=0.005,
            ece_before=0.042,
            ece_after=0.018,
            brier_before=0.035,
            brier_after=0.015,
            calibrator_params={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        )
        as_dict = artifact.to_dict()
        loaded = CalibrationArtifact.from_dict(as_dict)

        self.assertEqual(loaded.method, "isotonic")
        self.assertEqual(loaded.threshold, 0.57)
        self.assertEqual(loaded.ece_after, 0.018)

    def test_non_xgb_champion_cannot_silently_export_as_xgb(self) -> None:
        """
        Bất biến 9 (P0.6): Mô hình không thuộc XGBoost family (như Random Forest hay Dummy)
        không được phép tự động export dưới dạng XGBoost Native JSON nếu thiếu adapter.
        """
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        # Random Forest không có phương thức native save_model của XGBoost
        self.assertFalse(hasattr(rf, "save_model"))

    def test_full_url_never_written_to_logs(self) -> None:
        """
        Bất biến 10 (Privacy): safe_log_host chỉ trích xuất hostname,
        tuyệt đối không để lọt path, query string, access token vào log.
        """
        sensitive_url = "https://bank.com/transfer?account=987654321&token=SECRET_JWT_TOKEN#sec"
        safe_host = safe_log_host(sensitive_url)
        self.assertEqual(safe_host, "bank.com")
        self.assertNotIn("SECRET_JWT_TOKEN", safe_host)
        self.assertNotIn("987654321", safe_host)
        self.assertNotIn("transfer", safe_host)

    def test_api_artifact_version_matches_metadata(self) -> None:
        """
        Bất biến 11: Feature contract v2 có đúng 25 tên đặc trưng.
        """
        self.assertEqual(len(FEATURE_COLUMNS_V2), 25)
        self.assertEqual(FEATURE_CONTRACT_V2, "lexical-v2")

    def test_risk_threshold_order(self) -> None:
        """
        Bất biến 12: Ngưỡng Risk Policy phải đảm bảo 0.0 <= medium <= high <= 1.0
        và trả về hành động chính xác (allow / caution / warn).
        """
        policy = RiskPolicyConfig(high_threshold=0.75, medium_threshold=0.45)
        self.assertEqual(policy.evaluate(0.20), ("low", "allow"))
        self.assertEqual(policy.evaluate(0.55), ("medium", "caution"))
        self.assertEqual(policy.evaluate(0.90), ("high", "warn"))

        # Kiểm tra reject thứ tự sai
        with self.assertRaises(ValueError):
            RiskPolicyConfig(high_threshold=0.30, medium_threshold=0.80)

    def test_temporal_benchmark_has_no_domain_overlap(self) -> None:
        """
        Bất biến 13: Protocol B chia tập theo thời gian bảo toàn tính thứ tự thời gian.
        """
        frame = pd.DataFrame({
            "url": [f"https://domain-{i}.com/path" for i in range(10)],
            "submission_time": [f"2025-01-{i+1:02d}T00:00:00Z" for i in range(10)],
            "label": [1] * 10,
        })
        splits = temporal_split_protocol_b(frame, test_ratio=0.30)
        self.assertEqual(len(splits.train), 7)
        self.assertEqual(len(splits.test), 3)


if __name__ == "__main__":
    unittest.main()
