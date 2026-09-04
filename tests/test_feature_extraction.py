"""
Bộ kiểm thử tự động cho module trích xuất đặc trưng URL PhishGuard ML.
Tích hợp kiểm tra fixture hợp đồng đặc trưng tests/fixtures/feature_contract.json.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from phishguard.features import (
    FEATURE_COLUMNS,
    extract_features,
    first_directory_length,
    has_ip_address,
    has_redirection_pattern,
    path_depth,
    uses_shortening_service,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FEATURE_FIXTURE_PATH = FIXTURES_DIR / "feature_contract.json"


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_count_and_order_contract(self) -> None:
        """Đảm bảo hàm trả về đủ 12 đặc trưng đúng thứ tự cột của hợp đồng XGBoost."""
        features = extract_features("https://example.com/account/login?id=123")
        self.assertEqual(tuple(features), FEATURE_COLUMNS)
        self.assertEqual(len(features), 12)

    def test_detects_ip_address_hostname(self) -> None:
        """Kiểm tra khả năng nhận diện địa chỉ IPv4 và IPv6."""
        self.assertEqual(has_ip_address("http://192.168.1.1/login"), 1)
        self.assertEqual(has_ip_address("http://127.0.0.1:5000/health"), 1)
        self.assertEqual(has_ip_address("http://[2001:db8::1]/index.html"), 1)
        self.assertEqual(has_ip_address("https://example.com"), 0)

    def test_detects_shortened_url(self) -> None:
        """Kiểm tra nhận diện dịch vụ rút gọn liên kết."""
        self.assertEqual(uses_shortening_service("https://bit.ly/3xXyZ"), 1)
        self.assertEqual(uses_shortening_service("https://tinyurl.com/abc"), 1)
        self.assertEqual(uses_shortening_service("https://t.co/xyz123"), 1)
        self.assertEqual(uses_shortening_service("https://is.gd/test"), 1)
        self.assertEqual(uses_shortening_service("https://example.com/bit.ly/help"), 0)
        self.assertEqual(uses_shortening_service("https://notbit.ly/login"), 0)

    def test_golden_feature_contract_fixtures(self) -> None:
        """Kiểm tra toàn bộ 23 mẫu URL trong tests/fixtures/feature_contract.json."""
        self.assertTrue(FEATURE_FIXTURE_PATH.exists(), f"Không tìm thấy fixture file {FEATURE_FIXTURE_PATH}")
        with open(FEATURE_FIXTURE_PATH, encoding="utf-8") as f:
            fixtures = json.load(f)

        for case in fixtures:
            url = case["url"]
            expected = case["expected_features"]
            description = case.get("description", "")
            actual = extract_features(url)
            self.assertEqual(
                actual,
                expected,
                f"Mismatch fixture for URL '{url}' ({description}): expected {expected}, got {actual}",
            )

    def test_path_depth_and_first_directory_length(self) -> None:
        """Kiểm tra đếm độ sâu đường dẫn và độ dài thư mục đầu tiên."""
        url = "https://example.com/security/login/verify"
        self.assertEqual(path_depth(url), 3)
        self.assertEqual(first_directory_length(url), len("security"))

        root_url = "https://example.com"
        self.assertEqual(path_depth(root_url), 0)
        self.assertEqual(first_directory_length(root_url), 0)

    def test_redirection_pattern_detection(self) -> None:
        """Kiểm tra phát hiện mẫu chuyển hướng lừa đảo '//'."""
        self.assertEqual(has_redirection_pattern("https://example.com/path//next"), 1)
        self.assertEqual(has_redirection_pattern("https://example.com/path/next"), 0)

    def test_lexical_character_counts(self) -> None:
        """Kiểm tra đếm ký tự số, dấu chấm, @, gạch ngang, %, bằng."""
        url = "https://user@test-domain.com:8080/path/123?a=1&b=20%20"
        features = extract_features(url)
        self.assertGreater(features["Digit_Count"], 0)
        self.assertEqual(features["At_Count"], 1)
        self.assertEqual(features["Hyphen_Count"], 1)
        self.assertEqual(features["Equal_Count"], 2)
        self.assertEqual(features["Per_Count"], 1)

    def test_handles_unicode_or_malformed_urls_gracefully(self) -> None:
        """Kiểm tra không bị sập khi gặp URL chứa unicode hoặc cú pháp bất thường."""
        features = extract_features("https://tiếngviệt.vn/đăng-nhập")
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 12)


if __name__ == "__main__":
    unittest.main()
