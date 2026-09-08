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
        """Đảm bảo hàm trả về đủ 25 đặc trưng đúng thứ tự cột của hợp đồng lexical-v3."""
        features = extract_features("https://example.com/account/login?id=123")
        self.assertEqual(tuple(features), FEATURE_COLUMNS)
        self.assertEqual(len(features), 25)

        # Legacy v1 kiểm tra rõ ràng
        features_v1 = extract_features(
            "https://example.com/account/login?id=123", contract="lexical-v1"
        )
        self.assertEqual(len(features_v1), 12)

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
        """Kiểm tra toàn bộ 23 mẫu URL trong tests/fixtures/feature_contract.json theo hợp đồng v1."""
        self.assertTrue(
            FEATURE_FIXTURE_PATH.exists(), f"Không tìm thấy fixture file {FEATURE_FIXTURE_PATH}"
        )
        with open(FEATURE_FIXTURE_PATH, encoding="utf-8") as f:
            fixtures = json.load(f)

        for case in fixtures:
            url = case["url"]
            expected = case["expected_features"]
            description = case.get("description", "")
            actual = extract_features(url, contract="lexical-v1")
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
        """Kiểm tra đếm ký tự số, dấu chấm, @, gạch ngang, %, bằng trên v1 và v2."""
        url = "https://user@test-domain.com:8080/path/123?a=1&b=20%20"
        features_v1 = extract_features(url, contract="lexical-v1")
        self.assertGreater(features_v1["Digit_Count"], 0)
        self.assertEqual(features_v1["At_Count"], 1)
        self.assertEqual(features_v1["Hyphen_Count"], 1)
        self.assertEqual(features_v1["Equal_Count"], 2)
        self.assertEqual(features_v1["Per_Count"], 1)

        features_v2 = extract_features(url, contract="lexical-v2")
        self.assertGreater(features_v2["digit_ratio"], 0.0)
        self.assertEqual(features_v2["at_count"], 1)
        self.assertEqual(features_v2["hyphen_count"], 1)

    def test_handles_unicode_or_malformed_urls_gracefully(self) -> None:
        """Kiểm tra không bị sập khi gặp URL chứa unicode hoặc cú pháp bất thường."""
        features = extract_features("https://tiếngviệt.vn/đăng-nhập", contract="lexical-v1")
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 12)

    def test_lexical_v2_contract_and_structure(self) -> None:
        """Kiểm tra hợp đồng lexical-v2 đủ 25 đặc trưng theo 4 nhóm."""
        from phishguard.features import FEATURE_COLUMNS_V2, extract_features_v2

        features_v2 = extract_features_v2("https://sub.example.com/path/to/page?id=123&test=abc")
        self.assertEqual(len(features_v2), 25)
        self.assertEqual(tuple(features_v2.keys()), FEATURE_COLUMNS_V2)
        self.assertEqual(features_v2["subdomain_count"], 1)
        self.assertEqual(features_v2["path_depth"], 3)
        self.assertGreater(features_v2["url_entropy"], 0.0)
        self.assertGreater(features_v2["digit_ratio"], 0.0)

    def test_brand_impersonation_detection(self) -> None:
        """Kiểm tra phát hiện mạo danh thương hiệu (Brand Abuse) trong subdomain và path."""
        from phishguard.features import extract_features_v2

        # Phishing mạo danh PayPal trong subdomain
        phish_paypal = "https://paypal.com.account-verify.attacker.com/login"
        f_phish = extract_features_v2(phish_paypal)
        self.assertEqual(f_phish["brand_in_subdomain"], 1)
        self.assertEqual(f_phish["brand_not_registered_domain"], 1)

        # Domain PayPal hợp lệ
        legit_paypal = "https://www.paypal.com/signin"
        f_legit = extract_features_v2(legit_paypal)
        self.assertEqual(f_legit["brand_in_subdomain"], 0)
        self.assertEqual(f_legit["brand_not_registered_domain"], 0)

        # Phishing mạo danh Google trong path
        phish_google_path = "https://evil-server.net/google/login/oauth2"
        f_google = extract_features_v2(phish_google_path)
        self.assertEqual(f_google["brand_in_path"], 1)
        self.assertEqual(f_google["brand_not_registered_domain"], 1)

        # V4 phải tránh false positive substring: "pineapple" không phải token "apple".
        pineapple = extract_features_v2(
            "https://pineapple.example.com/login", strict_brand_matching=True
        )
        self.assertEqual(pineapple["brand_in_subdomain"], 0)
        self.assertEqual(pineapple["brand_not_registered_domain"], 0)

        v4 = extract_features("https://pineapple.example.com/login", contract="lexical-v4")
        self.assertEqual(v4["brand_in_subdomain"], 0)

    def test_punycode_and_suspicious_tld_features(self) -> None:
        """Kiểm tra nhận diện punycode và tên miền TLD đáng ngờ."""
        from phishguard.features import extract_features_v2

        puny_url = "https://xn--e1afmkfd.xn--p1ai/path"
        f_puny = extract_features_v2(puny_url)
        self.assertEqual(f_puny["has_punycode"], 1)

        suspicious_tld_url = "https://banking-secure-check.xyz/login"
        f_suspicious = extract_features_v2(suspicious_tld_url)
        self.assertEqual(f_suspicious["is_suspicious_tld"], 1)

        normal_url = "https://normal-domain.com/path"
        f_normal = extract_features_v2(normal_url)
        self.assertEqual(f_normal["has_punycode"], 0)
        self.assertEqual(f_normal["is_suspicious_tld"], 0)


if __name__ == "__main__":
    unittest.main()
