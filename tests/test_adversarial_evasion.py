"""
Bộ kiểm thử Adversarial & Evasion Cases cho PhishGuard ML.
Đảm bảo bộ trích xuất đặc trưng hoạt động ổn định, có tính tất định và không bị crash
trước các kỹ thuật lừa đảo tinh vi hoặc cố tình làm nhiễu URL.
"""

from __future__ import annotations

import unittest

from phishguard.features import extract_features_v1, extract_features_v2


class AdversarialEvasionTests(unittest.TestCase):
    def test_embedded_credentials_before_hostname(self) -> None:
        """Kiểm tra URL chứa thông tin xác thực giả mạo dạng user@domain."""
        url = "https://google.com@attacker-controlled-site.com/auth"
        f = extract_features_v2(url)
        self.assertEqual(f["at_count"], 1)
        self.assertEqual(
            f["brand_not_registered_domain"], 0
        )  # hostname is attacker-controlled-site.com

    def test_deceptive_subdomain_abuse(self) -> None:
        """Kiểm tra lạm dụng chuỗi subdomain sâu để giả mạo PayPal."""
        url = "https://paypal.com.verify.account.security-check.attacker.com/signin"
        f = extract_features_v2(url)
        self.assertEqual(f["brand_in_subdomain"], 1)
        self.assertEqual(f["brand_not_registered_domain"], 1)
        self.assertGreater(f["subdomain_count"], 3)

    def test_punycode_homograph_spoofing(self) -> None:
        """Kiểm tra phát hiện hostname sử dụng mã hóa ký tự quốc tế Punycode xn--."""
        url = "https://xn--pypal-4ve.com/account/security"
        f = extract_features_v2(url)
        self.assertEqual(f["has_punycode"], 1)

    def test_redirection_pattern_evasion(self) -> None:
        """Kiểm tra phát hiện dấu '//' trong path để bypass regex ngây thơ."""
        url = "https://legit-site.com//redirect-to-malicious.org"
        f = extract_features_v2(url)
        self.assertEqual(f["has_redirection_pattern"], 1)

    def test_percent_encoded_characters(self) -> None:
        """Kiểm tra các ký tự bị mã hóa phần trăm (%20, %2e, %2f)."""
        url = "https://example.com/%2e%2e%2fadmin%20login?token=%3D%3D"
        f = extract_features_v2(url)
        self.assertGreater(f["special_char_ratio"], 0.0)

    def test_mixed_case_url(self) -> None:
        """Kiểm tra URL viết hoa thường hỗn loạn (hTTpS://GoOGle.CoM)."""
        url = "hTTpS://GoOGle.CoM/Path/To/Page"
        f = extract_features_v2(url)
        self.assertEqual(f["brand_not_registered_domain"], 0)
        self.assertEqual(f["tld_length"], 3)

    def test_extremely_long_url_boundary(self) -> None:
        """Kiểm tra URL có độ dài cực lớn (1.800 ký tự) không làm tràn bộ nhớ hay treo CPU."""
        long_path = "a" * 1500
        url = f"https://example.com/{long_path}?key=val"
        f = extract_features_v2(url)
        self.assertGreater(f["url_length"], 1500)
        self.assertEqual(f["path_depth"], 1)

    def test_v1_and_v2_determinism(self) -> None:
        """Kiểm tra tính tất định: cùng 1 URL luôn cho ra kết quả đặc trưng giống hệt nhau."""
        url = "https://secure-login.test.org/verify?user=123"
        f1_a = extract_features_v1(url)
        f1_b = extract_features_v1(url)
        self.assertEqual(f1_a, f1_b)

        f2_a = extract_features_v2(url)
        f2_b = extract_features_v2(url)
        self.assertEqual(f2_a, f2_b)


if __name__ == "__main__":
    unittest.main()
