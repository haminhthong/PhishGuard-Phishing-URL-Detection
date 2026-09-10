"""
Kiểm thử bộ Hard Negatives (các URL hợp lệ nhưng có cấu trúc phức tạp như SSO, AWS S3, OAuth, CDN).
Đảm bảo bộ trích xuất đặc trưng xử lý chính xác và không sinh ra các giá trị ngoại lai bất thường.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from phishguard.features import extract_features

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "hard_negatives.json"


class HardNegativeBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(FIXTURE_PATH.is_file(), f"Không tìm thấy fixture {FIXTURE_PATH}")
        with open(FIXTURE_PATH, encoding="utf-8") as f:
            self.hard_negatives = json.load(f)

    def test_hard_negatives_extraction(self) -> None:
        """Đảm bảo mọi URL hard negative tuân thủ contract lexical-v4."""
        for item in self.hard_negatives:
            url = item["url"]
            category = item.get("category", "unknown")

            features = extract_features(url)
            self.assertEqual(len(features), 25)
            self.assertEqual(features["has_ip_address"], 0)
            self.assertEqual(features["has_punycode"], 0)
            # URL hợp lệ không được kích hoạt cờ mạo danh thương hiệu trái phép
            self.assertEqual(
                features["brand_not_registered_domain"],
                0,
                f"False positive brand abuse flag on legitimate URL: {url} ({category})",
            )


if __name__ == "__main__":
    unittest.main()
