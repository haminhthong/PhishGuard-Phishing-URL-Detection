"""
Kiểm thử bộ Hard Negatives (các URL hợp lệ nhưng có cấu trúc phức tạp như SSO, AWS S3, OAuth, CDN).
Đảm bảo bộ trích xuất đặc trưng xử lý chính xác và không sinh ra các giá trị ngoại lai bất thường.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from phishguard.features import extract_features_v1, extract_features_v2

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "hard_negatives.json"


class HardNegativeBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(FIXTURE_PATH.is_file(), f"Không tìm thấy fixture {FIXTURE_PATH}")
        with open(FIXTURE_PATH, encoding="utf-8") as f:
            self.hard_negatives = json.load(f)

    def test_hard_negatives_v1_and_v2_extraction(self) -> None:
        """Đảm bảo mọi URL trong hard_negatives đều trích xuất thành công cả v1 và v2."""
        for item in self.hard_negatives:
            url = item["url"]
            category = item.get("category", "unknown")

            # v1 contract
            f1 = extract_features_v1(url)
            self.assertEqual(len(f1), 12)
            self.assertEqual(f1["Having_IP"], 0)

            # v2 contract
            f2 = extract_features_v2(url)
            self.assertEqual(len(f2), 25)
            self.assertEqual(f2["has_ip_address"], 0)
            self.assertEqual(f2["has_punycode"], 0)
            # URL hợp lệ không được kích hoạt cờ mạo danh thương hiệu trái phép
            self.assertEqual(
                f2["brand_not_registered_domain"],
                0,
                f"False positive brand abuse flag on legitimate URL: {url} ({category})",
            )


if __name__ == "__main__":
    unittest.main()
