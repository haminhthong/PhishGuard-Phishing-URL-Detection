"""Kiểm thử pipeline dữ liệu, Public Suffix List (PSL) và hàng rào chống leakage."""

import unittest

import pandas as pd

from phishguard.training.data import (
    clean_dataset,
    registered_domain,
    split_by_domain,
)


class TrainingDataTests(unittest.TestCase):
    def test_removes_duplicate_and_conflicting_domains(self):
        frame = pd.DataFrame(
            {
                "url": [
                    "https://safe-example.com/a",
                    "https://safe-example.com/a",
                    # Multi-label domain nhưng URLs khác nhau -> BẢO TỒN
                    "https://shared-host.com/legit-page",
                    "https://sub.shared-host.com/phish-page",
                    # Exact canonical URL conflict (cùng URL mang cả nhãn 0 và 1) -> BỊ LOẠI
                    "https://exact-conflict.com/same",
                    "https://exact-conflict.com/same",
                    "not-a-url",
                ],
                "label": [0, 0, 0, 1, 0, 1, 0],
            }
        )
        cleaned = clean_dataset(frame)
        self.assertIn("https://safe-example.com/a", cleaned["url"].tolist())
        self.assertIn("https://shared-host.com/legit-page", cleaned["url"].tolist())
        self.assertIn("https://sub.shared-host.com/phish-page", cleaned["url"].tolist())
        self.assertNotIn("https://exact-conflict.com/same", cleaned["url"].tolist())
        self.assertNotIn("not-a-url", cleaned["url"].tolist())
        self.assertEqual(len(cleaned), 3)

    def test_split_has_no_domain_overlap(self):
        rows = []
        for index in range(80):
            label = index % 2
            rows.append({"url": f"https://sample-{index}-class-{label}.com/path", "label": label})
        splits = split_by_domain(pd.DataFrame(rows), random_state=7)
        train_domains = set(splits.train["domain"])
        validation_domains = set(splits.validation["domain"])
        calibration_domains = set(splits.calibration["domain"])
        threshold_domains = set(splits.threshold_validation["domain"])
        test_domains = set(splits.test["domain"])
        self.assertTrue(train_domains.isdisjoint(validation_domains))
        self.assertTrue(train_domains.isdisjoint(test_domains))
        self.assertTrue(validation_domains.isdisjoint(test_domains))
        self.assertTrue(calibration_domains.isdisjoint(threshold_domains))
        self.assertTrue(threshold_domains.isdisjoint(test_domains))

    def test_registered_domain_extraction_psl(self):
        """Kiểm tra registered domain tuân thủ ngữ nghĩa Mozilla Public Suffix List (PSL)."""
        # Multi-part ccTLD
        self.assertEqual(registered_domain("https://sub.example.co.uk/path"), "example.co.uk")
        self.assertEqual(registered_domain("https://deep.sub.portal.gov.uk/index"), "portal.gov.uk")
        self.assertEqual(registered_domain("https://www.google.com.vn/search"), "google.com.vn")
        # Standard gTLD
        self.assertEqual(registered_domain("https://login.paypal.com/signin"), "paypal.com")
        # IP Address fallback
        self.assertEqual(registered_domain("http://192.168.1.1:8080/admin"), "192.168.1.1")
        # Invalid input
        self.assertEqual(registered_domain("invalid-string"), "")

    def test_rejects_invalid_split_ratio(self):
        frame = pd.DataFrame({"url": ["https://a.com"], "label": [0]})
        with self.assertRaises(ValueError):
            split_by_domain(frame, train_size=0.6, validation_size=0.5)


if __name__ == "__main__":
    unittest.main()
