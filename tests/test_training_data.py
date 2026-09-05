"""Kiểm thử pipeline dữ liệu, Public Suffix List (PSL) và hàng rào chống leakage."""

import unittest

import pandas as pd

from phishguard.training.data import (
    clean_dataset,
    registered_domain,
    split_by_domain,
    split_by_domain_4way,
    temporal_split_protocol_b,
)


class TrainingDataTests(unittest.TestCase):
    def test_removes_duplicate_and_conflicting_domains(self):
        frame = pd.DataFrame(
            {
                "url": [
                    "https://safe-example.com/a",
                    "https://safe-example.com/a",
                    "https://conflict-example.com/one",
                    "https://sub.conflict-example.com/two",
                    "not-a-url",
                ],
                "label": [0, 0, 0, 1, 1],
            }
        )
        cleaned = clean_dataset(frame)
        self.assertEqual(cleaned["url"].tolist(), ["https://safe-example.com/a"])

    def test_split_has_no_domain_overlap(self):
        rows = []
        for index in range(80):
            label = index % 2
            rows.append(
                {"url": f"https://sample-{index}-class-{label}.com/path", "label": label}
            )
        splits = split_by_domain(pd.DataFrame(rows), random_state=7)
        train_domains = set(splits.train["domain"])
        validation_domains = set(splits.validation["domain"])
        test_domains = set(splits.test["domain"])
        self.assertTrue(train_domains.isdisjoint(validation_domains))
        self.assertTrue(train_domains.isdisjoint(test_domains))
        self.assertTrue(validation_domains.isdisjoint(test_domains))

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

    def test_split_by_domain_4way_no_overlap(self):
        """Kiểm tra chia 4 tập (Train/Val/Calibration/Test) tuyệt đối zero domain overlap."""
        rows = []
        for index in range(120):
            label = index % 2
            rows.append(
                {"url": f"https://domain-{index}.org/subpath", "label": label}
            )
        splits = split_by_domain_4way(
            pd.DataFrame(rows),
            test_size=0.10,
            calibration_size=0.10,
            validation_size=0.15,
            random_state=42,
        )
        sets = [
            set(splits.train["domain"]),
            set(splits.validation["domain"]),
            set(splits.calibration["domain"]),
            set(splits.test["domain"]),
        ]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertTrue(sets[i].isdisjoint(sets[j]), f"Domain overlap between split {i} and {j}")

    def test_temporal_split_protocol_b(self):
        """Kiểm tra chia tập theo trình tự thời gian (Protocol B - Temporal Holdout)."""
        frame = pd.DataFrame({
            "url": [
                "https://d1.com", "https://d2.com", "https://d3.com",
                "https://d4.com", "https://d5.com",
            ],
            "submission_time": [
                "2025-01-01T10:00:00Z",
                "2025-01-02T10:00:00Z",
                "2025-01-03T10:00:00Z",
                "2025-02-01T10:00:00Z",
                "2025-02-02T10:00:00Z",
            ],
            "label": [1, 1, 1, 1, 1],
        })
        temporal = temporal_split_protocol_b(frame, test_ratio=0.40)
        self.assertEqual(len(temporal.train), 3)
        self.assertEqual(len(temporal.test), 2)
        self.assertEqual(temporal.test["url"].tolist(), ["https://d4.com", "https://d5.com"])

    def test_rejects_invalid_split_ratio(self):
        frame = pd.DataFrame({"url": ["https://a.com"], "label": [0]})
        with self.assertRaises(ValueError):
            split_by_domain(frame, test_size=0.6, validation_size=0.5)


if __name__ == "__main__":
    unittest.main()
