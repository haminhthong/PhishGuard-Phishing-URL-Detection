"""Kiểm thử pipeline dữ liệu và hàng rào chống leakage."""

import unittest

import pandas as pd

from phishguard.training.data import clean_dataset, split_by_domain


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

    def test_rejects_invalid_split_ratio(self):
        frame = pd.DataFrame({"url": ["https://a.com"], "label": [0]})
        with self.assertRaises(ValueError):
            split_by_domain(frame, test_size=0.6, validation_size=0.5)


if __name__ == "__main__":
    unittest.main()
