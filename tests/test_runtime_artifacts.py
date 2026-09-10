"""Kiểm tra runtime bundle tối thiểu mà API cần để khởi động."""

from __future__ import annotations

import unittest

from API.services.model_loader import load_phishguard_model
from phishguard.features import FEATURE_COLUMNS, FEATURE_CONTRACT_VERSION


class RuntimeArtifactTests(unittest.TestCase):
    def test_runtime_bundle_loads_with_expected_feature_contract(self) -> None:
        loaded = load_phishguard_model()

        self.assertEqual(loaded.feature_count, len(FEATURE_COLUMNS))
        self.assertEqual(loaded.feature_contract, FEATURE_CONTRACT_VERSION)
        self.assertEqual(loaded.feature_count, 25)
        self.assertTrue(loaded.calibrator.is_fitted)
        self.assertLess(
            loaded.thresholds.caution_threshold,
            loaded.thresholds.block_threshold,
        )


if __name__ == "__main__":
    unittest.main()
