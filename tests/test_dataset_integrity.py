from __future__ import annotations

import unittest

from scripts.check_dataset_integrity import build_integrity_report


class DatasetIntegrityReportTests(unittest.TestCase):
    def test_manifest_contract_matches_te_benchmark_shape(self) -> None:
        report = build_integrity_report("configs/data/te_da.yaml")

        self.assertTrue(report["passed"])
        self.assertEqual(report["domain_count"], 6)
        self.assertEqual(report["normalization_scope"], "domain")
        self.assertFalse(report["default_use_target_labels"])
        for domain in report["domains"]:
            self.assertEqual(domain["signals_shape"][1:], [600, 34])
            self.assertEqual(domain["label_range"], {"min": 0, "max": 28})
            self.assertEqual(domain["label_unique_count"], 29)


if __name__ == "__main__":
    unittest.main()
