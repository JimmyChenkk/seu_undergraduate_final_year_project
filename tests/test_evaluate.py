from __future__ import annotations

import unittest

from src.evaluation.evaluate import sort_comparison_rows


class EvaluateSummaryTests(unittest.TestCase):
    def test_sort_comparison_rows_uses_baseline_method_order_per_scenario(self) -> None:
        rows = [
            {"scenario_id": "mode1_to_mode2", "method": "target_only"},
            {"scenario_id": "mode1_to_mode2", "method": "dsan"},
            {"scenario_id": "mode1_to_mode2", "method": "source_only"},
            {"scenario_id": "mode1_to_mode2", "method": "raincoat"},
            {"scenario_id": "mode2_to_mode1", "method": "target_only"},
            {"scenario_id": "mode2_to_mode1", "method": "source_only"},
        ]

        sorted_rows = sort_comparison_rows(rows)

        self.assertEqual(
            [row["method"] for row in sorted_rows],
            ["source_only", "dsan", "raincoat", "target_only", "source_only", "target_only"],
        )

    def test_sort_comparison_rows_keeps_ca_ccsr_variants_near_parent_method(self) -> None:
        rows = [
            {"scenario_id": "mode1-mode5_to_mode2", "method": "ca_ccsr_wjdot_refine20"},
            {"scenario_id": "mode1-mode5_to_mode2", "method": "codats"},
            {"scenario_id": "mode1-mode5_to_mode2", "method": "ca_ccsr_wjdot_otlite20"},
        ]

        sorted_rows = sort_comparison_rows(rows)

        self.assertEqual(
            [row["method"] for row in sorted_rows],
            ["codats", "ca_ccsr_wjdot_otlite20", "ca_ccsr_wjdot_refine20"],
        )


if __name__ == "__main__":
    unittest.main()
