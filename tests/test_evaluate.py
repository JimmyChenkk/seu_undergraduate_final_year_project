from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluate import build_rows, sort_comparison_rows


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

    def test_build_rows_keeps_single_source_methods_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for method_name in ("deepjdot", "tpu_deepjdot", "cbtpu_deepjdot"):
                tables_dir = root / method_name / "tables"
                tables_dir.mkdir(parents=True)
                payload = {
                    "method_name": method_name,
                    "setting": "single_source",
                    "scenario_id": "mode1_to_mode2",
                    "backbone_name": "fcn",
                    "source_domains": ["mode1"],
                    "target_domain": "mode2",
                    "source_fold": "Fold 1",
                    "target_fold": "Fold 1",
                    "result": {
                        "source_train_acc": 0.8,
                        "source_eval_acc": 0.7,
                        "target_eval_acc": 0.6,
                        "target_eval_balanced_acc": 0.5,
                    },
                }
                (tables_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            rows = build_rows(root)

        self.assertTrue(any(row["method"] == "deepjdot" for row in rows))
        self.assertTrue(any(row["method"] == "tpu_deepjdot" for row in rows))
        self.assertTrue(any(row["method"] == "cbtpu_deepjdot" for row in rows))


if __name__ == "__main__":
    unittest.main()
