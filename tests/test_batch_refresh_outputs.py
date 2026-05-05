from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.automation.run_small_scale_round import (
    _refresh_batch_outputs as refresh_automation_batch_outputs,
)
from src.trainers.train_benchmark import (
    _refresh_batch_outputs as refresh_training_batch_outputs,
)


class BatchRefreshOutputTests(unittest.TestCase):
    def _assert_refresh_exports_all_methods(self, refresh_func) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_root = Path(temp_dir) / "batch"
            summary_dir = batch_root / "comparison_summary" / "tables"

            with (
                mock.patch(
                    "src.evaluation.evaluate.export_comparison_summary",
                    return_value=summary_dir,
                ) as export_summary,
                mock.patch(
                    "src.evaluation.report_figures.export_summary_figures"
                ) as export_figures,
            ):
                refresh_func(batch_root)

            export_summary.assert_called_once_with(batch_root)
            export_figures.assert_called_once_with(
                batch_root,
                summary_dir.parent / "figures",
                include_all_methods=True,
            )

    def test_training_refresh_exports_all_methods_summary_figures(self) -> None:
        self._assert_refresh_exports_all_methods(refresh_training_batch_outputs)

    def test_automation_refresh_exports_all_methods_summary_figures(self) -> None:
        self._assert_refresh_exports_all_methods(refresh_automation_batch_outputs)


if __name__ == "__main__":
    unittest.main()
