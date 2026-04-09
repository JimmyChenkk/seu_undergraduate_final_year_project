from __future__ import annotations

import unittest

from src.evaluation.report_figures import _domain_visual_styles


class ReportFiguresTests(unittest.TestCase):
    def test_domain_visual_styles_use_high_contrast_source_target_defaults(self) -> None:
        styles = _domain_visual_styles(["source", "target"])

        self.assertEqual(styles["source"]["color"], "#D55E00")
        self.assertEqual(styles["source"]["marker"], "o")
        self.assertEqual(styles["target"]["color"], "#0072B2")
        self.assertEqual(styles["target"]["marker"], "^")

    def test_domain_visual_styles_keep_source_and_target_in_separate_color_families(self) -> None:
        styles = _domain_visual_styles(
            [
                "source:mode1",
                "source:mode2",
                "target:mode4",
                "target:mode5",
            ]
        )

        self.assertEqual(styles["source:mode1"]["color"], "#D55E00")
        self.assertEqual(styles["source:mode2"]["color"], "#E69F00")
        self.assertEqual(styles["target:mode4"]["color"], "#0072B2")
        self.assertEqual(styles["target:mode5"]["color"], "#009E73")


if __name__ == "__main__":
    unittest.main()
