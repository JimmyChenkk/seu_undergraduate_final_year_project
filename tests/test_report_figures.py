from __future__ import annotations

import unittest

from src.evaluation.report_figures import _compact_scenario_label, _domain_visual_styles, _sort_methods_for_mean_chart


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

    def test_mean_accuracy_method_order_places_source_only_first(self) -> None:
        methods = _sort_methods_for_mean_chart(["target_only", "dan", "source_only", "deepjdot"])

        self.assertEqual(methods[0], "source_only")
        self.assertEqual(methods[1:], ["dan", "deepjdot", "target_only"])

    def test_compact_scenario_label_shortens_mode_transition_labels(self) -> None:
        self.assertEqual(_compact_scenario_label("mode1_to_mode4"), "m1_m4")
        self.assertEqual(_compact_scenario_label("mode6_to_mode5"), "m6_m5")
        self.assertEqual(_compact_scenario_label("other_label"), "other_label")


if __name__ == "__main__":
    unittest.main()
