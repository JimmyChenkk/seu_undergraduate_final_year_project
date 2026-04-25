from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.evaluation.report_figures import (
    _build_figure_output_path,
    _compact_scenario_label,
    _domain_visual_styles,
    _resolve_primary_figure_format,
    _save_figure,
    _sort_methods_for_heatmap,
    _sort_methods_for_mean_chart,
)


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

    def test_heatmap_method_order_keeps_rcta_ablation_stages_in_sequence(self) -> None:
        methods = _sort_methods_for_heatmap(
            [
                "rcta_m3_dual_proto_static",
                "rcta_m1_temporal_mt",
                "rcta_m4_full",
                "rcta_m0_base_da",
                "rcta_m2_reliability_gate",
            ]
        )

        self.assertEqual(
            methods,
            [
                "rcta_m0_base_da",
                "rcta_m1_temporal_mt",
                "rcta_m2_reliability_gate",
                "rcta_m3_dual_proto_static",
                "rcta_m4_full",
            ],
        )

    def test_heatmap_method_order_keeps_compact_three_stage_ablation_sequence(self) -> None:
        methods = _sort_methods_for_heatmap(["rcta_abc_full", "rcta_a_temporal", "rcta_ab_reliable"])

        self.assertEqual(methods, ["rcta_a_temporal", "rcta_ab_reliable", "rcta_abc_full"])

    def test_compact_scenario_label_shortens_mode_transition_labels(self) -> None:
        self.assertEqual(_compact_scenario_label("mode1_to_mode4"), "m1_m4")
        self.assertEqual(_compact_scenario_label("mode6_to_mode5"), "m6_m5")
        self.assertEqual(_compact_scenario_label("other_label"), "other_label")

    def test_primary_figure_format_defaults_to_svg(self) -> None:
        self.assertEqual(_resolve_primary_figure_format(None), "svg")

    def test_build_figure_output_path_uses_primary_format_suffix(self) -> None:
        self.assertEqual(_build_figure_output_path(Path("figures/chart")), Path("figures/chart.svg"))
        self.assertEqual(
            _build_figure_output_path(Path("figures/chart.png"), figure_format="pdf"),
            Path("figures/chart.pdf"),
        )

    def test_save_figure_exports_svg_only_by_default(self) -> None:
        fake_plt = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chart.svg"
            with mock.patch(
                "src.evaluation.report_figures._runtime_dependencies",
                return_value=(None, fake_plt, None),
            ):
                _save_figure(output_path)

        self.assertEqual(
            fake_plt.savefig.call_args_list,
            [
                mock.call(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight"),
            ],
        )

    def test_save_figure_keeps_requested_primary_format_and_svg_companion(self) -> None:
        fake_plt = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chart.pdf"
            with mock.patch(
                "src.evaluation.report_figures._runtime_dependencies",
                return_value=(None, fake_plt, None),
            ):
                _save_figure(output_path, figure_format="pdf")

        self.assertEqual(
            fake_plt.savefig.call_args_list,
            [
                mock.call(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight"),
                mock.call(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
