from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.evaluation.report_figures import (
    _build_figure_output_path,
    _compact_scenario_label,
    _configure_matplotlib_fonts,
    _domain_visual_styles,
    _resolve_primary_figure_format,
    _save_figure,
    _visible_figure_methods,
    _sort_methods_for_heatmap,
    _sort_methods_for_mean_chart,
    _sort_scenarios_for_heatmap,
    FINAL_MAIN_METHODS,
    THESIS_FIGURE_FONT,
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

    def test_mode125_baseline_method_order_keeps_target_only_last(self) -> None:
        methods = _sort_methods_for_mean_chart(
            ["target_only", "raincoat", "source_only", "dsan", "cdan_ts", "codats", "deepjdot"]
        )

        self.assertEqual(
            methods,
            ["source_only", "dsan", "cdan_ts", "codats", "deepjdot", "raincoat", "target_only"],
        )

    def test_final_main_method_order_is_five_method_thesis_table(self) -> None:
        methods = _visible_figure_methods(
            _sort_methods_for_mean_chart(
                [
                    "target_ref",
                    "sa_ccsr_wjdot_train",
                    "codats",
                    "source_only",
                    "wjdot",
                    "ca_ccsr_wjdot",
                    "deepjdot",
                ]
            )
        )

        self.assertEqual(tuple(methods), FINAL_MAIN_METHODS)

    def test_ca_ccsr_tuning_variants_are_visible_next_to_parent_method(self) -> None:
        methods = _visible_figure_methods(
            _sort_methods_for_mean_chart(
                [
                    "ca_ccsr_wjdot_otlite20",
                    "codats",
                    "ca_ccsr_wjdot_refine20",
                    "deepjdot",
                ]
            )
        )

        self.assertEqual(methods, ["codats", "ca_ccsr_wjdot_otlite20", "ca_ccsr_wjdot_refine20"])

    def test_include_all_methods_keeps_full_summary_figure_method_set(self) -> None:
        methods = _visible_figure_methods(
            _sort_methods_for_mean_chart(
                [
                    "target_only",
                    "cbtpu_deepjdot",
                    "codats",
                    "source_only",
                    "tpu_deepjdot",
                    "cdan_ts",
                    "deepjdot",
                    "dsan",
                ]
            ),
            include_all_methods=True,
        )

        self.assertEqual(
            methods,
            [
                "source_only",
                "dsan",
                "cdan_ts",
                "codats",
                "deepjdot",
                "tpu_deepjdot",
                "cbtpu_deepjdot",
                "target_only",
            ],
        )

    def test_mode125_paper_order_matches_requested_heatmap_sequence(self) -> None:
        methods = _sort_methods_for_heatmap(
            [
                "target_only",
                "cbtpu_deepjdot",
                "codats",
                "source_only",
                "tpu_deepjdot",
                "cdan_ts",
                "deepjdot",
                "dsan",
            ]
        )

        self.assertEqual(
            methods,
            [
                "source_only",
                "dsan",
                "cdan_ts",
                "codats",
                "deepjdot",
                "tpu_deepjdot",
                "cbtpu_deepjdot",
                "target_only",
            ],
        )

    def test_deepjdot_innovation_method_order_keeps_progressive_chain(self) -> None:
        methods = _sort_methods_for_mean_chart(
            ["cbtp_deepjdot", "source_only", "target_only", "tp_deepjdot", "deepjdot"]
        )

        self.assertEqual(
            methods,
            ["source_only", "deepjdot", "tp_deepjdot", "cbtp_deepjdot", "target_only"],
        )

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

    def test_heatmap_scenario_order_keeps_two_source_before_five_source_stage_sequence(self) -> None:
        def row(scenario_id: str, source_domains: list[str], target_domain: str) -> dict:
            return {
                "scenario_id": scenario_id,
                "source_domains": source_domains,
                "target_domain": target_domain,
            }

        rows = [
            row("mode1-mode2-mode3-mode4-mode6_to_mode5", ["mode1", "mode2", "mode3", "mode4", "mode6"], "mode5"),
            row("mode1-mode2_to_mode5", ["mode1", "mode2"], "mode5"),
            row("mode1-mode5_to_mode2", ["mode1", "mode5"], "mode2"),
            row("mode2-mode3-mode4-mode5-mode6_to_mode1", ["mode2", "mode3", "mode4", "mode5", "mode6"], "mode1"),
            row("mode1-mode2-mode4-mode5-mode6_to_mode3", ["mode1", "mode2", "mode4", "mode5", "mode6"], "mode3"),
            row("mode2-mode5_to_mode1", ["mode2", "mode5"], "mode1"),
            row("mode1-mode2-mode3-mode4-mode5_to_mode6", ["mode1", "mode2", "mode3", "mode4", "mode5"], "mode6"),
            row("mode1-mode3-mode4-mode5-mode6_to_mode2", ["mode1", "mode3", "mode4", "mode5", "mode6"], "mode2"),
            row("mode1-mode2-mode3-mode5-mode6_to_mode4", ["mode1", "mode2", "mode3", "mode5", "mode6"], "mode4"),
        ]

        self.assertEqual(
            _sort_scenarios_for_heatmap(rows),
            [
                "mode1-mode2_to_mode5",
                "mode2-mode5_to_mode1",
                "mode1-mode5_to_mode2",
                "mode2-mode3-mode4-mode5-mode6_to_mode1",
                "mode1-mode3-mode4-mode5-mode6_to_mode2",
                "mode1-mode2-mode4-mode5-mode6_to_mode3",
                "mode1-mode2-mode3-mode5-mode6_to_mode4",
                "mode1-mode2-mode3-mode4-mode6_to_mode5",
                "mode1-mode2-mode3-mode4-mode5_to_mode6",
            ],
        )

    def test_compact_scenario_label_shortens_mode_transition_labels(self) -> None:
        self.assertEqual(_compact_scenario_label("mode1_to_mode4"), "m1_m4")
        self.assertEqual(_compact_scenario_label("mode6_to_mode5"), "m6_m5")
        self.assertEqual(_compact_scenario_label("other_label"), "other_label")

    def test_configure_matplotlib_fonts_uses_times_new_roman(self) -> None:
        fake_plt = mock.Mock()
        fake_plt.rcParams = {}

        with mock.patch("matplotlib.font_manager.findfont", return_value="/fonts/times.ttf"):
            _configure_matplotlib_fonts(fake_plt)

        self.assertEqual(fake_plt.rcParams["font.family"], THESIS_FIGURE_FONT)
        self.assertEqual(fake_plt.rcParams["font.serif"], [THESIS_FIGURE_FONT])
        self.assertEqual(fake_plt.rcParams["svg.fonttype"], "none")
        self.assertEqual(fake_plt.rcParams["pdf.fonttype"], 42)

    def test_primary_figure_format_defaults_to_pdf(self) -> None:
        self.assertEqual(_resolve_primary_figure_format(None), "pdf")

    def test_build_figure_output_path_uses_primary_format_suffix(self) -> None:
        self.assertEqual(_build_figure_output_path(Path("figures/chart")), Path("figures/chart.pdf"))
        self.assertEqual(
            _build_figure_output_path(Path("figures/chart.png"), figure_format="pdf"),
            Path("figures/chart.pdf"),
        )

    def test_save_figure_exports_pdf_only_by_default(self) -> None:
        fake_plt = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chart.pdf"
            with mock.patch(
                "src.evaluation.report_figures._runtime_dependencies",
                return_value=(None, fake_plt, None),
            ):
                _save_figure(output_path)

        self.assertEqual(
            fake_plt.savefig.call_args_list,
            [
                mock.call(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight"),
            ],
        )

    def test_save_figure_keeps_requested_primary_format_and_pdf_companion(self) -> None:
        fake_plt = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chart.svg"
            with mock.patch(
                "src.evaluation.report_figures._runtime_dependencies",
                return_value=(None, fake_plt, None),
            ):
                _save_figure(output_path, figure_format="svg")

        self.assertEqual(
            fake_plt.savefig.call_args_list,
            [
                mock.call(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight"),
                mock.call(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
