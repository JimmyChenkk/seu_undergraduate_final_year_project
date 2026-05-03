#!/usr/bin/env python
"""Export thesis placeholder figures and optional benchmark figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PLACEHOLDERS = {
    "tc_cdan_confusion_placeholder.svg": "TC-CDAN vs Source-Only confusion matrix\n示意图，待实验结果替换",
    "tc_cdan_tsne_domain_placeholder.svg": "TC-CDAN domain t-SNE/UMAP\n示意图，待实验结果替换",
    "tc_cdan_tsne_class_placeholder.svg": "TC-CDAN class t-SNE/UMAP\n示意图，待实验结果替换",
    "tc_cdan_metrics_placeholder.svg": "TC-CDAN metric bars/curves\n待实验结果替换",
    "rpl_tc_cdan_confusion_placeholder.svg": "RPL-TC-CDAN vs Source-Only confusion matrix\n示意图，待实验结果替换",
    "rpl_tc_cdan_tsne_domain_placeholder.svg": "RPL-TC-CDAN domain t-SNE/UMAP\n示意图，待实验结果替换",
    "rpl_tc_cdan_tsne_class_placeholder.svg": "RPL-TC-CDAN class t-SNE/UMAP\n示意图，待实验结果替换",
    "rpl_tc_cdan_pseudo_acceptance_placeholder.svg": "RPL-TC-CDAN pseudo-label acceptance rate\n待实验结果替换",
    "ccs_rpl_tc_cdan_confusion_placeholder.svg": "CCS-RPL-TC-CDAN vs Source-Only confusion matrix\n示意图，待实验结果替换",
    "ccs_rpl_tc_cdan_tsne_domain_placeholder.svg": "CCS-RPL-TC-CDAN domain t-SNE/UMAP\n示意图，待实验结果替换",
    "ccs_rpl_tc_cdan_tsne_class_placeholder.svg": "CCS-RPL-TC-CDAN class t-SNE/UMAP\n示意图，待实验结果替换",
    "ccs_rpl_tc_cdan_prototype_distance_placeholder.svg": "CCS-RPL-TC-CDAN prototype distance / inter-intra ratio\n待实验结果替换",
    "benchmark_90_table_placeholder.svg": "90-run benchmark table view\n待实验结果替换",
    "ablation_table_placeholder.svg": "Progressive ablation table\n待实验结果替换",
    "deepjdot_stability_placeholder.svg": "DeepJDOT stability curves\n待实验结果替换",
}


def export_placeholders(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in PLACEHOLDERS.items():
        lines = text.splitlines()
        tspan = "\n".join(
            f'<tspan x="360" dy="{0 if index == 0 else 28}">{line}</tspan>'
            for index, line in enumerate(lines)
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <rect width="720" height="420" fill="#ffffff"/>
  <rect x="72" y="86" width="576" height="248" rx="8" fill="#f7f7f7" stroke="#555555" stroke-width="2"/>
  <text x="360" y="184" text-anchor="middle" font-family="'Times New Roman', Times, serif" font-size="22" fill="#222222">
{tspan}
  </text>
</svg>
"""
        (output_dir / filename).write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figs/placeholders"))
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_placeholders(args.output_dir)
    if args.results_dir is not None:
        from src.evaluation.report_figures import export_summary_figures

        export_summary_figures(args.results_dir, args.output_dir / "benchmark")
    print(f"Wrote thesis placeholder figures to {args.output_dir}")


if __name__ == "__main__":
    main()
