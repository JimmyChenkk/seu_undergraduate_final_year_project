"""Aggregate per-run JSON results into a compact comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.run_layout import find_result_json_paths, resolve_comparison_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TE benchmark result JSON files.")
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Optional directory to save comparison JSON/Markdown files.",
    )
    return parser.parse_args()


def _build_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in find_result_json_paths(results_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        if not isinstance(result, dict) or not payload.get("method_name"):
            continue
        rows.append(
            {
                "file": str(path.relative_to(results_dir)) if results_dir.is_dir() and path.is_relative_to(results_dir) else path.name,
                "method": payload.get("method_name"),
                "setting": payload.get("setting"),
                "scenario_id": payload.get("scenario_id"),
                "backbone": payload.get("backbone_name"),
                "fold": payload.get("fold_name"),
                "source": ",".join(payload.get("source_domains", [])),
                "target": payload.get("target_domain"),
                "target_eval_acc": result.get("final_target_eval_acc", result.get("target_eval_acc")),
                "run_root": payload.get("run_root"),
            }
        )
    return rows


def _render_markdown_table(rows: list[dict]) -> str:
    header = ["file", "method", "setting", "scenario_id", "backbone", "fold", "source", "target", "target_eval_acc"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key)) for key in header) + " |")
    return "\n".join(lines)


def _save_summary(summary_dir: Path, rows: list[dict], markdown_table: str) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "row_count": len(rows),
        "rows": rows,
    }
    (summary_dir / "comparison.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (summary_dir / "comparison.md").write_text(markdown_table + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = _build_rows(args.results_dir)

    if not rows:
        print("No result JSON files found.")
        return

    markdown_table = _render_markdown_table(rows)
    print(markdown_table)

    summary_dir = args.summary_dir
    if summary_dir is None and len(rows) > 1:
        comparison_root = resolve_comparison_root(args.results_dir)
        if comparison_root is not None:
            summary_dir = comparison_root / "tables"
    if summary_dir is not None:
        _save_summary(summary_dir, rows, markdown_table)
        print(f"\nSaved comparison summary to {summary_dir}")


if __name__ == "__main__":
    main()
