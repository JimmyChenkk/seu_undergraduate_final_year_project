"""Aggregate per-run JSON results into comparison tables and round reviews."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.report_figures import FINAL_MAIN_METHODS, MAIN_TABLE_METHOD_PREFIXES, METHOD_ORDER
from src.evaluation.review import extract_core_metrics, load_review
from src.utils.run_layout import find_result_json_paths, resolve_comparison_root


METHOD_RANK = {method_name: index for index, method_name in enumerate(METHOD_ORDER)}
MAIN_TABLE_METHODS = set(FINAL_MAIN_METHODS)


def _display_method_name(method_name: Any) -> str:
    normalized = str(method_name)
    if normalized == "target_only":
        return "target_ref"
    return normalized


def _method_sort_anchor(method_name: Any) -> str:
    display_name = _display_method_name(method_name)
    for prefix in MAIN_TABLE_METHOD_PREFIXES:
        if display_name.startswith(prefix):
            return prefix.rstrip("_")
    return display_name


def _is_main_table_method(method_name: Any) -> bool:
    display_name = _display_method_name(method_name)
    return display_name in MAIN_TABLE_METHODS or any(
        display_name.startswith(prefix) for prefix in MAIN_TABLE_METHOD_PREFIXES
    )


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


def _relative_or_name(path: Path, base_dir: Path) -> str:
    if base_dir.is_dir() and path.is_relative_to(base_dir):
        return str(path.relative_to(base_dir))
    return path.name


def _load_run_review(result_payload: dict[str, Any], result_path: Path) -> dict[str, Any] | None:
    review_path_value = result_payload.get("review_path")
    review_path = Path(str(review_path_value)) if review_path_value else result_path.parent / "review.json"
    return load_review(review_path)


def build_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in find_result_json_paths(results_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "result" not in payload:
            continue
        result = payload.get("result", {})
        if not isinstance(result, dict) or not payload.get("method_name"):
            continue

        metrics = extract_core_metrics(payload)
        review_payload = _load_run_review(payload, path)
        figure_paths = payload.get("figure_paths", {})
        method_name = _display_method_name(payload.get("method_name"))
        if not _is_main_table_method(method_name):
            continue
        rows.append(
            {
                "file": _relative_or_name(path, results_dir),
                "method": method_name,
                "setting": payload.get("setting"),
                "scenario_id": payload.get("scenario_id"),
                "backbone": payload.get("backbone_name"),
                "fold": payload.get("fold_name"),
                "selected_fold": payload.get("selected_fold", payload.get("fold_name")),
                "source_fold": payload.get("source_fold"),
                "target_fold": payload.get("target_fold"),
                "fold_strategy": payload.get("fold_strategy"),
                "random_fold_enabled": payload.get("random_fold_enabled"),
                "source": ",".join(payload.get("source_domains", [])),
                "target": payload.get("target_domain"),
                "source_train_acc": metrics["source_train_acc"],
                "source_eval_acc": metrics["source_eval_acc"],
                "target_eval_acc": metrics["target_eval_acc"],
                "target_eval_macro_f1": result.get("target_eval_macro_f1"),
                "target_eval_balanced_acc": metrics["target_eval_balanced_acc"],
                "run_root": payload.get("run_root"),
                "figure_paths": figure_paths,
                "review_flags": review_payload.get("flags", {}) if review_payload else {},
                "next_focus": review_payload.get("next_focus", []) if review_payload else [],
            }
        )
    return rows


def annotate_rows_with_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_rows = {
        row["scenario_id"]: row
        for row in rows
        if row.get("method") == "source_only"
    }

    annotated: list[dict[str, Any]] = []
    for row in rows:
        baseline = baseline_rows.get(row["scenario_id"])
        baseline_target = baseline.get("target_eval_acc") if baseline else None
        target_value = row.get("target_eval_acc")
        delta = None
        if baseline_target is not None and target_value is not None:
            delta = float(target_value - baseline_target)
        annotated.append(
            {
                **row,
                "delta_target_vs_source_only": delta,
            }
        )
    return annotated


def sort_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_rank: dict[str, int] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id"))
        scenario_rank.setdefault(scenario_id, len(scenario_rank))

    return sorted(
        rows,
        key=lambda row: (
            scenario_rank[str(row.get("scenario_id"))],
            METHOD_RANK.get(_method_sort_anchor(row.get("method")), len(METHOD_RANK)),
            str(row.get("method")),
        ),
    )


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = [
        "method",
        "scenario_id",
        "source_train_acc",
        "source_eval_acc",
        "target_eval_acc",
        "delta_target_vs_source_only",
        "backbone",
        "source_fold",
        "target_fold",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key)) for key in header) + " |")
    return "\n".join(lines)


def build_round_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["scenario_id"]),
            METHOD_RANK.get(_method_sort_anchor(item["method"]), len(METHOD_RANK)),
            str(item["method"]),
        ),
    ):
        item = {
            "method": row["method"],
            "scenario_id": row["scenario_id"],
            "source_train_acc": row["source_train_acc"],
            "source_eval_acc": row["source_eval_acc"],
            "target_eval_acc": row["target_eval_acc"],
            "delta_target_vs_source_only": row["delta_target_vs_source_only"],
            "source_fold": row.get("source_fold"),
            "target_fold": row.get("target_fold"),
            "fold_strategy": row.get("fold_strategy"),
            "review_flags": row["review_flags"],
            "next_focus": row["next_focus"],
            "run_root": row["run_root"],
            "figure_paths": row["figure_paths"],
        }
        items.append(item)

    best_by_scenario = {}
    for row in rows:
        scenario_id = str(row["scenario_id"])
        target_eval_acc = row.get("target_eval_acc")
        if target_eval_acc is None:
            continue
        best_row = best_by_scenario.get(scenario_id)
        if best_row is None or float(target_eval_acc) > float(best_row["target_eval_acc"]):
            best_by_scenario[scenario_id] = {
                "method": row["method"],
                "target_eval_acc": float(target_eval_acc),
                "delta_target_vs_source_only": row.get("delta_target_vs_source_only"),
            }

    recommended_rechecks = []
    for item in items:
        if item["method"] == "source_only":
            continue
        flags = item["review_flags"]
        delta = item["delta_target_vs_source_only"]
        if flags.get("source_training_weak"):
            reason = "source_training_weak"
        elif flags.get("source_generalization_weak"):
            reason = "source_generalization_weak"
        elif flags.get("target_transfer_weak"):
            reason = "target_transfer_weak"
        elif delta is not None and delta < 0:
            reason = "underperforming_baseline"
        else:
            continue
        recommended_rechecks.append(
            {
                "method": item["method"],
                "scenario_id": item["scenario_id"],
                "reason": reason,
                "next_focus": item["next_focus"],
            }
        )

    return {
        "row_count": len(rows),
        "baseline_method": "source_only",
        "scenarios": sorted(set(str(row["scenario_id"]) for row in rows)),
        "items": items,
        "best_by_scenario": best_by_scenario,
        "recommended_rechecks": recommended_rechecks,
    }


def _resolve_summary_dir(results_dir: Path, summary_dir: Path | None) -> Path | None:
    if summary_dir is not None:
        return summary_dir
    comparison_root = resolve_comparison_root(results_dir)
    if comparison_root is None:
        return None
    return comparison_root / "tables"


def _save_summary(summary_dir: Path, rows: list[dict[str, Any]], markdown_table: str) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    comparison_payload = {
        "row_count": len(rows),
        "rows": rows,
    }
    (summary_dir / "comparison.json").write_text(
        json.dumps(comparison_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (summary_dir / "comparison.md").write_text(markdown_table + "\n", encoding="utf-8")
    (summary_dir / "round_review.json").write_text(
        json.dumps(build_round_review(rows), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "method",
        "setting",
        "scenario_id",
        "source",
        "target",
        "source_fold",
        "target_fold",
        "target_eval_acc",
        "target_eval_macro_f1",
        "target_eval_balanced_acc",
        "delta_target_vs_source_only",
        "run_root",
    ]
    for filename in ("comparison.csv", "multisource_summary.csv"):
        with (summary_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
    with (summary_dir / "multisource_v2_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    capacity_methods = {
        "codats_128",
        "codats_500",
        "pooled_wjdot_128",
        "pooled_wjdot_500",
        "sourceaware_wjdot_128",
        "sourceaware_wjdot_500",
        "sa_ccsr_wjdot_train_128",
        "sa_ccsr_wjdot_train_500",
    }
    with (summary_dir / "capacity_fairness_probe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("method") in capacity_methods:
                writer.writerow({key: row.get(key) for key in fieldnames})

    by_scene_method = {
        (str(row.get("scenario_id")), str(row.get("method"))): row
        for row in rows
    }
    compare_fields = [
        "scenario_id",
        "pooled_wjdot_acc",
        "sourceaware_shared_head_acc",
        "sourceaware_multi_head_acc",
        "delta_shared_vs_pooled",
        "delta_multi_vs_pooled",
    ]
    with (summary_dir / "sourceaware_vs_pooled_wjdot.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compare_fields)
        writer.writeheader()
        for scenario_id in sorted({str(row.get("scenario_id")) for row in rows}):
            pooled = by_scene_method.get((scenario_id, "pooled_wjdot")) or by_scene_method.get(
                (scenario_id, "pooled_wjdot_128")
            )
            shared = by_scene_method.get((scenario_id, "sourceaware_wjdot_shared_head"))
            multi = by_scene_method.get((scenario_id, "sourceaware_wjdot_multi_head")) or by_scene_method.get(
                (scenario_id, "sourceaware_wjdot_128")
            )
            pooled_acc = pooled.get("target_eval_acc") if pooled else None
            shared_acc = shared.get("target_eval_acc") if shared else None
            multi_acc = multi.get("target_eval_acc") if multi else None
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "pooled_wjdot_acc": pooled_acc,
                    "sourceaware_shared_head_acc": shared_acc,
                    "sourceaware_multi_head_acc": multi_acc,
                    "delta_shared_vs_pooled": (
                        None if pooled_acc is None or shared_acc is None else float(shared_acc - pooled_acc)
                    ),
                    "delta_multi_vs_pooled": (
                        None if pooled_acc is None or multi_acc is None else float(multi_acc - pooled_acc)
                    ),
                }
            )

    gate_fields = [
        "scenario_id",
        "ccsr_raw_acc",
        "ccsr_safe_acc",
        "ccsr_calibrated_override_acc",
        "delta_calibrated_vs_safe",
        "delta_raw_vs_safe",
    ]
    with (summary_dir / "ccsr_gate_ablation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=gate_fields)
        writer.writeheader()
        for scenario_id in sorted({str(row.get("scenario_id")) for row in rows}):
            raw = by_scene_method.get((scenario_id, "ccsr_raw"))
            safe = by_scene_method.get((scenario_id, "ccsr_safe"))
            calibrated = by_scene_method.get((scenario_id, "ccsr_calibrated_override"))
            raw_acc = raw.get("target_eval_acc") if raw else None
            safe_acc = safe.get("target_eval_acc") if safe else None
            calibrated_acc = calibrated.get("target_eval_acc") if calibrated else None
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "ccsr_raw_acc": raw_acc,
                    "ccsr_safe_acc": safe_acc,
                    "ccsr_calibrated_override_acc": calibrated_acc,
                    "delta_calibrated_vs_safe": (
                        None if calibrated_acc is None or safe_acc is None else float(calibrated_acc - safe_acc)
                    ),
                    "delta_raw_vs_safe": None if raw_acc is None or safe_acc is None else float(raw_acc - safe_acc),
                }
            )


def export_comparison_summary(results_dir: Path, summary_dir: Path | None = None) -> Path | None:
    rows = sort_comparison_rows(annotate_rows_with_baseline(build_rows(results_dir)))
    if not rows:
        return None
    resolved_summary_dir = _resolve_summary_dir(results_dir, summary_dir)
    if resolved_summary_dir is None:
        return None
    _save_summary(resolved_summary_dir, rows, render_markdown_table(rows))
    return resolved_summary_dir


def main() -> None:
    args = parse_args()
    rows = sort_comparison_rows(annotate_rows_with_baseline(build_rows(args.results_dir)))

    if not rows:
        print("No result JSON files found.")
        return

    markdown_table = render_markdown_table(rows)
    print(markdown_table)

    summary_dir = export_comparison_summary(args.results_dir, args.summary_dir)
    if summary_dir is not None:
        print(f"\nSaved comparison summary to {summary_dir}")


if __name__ == "__main__":
    main()
