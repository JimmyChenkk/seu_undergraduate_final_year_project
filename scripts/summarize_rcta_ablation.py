"""Summarize staged RCTA ablation batches for paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


STAGE_ORDER = [
    "rcta_a_temporal",
    "rcta_ab_reliable",
    "rcta_abc_full",
    "rcta_m0_base_da",
    "rcta_m1_temporal_mt",
    "rcta_m2_reliability_gate",
    "rcta_m3_dual_proto_static",
    "rcta_m4_full",
]

STAGE_LABELS = {
    "rcta_a_temporal": "A Temporal",
    "rcta_ab_reliable": "A+B Reliability",
    "rcta_abc_full": "A+B+C Full RCTA",
    "rcta_m0_base_da": "M0 Base DA",
    "rcta_m1_temporal_mt": "M1 + MT",
    "rcta_m2_reliability_gate": "M2 + Gate",
    "rcta_m3_dual_proto_static": "M3 + Proto",
    "rcta_m4_full": "M4 Full RCTA",
}

HISTORY_KEYS = [
    "gate_accept_ratio",
    "pseudo_kept_mean_reliability",
    "pseudo_label_kept",
    "pseudo_label_kept_classes",
    "target_prototype_active_classes",
    "source_weight_entropy",
    "source_weight_min",
    "source_weight_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an RCTA mode1/2/5 ablation batch.")
    parser.add_argument("results_dir", type=Path, help="Batch root, e.g. runs/202604xx_rcta_mode125_ablation_fixedfold")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _selected_history(result: dict[str, Any]) -> dict[str, Any]:
    history = result.get("history", [])
    selected_epoch = result.get("selected_epoch")
    if selected_epoch is not None:
        for item in history:
            if item.get("epoch") == selected_epoch:
                return item
    return history[-1] if history else {}


def _find_result_paths(results_dir: Path) -> list[Path]:
    if results_dir.is_file():
        return [results_dir]
    return sorted(results_dir.rglob("tables/result.json"))


def load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _find_result_paths(results_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        method = str(payload.get("method_name") or result.get("method_name") or "")
        if method not in STAGE_ORDER:
            continue
        selected = _selected_history(result)
        row = {
            "method": method,
            "stage": STAGE_LABELS.get(method, method),
            "setting": payload.get("setting"),
            "scenario_id": payload.get("scenario_id"),
            "source_fold": payload.get("source_fold"),
            "target_fold": payload.get("target_fold"),
            "target_eval_acc": _as_float(result.get("selected_target_eval_acc", result.get("target_eval_acc"))),
            "target_eval_balanced_acc": _as_float(result.get("target_eval_balanced_acc")),
            "selected_epoch": result.get("selected_epoch"),
            "run_root": payload.get("run_root"),
        }
        for key in HISTORY_KEYS:
            row[key] = _as_float(selected.get(key))
        rows.append(row)
    return rows


def _active_stage_order(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["method"]) for row in rows}
    ordered = [method for method in STAGE_ORDER if method in present]
    extra = sorted(present - set(STAGE_ORDER))
    return [*ordered, *extra]


def add_stage_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), {})[row["method"]] = row

    enriched: list[dict[str, Any]] = []
    for row in rows:
        scenario_rows = by_scenario[str(row["scenario_id"])]
        scenario_order = _active_stage_order(list(scenario_rows.values()))
        first_method = scenario_order[0] if scenario_order else row["method"]
        base_acc = scenario_rows.get(first_method, {}).get("target_eval_acc")
        row_acc = row.get("target_eval_acc")
        row["first_stage"] = STAGE_LABELS.get(first_method, first_method)
        row["delta_vs_first_stage"] = (
            None if base_acc is None or row_acc is None else float(row_acc - base_acc)
        )
        row["delta_vs_m0"] = (
            row["delta_vs_first_stage"] if first_method == "rcta_m0_base_da" else None
        )
        current_index = scenario_order.index(row["method"])
        if current_index == 0:
            row["delta_vs_prev"] = None
        else:
            prev_acc = scenario_rows.get(scenario_order[current_index - 1], {}).get("target_eval_acc")
            row["delta_vs_prev"] = (
                None if prev_acc is None or row_acc is None else float(row_acc - prev_acc)
            )
        enriched.append(row)
    return enriched


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(mean(clean))


def build_stage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in _active_stage_order(rows):
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        single_rows = [row for row in method_rows if row.get("setting") == "single_source"]
        multi_rows = [row for row in method_rows if row.get("setting") == "multi_source"]
        summary.append(
            {
                "method": method,
                "stage": STAGE_LABELS.get(method, method),
                "runs": len(method_rows),
                "single_source_acc": _mean([row.get("target_eval_acc") for row in single_rows]),
                "multi_source_acc": _mean([row.get("target_eval_acc") for row in multi_rows]),
                "mean_acc": _mean([row.get("target_eval_acc") for row in method_rows]),
                "mean_balanced_acc": _mean([row.get("target_eval_balanced_acc") for row in method_rows]),
                "mean_delta_vs_first_stage": _mean([row.get("delta_vs_first_stage") for row in method_rows]),
                "mean_delta_vs_m0": _mean([row.get("delta_vs_m0") for row in method_rows]),
                "mean_delta_vs_prev": _mean([row.get("delta_vs_prev") for row in method_rows]),
                "mean_gate_accept_ratio": _mean([row.get("gate_accept_ratio") for row in method_rows]),
                "mean_pseudo_reliability": _mean([row.get("pseudo_kept_mean_reliability") for row in method_rows]),
                "mean_kept_classes": _mean([row.get("pseudo_label_kept_classes") for row in method_rows]),
                "mean_source_weight_entropy": _mean([row.get("source_weight_entropy") for row in method_rows]),
            }
        )
    return summary


def build_monotonicity_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), {})[str(row["method"])] = row

    sequences: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for scenario_id in sorted(by_scenario):
        scenario_rows = by_scenario[scenario_id]
        scenario_order = _active_stage_order(list(scenario_rows.values()))
        sequence = []
        previous_item: dict[str, Any] | None = None
        scenario_violations = []
        for method in scenario_order:
            row = scenario_rows[method]
            acc = row.get("target_eval_acc")
            item = {
                "method": method,
                "stage": STAGE_LABELS.get(method, method),
                "target_eval_acc": acc,
            }
            sequence.append(item)
            if (
                previous_item is not None
                and acc is not None
                and previous_item.get("target_eval_acc") is not None
                and float(acc) < float(previous_item["target_eval_acc"])
            ):
                violation = {
                    "scenario_id": scenario_id,
                    "prev_stage": previous_item["stage"],
                    "stage": item["stage"],
                    "prev_acc": previous_item["target_eval_acc"],
                    "acc": acc,
                    "delta": float(acc) - float(previous_item["target_eval_acc"]),
                }
                scenario_violations.append(violation)
                violations.append(violation)
            previous_item = item
        sequences.append(
            {
                "scenario_id": scenario_id,
                "monotonic": not scenario_violations,
                "sequence": sequence,
                "violations": scenario_violations,
            }
        )

    return {
        "scenario_count": len(sequences),
        "monotonic_scenario_count": sum(1 for item in sequences if item["monotonic"]),
        "violation_count": len(violations),
        "sequences": sequences,
        "violations": violations,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_monotonicity_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| scenario_id | monotonic | accuracy_sequence | violations |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.get("sequences", []):
        sequence = " -> ".join(
            f"{stage['stage']}={float(stage['target_eval_acc']):.4f}"
            for stage in item.get("sequence", [])
            if stage.get("target_eval_acc") is not None
        )
        violations = "; ".join(
            f"{violation['prev_stage']}->{violation['stage']} {float(violation['delta']):.4f}"
            for violation in item.get("violations", [])
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("scenario_id")),
                    str(bool(item.get("monotonic"))),
                    sequence,
                    violations,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = add_stage_deltas(load_rows(args.results_dir))
    summary = build_stage_summary(rows)
    monotonicity = build_monotonicity_report(rows)
    output_dir = args.output_dir or args.results_dir / "comparison_summary" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "rcta_ablation_rows.csv", rows)
    write_csv(output_dir / "rcta_ablation_summary.csv", summary)
    (output_dir / "rcta_ablation_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "rcta_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "rcta_ablation_monotonicity.json").write_text(
        json.dumps(monotonicity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "rcta_ablation_summary.md",
        summary,
        [
            "stage",
            "runs",
            "single_source_acc",
            "multi_source_acc",
            "mean_acc",
            "mean_balanced_acc",
            "mean_delta_vs_first_stage",
            "mean_delta_vs_m0",
            "mean_delta_vs_prev",
            "mean_gate_accept_ratio",
            "mean_pseudo_reliability",
            "mean_kept_classes",
            "mean_source_weight_entropy",
        ],
    )
    write_monotonicity_markdown(output_dir / "rcta_ablation_monotonicity.md", monotonicity)
    print(f"Wrote RCTA ablation summary to {output_dir}")


if __name__ == "__main__":
    main()
