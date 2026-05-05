#!/usr/bin/env python
"""Diagnose per-scene method ordering across completed experiment rounds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


MULTISOURCE_METHODS = [
    "source_only",
    "codats",
    "wjdot",
    "ca_ccsr_wjdot_prior20",
    "target_ref",
]
SINGLE_SOURCE_METHODS = [
    "source_only",
    "dsan",
    "cdan_ts",
    "codats",
    "deepjdot",
    "tpu_deepjdot",
    "cbtpu_deepjdot",
    "target_only",
]


def _load_results(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/tables/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        if not isinstance(result, dict):
            continue
        ca_summary = result.get("ca_ccsr_wjdot") if isinstance(result.get("ca_ccsr_wjdot"), dict) else {}
        rows.append(
            {
                "round_root": root.name,
                "result_path": str(path),
                "run_root": payload.get("run_root"),
                "method": str(payload.get("method_name")),
                "method_base_name": str(payload.get("method_base_name", payload.get("method_name"))),
                "setting": payload.get("setting"),
                "scenario_id": payload.get("scenario_id"),
                "scene_label": payload.get("scene_label"),
                "source": ",".join(str(item) for item in payload.get("source_domains", [])),
                "target": payload.get("target_domain"),
                "source_count": len(payload.get("source_domains", [])),
                "source_fold": payload.get("source_fold"),
                "target_fold": payload.get("target_fold"),
                "target_eval_acc": result.get("target_eval_acc"),
                "target_eval_macro_f1": result.get("target_eval_macro_f1"),
                "target_eval_balanced_acc": result.get("target_eval_balanced_acc"),
                "source_eval_acc": result.get("source_eval_acc"),
                "selected_epoch": result.get("selected_epoch"),
                "epochs_requested": result.get("epochs_requested"),
                "model_selection": result.get("model_selection"),
                "selected_model_selection_score": result.get("selected_model_selection_score"),
                "selected_target_train_mean_confidence": result.get("selected_target_train_mean_confidence"),
                "selected_target_train_mean_entropy": result.get("selected_target_train_mean_entropy"),
                "teacher_checkpoint_loaded": result.get("teacher_checkpoint_loaded"),
                "teacher_checkpoint_path": result.get("teacher_checkpoint_path"),
                "student_target_eval_acc": ca_summary.get("student_target_eval_acc"),
                "teacher_target_eval_acc": ca_summary.get("teacher_target_eval_acc"),
                "accuracy_gain_vs_codats": ca_summary.get("accuracy_gain_vs_codats"),
                "accuracy_gain_vs_wjdot": ca_summary.get("accuracy_gain_vs_wjdot"),
                "fusion_base": ca_summary.get("fusion_base"),
                "prior_balance_strength": ca_summary.get("prior_balance_strength"),
                "prior_balance_student_mix": ca_summary.get("prior_balance_student_mix"),
                "override_count": ca_summary.get("override_count"),
                "mean_eta": ca_summary.get("mean_eta"),
            }
        )
    return rows


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(left: Any, right: Any) -> float | None:
    left_value = _as_float(left)
    right_value = _as_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _rank_map(values: dict[str, Any]) -> dict[str, int]:
    valid = [
        (method, _as_float(value))
        for method, value in values.items()
        if _as_float(value) is not None
    ]
    valid.sort(key=lambda item: item[1], reverse=True)
    return {method: rank for rank, (method, _) in enumerate(valid, start=1)}


def _format_float(value: Any) -> str:
    numeric = _as_float(value)
    return "" if numeric is None else f"{numeric:.6f}"


def _pivot(rows: list[dict[str, Any]], methods: list[str]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    pivot: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            continue
        key = (str(row["round_root"]), str(row["scenario_id"]))
        pivot.setdefault(key, {})[method] = row
    return pivot


def build_multisource_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (round_root, scenario_id), method_rows in sorted(_pivot(rows, MULTISOURCE_METHODS).items()):
        accuracies = {
            method: method_rows.get(method, {}).get("target_eval_acc")
            for method in MULTISOURCE_METHODS
        }
        ranks = _rank_map(accuracies)
        prior_row = method_rows.get("ca_ccsr_wjdot_prior20", {})
        target_ref_row = method_rows.get("target_ref", {})
        first_row = next(iter(method_rows.values()))
        row = {
            "round_root": round_root,
            "scenario_id": scenario_id,
            "source": first_row.get("source"),
            "target": first_row.get("target"),
            "source_count": first_row.get("source_count"),
            "source_only_acc": accuracies.get("source_only"),
            "codats_acc": accuracies.get("codats"),
            "wjdot_acc": accuracies.get("wjdot"),
            "prior20_acc": accuracies.get("ca_ccsr_wjdot_prior20"),
            "target_ref_acc": accuracies.get("target_ref"),
            "prior20_minus_codats": _delta(accuracies.get("ca_ccsr_wjdot_prior20"), accuracies.get("codats")),
            "prior20_minus_wjdot": _delta(accuracies.get("ca_ccsr_wjdot_prior20"), accuracies.get("wjdot")),
            "source_only_minus_target_ref": _delta(accuracies.get("source_only"), accuracies.get("target_ref")),
            "target_ref_minus_best_non_ref": None,
            "prior20_rank": ranks.get("ca_ccsr_wjdot_prior20"),
            "target_ref_rank": ranks.get("target_ref"),
            "source_only_rank": ranks.get("source_only"),
            "prior20_beats_codats": _delta(accuracies.get("ca_ccsr_wjdot_prior20"), accuracies.get("codats")) is not None
            and _delta(accuracies.get("ca_ccsr_wjdot_prior20"), accuracies.get("codats")) > 0,
            "prior20_beats_wjdot": _delta(accuracies.get("ca_ccsr_wjdot_prior20"), accuracies.get("wjdot")) is not None
            and _delta(accuracies.get("ca_ccsr_wjdot_prior20"), accuracies.get("wjdot")) > 0,
            "target_ref_is_top": ranks.get("target_ref") == 1,
            "source_only_is_lowest": ranks.get("source_only") == len(ranks),
            "prior20_selected_epoch": prior_row.get("selected_epoch"),
            "prior20_model_selection": prior_row.get("model_selection"),
            "prior20_teacher_checkpoint_loaded": prior_row.get("teacher_checkpoint_loaded"),
            "prior20_student_acc": prior_row.get("student_target_eval_acc"),
            "prior20_teacher_acc": prior_row.get("teacher_target_eval_acc"),
            "prior20_fusion_gain_vs_codats": prior_row.get("accuracy_gain_vs_codats"),
            "prior20_fusion_gain_vs_wjdot": prior_row.get("accuracy_gain_vs_wjdot"),
            "prior20_fusion_base": prior_row.get("fusion_base"),
            "prior20_prior_balance_strength": prior_row.get("prior_balance_strength"),
            "prior20_prior_balance_student_mix": prior_row.get("prior_balance_student_mix"),
            "prior20_override_count": prior_row.get("override_count"),
            "target_ref_model_selection": target_ref_row.get("model_selection"),
            "target_ref_selected_epoch": target_ref_row.get("selected_epoch"),
            "target_ref_epochs_requested": target_ref_row.get("epochs_requested"),
            "target_ref_selected_confidence": target_ref_row.get("selected_target_train_mean_confidence"),
        }
        non_ref_values = [
            _as_float(accuracies.get(method))
            for method in MULTISOURCE_METHODS
            if method != "target_ref"
        ]
        non_ref_values = [value for value in non_ref_values if value is not None]
        if non_ref_values:
            row["target_ref_minus_best_non_ref"] = _delta(accuracies.get("target_ref"), max(non_ref_values))
        output.append(row)
    return output


def build_single_source_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (round_root, scenario_id), method_rows in sorted(_pivot(rows, SINGLE_SOURCE_METHODS).items()):
        accuracies = {
            method: method_rows.get(method, {}).get("target_eval_acc")
            for method in SINGLE_SOURCE_METHODS
        }
        ranks = _rank_map(accuracies)
        uda_values = {
            method: value
            for method, value in accuracies.items()
            if method != "target_only" and _as_float(value) is not None
        }
        best_uda_method = ""
        best_uda_acc = None
        if uda_values:
            best_uda_method, best_uda_acc = max(
                uda_values.items(),
                key=lambda item: _as_float(item[1]) or float("-inf"),
            )
        deepjdot = _as_float(accuracies.get("deepjdot"))
        tpu = _as_float(accuracies.get("tpu_deepjdot"))
        cbtpu = _as_float(accuracies.get("cbtpu_deepjdot"))
        row = {
            "round_root": round_root,
            "scenario_id": scenario_id,
            "source": next(iter(method_rows.values())).get("source"),
            "target": next(iter(method_rows.values())).get("target"),
            "source_only_acc": accuracies.get("source_only"),
            "dsan_acc": accuracies.get("dsan"),
            "cdan_ts_acc": accuracies.get("cdan_ts"),
            "codats_acc": accuracies.get("codats"),
            "deepjdot_acc": accuracies.get("deepjdot"),
            "tpu_deepjdot_acc": accuracies.get("tpu_deepjdot"),
            "cbtpu_deepjdot_acc": accuracies.get("cbtpu_deepjdot"),
            "target_only_acc": accuracies.get("target_only"),
            "tpu_minus_deepjdot": _delta(accuracies.get("tpu_deepjdot"), accuracies.get("deepjdot")),
            "cbtpu_minus_tpu": _delta(accuracies.get("cbtpu_deepjdot"), accuracies.get("tpu_deepjdot")),
            "cbtpu_minus_best_uda": _delta(accuracies.get("cbtpu_deepjdot"), best_uda_acc),
            "best_uda_method": best_uda_method,
            "best_uda_acc": best_uda_acc,
            "chain_ok": deepjdot is not None and tpu is not None and cbtpu is not None and deepjdot < tpu < cbtpu,
            "cbtpu_rank": ranks.get("cbtpu_deepjdot"),
            "target_only_rank": ranks.get("target_only"),
            "target_only_is_top": ranks.get("target_only") == 1,
            "cbtpu_is_best_uda": best_uda_method == "cbtpu_deepjdot",
            "target_only_model_selection": method_rows.get("target_only", {}).get("model_selection"),
            "target_only_selected_epoch": method_rows.get("target_only", {}).get("selected_epoch"),
            "deepjdot_model_selection": method_rows.get("deepjdot", {}).get("model_selection"),
            "tpu_model_selection": method_rows.get("tpu_deepjdot", {}).get("model_selection"),
            "cbtpu_model_selection": method_rows.get("cbtpu_deepjdot", {}).get("model_selection"),
        }
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return None if not values else float(mean(values))


def _count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(_format_float(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_summary_markdown(
    *,
    multisource_rows: list[dict[str, Any]],
    single_source_rows: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    lines: list[str] = [
        "# Experiment Ordering Diagnostics",
        "",
        f"Output directory: `{output_dir}`",
        "",
    ]
    if multisource_rows:
        weak_prior = [
            row
            for row in multisource_rows
            if not row.get("prior20_beats_codats") or not row.get("prior20_beats_wjdot")
        ]
        target_ref_not_top = [row for row in multisource_rows if not row.get("target_ref_is_top")]
        source_only_not_lowest = [row for row in multisource_rows if not row.get("source_only_is_lowest")]
        target_ref_bad_selection = [
            row for row in multisource_rows if row.get("target_ref_model_selection") == "best_source_eval"
        ]
        lines.extend(
            [
                "## Multi-source",
                "",
                f"- Rows: {len(multisource_rows)}",
                f"- Mean prior20 minus CoDATS: {_format_float(_mean_numeric(multisource_rows, 'prior20_minus_codats'))}",
                f"- Mean prior20 minus WJDOT: {_format_float(_mean_numeric(multisource_rows, 'prior20_minus_wjdot'))}",
                f"- prior20 beats both CoDATS and WJDOT: {len(multisource_rows) - len(weak_prior)}/{len(multisource_rows)}",
                f"- target_ref is top: {_count_true(multisource_rows, 'target_ref_is_top')}/{len(multisource_rows)}",
                f"- source_only is lowest: {_count_true(multisource_rows, 'source_only_is_lowest')}/{len(multisource_rows)}",
                f"- target_ref selected by best_source_eval: {len(target_ref_bad_selection)}/{len(multisource_rows)}",
                "",
            ]
        )
        if weak_prior:
            lines.extend(["### Multi-source Weak Prior Scenes", ""])
            lines.extend(
                _markdown_table(
                    weak_prior,
                    [
                        "round_root",
                        "scenario_id",
                        "prior20_minus_codats",
                        "prior20_minus_wjdot",
                        "prior20_student_acc",
                        "prior20_teacher_acc",
                        "prior20_acc",
                    ],
                )
            )
            lines.append("")
        if target_ref_not_top:
            lines.extend(["### Target Ref Not Top", ""])
            lines.extend(
                _markdown_table(
                    target_ref_not_top,
                    [
                        "round_root",
                        "scenario_id",
                        "target_ref_minus_best_non_ref",
                        "target_ref_model_selection",
                        "target_ref_selected_epoch",
                        "target_ref_acc",
                    ],
                )
            )
            lines.append("")
        if source_only_not_lowest:
            lines.extend(["### Source Only Not Lowest", ""])
            lines.extend(
                _markdown_table(
                    source_only_not_lowest,
                    [
                        "round_root",
                        "scenario_id",
                        "source_only_minus_target_ref",
                        "source_only_acc",
                        "target_ref_acc",
                        "source_only_rank",
                    ],
                )
            )
            lines.append("")
    if single_source_rows:
        chain_failures = [row for row in single_source_rows if not row.get("chain_ok")]
        cbtpu_not_best = [row for row in single_source_rows if not row.get("cbtpu_is_best_uda")]
        target_only_not_top = [row for row in single_source_rows if not row.get("target_only_is_top")]
        lines.extend(
            [
                "## Single-source",
                "",
                f"- Rows: {len(single_source_rows)}",
                f"- Mean TPU minus DeepJDOT: {_format_float(_mean_numeric(single_source_rows, 'tpu_minus_deepjdot'))}",
                f"- Mean CBTPU minus TPU: {_format_float(_mean_numeric(single_source_rows, 'cbtpu_minus_tpu'))}",
                f"- Chain DeepJDOT < TPU < CBTPU: {_count_true(single_source_rows, 'chain_ok')}/{len(single_source_rows)}",
                f"- CBTPU best UDA: {_count_true(single_source_rows, 'cbtpu_is_best_uda')}/{len(single_source_rows)}",
                f"- target_only top: {_count_true(single_source_rows, 'target_only_is_top')}/{len(single_source_rows)}",
                "",
            ]
        )
        if chain_failures:
            lines.extend(["### Single-source Chain Failures", ""])
            lines.extend(
                _markdown_table(
                    chain_failures,
                    [
                        "round_root",
                        "scenario_id",
                        "tpu_minus_deepjdot",
                        "cbtpu_minus_tpu",
                        "deepjdot_acc",
                        "tpu_deepjdot_acc",
                        "cbtpu_deepjdot_acc",
                    ],
                )
            )
            lines.append("")
        if cbtpu_not_best:
            lines.extend(["### CBTPU Not Best UDA", ""])
            lines.extend(
                _markdown_table(
                    cbtpu_not_best,
                    [
                        "round_root",
                        "scenario_id",
                        "best_uda_method",
                        "cbtpu_minus_best_uda",
                        "cbtpu_rank",
                    ],
                )
            )
            lines.append("")
        if target_only_not_top:
            lines.extend(["### Target Only Not Top", ""])
            lines.extend(
                _markdown_table(
                    target_only_not_top,
                    ["round_root", "scenario_id", "target_only_rank", "target_only_acc", "best_uda_method"],
                )
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multisource-root", action="append", type=Path, default=[])
    parser.add_argument("--single-source-root", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multisource_results = [
        row
        for root in args.multisource_root
        for row in _load_results(root)
    ]
    single_source_results = [
        row
        for root in args.single_source_root
        for row in _load_results(root)
    ]
    multisource_rows = build_multisource_diagnostics(multisource_results)
    single_source_rows = build_single_source_diagnostics(single_source_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if multisource_rows:
        _write_csv(args.output_dir / "multisource_scene_diagnostics.csv", multisource_rows)
    if single_source_rows:
        _write_csv(args.output_dir / "single_source_scene_diagnostics.csv", single_source_rows)
    summary = build_summary_markdown(
        multisource_rows=multisource_rows,
        single_source_rows=single_source_rows,
        output_dir=args.output_dir,
    )
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
