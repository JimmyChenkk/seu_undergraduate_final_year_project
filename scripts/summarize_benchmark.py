#!/usr/bin/env python
"""Summarize benchmark result.json files for thesis tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.report_figures import METHOD_ORDER
from src.utils.run_layout import find_result_json_paths


METHOD_RANK = {method_name: index for index, method_name in enumerate(METHOD_ORDER)}


def _load_analysis_metrics(path_value: str | None) -> dict[str, float | None]:
    if not path_value:
        return {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    path = Path(path_value)
    if not path.exists():
        return {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    payload = np.load(path, allow_pickle=False)
    labels = payload["target_labels"]
    predictions = payload["target_predictions"]
    if labels.size == 0:
        return {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def _load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in find_result_json_paths(results_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "result" not in payload:
            continue
        result = payload.get("result", {})
        if not isinstance(result, dict) or not payload.get("method_name"):
            continue
        analysis_metrics = _load_analysis_metrics(result.get("analysis_path"))
        rows.append(
            {
                "method": payload.get("method_name"),
                "scene": payload.get("scene_label") or payload.get("scenario_id"),
                "scenario_id": payload.get("scenario_id"),
                "source_domains": payload.get("source_domains", []),
                "target_domain": payload.get("target_domain"),
                "fold": payload.get("selected_fold") or payload.get("fold_name"),
                "seed": payload.get("seed"),
                "accuracy": analysis_metrics["accuracy"] if analysis_metrics["accuracy"] is not None else result.get("target_eval_acc"),
                "macro_f1": analysis_metrics["macro_f1"],
                "balanced_accuracy": analysis_metrics["balanced_accuracy"] if analysis_metrics["balanced_accuracy"] is not None else result.get("target_eval_balanced_acc"),
                "train_seconds": (result.get("timing") or {}).get("train_step_seconds"),
                "total_seconds": (result.get("timing") or {}).get("total_run_seconds"),
                "path": str(path),
            }
        )
    return rows


def _mean_std(values: list[float]) -> dict[str, float | None]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(mean(values)), "std": float(pstdev(values) if len(values) > 1 else 0.0)}


def summarize(results_dir: Path) -> dict:
    rows = _load_rows(results_dir)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["scenario_id"])), []).append(row)

    scenario_methods: dict[str, list[dict]] = {}
    for row in rows:
        scenario_methods.setdefault(str(row["scenario_id"]), []).append(row)

    source_only_by_scene = {
        scene: max((row for row in scene_rows if row["method"] == "source_only"), key=lambda item: item.get("accuracy") or -1)
        for scene, scene_rows in scenario_methods.items()
        if any(row["method"] == "source_only" for row in scene_rows)
    }

    method_summary = {}
    for method in sorted(
        set(str(row["method"]) for row in rows),
        key=lambda name: (METHOD_RANK.get(name, len(METHOD_RANK)), name),
    ):
        method_rows = [row for row in rows if str(row["method"]) == method]
        method_summary[method] = {
            "accuracy": _mean_std([row.get("accuracy") for row in method_rows]),
            "macro_f1": _mean_std([row.get("macro_f1") for row in method_rows]),
            "balanced_accuracy": _mean_std([row.get("balanced_accuracy") for row in method_rows]),
            "negative_transfer_count": sum(
                1
                for row in method_rows
                if row["scenario_id"] in source_only_by_scene
                and row.get("accuracy") is not None
                and source_only_by_scene[row["scenario_id"]].get("accuracy") is not None
                and float(row["accuracy"]) < float(source_only_by_scene[row["scenario_id"]]["accuracy"])
            ),
        }

    rank_rows = []
    for scene, scene_rows in scenario_methods.items():
        valid = [row for row in scene_rows if row.get("accuracy") is not None]
        valid.sort(key=lambda item: float(item["accuracy"]), reverse=True)
        for rank, row in enumerate(valid, start=1):
            rank_rows.append({"scene": scene, "method": row["method"], "rank": rank, "accuracy": row["accuracy"]})
    for method in method_summary:
        ranks = [row["rank"] for row in rank_rows if row["method"] == method]
        method_summary[method]["average_rank"] = float(mean(ranks)) if ranks else None

    return {"row_count": len(rows), "rows": rows, "method_summary": method_summary, "ranks": rank_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(args.results_dir)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
