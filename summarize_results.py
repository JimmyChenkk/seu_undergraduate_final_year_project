"""Summarize TEP DA result CSV files by task and method."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ["accuracy", "macro_f1", "balanced_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TEP DA CSV results.")
    parser.add_argument("results_csv", nargs="+", type=Path, help="One or more result CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tep_ot_summary"))
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    return rows


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_csv)
    by_task_method: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task_method[(row["task"], row["method"])].append(row)

    summary_rows = []
    for (task, method), group in sorted(by_task_method.items()):
        output = {"task": task, "method": method, "n": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) not in {"", None}]
            metric_mean, metric_std = _stats(values)
            output[f"{metric}_mean"] = metric_mean
            output[f"{metric}_std"] = metric_std
        summary_rows.append(output)

    by_method_task_means: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        method = str(row["method"])
        for metric in METRICS:
            by_method_task_means[method][metric].append(float(row[f"{metric}_mean"]))

    overall_rows = []
    for method, metric_values in sorted(by_method_task_means.items()):
        output = {"method": method, "task_count": len(metric_values[METRICS[0]])}
        for metric in METRICS:
            metric_mean, metric_std = _stats(metric_values[metric])
            output[f"{metric}_mean"] = metric_mean
            output[f"{metric}_std"] = metric_std
        overall_rows.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_task_path = args.output_dir / "summary_by_task_method.csv"
    overall_path = args.output_dir / "summary_overall.csv"
    write_csv(by_task_path, summary_rows, list(summary_rows[0].keys()) if summary_rows else ["task", "method", "n"])
    write_csv(overall_path, overall_rows, list(overall_rows[0].keys()) if overall_rows else ["method", "task_count"])

    print(f"task-method summary: {by_task_path}")
    print(f"overall summary: {overall_path}")
    for row in overall_rows:
        print(
            f"{row['method']}: acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}, "
            f"macro_f1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}, "
            f"bal_acc={row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f}"
        )


if __name__ == "__main__":
    main()
