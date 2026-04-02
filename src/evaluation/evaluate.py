"""Aggregate per-run JSON results into a compact comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TE benchmark result JSON files.")
    parser.add_argument("--results-dir", type=Path, default=Path("runs/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        rows.append(
            {
                "file": path.name,
                "method": payload.get("method_name"),
                "setting": payload.get("setting"),
                "source": ",".join(payload.get("source_domains", [])),
                "target": payload.get("target_domain"),
                "target_eval_acc": result.get("final_target_eval_acc", result.get("target_eval_acc")),
            }
        )

    if not rows:
        print("No result JSON files found.")
        return

    header = ["file", "method", "setting", "source", "target", "target_eval_acc"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[key]) for key in header) + " |")


if __name__ == "__main__":
    main()
