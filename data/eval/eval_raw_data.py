from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_TOP_LEVEL_KEYS = ("Signals", "Labels", "Folds")


def canonicalize_domain_id(name: str) -> str:
    stem = Path(name).stem.lower()
    match = re.search(r"mode\s*[_-]?\s*(\d+)", stem)
    if match:
        return f"mode{int(match.group(1))}"
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return f"mode{int(digits)}"
    return stem


def summarize_array(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    summary: dict[str, Any] = {
        "python_type": type(value).__name__,
        "shape": [int(x) for x in shape] if shape is not None else None,
        "length": None,
        "dtype": str(dtype) if dtype is not None else None,
    }
    try:
        summary["length"] = len(value)
    except Exception:
        pass
    return summary


def summarize_labels(labels: Any) -> dict[str, Any]:
    summary = summarize_array(labels)
    if isinstance(labels, np.ndarray) and labels.size > 0:
        unique = np.unique(labels)
        summary["label_stats"] = {
            "size": int(labels.size),
            "min": int(labels.min()),
            "max": int(labels.max()),
            "unique_count": int(unique.size),
            "unique_values_preview": [int(v) for v in unique[:20]],
        }
        if unique.size > 20:
            summary["label_stats"]["unique_values_truncated"] = True
    return summary


def summarize_folds(folds: Any) -> dict[str, Any]:
    summary = summarize_array(folds)
    if not isinstance(folds, dict):
        summary["fold_names"] = []
        summary["fold_count"] = 0
        return summary
    entries: dict[str, Any] = {}
    for fold_name, indices in folds.items():
        fold_summary = summarize_array(indices)
        if isinstance(indices, np.ndarray) and indices.size > 0:
            fold_summary["index_range"] = {
                "size": int(indices.size),
                "min": int(indices.min()),
                "max": int(indices.max()),
            }
        entries[str(fold_name)] = fold_summary
    summary["fold_names"] = list(entries.keys())
    summary["fold_count"] = len(entries)
    summary["entries"] = entries
    return summary


def inspect_pickle(path: Path, raw_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file_name": path.name,
        "domain_id": canonicalize_domain_id(path.name),
        "source_path": str(path.relative_to(raw_dir.parent)),
        "size_bytes": path.stat().st_size,
    }
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["python_type"] = type(payload).__name__
    if not isinstance(payload, dict):
        result["top_level_keys"] = []
        result["root_summary"] = summarize_array(payload)
        return result

    result["top_level_keys"] = [str(k) for k in payload.keys()]
    result["required_keys_present"] = {key: key in payload for key in REQUIRED_TOP_LEVEL_KEYS}
    key_summaries: dict[str, Any] = {}
    for key, value in payload.items():
        key_name = str(key)
        if key_name == "Labels":
            key_summaries[key_name] = summarize_labels(value)
        elif key_name == "Folds":
            key_summaries[key_name] = summarize_folds(value)
        else:
            key_summaries[key_name] = summarize_array(value)
    result["key_summaries"] = key_summaries
    return result


def inspect_raw_dir(raw_dir: Path, pattern: str = "*.pickle") -> dict[str, Any]:
    files = sorted(raw_dir.glob(pattern)) if raw_dir.exists() else []
    files = [p for p in files if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".Zone.Identifier")]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "file_count": len(files),
        "files": [inspect_pickle(path, raw_dir) for path in files],
    }


def render_markdown(inspection: dict[str, Any]) -> str:
    lines: list[str] = [
        "# TE Raw Data Inspection",
        "",
        f"Generated at: `{inspection.get('generated_at')}`",
        f"Raw dir: `{inspection.get('raw_dir')}`",
        "",
        "## File Summary",
        "",
        "| File | Domain | Type | Top-level keys | Error |",
        "| --- | --- | --- | --- | --- |",
    ]
    files = inspection.get("files", [])
    for item in files:
        keys = ", ".join(item.get("top_level_keys", [])) or "N/A"
        lines.append(
            f"| {item['file_name']} | {item.get('domain_id', 'N/A')} | {item.get('python_type', 'N/A')} | {keys} | {item.get('error', 'None')} |"
        )

    all_signals_shapes: set[tuple[int, ...]] = set()
    all_folds: set[str] = set()
    label_counter: Counter[int] = Counter()

    for item in files:
        lines.extend([
            "",
            f"## {item['file_name']}",
            "",
            f"- Domain ID: `{item.get('domain_id', 'N/A')}`",
            f"- Source path: `{item.get('source_path', 'N/A')}`",
            f"- Size (bytes): `{item.get('size_bytes', 'N/A')}`",
            f"- Python type: `{item.get('python_type', 'N/A')}`",
        ])
        if item.get("error"):
            lines.append(f"- Error: `{item['error']}`")
            continue
        lines.append(f"- Required keys present: `{item.get('required_keys_present', {})}`")
        lines.append("")
        lines.append("| Key | Type | Shape | Length | Dtype | Extra |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for key in item.get("top_level_keys", []):
            summary = item.get("key_summaries", {}).get(key, {})
            extra = []
            if key == "Signals" and summary.get("shape"):
                all_signals_shapes.add(tuple(summary["shape"]))
            if key == "Labels" and summary.get("label_stats"):
                preview = summary["label_stats"].get("unique_values_preview", [])
                label_counter.update(int(v) for v in preview)
            if key == "Folds":
                all_folds.update(summary.get("fold_names", []))
                extra.append(f"fold_names={summary.get('fold_names', [])}")
                extra.append(f"fold_count={summary.get('fold_count', 0)}")
            lines.append(
                f"| {key} | {summary.get('python_type')} | {summary.get('shape')} | {summary.get('length')} | {summary.get('dtype')} | {'; '.join(extra) if extra else 'None'} |"
            )
            if key == "Folds" and summary.get("entries"):
                lines.append("")
                lines.append("| Fold | Shape | Length | Dtype | Index range |")
                lines.append("| --- | --- | --- | --- | --- |")
                for fold_name, fold_summary in summary["entries"].items():
                    lines.append(
                        f"| {fold_name} | {fold_summary.get('shape')} | {fold_summary.get('length')} | {fold_summary.get('dtype')} | {fold_summary.get('index_range', 'None')} |"
                    )

    lines.extend([
        "",
        "## Quick takeaways",
        "",
        f"- Signals shape variants: `{sorted(all_signals_shapes) if all_signals_shapes else 'N/A'}`",
        f"- Fold names observed: `{sorted(all_folds) if all_folds else 'N/A'}`",
        f"- Label preview frequency (not class counts): `{dict(label_counter) if label_counter else 'N/A'}`",
        "- Next step: inspect domain-wise sample counts, class balance, and signal normalization strategy.",
        "",
        "## Mermaid overview",
        "",
        "```mermaid",
        "flowchart TD",
        "    A[Raw pickle files in data/raw] --> B[pickle.load(handle)]",
        "    B --> C{Top-level dict?}",
        "    C -->|Yes| D[Signals array]",
        "    C -->|Yes| E[Labels array]",
        "    C -->|Yes| F[Folds dict]",
        "    D --> G[Benchmark manifest / dataset loader]",
        "    E --> G",
        "    F --> G",
        "    G --> H[Normalization / split selection]",
        "    H --> I[Single-source or multi-source batches]",
        "    I --> J[Training methods: source_only / coral / dan / dann / cdan / deepjdot / rcta]",
        "```",
    ])
    return "\n".join(lines) + "\n"


def build_mermaid() -> str:
    return "\n".join([
        "flowchart TD",
        "    A[Raw pickle files in data/raw] --> B[pickle.load(handle)]",
        "    B --> C{Top-level dict?}",
        "    C -->|Yes| D[Signals array]",
        "    C -->|Yes| E[Labels array]",
        "    C -->|Yes| F[Folds dict]",
        "    D --> G[Benchmark manifest / dataset loader]",
        "    E --> G",
        "    F --> G",
        "    G --> H[Normalization / split selection]",
        "    H --> I[Single-source or multi-source batches]",
        "    I --> J[Training methods: source_only / coral / dan / dann / cdan / deepjdot / rcta]",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TE raw pickle files and render a markdown report.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-md", type=Path, default=Path("data/eval/raw_data_report.md"))
    parser.add_argument("--output-json", type=Path, default=Path("data/eval/raw_data_report.json"))
    parser.add_argument("--output-mermaid", type=Path, default=Path("data/eval/raw_data_flow.mmd"))
    args = parser.parse_args()

    inspection = inspect_raw_dir(args.raw_dir)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_mermaid.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(inspection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(inspection), encoding="utf-8")
    args.output_mermaid.write_text(build_mermaid() + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_mermaid}")


if __name__ == "__main__":
    main()
