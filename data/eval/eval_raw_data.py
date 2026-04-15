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

SCRIPT_PATH = Path(__file__).resolve()
DATA_EVAL_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_MD = DATA_EVAL_DIR / "raw_data_report.md"
DEFAULT_OUTPUT_JSON = DATA_EVAL_DIR / "raw_data_report.json"

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
        "# TE Raw Data Inspection / TE 原始数据检查",
        "",
        f"Generated at / 生成时间: `{inspection.get('generated_at')}`",
        f"Raw dir / 原始数据目录: `{inspection.get('raw_dir')}`",
        "",
        "## File Summary / 文件汇总",
        "",
        "| File / 文件 | Domain / 域 | Type / 类型 | Top-level keys / 顶层键 | Error / 错误 |",
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
            f"## {item['file_name']} / 文件详情",
            "",
            f"- Domain ID / 域编号: `{item.get('domain_id', 'N/A')}`",
            f"- Source path / 源路径: `{item.get('source_path', 'N/A')}`",
            f"- Size (bytes) / 文件大小(字节): `{item.get('size_bytes', 'N/A')}`",
            f"- Python type / Python类型: `{item.get('python_type', 'N/A')}`",
        ])
        if item.get("error"):
            lines.append(f"- Error: `{item['error']}`")
            continue
        lines.append(f"- Required keys present / 必需键是否存在: `{item.get('required_keys_present', {})}`")
        lines.append("")
        lines.append("| Key / 键 | Type / 类型 | Shape / 形状 | Length / 长度 | Dtype / 数据类型 | Extra / 额外信息 |")
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
        "## Mermaid overview / 流程总览",
        "",
        "```mermaid",
        "flowchart TD",
        "    A[\"Raw pickle files\"] --> B[\"pickle load\"]",
        "    B --> C{\"Top-level dict?\"}",
        "    C -->|Yes| D[\"Signals\"]",
        "    C -->|Yes| E[\"Labels\"]",
        "    C -->|Yes| F[\"Folds\"]",
        "    D --> G[\"Benchmark manifest\"]",
        "    E --> G",
        "    F --> G",
        "    G --> H[\"Domain resolver\"]",
        "    H --> I[\"Normalization and fold selection\"]",
        "    I --> J[\"Single-source pipeline\"]",
        "    I --> K[\"Multi-source pipeline\"]",
        "    J --> L[\"Training methods\"]",
        "    K --> L",
        "    L --> M[\"source_only coral dan dann cdan deepjdot rcta\"]",
        "```",
        "",
        "## Raw data structure explained / 原始数据结构说明",
        "",
        "```mermaid",
        "erDiagram",
        "    PICKLE_FILE ||--|| SIGNALS : contains",
        "    PICKLE_FILE ||--|| LABELS : contains",
        "    PICKLE_FILE ||--|| FOLDS : contains",
        "    SIGNALS {",
        "        int sample_count",
        "        int time_steps",
        "        int channels",
        "    }",
        "    LABELS {",
        "        int sample_count",
        "        int[] label_values",
        "    }",
        "    FOLDS {",
        "        string fold_name",
        "        int[] sample_indices",
        "    }",
        "    PICKLE_FILE {",
        "        string file_name",
        "        string domain_id",
        "    }",
        "```",
    ])
    return "\n".join(lines) + "\n"
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TE raw pickle files and render a markdown report.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    inspection = inspect_raw_dir(args.raw_dir)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(inspection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(inspection), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
