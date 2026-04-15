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


def sample_preview(value: Any, *, max_items: int = 3) -> Any:
    """Return a compact JSON-friendly preview of a large value."""

    if isinstance(value, np.ndarray):
        preview: dict[str, Any] = {
            "python_type": "ndarray",
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype),
        }
        if value.size == 0:
            preview["example"] = []
            return preview
        flat = value.reshape(-1)
        head = flat[:max_items]
        preview["example"] = [item.item() if hasattr(item, "item") else item for item in head]
        return preview
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        return {str(key): sample_preview(val, max_items=max_items) for key, val in items}
    if isinstance(value, (list, tuple)):
        return [sample_preview(item, max_items=max_items) for item in list(value)[:max_items]]
    return value


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
        result["root_preview"] = sample_preview(payload)
        return result

    result["top_level_keys"] = [str(k) for k in payload.keys()]
    result["required_keys_present"] = {key: key in payload for key in REQUIRED_TOP_LEVEL_KEYS}
    key_summaries: dict[str, Any] = {}
    key_previews: dict[str, Any] = {}
    for key, value in payload.items():
        key_name = str(key)
        if key_name == "Labels":
            key_summaries[key_name] = summarize_labels(value)
        elif key_name == "Folds":
            key_summaries[key_name] = summarize_folds(value)
        else:
            key_summaries[key_name] = summarize_array(value)
        key_previews[key_name] = sample_preview(value)
    result["key_summaries"] = key_summaries
    result["key_previews"] = key_previews
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
    def fmt_shape(summary: dict[str, Any] | None) -> str:
        shape = (summary or {}).get("shape")
        return " × ".join(str(x) for x in shape) if shape else "N/A"

    def shape_dim(summary: dict[str, Any] | None, index: int) -> str:
        shape = (summary or {}).get("shape")
        if not shape or len(shape) <= index:
            return "N/A"
        return str(shape[index])

    files = inspection.get("files", [])
    first = next((item for item in files if not item.get("error")), {})
    first_summaries = first.get("key_summaries", {}) if isinstance(first, dict) else {}
    first_previews = first.get("key_previews", {}) if isinstance(first, dict) else {}
    signals_summary = first_summaries.get("Signals", {})
    labels_summary = first_summaries.get("Labels", {})
    folds_summary = first_summaries.get("Folds", {})

    signals_preview = first_previews.get("Signals")
    labels_preview = first_previews.get("Labels")
    folds_preview = first_previews.get("Folds")
    fold_names = folds_summary.get("fold_names", [])
    first_fold_name = fold_names[0] if fold_names else "Fold 1"

    lines: list[str] = [
        "# TE Raw Data Inspection / TE 原始数据检查",
        "",
        f"Generated / 生成时间: `{inspection.get('generated_at')}`",
        f"Raw dir / 原始数据目录: `{inspection.get('raw_dir')}`",
        "",
        "## What is inside one `.pickle`? / 一个 `.pickle` 里有什么？",
        "",
        "```mermaid",
        "flowchart TB",
        "    P[\"One pickle file / 一个pickle文件\"]",
        f"    P --> S[\"Signals ({fmt_shape(signals_summary)})\"]",
        f"    P --> L[\"Labels ({fmt_shape(labels_summary)})\"]",
        f"    P --> F[\"Folds ({folds_summary.get('fold_count', 'N/A')} folds)\"]",
        f"    S --> S1[\"3D array: samples × time × channels = {fmt_shape(signals_summary)}\"]",
        f"    L --> L1[\"1D array: one label per sample = {fmt_shape(labels_summary)}\"]",
        f"    F --> F1[\"dict: fold name -> sample indices\"]",
        "```",
        "",
        "## Counts and dimensions / 数量与维度",
        "",
        f"- Signals count / Signals 数量: `{shape_dim(signals_summary, 0)}` samples",
        f"- One signal shape / 单个 signal 的形状: `{shape_dim(signals_summary, 1)} × {shape_dim(signals_summary, 2)}`",
        f"- Labels count / Labels 数量: `{shape_dim(labels_summary, 0)}` labels",
        f"- Folds count / Fold 数量: `{folds_summary.get('fold_count', 'N/A')}`",
        f"- Each fold stores sample indices / 每个 fold 存样本索引: yes",
        "",
        "## Three examples / 三个小例子",
        "",
        "### 1) Signals example / Signals 示例",
        "",
        "```json",
        json.dumps(signals_preview, ensure_ascii=False, indent=2)[:600],
        "```",
        "",
        "### 2) Labels example / Labels 示例",
        "",
        "```json",
        json.dumps(labels_preview, ensure_ascii=False, indent=2)[:400],
        "```",
        "",
        f"### 3) Folds example / Folds 示例 ({first_fold_name})",
        "",
        "```json",
        json.dumps({first_fold_name: (folds_preview or {}).get(first_fold_name)}, ensure_ascii=False, indent=2)[:600],
        "```",
        "",
        "## How to read them / 怎么理解",
        "",
        "```mermaid",
        "flowchart LR",
        "    S0[\"Signals[i]\"] --- L0[\"Labels[i]\"]",
        "    S0 --> X[\"same sample index\"]",
        "    L0 --> X",
        "    F1[\"Folds['Fold 1']\"] --> I1[\"sample indices\"]",
        "    I1 --> T1[\"train / val / test split\"]",
        "```",
        "",
        "```mermaid",
        "flowchart TB",
        "    A[\"All sample indices\"] --> B[\"Fold indices\"]",
        "    A --> C[\"Remaining training indices\"]",
        "    B --> D[\"held-out set\"]",
        "    C --> E[\"training set\"]",
        "```",
        "",
        "## Your understanding / 你的理解记录",
        "",
        "- One sample is a `600 × 34` matrix / 一个 sample 是 `600 × 34` 的矩阵。",
        "- There are about `2900` such samples / 大约有 `2900` 个这样的 sample。",
        "- If time is concatenated continuously, the dataset can be seen as `34 × (600 × 2900)` over the time axis / 如果把时间连续拼接，整个数据集可理解为沿时间轴的 `34 × (600 × 2900)` 二维形式。",
        "- The raw data is now segmented into `2900` samples / 原始数据现在被切成了 `2900` 份样本。",
        "- Each fold has about `2900 / 5 = 580` samples / 每个 fold 大约有 `2900 / 5 = 580` 个 sample。",
        "- Fold membership is not sequential slicing / fold 不是按顺序切片。",
        "- Sample IDs are randomly assigned into folds, for example a fold may contain a mixed set of sample indices instead of `1-580`, `581-1160`, ... / sample 编号是随机分配到各个 fold 中的，不是按 `1-580`、`581-1160` 这样的连续区间。",
        "- `Folds` therefore store sample index sets, not raw signals / 因此 `Folds` 存的是样本索引集合，而不是原始信号。",
        "",
        "## Notes / 备注",
        "",
        "- `Signals` 是样本本体 / `Signals` is the sample tensor.",
        "- `Labels` 与样本一一对齐 / `Labels` aligns with `Signals` by index.",
        "- `Folds` 存的是索引，不是原始信号 / `Folds` stores indices, not raw signals.",
    ]
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
    args.output_json.write_text(
        json.dumps(inspection, indent=2, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(inspection), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
