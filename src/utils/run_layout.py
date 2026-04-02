"""Helpers for the benchmark run directory layout under ``runs/``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


COMPARISON_DIR_NAME = "comparison_summary"


@dataclass(frozen=True)
class RunLayout:
    """Concrete directories for one benchmark run."""

    timestamp: str
    run_name: str
    output_root: Path
    batch_root: Path | None
    run_root: Path
    artifacts_dir: Path
    tables_dir: Path
    figures_dir: Path
    logs_dir: Path
    checkpoints_dir: Path


def build_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_token(value: str, *, lowercase: bool = False, compact_spaces: bool = False) -> str:
    """Convert arbitrary labels into stable directory-name fragments."""

    text = str(value).strip()
    if lowercase:
        text = text.lower()
    text = re.sub(r"\s+", "" if compact_spaces else "-", text)
    text = text.replace("/", "-").replace("\\", "-").replace(":", "-").replace(",", "-")
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    return text or "unknown"


def normalize_fold_name(fold_name: str) -> str:
    return normalize_token(fold_name, lowercase=True, compact_spaces=True)


def build_run_name(
    *,
    timestamp: str,
    method_name: str,
    scenario_id: str,
    backbone_name: str,
    fold_name: str,
) -> str:
    """Build names like ``20260402_120000_dann_mode2_to_mode5_fcn_fold1``."""

    parts = [
        timestamp,
        normalize_token(method_name, lowercase=True),
        normalize_token(scenario_id, lowercase=True),
        normalize_token(backbone_name, lowercase=True),
        normalize_fold_name(fold_name),
    ]
    return "_".join(parts)


def build_run_layout(
    *,
    output_dir: Path,
    method_name: str,
    scenario_id: str,
    backbone_name: str,
    fold_name: str,
    timestamp: str | None = None,
    batch_root_name: str | None = None,
) -> RunLayout:
    """Create the directory layout for one run, optionally inside a batch root."""

    resolved_timestamp = timestamp or build_timestamp()
    batch_root = output_dir / normalize_token(batch_root_name) if batch_root_name else None
    run_name = build_run_name(
        timestamp=resolved_timestamp,
        method_name=method_name,
        scenario_id=scenario_id,
        backbone_name=backbone_name,
        fold_name=fold_name,
    )
    run_root = (batch_root or output_dir) / run_name
    layout = RunLayout(
        timestamp=resolved_timestamp,
        run_name=run_name,
        output_root=output_dir,
        batch_root=batch_root,
        run_root=run_root,
        artifacts_dir=run_root / "artifacts",
        tables_dir=run_root / "tables",
        figures_dir=run_root / "figures",
        logs_dir=run_root / "logs",
        checkpoints_dir=run_root / "checkpoints",
    )
    for directory in [
        layout.run_root,
        layout.artifacts_dir,
        layout.tables_dir,
        layout.figures_dir,
        layout.logs_dir,
        layout.checkpoints_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def is_run_root(path: Path) -> bool:
    return path.is_dir() and (path / "artifacts").is_dir() and (path / "tables").is_dir()


def find_result_json_paths(base_path: Path) -> list[Path]:
    """Locate per-run result JSON files in the new or legacy layouts."""

    if base_path.is_file():
        return [base_path]
    if not base_path.exists():
        return []
    if base_path.name == "tables":
        return sorted(path for path in base_path.glob("*.json") if path.is_file())
    if is_run_root(base_path):
        tables_dir = base_path / "tables"
        return sorted(path for path in tables_dir.glob("*.json") if path.is_file())
    return sorted(path for path in base_path.rglob("tables/*.json") if path.is_file())


def resolve_comparison_root(base_path: Path) -> Path | None:
    """Infer where cross-run comparison outputs should live."""

    resolved = base_path
    if resolved.is_file():
        resolved = resolved.parent
    if resolved.name == "tables":
        resolved = resolved.parent
    if is_run_root(resolved):
        return None
    if resolved.name == COMPARISON_DIR_NAME:
        return resolved
    return resolved / COMPARISON_DIR_NAME
