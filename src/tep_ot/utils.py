"""Small utilities for the focused TEP OT experiments."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def canonical_domain(domain: int | str) -> str:
    """Normalize ``1``, ``m1`` or ``mode1`` into ``mode1``."""

    text = str(domain).strip().lower()
    if text.startswith("mode"):
        return f"mode{int(text.replace('mode', ''))}"
    if text.startswith("m"):
        return f"mode{int(text[1:])}"
    return f"mode{int(text)}"


def short_domain(domain: int | str) -> str:
    return canonical_domain(domain).replace("mode", "")


def fold_to_name(fold: int | str) -> str:
    """Map CLI folds to raw pickle fold names.

    Integer folds are zero-based for the requested command style, so ``0`` is
    ``Fold 1`` and ``4`` is ``Fold 5``. Existing names pass through unchanged.
    """

    if isinstance(fold, int):
        return f"Fold {fold + 1}"
    text = str(fold).strip()
    if text.lower().startswith("fold"):
        number = int(text.split()[-1])
        return f"Fold {number}"
    return f"Fold {int(text) + 1}"


def fold_to_index(fold: int | str) -> int:
    if isinstance(fold, int):
        return fold
    text = str(fold).strip()
    if text.lower().startswith("fold"):
        return int(text.split()[-1]) - 1
    return int(text)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                fieldnames = next(reader)
            except StopIteration:
                fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def task_label(sources: list[str], target: str) -> str:
    source_part = "+".join(f"m{short_domain(source)}" for source in sources)
    return f"{source_part}->m{short_domain(target)}"
