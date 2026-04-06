"""Agent-oriented review helpers for autonomous small-scale experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _first_metric(result: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        metric = _coerce_float(result.get(key))
        if metric is not None:
            return metric
    return None


def extract_core_metrics(result_payload: dict[str, Any]) -> dict[str, float | None]:
    """Normalize the small set of metrics used in the autonomous review loop."""

    result = result_payload.get("result", {})
    return {
        "source_train_acc": _first_metric(
            result,
            "selected_source_train_acc",
            "final_source_train_acc",
            "source_train_acc",
        ),
        "source_eval_acc": _first_metric(
            result,
            "selected_source_eval_acc",
            "final_source_eval_acc",
            "source_eval_acc",
        ),
        "target_eval_acc": _first_metric(
            result,
            "selected_target_eval_acc",
            "final_target_eval_acc",
            "target_eval_acc",
        ),
        "target_eval_balanced_acc": _coerce_float(result.get("target_eval_balanced_acc")),
    }


def build_run_review(
    result_payload: dict[str, Any],
    *,
    figure_paths: dict[str, str | None],
) -> dict[str, Any]:
    """Build a lightweight advisory summary for one run."""

    metrics = extract_core_metrics(result_payload)
    source_train_acc = metrics["source_train_acc"]
    source_eval_acc = metrics["source_eval_acc"]
    target_eval_acc = metrics["target_eval_acc"]

    generalization_gap = (
        None
        if source_train_acc is None or source_eval_acc is None
        else float(source_train_acc - source_eval_acc)
    )
    transfer_gap = (
        None
        if source_eval_acc is None or target_eval_acc is None
        else float(source_eval_acc - target_eval_acc)
    )

    flags = {
        "source_training_weak": source_train_acc is None or source_train_acc < 0.75,
        "source_generalization_weak": (
            source_eval_acc is None
            or source_eval_acc < 0.70
            or (generalization_gap is not None and generalization_gap > 0.12)
        ),
        "target_transfer_weak": (
            target_eval_acc is None
            or target_eval_acc < 0.55
            or (transfer_gap is not None and transfer_gap > 0.15)
        ),
        "over_alignment_suspect": (
            source_eval_acc is not None
            and target_eval_acc is not None
            and source_eval_acc >= 0.80
            and transfer_gap is not None
            and transfer_gap > 0.25
        ),
        "visual_artifacts_missing": any(path is None for path in figure_paths.values()),
    }

    next_focus: list[str] = []
    if flags["source_training_weak"]:
        next_focus.append("stabilize_source_training")
    elif flags["source_generalization_weak"]:
        next_focus.append("improve_source_generalization")
    elif flags["target_transfer_weak"]:
        next_focus.append("strengthen_domain_alignment")
    else:
        next_focus.append("validate_small_scale_gain")

    if flags["over_alignment_suspect"]:
        next_focus.append("reduce_alignment_pressure")

    next_focus.extend(
        [
            "inspect_confusion_matrix",
            "inspect_tsne_domain",
            "inspect_tsne_class",
        ]
    )

    return {
        "method_name": result_payload.get("method_name"),
        "scenario_id": result_payload.get("scenario_id"),
        "setting": result_payload.get("setting"),
        "metrics": metrics,
        "gaps": {
            "source_train_minus_eval": generalization_gap,
            "source_eval_minus_target_eval": transfer_gap,
        },
        "flags": flags,
        "next_focus": next_focus,
        "manual_visual_review_required": True,
        "figures": figure_paths,
        "run_root": result_payload.get("run_root"),
    }


def save_review(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_review(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
