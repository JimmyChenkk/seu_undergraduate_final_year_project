"""Shared fold policy helpers for benchmark planning and execution."""

from __future__ import annotations

from typing import Any
import random


DEFAULT_FOLD_NAME = "Fold 1"


def canonicalize_fold_name(fold_name: Any) -> str:
    text = str(fold_name).strip()
    if not text:
        return DEFAULT_FOLD_NAME
    lowered = text.lower()
    if lowered.startswith("fold"):
        suffix = text[4:].strip()
        return f"Fold {suffix}" if suffix else DEFAULT_FOLD_NAME
    return f"Fold {text}"


def canonicalize_fold_choice(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return DEFAULT_FOLD_NAME
    if text.lower().startswith("fold"):
        return canonicalize_fold_name(text)
    if text.isdigit():
        return f"Fold {int(text)}"
    return canonicalize_fold_name(text)


def resolve_fold_policy(protocol_payload: dict[str, Any]) -> dict[str, Any]:
    fold_sampling = protocol_payload.get("fold_sampling", {})
    if not isinstance(fold_sampling, dict):
        fold_sampling = {}

    random_fold_enabled = bool(protocol_payload.get("random_fold_enabled", False))
    strategy = str(fold_sampling.get("strategy", "fixed" if not random_fold_enabled else "random")).strip().lower()
    if not strategy:
        strategy = "fixed" if not random_fold_enabled else "random"

    random_per_scene = bool(fold_sampling.get("random_per_scene", random_fold_enabled))
    random_per_run = bool(fold_sampling.get("random_per_run", False))
    source_fold_choices = fold_sampling.get("source_fold_choices", protocol_payload.get("source_folds", [1, 2, 3, 4, 5]))
    target_fold_choices = fold_sampling.get("target_fold_choices", protocol_payload.get("target_folds", [1, 2, 3, 4, 5]))

    return {
        "enabled": random_fold_enabled,
        "strategy": strategy if random_fold_enabled else "fixed",
        "random_per_scene": bool(random_per_scene if random_fold_enabled else False),
        "random_per_run": bool(random_per_run if random_fold_enabled else False),
        "source_fold_choices": [canonicalize_fold_choice(item) for item in source_fold_choices],
        "target_fold_choices": [canonicalize_fold_choice(item) for item in target_fold_choices],
        "preferred_fold": canonicalize_fold_name(protocol_payload.get("preferred_fold", DEFAULT_FOLD_NAME)),
    }


def sample_fold_pair(
    protocol_payload: dict[str, Any],
    *,
    rng: random.Random,
    default_fold: str = DEFAULT_FOLD_NAME,
) -> dict[str, str | bool]:
    fold_policy = resolve_fold_policy(protocol_payload)
    selected_fold = canonicalize_fold_name(protocol_payload.get("preferred_fold", default_fold) or default_fold)
    source_fold_default = canonicalize_fold_name(protocol_payload.get("source_fold", selected_fold) or selected_fold)
    target_fold_default = canonicalize_fold_name(protocol_payload.get("target_fold", selected_fold) or selected_fold)

    if fold_policy["enabled"] and protocol_payload.get("source_fold") is None:
        source_fold = canonicalize_fold_name(rng.choice(list(fold_policy["source_fold_choices"])))
    else:
        source_fold = source_fold_default

    if fold_policy["enabled"] and protocol_payload.get("target_fold") is None:
        target_fold = canonicalize_fold_name(rng.choice(list(fold_policy["target_fold_choices"])))
    else:
        target_fold = target_fold_default

    fold_strategy = fold_policy["strategy"]
    if not fold_policy["enabled"]:
        fold_strategy = "fixed"
    elif fold_policy["random_per_scene"]:
        fold_strategy = "random_per_scene"
    elif fold_policy["random_per_run"]:
        fold_strategy = "random_per_run"

    return {
        "selected_fold": selected_fold,
        "source_fold": source_fold,
        "target_fold": target_fold,
        "random_fold_enabled": fold_policy["enabled"],
        "fold_strategy": fold_strategy,
        "random_per_scene": fold_policy["random_per_scene"],
        "random_per_run": fold_policy["random_per_run"],
    }
