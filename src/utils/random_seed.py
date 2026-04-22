"""Helpers for resolving fixed or auto-generated experiment seeds."""

from __future__ import annotations

from typing import Any
import random


MAX_NUMPY_SEED = 2**32 - 1
AUTO_SEED_TOKENS = {"", "auto", "random", "none", "null"}


def _generate_seed() -> int:
    return random.SystemRandom().randrange(0, MAX_NUMPY_SEED + 1)


def resolve_seed(seed_value: Any) -> tuple[int, str]:
    """Return a concrete seed and whether it came from a fixed or auto request."""

    if seed_value is None:
        return _generate_seed(), "auto"

    if isinstance(seed_value, str):
        text = seed_value.strip()
        if text.lower() in AUTO_SEED_TOKENS:
            return _generate_seed(), "auto"
        seed = int(text)
    else:
        seed = int(seed_value)

    if seed < 0 or seed > MAX_NUMPY_SEED:
        raise ValueError(f"Seed must be in [0, {MAX_NUMPY_SEED}], got {seed}.")
    return seed, "fixed"
