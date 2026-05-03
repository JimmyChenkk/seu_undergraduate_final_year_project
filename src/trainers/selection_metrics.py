"""Registry-backed checkpoint-selection and early-stopping metrics."""

from __future__ import annotations

import math
from typing import Callable


MetricResolver = Callable[
    [dict[str, float], dict[str, float], dict[str, float]],
    float | None,
]

_SELECTION_METRICS: dict[str, MetricResolver] = {}


def _finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def register_selection_metric(*names: str) -> Callable[[MetricResolver], MetricResolver]:
    """Register one metric resolver under one or more case-insensitive names."""

    normalized_names = [str(name).strip().lower() for name in names if str(name).strip()]
    if not normalized_names:
        raise ValueError("Selection metrics require at least one non-empty name.")

    def decorator(func: MetricResolver) -> MetricResolver:
        for name in normalized_names:
            _SELECTION_METRICS[name] = func
        return func

    return decorator


def list_selection_metrics() -> tuple[str, ...]:
    """Return the currently registered metric names."""

    return tuple(sorted(_SELECTION_METRICS))


def resolve_selection_metric(
    summary: dict[str, float],
    metric_name: str,
    *,
    weights: dict[str, float] | None = None,
    params: dict[str, float] | None = None,
) -> float | None:
    """Resolve one score-to-maximize from epoch summary metrics."""

    normalized = str(metric_name).strip().lower()
    resolver = _SELECTION_METRICS.get(normalized)
    if resolver is None:
        raise KeyError(
            "Unsupported metric for model selection / early stopping: "
            + metric_name
            + ". Registered metrics: "
            + ", ".join(list_selection_metrics())
        )
    return resolver(summary, weights or {}, params or {})


@register_selection_metric("source_train", "best_source_train")
def _metric_source_train(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    del weights, params
    return _finite_or_none(summary.get("acc_source_train"))


@register_selection_metric("source_eval", "best_source_eval")
def _metric_source_eval(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    del weights, params
    return _finite_or_none(summary.get("acc_source_eval"))


@register_selection_metric("target_eval", "best_target_eval", "best_target_eval_oracle")
def _metric_target_eval(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    del weights, params
    return _finite_or_none(summary.get("target_eval_acc"))


@register_selection_metric("target_confidence", "best_target_confidence")
def _metric_target_confidence(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    del weights, params
    return _finite_or_none(summary.get("target_train_mean_confidence"))


@register_selection_metric("target_entropy", "lowest_target_entropy", "min_target_entropy")
def _metric_target_entropy(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    del weights, params
    value = _finite_or_none(summary.get("target_train_mean_entropy"))
    return None if value is None else -value


@register_selection_metric("hybrid_source_eval_target_confidence")
def _metric_hybrid_source_eval_target_confidence(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    del params
    source_eval = _finite_or_none(summary.get("acc_source_eval"))
    target_confidence = _finite_or_none(summary.get("target_train_mean_confidence"))
    if source_eval is None or target_confidence is None:
        return None
    source_weight = float(weights.get("source_eval", 0.7))
    target_weight = float(weights.get("target_confidence", 0.3))
    return source_weight * source_eval + target_weight * target_confidence


@register_selection_metric("hybrid_source_eval_inverse_entropy")
def _metric_hybrid_source_eval_inverse_entropy(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    del params
    source_eval = _finite_or_none(summary.get("acc_source_eval"))
    target_entropy = _finite_or_none(summary.get("target_train_mean_entropy"))
    if source_eval is None or target_entropy is None:
        return None
    source_weight = float(weights.get("source_eval", 0.7))
    entropy_weight = float(weights.get("target_entropy", 0.3))
    return source_weight * source_eval - entropy_weight * target_entropy


@register_selection_metric("hybrid_source_eval_entropy_guard_domain_gap")
def _metric_hybrid_source_eval_entropy_guard_domain_gap(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    """Penalize low target entropy more when domain alignment is still weak.

    This is useful for adversarial DA methods that can become over-confident on
    target predictions while the domain discriminator is still far from the
    desired confusion point.
    """

    source_eval = _finite_or_none(summary.get("acc_source_eval"))
    target_entropy = _finite_or_none(summary.get("target_train_mean_entropy"))
    domain_accuracy = _finite_or_none(summary.get("acc_domain"))
    if source_eval is None or target_entropy is None or domain_accuracy is None:
        return None

    source_weight = float(weights.get("source_eval", 1.0))
    entropy_shortfall_weight = float(weights.get("entropy_shortfall", 1.0))
    domain_gap_weight = float(weights.get("domain_gap", 1.0))
    entropy_floor = float(params.get("entropy_floor", 0.5))
    domain_confusion_target = float(params.get("domain_confusion_target", 0.5))

    entropy_shortfall = max(0.0, entropy_floor - target_entropy)
    domain_gap = abs(domain_accuracy - domain_confusion_target)
    penalty = entropy_shortfall_weight * entropy_shortfall * domain_gap_weight * domain_gap
    return source_weight * source_eval - penalty


@register_selection_metric("hybrid_source_eval_confidence_guard")
def _metric_hybrid_source_eval_confidence_guard(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    """Prefer strong source validation while avoiding over-confident target collapse."""

    source_eval = _finite_or_none(summary.get("acc_source_eval"))
    target_confidence = _finite_or_none(summary.get("target_train_mean_confidence"))
    if source_eval is None or target_confidence is None:
        return None

    source_weight = float(weights.get("source_eval", 0.85))
    confidence_weight = float(weights.get("target_confidence", 0.15))
    overconfidence_weight = float(weights.get("overconfidence", 1.0))
    confidence_ceiling = float(params.get("confidence_ceiling", 0.9))
    overconfidence = max(0.0, target_confidence - confidence_ceiling)

    score = source_weight * source_eval + confidence_weight * target_confidence
    score -= overconfidence_weight * overconfidence

    domain_accuracy = _finite_or_none(summary.get("acc_domain"))
    if domain_accuracy is not None:
        domain_gap_weight = float(weights.get("domain_gap", 0.0))
        domain_confusion_target = float(params.get("domain_confusion_target", 0.5))
        domain_gap_tolerance = max(float(params.get("domain_gap_tolerance", 0.0)), 0.0)
        domain_gap = max(0.0, abs(domain_accuracy - domain_confusion_target) - domain_gap_tolerance)
        score -= domain_gap_weight * domain_gap

    return score


@register_selection_metric(
    "target_free_checkpoint_guard",
    "hybrid_source_eval_target_free_guard",
    "source_val_target_free_guard",
)
def _metric_target_free_checkpoint_guard(
    summary: dict[str, float],
    weights: dict[str, float],
    params: dict[str, float],
) -> float | None:
    """Target-label-free checkpoint score with collapse and instability guards."""

    source_eval = _finite_or_none(summary.get("acc_source_eval"))
    if source_eval is None:
        return None

    target_entropy = _finite_or_none(summary.get("target_train_mean_entropy"))
    target_confidence = _finite_or_none(summary.get("target_train_mean_confidence"))
    class_entropy = _finite_or_none(summary.get("target_train_pred_class_entropy"))
    ot_instability = _finite_or_none(summary.get("ot_cost_instability")) or 0.0
    embedding_instability = _finite_or_none(summary.get("embedding_norm_instability")) or 0.0
    ot_class_collapse = _finite_or_none(summary.get("ot_class_collapse_penalty")) or 0.0

    source_weight = float(weights.get("source_eval", 1.0))
    entropy_weight = float(weights.get("target_entropy_badness", weights.get("target_entropy", 0.10)))
    confidence_weight = float(weights.get("confidence_collapse", weights.get("overconfidence", 1.0)))
    class_weight = float(weights.get("class_distribution_collapse", 0.50))
    ot_weight = float(weights.get("ot_cost_instability", 0.10))
    embedding_weight = float(weights.get("embedding_norm_instability", 0.05))

    entropy_floor = float(params.get("target_entropy_floor", 0.15))
    confidence_ceiling = float(params.get("confidence_ceiling", 0.90))
    class_entropy_floor = float(params.get("class_entropy_floor", 0.35))

    entropy_badness = 0.0
    if target_entropy is not None:
        entropy_badness = max(0.0, entropy_floor - target_entropy)

    confidence_collapse = 0.0
    if target_confidence is not None:
        confidence_collapse = max(0.0, target_confidence - confidence_ceiling)

    class_collapse = 0.0
    if class_entropy is not None:
        class_collapse = max(0.0, class_entropy_floor - class_entropy)

    score = source_weight * source_eval
    score -= entropy_weight * entropy_badness
    score -= confidence_weight * confidence_collapse
    score -= class_weight * class_collapse
    score -= ot_weight * max(0.0, ot_instability)
    score -= embedding_weight * max(0.0, embedding_instability)
    score -= class_weight * max(0.0, ot_class_collapse)
    return score
