"""DeepJDOT implementation using minibatch OT on GPU-friendly torch tensors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any
import warnings

import torch
import torch.nn.functional as F

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


EPS = 1e-8


@dataclass
class _TransportOutput:
    loss: torch.Tensor
    gamma: torch.Tensor
    q_ot: torch.Tensor
    q_ot_entropy: torch.Tensor
    metrics: dict[str, float]


def _ramped_weight(base_weight: float, *, step: int, start_step: int, warmup_steps: int) -> float:
    if base_weight <= 0 or step <= start_step:
        return 0.0
    progress = min((step - start_step) / float(max(warmup_steps, 1)), 1.0)
    return float(base_weight) * progress


def _sanitize_features(features: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)


def _prototype_features(features: torch.Tensor, distance: str) -> torch.Tensor:
    values = _sanitize_features(features)
    if str(distance).strip().lower() in {"cosine", "cos", "normalized_l2", "normalized_l2_sq"}:
        return F.normalize(values, p=2, dim=1, eps=1e-6)
    return values


def _class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.detach().long()
    valid_mask = (labels >= 0) & (labels < num_classes)
    prototypes = features.new_zeros((num_classes, features.shape[1]))
    counts = features.new_zeros((num_classes,))
    if bool(valid_mask.any()):
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]
        prototypes.index_add_(0, valid_labels, valid_features)
        counts.index_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=features.dtype))
    present = counts > 0
    prototypes = prototypes / counts.clamp_min(1.0).unsqueeze(1)
    return prototypes, present


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    num_classes = int(probabilities.shape[1])
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=1)
    return entropy / math.log(max(num_classes, 2))


def _renormalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(probabilities.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    total = values.sum(dim=1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / float(max(values.shape[1], 1)))
    normalized = values / total.clamp_min(EPS)
    return torch.where(total > 0, normalized, uniform)


def _target_prediction_class_entropy(probabilities: torch.Tensor) -> float:
    if probabilities.numel() == 0:
        return 0.0
    predictions = probabilities.detach().argmax(dim=1)
    counts = torch.bincount(predictions, minlength=int(probabilities.shape[1])).to(
        device=probabilities.device,
        dtype=torch.float32,
    )
    distribution = counts / counts.sum().clamp_min(1.0)
    entropy = -(distribution * distribution.clamp_min(EPS).log()).sum()
    return float((entropy / math.log(max(int(probabilities.shape[1]), 2))).detach().item())


def _safe_distribution(
    size: int,
    *,
    weights: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if weights is None:
        values = torch.ones((size,), device=device, dtype=dtype)
    else:
        values = weights[:size].to(device=device, dtype=dtype).clamp_min(0.0)
    if values.numel() == 0 or not torch.isfinite(values).all() or values.sum() <= 0:
        values = torch.ones((size,), device=device, dtype=dtype)
    return values / values.sum().clamp_min(EPS)


def _generalized_kl(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0.0)
    reference = reference.clamp_min(EPS)
    return (values * (values.clamp_min(EPS) / reference).log() - values + reference).sum()


def _solve_deepjdot_coupling(
    cost: torch.Tensor,
    *,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    solver: str,
    sinkhorn_reg: float,
    sinkhorn_num_iter_max: int,
    unbalanced: bool,
    uot_tau_s: float,
    uot_tau_t: float,
) -> torch.Tensor:
    try:
        import ot
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("DeepJDOT requires the POT package (import ot).") from exc

    solver_name = str(solver).strip().lower()
    cost_np = torch.nan_to_num(cost.detach(), nan=1e6, posinf=1e6, neginf=1e6).double().cpu().numpy()
    source_np = source_weights.detach().double().cpu().numpy()
    target_np = target_weights.detach().double().cpu().numpy()
    if unbalanced or solver_name in {"unbalanced", "sinkhorn_unbalanced", "uot"}:
        if solver_name not in {"sinkhorn", "entropic", "regularized", "unbalanced", "sinkhorn_unbalanced", "uot"}:
            raise ValueError(f"Unsupported unbalanced DeepJDOT solver: {solver}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sinkhorn did not converge.*")
            warnings.filterwarnings("ignore", message="Numerical errors.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                gamma_np = ot.unbalanced.sinkhorn_unbalanced(
                    source_np,
                    target_np,
                    cost_np,
                    reg=max(float(sinkhorn_reg), 1e-6),
                    reg_m=(max(float(uot_tau_s), 1e-6), max(float(uot_tau_t), 1e-6)),
                    numItermax=max(int(sinkhorn_num_iter_max), 20),
                    stopThr=1e-6,
                )
            except TypeError:
                gamma_np = ot.unbalanced.sinkhorn_unbalanced(
                    source_np,
                    target_np,
                    cost_np,
                    reg=max(float(sinkhorn_reg), 1e-6),
                    reg_m=max(float((uot_tau_s + uot_tau_t) * 0.5), 1e-6),
                    numItermax=max(int(sinkhorn_num_iter_max), 20),
                    stopThr=1e-6,
                )
    elif solver_name in {"sinkhorn", "entropic", "regularized"}:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sinkhorn did not converge.*")
            gamma_np = ot.sinkhorn(
                source_np,
                target_np,
                cost_np,
                reg=max(float(sinkhorn_reg), 1e-6),
                numItermax=max(int(sinkhorn_num_iter_max), 10),
                stopThr=1e-6,
            )
    elif solver_name in {"emd", "exact"}:
        gamma_np = ot.emd(source_np, target_np, cost_np)
    else:
        raise ValueError(f"Unsupported DeepJDOT solver: {solver}")

    gamma = torch.as_tensor(gamma_np, device=cost.device, dtype=cost.dtype)
    if not torch.isfinite(gamma).all():
        raise RuntimeError("DeepJDOT transport solver returned a non-finite coupling.")
    gamma = gamma.clamp_min(0.0)
    gamma_sum = gamma.sum()
    if gamma_sum <= 0:
        raise RuntimeError("DeepJDOT transport solver returned an empty coupling.")
    if not (unbalanced or solver_name in {"unbalanced", "sinkhorn_unbalanced", "uot"}):
        gamma = gamma / gamma_sum.clamp_min(EPS)
    return gamma


def _class_target_distances(
    source_prototypes: torch.Tensor,
    target_features: torch.Tensor,
    *,
    distance: str,
) -> torch.Tensor:
    metric = str(distance).strip().lower()
    prototypes = source_prototypes.detach().to(device=target_features.device, dtype=target_features.dtype)
    targets = _sanitize_features(target_features)
    if metric in {"cosine", "cos"}:
        cost = 1.0 - _prototype_features(prototypes, "normalized_l2").mm(
            _prototype_features(targets, "normalized_l2").t()
        )
        return torch.nan_to_num(cost.clamp_min(0.0), nan=0.0, posinf=1e4, neginf=0.0)
    if metric in {"normalized_l2", "normalized_l2_sq", "l2_normalized", "cosine_l2"}:
        prototypes = _prototype_features(prototypes, "normalized_l2")
        targets = _prototype_features(targets, "normalized_l2")
    return torch.nan_to_num(torch.cdist(prototypes, targets, p=2).pow(2), nan=0.0, posinf=1e4, neginf=0.0)


def _relative_prototype_cost(
    *,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    source_prototypes: torch.Tensor,
    active_mask: torch.Tensor,
    distance: str,
    min_value: float = -1.0,
    max_value: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    source_labels = source_labels.detach().long()
    active = active_mask.to(device=target_features.device, dtype=torch.bool)
    all_distances = _class_target_distances(source_prototypes, target_features, distance=distance)
    if not bool(active.any()):
        zero_cost = target_features.new_zeros((source_labels.numel(), target_features.shape[0]))
        return zero_cost, {
            "prototype_relative_cost_mean": 0.0,
            "prototype_relative_cost_std": 0.0,
            "prototype_relative_cost_min": 0.0,
            "prototype_relative_cost_max": 0.0,
        }, all_distances.t().new_zeros((target_features.shape[0], source_prototypes.shape[0]))

    active_distances = all_distances[active]
    source_class_distances = all_distances[source_labels.clamp(0, source_prototypes.shape[0] - 1)]
    active_source = active[source_labels.clamp(0, source_prototypes.shape[0] - 1)].view(-1, 1)
    active_sum = active_distances.sum(dim=0, keepdim=True)
    active_count = float(active_distances.shape[0])
    if active_distances.shape[0] > 1:
        other_mean = (active_sum - source_class_distances) / max(active_count - 1.0, 1.0)
    else:
        other_mean = source_class_distances
    std = active_distances.std(dim=0, keepdim=True, unbiased=False).clamp_min(EPS)
    relative = (source_class_distances - other_mean) / std
    relative = relative.clamp(min=float(min_value), max=float(max_value))
    relative = torch.where(active_source, relative, torch.zeros_like(relative))
    rel_detached = relative.detach()
    return relative, {
        "prototype_relative_cost_mean": float(rel_detached.mean().item()),
        "prototype_relative_cost_std": float(rel_detached.std(unbiased=False).item()),
        "prototype_relative_cost_min": float(rel_detached.min().item()),
        "prototype_relative_cost_max": float(rel_detached.max().item()),
    }, all_distances.t()


def _temporal_descriptor(x: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)
    mean = values.mean(dim=-1)
    std = values.std(dim=-1, unbiased=False)
    if values.shape[-1] > 1:
        diff = values.diff(dim=-1)
        diff_mean = diff.mean(dim=-1)
        diff_std = diff.std(dim=-1, unbiased=False)
    else:
        diff_mean = torch.zeros_like(mean)
        diff_std = torch.zeros_like(std)
    descriptor = torch.cat([mean, std, diff_mean, diff_std], dim=1)
    return F.normalize(descriptor, p=2, dim=1, eps=1e-6)


def _temporal_cost(source_x: torch.Tensor, target_x: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    source_descriptor = _temporal_descriptor(source_x)
    target_descriptor = _temporal_descriptor(target_x)
    cost = torch.cdist(source_descriptor, target_descriptor, p=2).pow(2)
    return torch.nan_to_num(cost / float(max(source_descriptor.shape[1], 1)), nan=0.0, posinf=1e4, neginf=0.0).to(
        dtype=dtype
    )


def _supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if features.shape[0] <= 1:
        return features.new_zeros(())
    labels = labels.detach().long()
    valid = labels >= 0
    if int(valid.sum().item()) <= 1:
        return features.new_zeros(())
    features = F.normalize(_sanitize_features(features[valid]).float(), p=2, dim=1, eps=1e-6)
    labels = labels[valid]
    logits = (features.mm(features.t()) / max(float(temperature), EPS)).float()
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    positive_mask = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not bool(valid_anchor.any()):
        return logits.new_zeros(())
    logits_masked = logits.masked_fill(self_mask, -1e9)
    log_prob = logits_masked - torch.logsumexp(logits_masked, dim=1, keepdim=True)
    positive_log_prob = (positive_mask.to(dtype=log_prob.dtype) * log_prob).sum(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp_min(1).to(dtype=log_prob.dtype)
    loss = -(positive_log_prob[valid_anchor] / positive_count[valid_anchor]).mean()
    return loss.to(dtype=features.dtype)


def _q_ot_from_gamma(
    gamma: torch.Tensor,
    source_labels: torch.Tensor,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    class_mass = gamma.detach().new_zeros((num_classes, gamma.shape[1]))
    class_mass.index_add_(0, source_labels.detach().long().clamp(0, num_classes - 1), gamma.detach())
    col_mass = gamma.detach().sum(dim=0).clamp_min(EPS)
    q_ot = (class_mass / col_mass.view(1, -1)).t()
    q_ot = q_ot / q_ot.sum(dim=1, keepdim=True).clamp_min(EPS)
    entropy = _normalized_entropy(q_ot)
    return q_ot, entropy


def _soft_label_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets.detach() * F.log_softmax(logits, dim=1)).sum(dim=1)


def _js_disagreement(probabilities: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack([item.clamp_min(EPS) for item in probabilities], dim=0)
    mixture = stacked.mean(dim=0).clamp_min(EPS)
    kl_values = (stacked * (stacked / mixture.unsqueeze(0)).log()).sum(dim=2)
    return kl_values.mean(dim=0) / math.log(max(int(stacked.shape[-1]), 2))


def _transport_with_diagnostics(
    *,
    source_labels: torch.Tensor,
    logits_target: torch.Tensor,
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    reg_dist: float,
    reg_cl: float,
    normalize_feature_cost: bool,
    solver: str,
    sinkhorn_reg: float,
    sinkhorn_num_iter_max: int,
    unbalanced: bool,
    uot_tau_s: float,
    uot_tau_t: float,
    source_sample_weights: torch.Tensor | None = None,
    target_sample_weights: torch.Tensor | None = None,
    use_ce_cost_for_plan: bool = False,
    source_prototypes: torch.Tensor | None = None,
    source_prototype_active: torch.Tensor | None = None,
    prototype_distance: str = "normalized_l2",
    prototype_cost_weight: float = 0.0,
    prototype_relative_min: float = -1.0,
    prototype_relative_max: float = 3.0,
    source_x: torch.Tensor | None = None,
    target_x: torch.Tensor | None = None,
    temporal_cost_weight: float = 0.0,
) -> _TransportOutput:
    source_size = min(features_source.shape[0], source_labels.shape[0])
    target_size = min(features_target.shape[0], logits_target.shape[0])
    if source_size == 0 or target_size == 0:
        zero = logits_target.new_zeros(())
        q_ot = logits_target.new_zeros((target_size, max(int(logits_target.shape[1]), 1)))
        return _TransportOutput(zero, logits_target.new_zeros((source_size, target_size)), q_ot, zero.view(1), {})

    source_labels = source_labels[:source_size].long()
    features_source = _sanitize_features(features_source[:source_size])
    features_target = _sanitize_features(features_target[:target_size])
    logits_target = torch.nan_to_num(logits_target[:target_size], nan=0.0, posinf=1e6, neginf=-1e6)
    num_classes = max(int(logits_target.shape[1]), 1)
    if source_labels.min() < 0 or source_labels.max() >= num_classes:
        raise ValueError(
            f"DeepJDOT source labels must be in [0, {num_classes - 1}], "
            f"got min={int(source_labels.min())}, max={int(source_labels.max())}."
        )

    feature_cost = torch.cdist(features_source, features_target, p=2).pow(2)
    if normalize_feature_cost:
        feature_cost = feature_cost / float(max(features_source.shape[1], 1))
    feature_cost = torch.nan_to_num(feature_cost, nan=1e6, posinf=1e6, neginf=1e6)

    target_log_probs = F.log_softmax(logits_target.float(), dim=1).to(dtype=logits_target.dtype)
    target_probabilities = target_log_probs.exp()
    source_one_hot = F.one_hot(source_labels, num_classes=num_classes).to(dtype=target_probabilities.dtype)
    plan_class_cost = -target_log_probs[:, source_labels].transpose(0, 1) if use_ce_cost_for_plan else torch.cdist(
        source_one_hot,
        target_probabilities,
        p=2,
    ).pow(2)
    plan_class_cost = torch.nan_to_num(plan_class_cost, nan=1e6, posinf=1e6, neginf=1e6)
    class_loss_cost = -target_log_probs[:, source_labels].transpose(0, 1)
    class_loss_cost = torch.nan_to_num(class_loss_cost, nan=1e6, posinf=1e6, neginf=1e6)

    base_plan_cost = float(reg_dist) * feature_cost + float(reg_cl) * plan_class_cost
    loss_cost = float(reg_dist) * feature_cost + float(reg_cl) * class_loss_cost
    prototype_metrics: dict[str, float] = {
        "prototype_relative_cost_mean": 0.0,
        "prototype_relative_cost_std": 0.0,
        "prototype_relative_cost_min": 0.0,
        "prototype_relative_cost_max": 0.0,
    }
    prototype_plan_cost = torch.zeros_like(base_plan_cost)
    prototype_all_distances = logits_target.new_zeros((target_size, num_classes))
    if (
        source_prototypes is not None
        and source_prototype_active is not None
        and float(prototype_cost_weight) > 0
    ):
        prototype_plan_cost, prototype_metrics, prototype_all_distances = _relative_prototype_cost(
            source_labels=source_labels,
            target_features=features_target,
            source_prototypes=source_prototypes,
            active_mask=source_prototype_active,
            distance=prototype_distance,
            min_value=prototype_relative_min,
            max_value=prototype_relative_max,
        )
        base_plan_cost = base_plan_cost + float(prototype_cost_weight) * prototype_plan_cost
        loss_cost = loss_cost + float(prototype_cost_weight) * prototype_plan_cost.detach()

    temporal_plan_cost = torch.zeros_like(base_plan_cost)
    if source_x is not None and target_x is not None and float(temporal_cost_weight) > 0:
        temporal_plan_cost = _temporal_cost(source_x[:source_size], target_x[:target_size], dtype=base_plan_cost.dtype)
        base_plan_cost = base_plan_cost + float(temporal_cost_weight) * temporal_plan_cost
        loss_cost = loss_cost + float(temporal_cost_weight) * temporal_plan_cost.detach()

    if not torch.isfinite(base_plan_cost).all() or not torch.isfinite(loss_cost).all():
        raise RuntimeError("DeepJDOT encountered a non-finite transport cost.")

    source_weights = _safe_distribution(
        source_size,
        weights=source_sample_weights,
        device=features_source.device,
        dtype=features_source.dtype,
    )
    target_weights = _safe_distribution(
        target_size,
        weights=target_sample_weights,
        device=features_target.device,
        dtype=features_target.dtype,
    )
    gamma = _solve_deepjdot_coupling(
        base_plan_cost,
        source_weights=source_weights,
        target_weights=target_weights,
        solver=solver,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_num_iter_max=sinkhorn_num_iter_max,
        unbalanced=unbalanced,
        uot_tau_s=uot_tau_s,
        uot_tau_t=uot_tau_t,
    )

    gamma_detached = gamma.detach()
    gamma_for_objective = gamma_detached.float()
    source_weights_objective = source_weights.float()
    target_weights_objective = target_weights.float()
    row_mass = gamma_for_objective.sum(dim=1)
    col_mass = gamma_for_objective.sum(dim=0)
    outer = torch.outer(source_weights_objective, target_weights_objective)
    transport_cost = (gamma_for_objective * loss_cost.float()).sum()
    kl_gamma = _generalized_kl(gamma_for_objective, outer)
    kl_row = _generalized_kl(row_mass, source_weights_objective)
    kl_col = _generalized_kl(col_mass, target_weights_objective)
    if unbalanced or str(solver).strip().lower() in {"unbalanced", "sinkhorn_unbalanced", "uot"}:
        loss = (
            transport_cost
            + float(sinkhorn_reg) * kl_gamma.to(dtype=transport_cost.dtype)
            + float(uot_tau_s) * kl_row.to(dtype=transport_cost.dtype)
            + float(uot_tau_t) * kl_col.to(dtype=transport_cost.dtype)
        )
    else:
        loss = transport_cost

    q_ot, q_ot_entropy = _q_ot_from_gamma(gamma_detached, source_labels, num_classes=num_classes)
    metrics = {
        "loss_uot": float(loss.detach().item()),
        "loss_transport_cost": float(transport_cost.detach().item()),
        "uot_kl_gamma": float(kl_gamma.detach().item()),
        "uot_kl_row": float(kl_row.detach().item()),
        "uot_kl_col": float(kl_col.detach().item()),
        "uot_transported_mass": float(gamma_detached.sum().item()),
        "ot_transport_mass": float(gamma_detached.sum().item()),
        "uot_row_mass_deviation": float((row_mass - source_weights).abs().mean().detach().item()),
        "uot_column_mass_deviation": float((col_mass - target_weights).abs().mean().detach().item()),
        "uot_row_mass_max_deviation": float((row_mass - source_weights).abs().max().detach().item()),
        "uot_column_mass_max_deviation": float((col_mass - target_weights).abs().max().detach().item()),
        "uot_gamma_max": float(gamma_detached.max().item()),
        "uot_unbalanced": float(bool(unbalanced)),
        "q_ot_entropy_mean": float(q_ot_entropy.detach().mean().item()),
        "q_ot_entropy_min": float(q_ot_entropy.detach().min().item()),
        "q_ot_entropy_max": float(q_ot_entropy.detach().max().item()),
        "target_pred_class_entropy": _target_prediction_class_entropy(target_probabilities.detach()),
        "target_prediction_class_entropy": _target_prediction_class_entropy(target_probabilities.detach()),
        "ot_feature_cost": float((gamma_detached * feature_cost.detach()).sum().item()),
        "ot_label_cost": float((gamma_detached * class_loss_cost.detach()).sum().item()),
        "ot_prototype_plan_cost": float((gamma_detached * prototype_plan_cost.detach()).sum().item()),
        "ot_temporal_cost": float((gamma_detached * temporal_plan_cost.detach()).sum().item()),
        "prototype_cost_weight": float(prototype_cost_weight),
        "temporal_cost_weight": float(temporal_cost_weight),
    }
    metrics.update(prototype_metrics)
    if prototype_all_distances.numel() > 0:
        proto_probs = torch.softmax(-prototype_all_distances.detach(), dim=1)
        metrics["q_proto_entropy_mean"] = float(_normalized_entropy(proto_probs).mean().detach().item())
    return _TransportOutput(loss, gamma_detached, q_ot.detach(), q_ot_entropy.detach(), metrics)


def _inverse_sqrt_class_weights(
    labels: torch.Tensor,
    *,
    mask: torch.Tensor,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    compute_dtype = torch.float32
    weights = torch.ones((num_classes,), device=device, dtype=compute_dtype)
    if not bool(mask.any()):
        return weights.to(dtype=dtype)
    accepted_labels = labels[mask].detach().long().clamp(0, num_classes - 1)
    counts = torch.bincount(accepted_labels, minlength=num_classes).to(device=device, dtype=compute_dtype)
    nonzero = counts > 0
    if not bool(nonzero.any()):
        return weights.to(dtype=dtype)
    values = torch.zeros_like(counts)
    values[nonzero] = counts[nonzero].rsqrt()
    values[nonzero] = values[nonzero] / values[nonzero].mean().clamp_min(EPS)
    weights[nonzero] = values[nonzero]
    return weights.to(dtype=dtype)


class DeepJDOTMethod(SingleSourceMethodBase):
    """Deep joint distribution optimal transport with a shared encoder."""

    method_name = "deepjdot"

    def __init__(
        self,
        *,
        adaptation_weight: float = 1.0,
        adaptation_schedule: str = "constant",
        adaptation_max_steps: int = 2000,
        adaptation_schedule_alpha: float = 10.0,
        reg_dist: float = 0.1,
        reg_cl: float = 1.0,
        normalize_feature_cost: bool = False,
        transport_solver: str = "emd",
        sinkhorn_reg: float = 0.05,
        sinkhorn_num_iter_max: int = 300,
        unbalanced_transport: bool = False,
        uot_tau_s: float = 1.0,
        uot_tau_t: float = 1.0,
        use_ce_cost_for_plan: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = int(kwargs.get("num_classes", 1))
        self.alignment_scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        self.reg_dist = reg_dist
        self.reg_cl = reg_cl
        self.normalize_feature_cost = normalize_feature_cost
        self.transport_solver = transport_solver
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_num_iter_max = sinkhorn_num_iter_max
        self.unbalanced_transport = bool(unbalanced_transport)
        self.uot_tau_s = float(uot_tau_s)
        self.uot_tau_t = float(uot_tau_t)
        self.use_ce_cost_for_plan = bool(use_ce_cost_for_plan)
        self.global_step = 0

    def extra_regularization_loss(
        self,
        *,
        source_y: torch.Tensor,
        logits_target: torch.Tensor,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return logits_target.new_zeros(()), {}

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        self.global_step += 1
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)

        loss_cls = F.cross_entropy(logits_source, source_y)
        transport_output = _transport_with_diagnostics(
            source_labels=source_y,
            logits_target=logits_target,
            features_source=features_source,
            features_target=features_target,
            reg_dist=self.reg_dist,
            reg_cl=self.reg_cl,
            normalize_feature_cost=self.normalize_feature_cost,
            solver=self.transport_solver,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_num_iter_max=self.sinkhorn_num_iter_max,
            unbalanced=self.unbalanced_transport,
            uot_tau_s=self.uot_tau_s,
            uot_tau_t=self.uot_tau_t,
            use_ce_cost_for_plan=self.use_ce_cost_for_plan,
        )
        loss_alignment = torch.as_tensor(transport_output.loss, device=loss_cls.device, dtype=loss_cls.dtype)

        current_weight = self.alignment_scheduler.step()
        loss_extra, extra_metrics = self.extra_regularization_loss(
            source_y=source_y,
            logits_target=logits_target,
            features_source=features_source,
            features_target=features_target,
        )
        loss_total = loss_cls + current_weight * loss_alignment + loss_extra
        if not torch.isfinite(loss_total).item():
            raise RuntimeError("DeepJDOT produced a non-finite total loss.")

        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_alignment.item()),
                "lambda_alignment": current_weight,
                "acc_source": accuracy_from_logits(logits_source, source_y),
                **transport_output.metrics,
                **extra_metrics,
            },
        )


class UDeepJDOTMethod(DeepJDOTMethod):
    """DeepJDOT with JUMBOT-style unbalanced minibatch transport."""

    method_name = "u_deepjdot"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("unbalanced_transport", True)
        kwargs.setdefault("use_ce_cost_for_plan", False)
        super().__init__(**kwargs)


class TPDeepJDOTMethod(DeepJDOTMethod):
    """DeepJDOT plus delayed source/target prototype residual regularization."""

    method_name = "tp_deepjdot"

    def __init__(
        self,
        *,
        prototype_weight: float = 0.02,
        prototype_source_weight: float = 0.25,
        prototype_target_weight: float = 1.0,
        prototype_start_step: int = 1200,
        prototype_warmup_steps: int = 1600,
        prototype_distance: str = "normalized_l2",
        prototype_confidence_threshold: float = 0.0,
        prototype_target_confidence_power: float = 1.0,
        prototype_probability_temperature: float = 1.0,
        prototype_class_balance: bool = False,
        prototype_class_balance_clip_min: float = 0.0,
        prototype_class_balance_clip_max: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prototype_weight = float(prototype_weight)
        self.prototype_source_weight = float(prototype_source_weight)
        self.prototype_target_weight = float(prototype_target_weight)
        self.prototype_start_step = max(int(prototype_start_step), 0)
        self.prototype_warmup_steps = max(int(prototype_warmup_steps), 1)
        self.prototype_distance = str(prototype_distance)
        self.prototype_confidence_threshold = min(max(float(prototype_confidence_threshold), 0.0), 1.0)
        self.prototype_target_confidence_power = max(float(prototype_target_confidence_power), 0.0)
        self.prototype_probability_temperature = max(float(prototype_probability_temperature), 1e-3)
        self.prototype_class_balance = bool(prototype_class_balance)
        self.prototype_class_balance_clip_min = max(float(prototype_class_balance_clip_min), 0.0)
        self.prototype_class_balance_clip_max = max(float(prototype_class_balance_clip_max), 0.0)

    def current_prototype_weight(self) -> float:
        return _ramped_weight(
            self.prototype_weight,
            step=self.global_step,
            start_step=self.prototype_start_step,
            warmup_steps=self.prototype_warmup_steps,
        )

    def _prototype_residual_loss(
        self,
        *,
        source_y: torch.Tensor,
        logits_target: torch.Tensor,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        current_weight = self.current_prototype_weight()
        zero = logits_target.new_zeros(())
        metrics = {
            "lambda_prototype": float(current_weight),
            "loss_prototype": 0.0,
            "loss_prototype_source": 0.0,
            "loss_prototype_target": 0.0,
            "prototype_acceptance": 0.0,
            "prototype_present_classes": 0.0,
        }
        if current_weight <= 0:
            return zero, metrics

        num_classes = int(logits_target.shape[1])
        source_features_for_loss = _prototype_features(features_source, self.prototype_distance)
        target_features_for_loss = _prototype_features(features_target, self.prototype_distance)
        source_features_for_proto = _prototype_features(features_source.detach(), self.prototype_distance)
        prototypes, present = _class_prototypes(
            source_features_for_proto,
            source_y,
            num_classes=num_classes,
        )
        if not bool(present.any()):
            return zero, metrics

        valid_source = (source_y >= 0) & (source_y < num_classes)
        if bool(valid_source.any()):
            source_residual = (
                source_features_for_loss[valid_source]
                - prototypes[source_y[valid_source].detach().long()]
            ).pow(2).sum(dim=1).mean()
        else:
            source_residual = zero

        probabilities = F.softmax(
            logits_target.detach().float() / self.prototype_probability_temperature,
            dim=1,
        ).to(dtype=logits_target.dtype)
        confidence = probabilities.max(dim=1).values
        target_gate = confidence >= self.prototype_confidence_threshold
        present_weights = present.to(device=probabilities.device, dtype=probabilities.dtype)
        masked_probabilities = probabilities * present_weights.view(1, -1)
        probability_mass = masked_probabilities.sum(dim=1).clamp_min(EPS)
        masked_probabilities = masked_probabilities / probability_mass.unsqueeze(1)
        prototype_distances = torch.cdist(target_features_for_loss, prototypes.to(target_features_for_loss), p=2).pow(2)
        target_residual_per_sample = (masked_probabilities * prototype_distances).sum(dim=1)
        target_weights = confidence.clamp_min(0.0).pow(self.prototype_target_confidence_power)
        target_weights = target_weights * target_gate.to(device=target_weights.device, dtype=target_weights.dtype)
        prototype_balance_mean = 1.0
        if self.prototype_class_balance and bool(target_gate.any()):
            pseudo_labels = masked_probabilities.detach().argmax(dim=1)
            class_weights = _inverse_sqrt_class_weights(
                pseudo_labels,
                mask=target_gate,
                num_classes=num_classes,
                device=target_weights.device,
                dtype=target_weights.dtype,
            )
            if self.prototype_class_balance_clip_min > 0:
                class_weights = class_weights.clamp_min(self.prototype_class_balance_clip_min)
            if self.prototype_class_balance_clip_max > 0:
                class_weights = class_weights.clamp_max(self.prototype_class_balance_clip_max)
            sample_balance = class_weights[pseudo_labels.to(device=target_weights.device, dtype=torch.long)]
            target_weights = target_weights * sample_balance
            prototype_balance_mean = float(sample_balance[target_gate].float().mean().detach().item())
        if target_weights.sum() > 0:
            target_residual = (
                target_residual_per_sample * target_weights.to(dtype=target_residual_per_sample.dtype)
            ).sum() / target_weights.sum().clamp_min(EPS)
        else:
            target_residual = zero

        combined = self.prototype_source_weight * source_residual + self.prototype_target_weight * target_residual
        loss = current_weight * combined
        metrics.update(
            {
                "loss_prototype": float(combined.detach().item()),
                "loss_prototype_source": float(source_residual.detach().item()),
                "loss_prototype_target": float(target_residual.detach().item()),
                "prototype_acceptance": float(target_gate.float().mean().detach().item()),
                "prototype_present_classes": float(present.float().sum().detach().item()),
                "prototype_probability_temperature": float(self.prototype_probability_temperature),
                "prototype_class_balance": float(self.prototype_class_balance),
                "prototype_balance_mean": prototype_balance_mean,
                "target_confidence": float(confidence.mean().detach().item()),
            }
        )
        return loss, metrics

    def extra_regularization_loss(
        self,
        *,
        source_y: torch.Tensor,
        logits_target: torch.Tensor,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self._prototype_residual_loss(
            source_y=source_y,
            logits_target=logits_target,
            features_source=features_source,
            features_target=features_target,
        )


class CBTPDeepJDOTMethod(TPDeepJDOTMethod):
    """TP-DeepJDOT plus conservative confidence-balanced target regularization."""

    method_name = "cbtp_deepjdot"

    def __init__(
        self,
        *,
        pseudo_weight: float = 0.015,
        pseudo_start_step: int | None = None,
        pseudo_warmup_steps: int = 1000,
        tau_start: float = 0.97,
        tau_end: float = 0.92,
        tau_steps: int = 1200,
        class_balance_clip_min: float = 0.5,
        class_balance_clip_max: float = 2.5,
        pseudo_confidence_power: float = 1.0,
        pseudo_min_acceptance: float = 0.0,
        pseudo_min_classes: int = 1,
        target_im_weight: float = 0.0,
        target_im_start_step: int | None = None,
        target_im_warmup_steps: int = 1000,
        target_im_temperature: float = 1.0,
        target_im_entropy_weight: float = 1.0,
        target_im_diversity_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pseudo_weight = float(pseudo_weight)
        self.pseudo_start_step = (
            self.prototype_start_step if pseudo_start_step is None else max(int(pseudo_start_step), 0)
        )
        self.pseudo_warmup_steps = max(int(pseudo_warmup_steps), 1)
        self.tau_start = min(max(float(tau_start), 0.0), 1.0)
        self.tau_end = min(max(float(tau_end), 0.0), 1.0)
        self.tau_steps = max(int(tau_steps), 1)
        self.class_balance_clip_min = max(float(class_balance_clip_min), 0.0)
        self.class_balance_clip_max = max(float(class_balance_clip_max), 0.0)
        self.pseudo_confidence_power = max(float(pseudo_confidence_power), 0.0)
        self.pseudo_min_acceptance = min(max(float(pseudo_min_acceptance), 0.0), 1.0)
        self.pseudo_min_classes = max(int(pseudo_min_classes), 1)
        self.target_im_weight = float(target_im_weight)
        self.target_im_start_step = (
            self.prototype_start_step if target_im_start_step is None else max(int(target_im_start_step), 0)
        )
        self.target_im_warmup_steps = max(int(target_im_warmup_steps), 1)
        self.target_im_temperature = max(float(target_im_temperature), 1e-3)
        self.target_im_entropy_weight = float(target_im_entropy_weight)
        self.target_im_diversity_weight = float(target_im_diversity_weight)

    def current_tau(self) -> float:
        pseudo_step = max(self.global_step - self.pseudo_start_step, 0)
        progress = min(pseudo_step / float(self.tau_steps), 1.0)
        return self.tau_start + progress * (self.tau_end - self.tau_start)

    def current_pseudo_weight(self) -> float:
        return _ramped_weight(
            self.pseudo_weight,
            step=self.global_step,
            start_step=self.pseudo_start_step,
            warmup_steps=self.pseudo_warmup_steps,
        )

    def current_target_im_weight(self) -> float:
        return _ramped_weight(
            self.target_im_weight,
            step=self.global_step,
            start_step=self.target_im_start_step,
            warmup_steps=self.target_im_warmup_steps,
        )

    def _target_regularization_loss(
        self,
        logits_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        train_probabilities = F.softmax(logits_target.float() / self.target_im_temperature, dim=1)
        detached_probabilities = train_probabilities.detach()
        confidence, pseudo_labels = detached_probabilities.max(dim=1)
        tau = self.current_tau()
        mask = confidence >= tau
        current_weight = self.current_pseudo_weight()
        current_im_weight = self.current_target_im_weight()
        zero = logits_target.new_zeros(())
        num_classes = int(logits_target.shape[1])
        accepted_classes = int(pseudo_labels[mask].unique().numel()) if bool(mask.any()) else 0
        pseudo_acceptance = float(mask.float().mean().detach().item())
        metrics = {
            "lambda_pseudo": float(current_weight),
            "lambda_target_im": float(current_im_weight),
            "tau": float(tau),
            "pseudo_acceptance": pseudo_acceptance,
            "pseudo_accepted_classes": float(accepted_classes),
            "loss_pseudo": 0.0,
            "loss_target_im": 0.0,
            "loss_target_entropy": 0.0,
            "loss_target_diversity": 0.0,
        }
        target_loss = zero

        if current_im_weight > 0:
            log_probabilities = train_probabilities.clamp_min(EPS).log()
            normalizer = torch.log(logits_target.new_tensor(float(max(num_classes, 2)))).to(
                dtype=train_probabilities.dtype
            )
            entropy_loss = -(train_probabilities * log_probabilities).sum(dim=1).mean() / normalizer
            mean_probabilities = train_probabilities.mean(dim=0).clamp_min(EPS)
            diversity_loss = (mean_probabilities * mean_probabilities.log()).sum() / normalizer
            im_loss = (
                self.target_im_entropy_weight * entropy_loss
                + self.target_im_diversity_weight * diversity_loss
            )
            target_loss = target_loss + current_im_weight * im_loss.to(dtype=logits_target.dtype)
            metrics.update(
                {
                    "loss_target_im": float(im_loss.detach().item()),
                    "loss_target_entropy": float(entropy_loss.detach().item()),
                    "loss_target_diversity": float(diversity_loss.detach().item()),
                }
            )

        pseudo_gate_open = (
            current_weight > 0
            and bool(mask.any())
            and pseudo_acceptance >= self.pseudo_min_acceptance
            and accepted_classes >= self.pseudo_min_classes
        )
        metrics["pseudo_gate_open"] = float(pseudo_gate_open)
        if not pseudo_gate_open:
            return target_loss, metrics

        class_weights = _inverse_sqrt_class_weights(
            pseudo_labels,
            mask=mask,
            num_classes=num_classes,
            device=logits_target.device,
            dtype=logits_target.dtype,
        )
        if self.class_balance_clip_min > 0:
            class_weights = class_weights.clamp_min(self.class_balance_clip_min)
        if self.class_balance_clip_max > 0:
            class_weights = class_weights.clamp_max(self.class_balance_clip_max)

        pseudo_losses = F.cross_entropy(
            logits_target[mask],
            pseudo_labels[mask].to(device=logits_target.device, dtype=torch.long),
            reduction="none",
        )
        accepted_confidence = confidence[mask].to(device=logits_target.device, dtype=logits_target.dtype)
        sample_weights = class_weights[pseudo_labels[mask].to(device=logits_target.device, dtype=torch.long)]
        sample_weights = sample_weights * accepted_confidence.clamp_min(0.0).pow(self.pseudo_confidence_power)
        pseudo_loss = (pseudo_losses * sample_weights).sum() / sample_weights.sum().clamp_min(EPS)
        metrics["loss_pseudo"] = float(pseudo_loss.detach().item())
        return target_loss + current_weight * pseudo_loss, metrics

    def extra_regularization_loss(
        self,
        *,
        source_y: torch.Tensor,
        logits_target: torch.Tensor,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prototype_loss, metrics = self._prototype_residual_loss(
            source_y=source_y,
            logits_target=logits_target,
            features_source=features_source,
            features_target=features_target,
        )
        target_loss, target_metrics = self._target_regularization_loss(logits_target)
        metrics.update(target_metrics)
        return prototype_loss + target_loss, metrics


class _DeepJDOTTemporalAugmenter:
    """Conservative weak/strong augmentations for target process windows."""

    def __init__(
        self,
        *,
        weak_jitter_std: float = 0.006,
        weak_scaling_std: float = 0.006,
        strong_jitter_std: float = 0.014,
        strong_scaling_std: float = 0.014,
        strong_time_mask_ratio: float = 0.06,
        strong_channel_dropout_prob: float = 0.05,
    ) -> None:
        self.weak_jitter_std = max(float(weak_jitter_std), 0.0)
        self.weak_scaling_std = max(float(weak_scaling_std), 0.0)
        self.strong_jitter_std = max(float(strong_jitter_std), 0.0)
        self.strong_scaling_std = max(float(strong_scaling_std), 0.0)
        self.strong_time_mask_ratio = min(max(float(strong_time_mask_ratio), 0.0), 1.0)
        self.strong_channel_dropout_prob = min(max(float(strong_channel_dropout_prob), 0.0), 1.0)

    @staticmethod
    def _jitter(x: torch.Tensor, std: float) -> torch.Tensor:
        return x if std <= 0 else x + torch.randn_like(x) * std

    @staticmethod
    def _scale(x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        scale = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) * std
        return x * scale

    @staticmethod
    def _time_mask(x: torch.Tensor, ratio: float) -> torch.Tensor:
        if ratio <= 0:
            return x
        mask_length = max(int(round(x.shape[-1] * ratio)), 1)
        if mask_length >= x.shape[-1]:
            return torch.zeros_like(x)
        masked = x.clone()
        max_start = x.shape[-1] - mask_length
        starts = torch.randint(0, max_start + 1, (x.shape[0],), device=x.device)
        for sample_index, start in enumerate(starts.tolist()):
            masked[sample_index, :, start : start + mask_length] = 0.0
        return masked

    @staticmethod
    def _channel_dropout(x: torch.Tensor, probability: float) -> torch.Tensor:
        if probability <= 0:
            return x
        keep = (
            torch.rand(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype) >= probability
        ).to(dtype=x.dtype)
        return x * keep

    def weak(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale(self._jitter(x, self.weak_jitter_std), self.weak_scaling_std)

    def strong(self, x: torch.Tensor) -> torch.Tensor:
        x = self._scale(self._jitter(x, self.strong_jitter_std), self.strong_scaling_std)
        x = self._time_mask(x, self.strong_time_mask_ratio)
        return self._channel_dropout(x, self.strong_channel_dropout_prob)

    def pair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weak(x), self.strong(x)


class TPUDeepJDOTMethod(DeepJDOTMethod):
    """Temporal-prototypical unbalanced DeepJDOT without target pseudo labels."""

    method_name = "tpu_deepjdot"

    def __init__(
        self,
        *,
        num_classes: int,
        prototype_cost_weight: float = 0.015,
        prototype_start_step: int = 500,
        prototype_warmup_steps: int = 800,
        prototype_distance: str = "normalized_l2",
        prototype_momentum: float = 0.95,
        prototype_relative_min: float = -1.0,
        prototype_relative_max: float = 3.0,
        temporal_cost_weight: float = 0.01,
        temporal_start_step: int | None = None,
        temporal_warmup_steps: int = 800,
        supcon_weight: float = 0.04,
        supcon_temperature: float = 0.10,
        source_warmup_steps: int = 500,
        supcon_warmup_only: bool = True,
        alignment_start_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("unbalanced_transport", True)
        kwargs.setdefault("use_ce_cost_for_plan", True)
        super().__init__(num_classes=num_classes, **kwargs)
        self.prototype_cost_weight = float(prototype_cost_weight)
        self.prototype_start_step = max(int(prototype_start_step), 0)
        self.prototype_warmup_steps = max(int(prototype_warmup_steps), 1)
        self.prototype_distance = str(prototype_distance)
        self.prototype_momentum = min(max(float(prototype_momentum), 0.0), 0.9999)
        self.prototype_relative_min = float(prototype_relative_min)
        self.prototype_relative_max = float(prototype_relative_max)
        self.temporal_cost_weight = float(temporal_cost_weight)
        self.temporal_start_step = self.prototype_start_step if temporal_start_step is None else max(int(temporal_start_step), 0)
        self.temporal_warmup_steps = max(int(temporal_warmup_steps), 1)
        self.supcon_weight = float(supcon_weight)
        self.supcon_temperature = max(float(supcon_temperature), EPS)
        self.source_warmup_steps = max(int(source_warmup_steps), 0)
        self.supcon_warmup_only = bool(supcon_warmup_only)
        self.alignment_start_step = self.source_warmup_steps if alignment_start_step is None else max(int(alignment_start_step), 0)
        self.register_buffer("source_prototypes", torch.zeros(num_classes, self.encoder.out_dim))
        self.register_buffer("source_prototype_active", torch.zeros(num_classes, dtype=torch.bool))

    def _scheduled_weight(self, base_weight: float, *, start_step: int, warmup_steps: int) -> float:
        return _ramped_weight(
            base_weight,
            step=self.global_step,
            start_step=start_step,
            warmup_steps=warmup_steps,
        )

    def current_prototype_cost_weight(self) -> float:
        return self._scheduled_weight(
            self.prototype_cost_weight,
            start_step=self.prototype_start_step,
            warmup_steps=self.prototype_warmup_steps,
        )

    def current_temporal_cost_weight(self) -> float:
        return self._scheduled_weight(
            self.temporal_cost_weight,
            start_step=self.temporal_start_step,
            warmup_steps=self.temporal_warmup_steps,
        )

    def current_supcon_weight(self) -> float:
        if self.supcon_warmup_only and self.source_warmup_steps > 0 and self.global_step > self.source_warmup_steps:
            return 0.0
        return max(float(self.supcon_weight), 0.0)

    def current_alignment_weight(self) -> float:
        scheduled = self.alignment_scheduler.step()
        if self.global_step <= self.alignment_start_step:
            return 0.0
        return scheduled

    @torch.no_grad()
    def _update_source_prototypes(self, features_source: torch.Tensor, source_y: torch.Tensor) -> None:
        proto_features = _prototype_features(features_source.detach(), self.prototype_distance)
        batch_prototypes, present = _class_prototypes(proto_features, source_y, num_classes=self.num_classes)
        present = present.to(device=self.source_prototype_active.device)
        for class_id in torch.where(present)[0].tolist():
            candidate = F.normalize(batch_prototypes[class_id], p=2, dim=0, eps=1e-6)
            if bool(self.source_prototype_active[class_id].item()):
                updated = (
                    self.prototype_momentum * self.source_prototypes[class_id].to(dtype=candidate.dtype)
                    + (1.0 - self.prototype_momentum) * candidate
                )
                self.source_prototypes[class_id].copy_(F.normalize(updated, p=2, dim=0, eps=1e-6).to(self.source_prototypes.dtype))
            else:
                self.source_prototypes[class_id].copy_(candidate.to(self.source_prototypes.dtype))
            self.source_prototype_active[class_id] = True

    def _compute_tpu_transport(
        self,
        *,
        source_y: torch.Tensor,
        logits_target: torch.Tensor,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> _TransportOutput:
        prototype_weight = self.current_prototype_cost_weight()
        temporal_weight = self.current_temporal_cost_weight()
        return _transport_with_diagnostics(
            source_labels=source_y,
            logits_target=logits_target,
            features_source=features_source,
            features_target=features_target,
            reg_dist=self.reg_dist,
            reg_cl=self.reg_cl,
            normalize_feature_cost=self.normalize_feature_cost,
            solver=self.transport_solver,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_num_iter_max=self.sinkhorn_num_iter_max,
            unbalanced=True,
            uot_tau_s=self.uot_tau_s,
            uot_tau_t=self.uot_tau_t,
            use_ce_cost_for_plan=True,
            source_prototypes=self.source_prototypes,
            source_prototype_active=self.source_prototype_active,
            prototype_distance=self.prototype_distance,
            prototype_cost_weight=prototype_weight,
            prototype_relative_min=self.prototype_relative_min,
            prototype_relative_max=self.prototype_relative_max,
            source_x=source_x,
            target_x=target_x,
            temporal_cost_weight=temporal_weight,
        )

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        self.global_step += 1
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)
        self._update_source_prototypes(features_source.detach(), source_y)

        loss_cls = F.cross_entropy(logits_source, source_y)
        source_supcon_loss = _supervised_contrastive_loss(
            _prototype_features(features_source, self.prototype_distance),
            source_y,
            temperature=self.supcon_temperature,
        ).to(device=loss_cls.device, dtype=loss_cls.dtype)
        lambda_supcon = self.current_supcon_weight()
        transport_output = self._compute_tpu_transport(
            source_y=source_y,
            logits_target=logits_target,
            features_source=features_source,
            features_target=features_target,
            source_x=source_x,
            target_x=target_x,
        )
        loss_alignment = transport_output.loss.to(device=loss_cls.device, dtype=loss_cls.dtype)
        lambda_alignment = self.current_alignment_weight()
        loss_total = loss_cls + lambda_supcon * source_supcon_loss + lambda_alignment * loss_alignment
        if not torch.isfinite(loss_total).item():
            raise RuntimeError("TPU-DeepJDOT produced a non-finite total loss.")

        metrics = {
            "loss_total": float(loss_total.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_alignment": float(loss_alignment.item()),
            "lambda_alignment": float(lambda_alignment),
            "loss_source_supcon": float(source_supcon_loss.detach().item()),
            "source_supcon_loss": float(source_supcon_loss.detach().item()),
            "lambda_source_supcon": float(lambda_supcon),
            "lambda_prototype_cost": float(self.current_prototype_cost_weight()),
            "lambda_temporal_cost": float(self.current_temporal_cost_weight()),
            "source_prototype_active_classes": float(self.source_prototype_active.float().sum().item()),
            "acc_source": accuracy_from_logits(logits_source, source_y),
            **transport_output.metrics,
        }
        return MethodStepOutput(loss=loss_total, metrics=metrics)


class CBTPUDeepJDOTMethod(TPUDeepJDOTMethod):
    """Confidence-balanced TPU-DeepJDOT with three-way target pseudo semantics."""

    method_name = "cbtpu_deepjdot"

    def __init__(
        self,
        *,
        num_classes: int,
        teacher_ema_decay: float = 0.995,
        teacher_temperature: float = 1.0,
        proto_temperature: float = 0.20,
        q_ot_power: float = 1.0,
        q_cls_power: float = 1.0,
        q_proto_power: float = 1.0,
        pseudo_weight: float = 0.015,
        pseudo_start_step: int = 800,
        pseudo_warmup_steps: int = 1000,
        tau_start: float = 0.95,
        tau_end: float = 0.85,
        tau_steps: int = 1500,
        q_ot_entropy_threshold: float = 0.70,
        js_threshold: float = 0.08,
        consistency_weight: float = 0.02,
        consistency_start_step: int = 800,
        consistency_warmup_steps: int = 1000,
        consistency_mid_tau: float = 0.55,
        logit_adjustment_eta: float = 0.10,
        infomax_weight: float = 0.0,
        infomax_start_ratio: float = 0.70,
        infomax_start_step: int | None = None,
        infomax_warmup_steps: int = 500,
        augment_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout, **kwargs)
        self.teacher_ema_decay = min(max(float(teacher_ema_decay), 0.0), 0.9999)
        self.teacher_temperature = max(float(teacher_temperature), EPS)
        self.proto_temperature = max(float(proto_temperature), EPS)
        self.q_ot_power = max(float(q_ot_power), 0.0)
        self.q_cls_power = max(float(q_cls_power), 0.0)
        self.q_proto_power = max(float(q_proto_power), 0.0)
        self.pseudo_weight = float(pseudo_weight)
        self.pseudo_start_step = max(int(pseudo_start_step), 0)
        self.pseudo_warmup_steps = max(int(pseudo_warmup_steps), 1)
        self.tau_start = min(max(float(tau_start), 0.0), 1.0)
        self.tau_end = min(max(float(tau_end), 0.0), 1.0)
        self.tau_steps = max(int(tau_steps), 1)
        self.q_ot_entropy_threshold = max(float(q_ot_entropy_threshold), 0.0)
        self.js_threshold = max(float(js_threshold), 0.0)
        self.consistency_weight = float(consistency_weight)
        self.consistency_start_step = max(int(consistency_start_step), 0)
        self.consistency_warmup_steps = max(int(consistency_warmup_steps), 1)
        self.consistency_mid_tau = min(max(float(consistency_mid_tau), 0.0), 1.0)
        self.logit_adjustment_eta = max(float(logit_adjustment_eta), 0.0)
        self.infomax_weight = float(infomax_weight)
        default_infomax_start = int(round(float(infomax_start_ratio) * float(max(self.alignment_scheduler.max_steps, 1))))
        self.infomax_start_step = default_infomax_start if infomax_start_step is None else max(int(infomax_start_step), 0)
        self.infomax_warmup_steps = max(int(infomax_warmup_steps), 1)
        self.augmenter = _DeepJDOTTemporalAugmenter(**(augment_kwargs or {}))
        self.teacher_encoder = deepcopy(self.encoder)
        self.teacher_classifier = deepcopy(self.classifier)
        self._set_teacher_requires_grad(False)
        self.teacher_encoder.eval()
        self.teacher_classifier.eval()

    def train(self, mode: bool = True) -> "CBTPUDeepJDOTMethod":
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_classifier.eval()
        return self

    def _set_teacher_requires_grad(self, requires_grad: bool) -> None:
        for module in (self.teacher_encoder, self.teacher_classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(requires_grad)

    def _teacher_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.teacher_encoder(x)
            logits = self.teacher_classifier(features)
        return logits, features

    def _teacher_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float() / self.teacher_temperature, dim=1)

    def current_tau(self) -> float:
        pseudo_step = max(self.global_step - self.pseudo_start_step, 0)
        progress = min(pseudo_step / float(self.tau_steps), 1.0)
        return self.tau_start + progress * (self.tau_end - self.tau_start)

    def current_pseudo_weight(self) -> float:
        return _ramped_weight(
            self.pseudo_weight,
            step=self.global_step,
            start_step=self.pseudo_start_step,
            warmup_steps=self.pseudo_warmup_steps,
        )

    def current_consistency_weight(self) -> float:
        return _ramped_weight(
            self.consistency_weight,
            step=self.global_step,
            start_step=self.consistency_start_step,
            warmup_steps=self.consistency_warmup_steps,
        )

    def current_infomax_weight(self) -> float:
        return _ramped_weight(
            self.infomax_weight,
            step=self.global_step,
            start_step=self.infomax_start_step,
            warmup_steps=self.infomax_warmup_steps,
        )

    def _prototype_probabilities(self, features_target: torch.Tensor) -> torch.Tensor:
        active = self.source_prototype_active.to(device=features_target.device, dtype=torch.bool)
        distances = _class_target_distances(
            self.source_prototypes,
            features_target,
            distance=self.prototype_distance,
        ).t()
        if not bool(active.any()):
            return torch.full(
                (features_target.shape[0], self.num_classes),
                1.0 / float(max(self.num_classes, 1)),
                device=features_target.device,
                dtype=features_target.dtype,
            )
        distances = distances / self.proto_temperature
        inactive_fill = torch.full_like(distances, 1e4)
        distances = torch.where(active.view(1, -1), distances, inactive_fill)
        return torch.softmax(-distances.float(), dim=1).to(dtype=features_target.dtype)

    def _logit_adjusted(self, logits: torch.Tensor, accepted_labels: torch.Tensor) -> torch.Tensor:
        if self.logit_adjustment_eta <= 0 or accepted_labels.numel() == 0:
            return logits
        counts = torch.bincount(accepted_labels.detach().long(), minlength=self.num_classes).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        freq = (counts + EPS) / (counts.sum() + EPS * float(self.num_classes))
        return logits - self.logit_adjustment_eta * freq.clamp_min(EPS).log().view(1, -1)

    def _infomax_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits.float(), dim=1)
        normalizer = math.log(max(int(probabilities.shape[1]), 2))
        entropy_loss = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=1).mean() / normalizer
        mean_probabilities = probabilities.mean(dim=0).clamp_min(EPS)
        diversity_loss = (mean_probabilities * mean_probabilities.log()).sum() / normalizer
        return (entropy_loss + diversity_loss).to(dtype=logits.dtype)

    @torch.no_grad()
    def _ema_update_teacher(self) -> None:
        for teacher_parameter, student_parameter in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            teacher_parameter.data.mul_(self.teacher_ema_decay).add_(
                student_parameter.data,
                alpha=1.0 - self.teacher_ema_decay,
            )
        for teacher_parameter, student_parameter in zip(self.teacher_classifier.parameters(), self.classifier.parameters()):
            teacher_parameter.data.mul_(self.teacher_ema_decay).add_(
                student_parameter.data,
                alpha=1.0 - self.teacher_ema_decay,
            )
        for teacher_buffer, student_buffer in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
            teacher_buffer.copy_(student_buffer)
        for teacher_buffer, student_buffer in zip(self.teacher_classifier.buffers(), self.classifier.buffers()):
            teacher_buffer.copy_(student_buffer)

    def after_optimizer_step(self) -> dict[str, float]:
        self._ema_update_teacher()
        return {
            "teacher_ema_decay": float(self.teacher_ema_decay),
            "source_prototype_active_classes": float(self.source_prototype_active.float().sum().item()),
        }

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        self.global_step += 1
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        weak_target, strong_target = self.augmenter.pair(target_x)

        logits_source, features_source = self.forward(source_x)
        logits_target_weak, features_target_weak = self.forward(weak_target)
        logits_target_strong, features_target_strong = self.forward(strong_target)
        teacher_logits_weak, _ = self._teacher_forward(weak_target)
        q_cls = self._teacher_probabilities(teacher_logits_weak).detach()
        self._update_source_prototypes(features_source.detach(), source_y)

        loss_cls = F.cross_entropy(logits_source, source_y)
        source_supcon_loss = _supervised_contrastive_loss(
            _prototype_features(features_source, self.prototype_distance),
            source_y,
            temperature=self.supcon_temperature,
        ).to(device=loss_cls.device, dtype=loss_cls.dtype)
        lambda_supcon = self.current_supcon_weight()
        transport_output = self._compute_tpu_transport(
            source_y=source_y,
            logits_target=logits_target_weak,
            features_source=features_source,
            features_target=features_target_weak,
            source_x=source_x,
            target_x=weak_target,
        )
        loss_alignment = transport_output.loss.to(device=loss_cls.device, dtype=loss_cls.dtype)
        lambda_alignment = self.current_alignment_weight()

        q_ot = transport_output.q_ot.to(device=logits_target_strong.device, dtype=logits_target_strong.dtype)
        q_proto = self._prototype_probabilities(features_target_weak.detach()).to(
            device=logits_target_strong.device,
            dtype=logits_target_strong.dtype,
        )
        q_ot = _renormalize_probabilities(q_ot)
        q_cls = _renormalize_probabilities(q_cls).to(device=q_ot.device, dtype=q_ot.dtype)
        q_proto = _renormalize_probabilities(q_proto).to(device=q_ot.device, dtype=q_ot.dtype)
        log_mix = (
            self.q_ot_power * q_ot.clamp_min(EPS).log()
            + self.q_cls_power * q_cls.to(dtype=q_ot.dtype).clamp_min(EPS).log()
            + self.q_proto_power * q_proto.clamp_min(EPS).log()
        )
        log_mix = torch.nan_to_num(log_mix.float(), nan=-60.0, posinf=60.0, neginf=-60.0)
        q_mix = _renormalize_probabilities(torch.softmax(log_mix, dim=1)).to(
            device=logits_target_strong.device,
            dtype=logits_target_strong.dtype,
        )
        q_ot_labels = q_ot.argmax(dim=1)
        q_cls_labels = q_cls.argmax(dim=1)
        q_proto_labels = q_proto.argmax(dim=1)
        q_mix_confidence, q_mix_labels = q_mix.max(dim=1)
        all_agree = (q_ot_labels == q_cls_labels) & (q_cls_labels == q_proto_labels)
        js_disagreement = _js_disagreement((q_ot.float(), q_cls.float(), q_proto.float())).to(device=q_mix.device)
        q_ot_entropy = transport_output.q_ot_entropy.to(device=q_mix.device, dtype=q_mix.dtype)
        tau = self.current_tau()
        accept_mask = (
            (all_agree | (js_disagreement <= self.js_threshold))
            & (q_mix_confidence >= tau)
            & (q_ot_entropy <= self.q_ot_entropy_threshold)
        )
        mid_confidence_mask = (
            (~accept_mask)
            & (q_mix_confidence >= self.consistency_mid_tau)
            & (q_ot_entropy <= self.q_ot_entropy_threshold)
        )
        consistency_mask = accept_mask | mid_confidence_mask

        lambda_pseudo = self.current_pseudo_weight()
        lambda_consistency = self.current_consistency_weight()
        lambda_infomax = self.current_infomax_weight()
        zero = loss_cls.new_zeros(())
        pseudo_loss = zero
        if lambda_pseudo > 0 and bool(accept_mask.any()):
            accepted_labels = q_mix_labels[accept_mask].detach().long()
            adjusted_logits = self._logit_adjusted(logits_target_strong[accept_mask].float(), accepted_labels)
            adjusted_logits = torch.nan_to_num(adjusted_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            accepted_targets = _renormalize_probabilities(q_mix[accept_mask].float()).to(
                device=adjusted_logits.device,
                dtype=adjusted_logits.dtype,
            )
            pseudo_values = torch.nan_to_num(_soft_label_ce(adjusted_logits, accepted_targets), nan=0.0, posinf=1e4, neginf=0.0)
            pseudo_loss = pseudo_values.mean().to(dtype=loss_cls.dtype)

        consistency_loss = zero
        if lambda_consistency > 0 and bool(consistency_mask.any()):
            consistency_values = F.kl_div(
                F.log_softmax(logits_target_strong[consistency_mask].float(), dim=1),
                q_cls[consistency_mask].float().detach(),
                reduction="none",
            ).sum(dim=1)
            consistency_loss = consistency_values.mean().to(dtype=loss_cls.dtype)

        infomax_loss = self._infomax_loss(logits_target_strong) if lambda_infomax > 0 else zero
        loss_total = (
            loss_cls
            + lambda_supcon * source_supcon_loss
            + lambda_alignment * loss_alignment
            + lambda_pseudo * pseudo_loss
            + lambda_consistency * consistency_loss
            + lambda_infomax * infomax_loss
        )
        if not torch.isfinite(loss_total).item():
            raise RuntimeError(
                "CBTPU-DeepJDOT produced a non-finite total loss: "
                f"loss_cls={float(loss_cls.detach().float().item())}, "
                f"source_supcon={float(source_supcon_loss.detach().float().item())}, "
                f"alignment={float(loss_alignment.detach().float().item())}, "
                f"pseudo={float(pseudo_loss.detach().float().item())}, "
                f"consistency={float(consistency_loss.detach().float().item())}, "
                f"infomax={float(infomax_loss.detach().float().item())}, "
                f"lambda_alignment={lambda_alignment}, "
                f"lambda_pseudo={lambda_pseudo}, "
                f"lambda_consistency={lambda_consistency}."
            )

        accepted_ratio = float(accept_mask.float().mean().detach().item())
        consistency_ratio = float(consistency_mask.float().mean().detach().item())
        accepted_hist = torch.bincount(
            q_mix_labels[accept_mask].detach().long(),
            minlength=self.num_classes,
        ).to(device=logits_target_strong.device, dtype=torch.float32)
        if accepted_hist.sum() > 0:
            accepted_hist = accepted_hist / accepted_hist.sum().clamp_min(1.0)
        metrics = {
            "loss_total": float(loss_total.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_alignment": float(loss_alignment.item()),
            "lambda_alignment": float(lambda_alignment),
            "loss_source_supcon": float(source_supcon_loss.detach().item()),
            "source_supcon_loss": float(source_supcon_loss.detach().item()),
            "lambda_source_supcon": float(lambda_supcon),
            "lambda_prototype_cost": float(self.current_prototype_cost_weight()),
            "lambda_temporal_cost": float(self.current_temporal_cost_weight()),
            "loss_pseudo": float(pseudo_loss.detach().item()),
            "lambda_pseudo": float(lambda_pseudo),
            "loss_consistency": float(consistency_loss.detach().item()),
            "lambda_consistency": float(lambda_consistency),
            "loss_infomax": float(infomax_loss.detach().item()),
            "lambda_infomax": float(lambda_infomax),
            "tau": float(tau),
            "q_ot_entropy_mean": float(q_ot_entropy.detach().mean().item()),
            "q_ot_q_cls_q_proto_agreement_rate": float(all_agree.float().mean().detach().item()),
            "q_ot_cls_proto_agreement_rate": float(all_agree.float().mean().detach().item()),
            "q_ot_q_cls_agreement_rate": float((q_ot_labels == q_cls_labels).float().mean().detach().item()),
            "q_ot_q_proto_agreement_rate": float((q_ot_labels == q_proto_labels).float().mean().detach().item()),
            "q_cls_q_proto_agreement_rate": float((q_cls_labels == q_proto_labels).float().mean().detach().item()),
            "js_disagreement_mean": float(js_disagreement.detach().mean().item()),
            "accepted_target_ratio": accepted_ratio,
            "pseudo_acceptance": accepted_ratio,
            "consistency_target_ratio": consistency_ratio,
            "q_mix_confidence_mean": float(q_mix_confidence.detach().mean().item()),
            "q_mix_entropy_mean": float(_normalized_entropy(q_mix.float()).mean().detach().item()),
            "target_pred_class_entropy": _target_prediction_class_entropy(torch.softmax(logits_target_strong.float(), dim=1)),
            "target_prediction_class_entropy": _target_prediction_class_entropy(torch.softmax(logits_target_strong.float(), dim=1)),
            "source_prototype_active_classes": float(self.source_prototype_active.float().sum().item()),
            "acc_source": accuracy_from_logits(logits_source, source_y),
            **transport_output.metrics,
        }
        for class_id in range(self.num_classes):
            metrics[f"accepted_pseudo_label_class_{class_id:02d}"] = float(accepted_hist[class_id].detach().item())
        return MethodStepOutput(loss=loss_total, metrics=metrics)
