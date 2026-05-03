"""Optimal transport losses used by JDOT/WJDOT and the proposed variants."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass
class OTLossConfig:
    feature_weight: float = 0.1
    label_weight: float = 1.0
    prototype_weight: float = 0.0
    prototype_coupling_weight: float | None = None
    prototype_in_coupling: bool | str = True
    prototype_mode: str = "legacy_pairwise"
    prototype_distance: str = "normalized_l2"
    prototype_cost_clip: str | float | None = None
    ot_class_entropy_gate: bool = False
    min_class_transport_mass: float = 1e-4
    class_mass_weight: str = "sqrt"
    class_mass_weight_min: float = 0.0
    class_mass_weight_max: float = 1.0
    relative_margin: float = 0.05
    relative_margin_temperature: float = 0.05
    relative_cost_min: float = 0.0
    relative_cost_max: float = 3.0
    normalize_costs: bool = True
    solver: str = "sinkhorn"
    sinkhorn_reg: float = 0.05
    sinkhorn_num_iter: int = 200
    unbalanced_transport: bool = False
    unbalanced_reg_m: float = 1.0


def normalize_cost(cost: torch.Tensor) -> torch.Tensor:
    scale = cost.detach().mean().clamp_min(EPS)
    return cost / scale


def _normalize_rows(features: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
    return F.normalize(values, p=2, dim=1, eps=1e-6)


def _parse_cost_clip(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text in {"none", "false", "off", "0"}:
            return None
        if text.startswith("p"):
            return min(max(float(text[1:]) / 100.0, 0.0), 1.0)
        if text.startswith("q"):
            return min(max(float(text[1:]), 0.0), 1.0)
        return float(text)
    return float(value)


def _clip_cost(cost: torch.Tensor, value: str | float | None) -> torch.Tensor:
    clip = _parse_cost_clip(value)
    if clip is None or clip <= 0:
        return cost
    finite = cost.detach()[torch.isfinite(cost.detach())]
    if finite.numel() == 0:
        return cost
    if clip <= 1.0:
        threshold = torch.quantile(finite.float(), clip).to(device=cost.device, dtype=cost.dtype)
    else:
        threshold = cost.new_tensor(clip)
    if not torch.isfinite(threshold):
        return cost
    return cost.clamp_max(threshold)


def _class_target_distances(
    source_prototypes: torch.Tensor,
    target_features: torch.Tensor,
    *,
    distance: str,
) -> torch.Tensor:
    metric = str(distance).strip().lower()
    prototypes = source_prototypes.detach().to(device=target_features.device, dtype=target_features.dtype)
    if metric in {"cosine", "cos"}:
        proto_norm = _normalize_rows(prototypes)
        target_norm = _normalize_rows(target_features)
        cost = 1.0 - proto_norm.mm(target_norm.t())
        return torch.nan_to_num(cost.clamp_min(0.0), nan=0.0, posinf=1e4, neginf=0.0)
    if metric in {"normalized_l2", "normalized_l2_sq", "l2_normalized", "cosine_l2"}:
        prototypes = _normalize_rows(prototypes)
        target_features = _normalize_rows(target_features)
    cost = torch.cdist(prototypes, target_features, p=2).pow(2)
    return torch.nan_to_num(cost, nan=0.0, posinf=1e4, neginf=0.0)


def _pairwise_prototype_cost(
    source_prototypes: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    *,
    distance: str,
) -> torch.Tensor:
    class_cost = _class_target_distances(
        source_prototypes,
        target_features,
        distance=distance,
    )
    return class_cost[source_labels.long()]


def _transport_class_mass(
    gamma: torch.Tensor,
    source_labels: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    class_mass = gamma.new_zeros((num_classes, gamma.shape[1]))
    class_mass.index_add_(0, source_labels.detach().long().clamp(0, num_classes - 1), gamma.detach())
    return class_mass


def _ot_class_entropy_gate(
    gamma: torch.Tensor,
    source_labels: torch.Tensor,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_mass = _transport_class_mass(gamma, source_labels, num_classes=num_classes)
    target_mass = gamma.detach().sum(dim=0).clamp_min(EPS)
    probabilities = class_mass / target_mass.view(1, -1)
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=0)
    if num_classes > 1:
        entropy = entropy / np.log(float(num_classes))
    gate = (1.0 - entropy).clamp(0.0, 1.0)
    return gate.to(dtype=gamma.dtype), entropy.to(dtype=gamma.dtype), class_mass


def _prototype_in_plan(value: bool | str, *, coupling_weight: float) -> bool:
    if coupling_weight <= 0:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"true", "1", "yes", "on", "late_only", "late"}
    return bool(value)


def _class_mass_weights(
    mass: torch.Tensor,
    *,
    mode: str,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    normalized = str(mode).strip().lower()
    values = mass.detach().clamp_min(0.0)
    if normalized in {"none", "uniform", "equal"}:
        weights = torch.ones_like(values)
    elif normalized in {"sqrt", "sqrt_or_clamp", "sqrt_clamp"}:
        weights = torch.sqrt(values.clamp_min(EPS))
    elif normalized in {"clamp", "clamped"}:
        weights = values
    else:
        weights = values
    if max_value > 0:
        weights = weights.clamp(max=float(max_value))
    if min_value > 0:
        weights = weights.clamp(min=float(min_value))
    return weights


def _prototype_vector_distance(
    values: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    distance: str,
) -> torch.Tensor:
    metric = str(distance).strip().lower()
    reference = prototypes.detach().to(device=values.device, dtype=values.dtype)
    if metric in {"cosine", "cos"}:
        return (1.0 - _normalize_rows(values).mul(_normalize_rows(reference)).sum(dim=1)).clamp_min(0.0)
    if metric in {"normalized_l2", "normalized_l2_sq", "l2_normalized", "cosine_l2"}:
        values = _normalize_rows(values)
        reference = _normalize_rows(reference)
    return (values - reference).pow(2).sum(dim=1)


def _safe_weights(size: int, weights: torch.Tensor | None, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if weights is None:
        values = torch.ones(size, device=device, dtype=dtype)
    else:
        values = weights[:size].to(device=device, dtype=dtype).clamp_min(0.0)
    total = values.sum()
    if not torch.isfinite(total) or total <= 0:
        values = torch.ones(size, device=device, dtype=dtype)
        total = values.sum()
    else:
        values = torch.where(values > 0, values, torch.full_like(values, EPS))
        total = values.sum()
    return values / total.clamp_min(EPS)


def _safe_unbalanced_source_weights(
    size: int,
    weights: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return non-negative source masses whose sum may be smaller than one."""

    if weights is None:
        return torch.full((size,), 1.0 / max(size, 1), device=device, dtype=dtype)
    values = weights[:size].to(device=device, dtype=dtype).clamp_min(0.0)
    if values.numel() == 0 or not torch.isfinite(values).all() or values.sum() <= 0:
        return torch.zeros(size, device=device, dtype=dtype)
    values = torch.where(values > 0, values, torch.full_like(values, EPS))
    return values / float(max(size, 1))


def solve_unbalanced_coupling(
    cost: torch.Tensor,
    *,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    solver: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
    unbalanced_reg_m: float = 1.0,
) -> torch.Tensor:
    """Solve an unbalanced OT coupling, allowing unreliable source mass to vanish."""

    source_size, target_size = int(cost.shape[0]), int(cost.shape[1])
    a = _safe_unbalanced_source_weights(source_size, source_weights, device=cost.device, dtype=cost.dtype)
    b = _safe_weights(target_size, target_weights, device=cost.device, dtype=cost.dtype)
    if a.sum() <= 0:
        return cost.new_zeros((source_size, target_size))
    fallback = torch.outer(a, b)
    try:
        import ot

        cost_np = torch.nan_to_num(cost.detach(), nan=1e6, posinf=1e6, neginf=1e6).double().cpu().numpy()
        a_np = a.detach().double().cpu().numpy()
        b_np = b.detach().double().cpu().numpy()
        solver_name = solver.strip().lower()
        if solver_name not in {"sinkhorn", "entropic", "regularized", "unbalanced", "sinkhorn_unbalanced"}:
            return fallback
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sinkhorn did not converge.*")
            warnings.filterwarnings("ignore", message="Numerical errors.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gamma_np = ot.unbalanced.sinkhorn_unbalanced(
                a_np,
                b_np,
                cost_np,
                reg=max(float(sinkhorn_reg), 1e-6),
                reg_m=max(float(unbalanced_reg_m), 1e-6),
                numItermax=200,
                stopThr=1e-6,
            )
        gamma = torch.as_tensor(gamma_np, device=cost.device, dtype=cost.dtype)
        if not torch.isfinite(gamma).all() or gamma.sum() < 0:
            return fallback
        return gamma.clamp_min(0.0)
    except Exception:
        return fallback


def solve_coupling(
    cost: torch.Tensor,
    *,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    solver: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
    sinkhorn_num_iter: int = 200,
) -> torch.Tensor:
    """Solve an OT coupling with POT, falling back to an outer product."""

    source_size, target_size = int(cost.shape[0]), int(cost.shape[1])
    a = _safe_weights(source_size, source_weights, device=cost.device, dtype=cost.dtype)
    b = _safe_weights(target_size, target_weights, device=cost.device, dtype=cost.dtype)
    fallback = torch.outer(a, b)
    try:
        import ot

        cost_np = torch.nan_to_num(cost.detach(), nan=1e6, posinf=1e6, neginf=1e6).double().cpu().numpy()
        a_np = a.detach().double().cpu().numpy()
        b_np = b.detach().double().cpu().numpy()
        solver_name = solver.strip().lower()
        if solver_name in {"emd", "exact"}:
            gamma_np = ot.emd(a_np, b_np, cost_np)
        elif solver_name in {"sinkhorn", "entropic", "regularized"}:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sinkhorn did not converge.*")
                gamma_np = ot.sinkhorn(
                    a_np,
                    b_np,
                    cost_np,
                    reg=max(float(sinkhorn_reg), 1e-6),
                    numItermax=max(int(sinkhorn_num_iter), 20),
                    stopThr=1e-6,
                )
        else:
            raise ValueError(f"Unsupported OT solver: {solver}")
        gamma = torch.as_tensor(gamma_np, device=cost.device, dtype=cost.dtype)
        if not torch.isfinite(gamma).all() or gamma.sum() <= 0:
            return fallback
        return gamma / gamma.sum().clamp_min(EPS)
    except Exception:
        return fallback


def compute_class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return class prototypes and a boolean mask of classes that are present."""

    feature_dim = int(features.shape[1])
    prototypes = features.new_zeros((num_classes, feature_dim))
    counts = features.new_zeros((num_classes, 1))
    if sample_weights is None:
        weights = features.new_ones((features.shape[0], 1))
    else:
        weights = sample_weights.to(device=features.device, dtype=features.dtype).view(-1, 1).clamp_min(0.0)
    for class_id in range(num_classes):
        mask = labels == class_id
        if not bool(mask.any()):
            continue
        class_weights = weights[mask]
        total = class_weights.sum().clamp_min(EPS)
        prototypes[class_id] = (features[mask] * class_weights).sum(dim=0) / total
        counts[class_id] = total
    return prototypes, counts.squeeze(1) > 0


def inverse_sqrt_class_weights(labels: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.detach().long().clamp_min(0), minlength=num_classes).to(
        device=labels.device,
        dtype=torch.float32,
    )
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts + EPS), torch.zeros_like(counts))
    present = counts > 0
    if bool(present.any()):
        weights[present] = weights[present] / weights[present].mean().clamp_min(EPS)
    return weights


def target_pseudo_class_weights(
    pseudo_labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    if not bool(mask.any()):
        return torch.ones(num_classes, device=pseudo_labels.device, dtype=torch.float32)
    counts = torch.bincount(pseudo_labels[mask].detach().long(), minlength=num_classes).to(
        device=pseudo_labels.device,
        dtype=torch.float32,
    )
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts + EPS), torch.zeros_like(counts))
    present = counts > 0
    if bool(present.any()):
        weights[present] = weights[present] / weights[present].mean().clamp_min(EPS)
    return torch.where(present, weights, torch.ones_like(weights))


def source_outlier_weights(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    proto = prototypes[labels.long()]
    dist = (features - proto).pow(2).sum(dim=1)
    scale = dist.detach().mean().clamp_min(EPS)
    weights = torch.exp(-dist.detach() / scale)
    return weights / weights.mean().clamp_min(EPS)


def jdot_transport_loss(
    *,
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_logits: torch.Tensor,
    num_classes: int,
    config: OTLossConfig,
    source_sample_weights: torch.Tensor | None = None,
    target_sample_weights: torch.Tensor | None = None,
    source_prototypes: torch.Tensor | None = None,
    label_gate: torch.Tensor | None = None,
    prototype_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a fixed-coupling JDOT-style minibatch loss.

    Target labels are never used. The label term is source-label cross entropy
    against the target predictions.
    """

    source_labels = source_labels.long()
    feature_cost = torch.cdist(source_features, target_features, p=2).pow(2)
    label_cost = -F.log_softmax(target_logits.float(), dim=1)[:, source_labels].transpose(0, 1)
    label_cost = label_cost.to(dtype=feature_cost.dtype)
    if label_gate is not None:
        label_cost = label_cost * label_gate.to(device=feature_cost.device, dtype=feature_cost.dtype).view(1, -1)

    prototype_mode = str(config.prototype_mode).strip().lower()
    prototype_weight = max(float(config.prototype_weight), 0.0)
    coupling_weight = (
        prototype_weight
        if config.prototype_coupling_weight is None
        else max(float(config.prototype_coupling_weight), 0.0)
    )
    use_prototypes = source_prototypes is not None and (prototype_weight > 0 or coupling_weight > 0)
    prototype_pair_cost = torch.zeros_like(feature_cost)
    prototype_plan_cost = torch.zeros_like(feature_cost)
    margin_gate = torch.ones(feature_cost.shape[1], device=feature_cost.device, dtype=feature_cost.dtype)

    if use_prototypes:
        assert source_prototypes is not None
        if prototype_mode == "tp_relative_margin":
            all_distances = _class_target_distances(
                source_prototypes,
                target_features,
                distance=config.prototype_distance,
            )
            present = torch.bincount(
                source_labels.detach().long().clamp_min(0),
                minlength=num_classes,
            ).to(device=feature_cost.device)[:num_classes] > 0
            valid_distances = all_distances[present]
            if valid_distances.shape[0] == 0:
                prototype_pair_cost = torch.zeros_like(feature_cost)
                prototype_plan_cost = torch.zeros_like(feature_cost)
            else:
                source_class_distances = all_distances[source_labels]
                valid_sum = valid_distances.sum(dim=0, keepdim=True)
                valid_count = float(valid_distances.shape[0])
                if valid_distances.shape[0] > 1:
                    other_mean = (valid_sum - source_class_distances) / max(valid_count - 1.0, 1.0)
                else:
                    other_mean = source_class_distances
                std = valid_distances.std(dim=0, keepdim=True, unbiased=False).clamp_min(EPS)
                relative_cost = (source_class_distances - other_mean) / std
                relative_cost = relative_cost.clamp(
                    min=float(config.relative_cost_min),
                    max=float(config.relative_cost_max),
                )
                sorted_distances = torch.sort(valid_distances, dim=0).values
                second_nearest = sorted_distances[min(1, sorted_distances.shape[0] - 1)]
                margin = float(config.relative_margin)
                temperature = max(float(config.relative_margin_temperature), EPS)
                margin_gate_pair = torch.sigmoid((second_nearest.view(1, -1) - source_class_distances - margin) / temperature)
                prototype_pair_cost = source_class_distances * margin_gate_pair
                prototype_plan_cost = relative_cost * (1.0 - margin_gate_pair.detach())
                margin_gate = margin_gate_pair.detach().mean(dim=0).to(dtype=feature_cost.dtype)
        else:
            prototype_pair_cost = _pairwise_prototype_cost(
                source_prototypes,
                source_labels,
                target_features,
                distance=config.prototype_distance,
            )
            prototype_plan_cost = prototype_pair_cost

        prototype_pair_cost = _clip_cost(prototype_pair_cost, config.prototype_cost_clip)
        prototype_plan_cost = _clip_cost(prototype_plan_cost, config.prototype_cost_clip)

    if config.normalize_costs:
        feature_plan = normalize_cost(feature_cost)
        label_plan = normalize_cost(label_cost)
        prototype_pair_plan = normalize_cost(prototype_pair_cost) if use_prototypes else prototype_pair_cost
        prototype_plan = normalize_cost(prototype_plan_cost) if use_prototypes else prototype_plan_cost
    else:
        feature_plan = feature_cost
        label_plan = label_cost
        prototype_pair_plan = prototype_pair_cost
        prototype_plan = prototype_plan_cost

    target_prototype_gate = None
    if prototype_gate is not None and use_prototypes:
        target_prototype_gate = prototype_gate.to(
            device=feature_cost.device,
            dtype=feature_cost.dtype,
        ).view(1, -1)
        prototype_pair_plan = prototype_pair_plan * target_prototype_gate
        prototype_plan = prototype_plan * target_prototype_gate

    base_plan_cost = float(config.feature_weight) * feature_plan + float(config.label_weight) * label_plan
    include_prototype_in_plan = _prototype_in_plan(config.prototype_in_coupling, coupling_weight=coupling_weight)
    plan_cost = base_plan_cost + float(coupling_weight) * prototype_plan if include_prototype_in_plan else base_plan_cost
    if bool(config.unbalanced_transport):
        gamma = solve_unbalanced_coupling(
            plan_cost,
            source_weights=source_sample_weights,
            target_weights=target_sample_weights,
            solver=config.solver,
            sinkhorn_reg=config.sinkhorn_reg,
            unbalanced_reg_m=config.unbalanced_reg_m,
        )
    else:
        gamma = solve_coupling(
            plan_cost,
            source_weights=source_sample_weights,
            target_weights=target_sample_weights,
            solver=config.solver,
            sinkhorn_reg=config.sinkhorn_reg,
            sinkhorn_num_iter=config.sinkhorn_num_iter,
        )

    entropy_gate = None
    entropy_values = None
    class_mass = None
    if bool(config.ot_class_entropy_gate) or prototype_mode == "tp_barycentric":
        entropy_gate, entropy_values, class_mass = _ot_class_entropy_gate(
            gamma,
            source_labels,
            num_classes=num_classes,
        )

    base_loss = (gamma.detach() * base_plan_cost).sum()
    prototype_loss = feature_cost.new_zeros(())
    active_classes = 0
    class_mass_total = 0.0
    if use_prototypes and prototype_weight > 0:
        if prototype_mode == "tp_barycentric":
            assert source_prototypes is not None
            if class_mass is None:
                _, _, class_mass = _ot_class_entropy_gate(gamma, source_labels, num_classes=num_classes)
            barycentric_mass = class_mass.to(device=target_features.device, dtype=target_features.dtype)
            if target_prototype_gate is not None:
                barycentric_mass = barycentric_mass * target_prototype_gate.detach()
            if bool(config.ot_class_entropy_gate) and entropy_gate is not None:
                barycentric_mass = barycentric_mass * entropy_gate.view(1, -1).detach()
            mass = barycentric_mass.sum(dim=1)
            valid = mass >= max(float(config.min_class_transport_mass), EPS)
            present = torch.bincount(
                source_labels.detach().long().clamp_min(0),
                minlength=num_classes,
            ).to(device=target_features.device)[:num_classes] > 0
            valid = valid & present
            if bool(valid.any()):
                barycenters = barycentric_mass.mm(target_features) / mass.clamp_min(EPS).view(-1, 1)
                distances = _prototype_vector_distance(
                    barycenters[valid],
                    source_prototypes.detach().to(device=target_features.device, dtype=target_features.dtype)[valid],
                    distance=config.prototype_distance,
                )
                distances = _clip_cost(distances, config.prototype_cost_clip)
                weights = _class_mass_weights(
                    mass[valid],
                    mode=config.class_mass_weight,
                    min_value=float(config.class_mass_weight_min),
                    max_value=float(config.class_mass_weight_max),
                ).to(device=target_features.device, dtype=target_features.dtype)
                prototype_loss = (weights * distances).sum() / weights.sum().clamp_min(EPS)
                active_classes = int(valid.sum().detach().item())
                class_mass_total = float(mass[valid].sum().detach().item())
        else:
            residual_plan = prototype_pair_plan
            if prototype_mode == "tp_relative_margin":
                residual_plan = prototype_pair_plan * margin_gate.view(1, -1)
            if bool(config.ot_class_entropy_gate) and entropy_gate is not None:
                residual_plan = residual_plan * entropy_gate.view(1, -1).detach()
            prototype_loss = (gamma.detach() * residual_plan).sum()

    loss = base_loss + prototype_weight * prototype_loss
    gamma_detached = gamma.detach()
    source_mass_by_class = gamma_detached.new_zeros(num_classes)
    source_mass_by_class.index_add_(
        0,
        source_labels.detach().long().clamp(0, num_classes - 1),
        gamma_detached.sum(dim=1),
    )
    per_class_ot_cost = gamma_detached.new_zeros(num_classes)
    weighted_cost_by_row = (gamma_detached * base_plan_cost.detach()).sum(dim=1)
    per_class_ot_cost.index_add_(
        0,
        source_labels.detach().long().clamp(0, num_classes - 1),
        weighted_cost_by_row,
    )
    per_class_ot_cost = per_class_ot_cost / source_mass_by_class.clamp_min(EPS)
    metrics = {
        "loss_ot": float(loss.detach().item()),
        "ot_feature_cost": float((gamma_detached * feature_plan).sum().detach().item()),
        "ot_label_cost": float((gamma_detached * label_plan).sum().detach().item()),
        "ot_prototype_cost": float(prototype_loss.detach().item()),
        "ot_prototype_plan_cost": float((gamma_detached * prototype_plan).sum().detach().item()),
        "ot_prototype_in_coupling": float(include_prototype_in_plan),
        "ot_prototype_mode_id": float(
            {
                "legacy_pairwise": 0,
                "tp_residual_safe": 1,
                "tp_barycentric": 2,
                "tp_relative_margin": 3,
            }.get(prototype_mode, -1)
        ),
        "ot_prototype_weight": float(prototype_weight),
        "ot_prototype_coupling_weight": float(coupling_weight),
        "ot_gamma_max": float(gamma_detached.max().item()),
        "ot_transport_mass": float(gamma_detached.sum().item()),
        "ot_unbalanced_transport": float(bool(config.unbalanced_transport)),
    }
    for class_id in range(num_classes):
        metrics[f"ot_transport_mass_class_{class_id:02d}"] = float(source_mass_by_class[class_id].item())
        metrics[f"ot_cost_class_{class_id:02d}"] = float(per_class_ot_cost[class_id].item())
    if entropy_gate is not None and entropy_values is not None:
        metrics.update(
            {
                "ot_class_entropy_mean": float(entropy_values.detach().mean().item()),
                "ot_class_entropy_gate_mean": float(entropy_gate.detach().mean().item()),
                "ot_class_entropy_gate_min": float(entropy_gate.detach().min().item()),
                "ot_class_entropy_gate_max": float(entropy_gate.detach().max().item()),
            }
        )
    if prototype_mode == "tp_barycentric":
        metrics.update(
            {
                "ot_barycentric_active_classes": float(active_classes),
                "ot_barycentric_mass": float(class_mass_total),
            }
        )
    if prototype_mode == "tp_relative_margin":
        metrics["ot_relative_margin_gate_mean"] = float(margin_gate.detach().mean().item())
    return loss, metrics


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def covariance(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=0, keepdim=True)
        return centered.t().mm(centered) / max(x.shape[0] - 1, 1)

    mean_loss = (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean()
    cov_loss = (covariance(source) - covariance(target)).pow(2).mean()
    return mean_loss + cov_loss


def mmd_loss(source: torch.Tensor, target: torch.Tensor, scales: tuple[float, ...] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    features = torch.cat([source, target], dim=0)
    distances = torch.cdist(features, features, p=2).pow(2)
    base = distances.detach().mean().clamp_min(EPS)
    kernels = [torch.exp(-distances / (base * float(scale)).clamp_min(EPS)) for scale in scales]
    kernel = sum(kernels)
    n_source = source.shape[0]
    k_ss = kernel[:n_source, :n_source].mean()
    k_tt = kernel[n_source:, n_source:].mean()
    k_st = kernel[:n_source, n_source:].mean()
    return k_ss + k_tt - 2.0 * k_st


def balanced_accuracy_np(labels: np.ndarray, predictions: np.ndarray, *, num_classes: int) -> float:
    recalls = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if not mask.any():
            continue
        recalls.append(float((predictions[mask] == class_id).mean()))
    return float(np.mean(recalls)) if recalls else 0.0
