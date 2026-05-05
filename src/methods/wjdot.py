"""WJDOT family methods integrated into the existing benchmark trainer."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.backbones import ClassifierHead
from src.losses import (
    DomainDiscriminator,
    GradientReverseLayer,
    WarmStartGradientReverseLayer,
    domain_adversarial_loss,
)
from src.tep_ot.augment import augment_signal
from src.tep_ot.ot_losses import (
    OTLossConfig,
    compute_class_prototypes,
    inverse_sqrt_class_weights,
    jdot_transport_loss,
    normalize_cost,
    solve_coupling,
    source_outlier_weights,
    target_pseudo_class_weights,
)

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits
from .codats import CoDATSClassifierHead


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels.long(), reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    class_count = max(int(probabilities.shape[1]), 2)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    return entropy / math.log(float(class_count))


def _minmax_normalize_by_class(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    normalized = values.new_zeros(values.shape)
    for class_id in range(values.shape[1]):
        valid = present[:, class_id] & torch.isfinite(values[:, class_id])
        if not bool(valid.any()):
            normalized[:, class_id] = 1.0
            continue
        column = values[valid, class_id]
        minimum = column.min()
        maximum = column.max()
        if float((maximum - minimum).detach().item()) <= 1e-8:
            normalized[valid, class_id] = 0.0
        else:
            normalized[valid, class_id] = (column - minimum) / (maximum - minimum).clamp_min(1e-8)
        normalized[~valid, class_id] = 1.0
    return normalized


def _softmax_source_floor(
    reliability: torch.Tensor,
    *,
    temperature: float,
    floor: float,
    top_m: int,
) -> torch.Tensor:
    source_count, num_classes = int(reliability.shape[0]), int(reliability.shape[1])
    raw = F.softmax(-reliability / max(float(temperature), 1e-8), dim=0)
    floor_value = max(float(floor), 0.0)
    alpha = raw.new_zeros(raw.shape)
    for class_id in range(num_classes):
        values = raw[:, class_id].clamp_min(floor_value)
        if 0 < int(top_m) < source_count:
            keep = torch.topk(raw[:, class_id], k=int(top_m), largest=True).indices
            filtered = raw.new_full((source_count,), floor_value)
            filtered[keep] = raw[keep, class_id].clamp_min(floor_value)
            values = filtered
        alpha[:, class_id] = values / values.sum().clamp_min(1e-8)
    return alpha


def _source_class_recall_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.detach().argmax(dim=1)
    errors = logits.new_ones(num_classes)
    supports = logits.new_zeros(num_classes)
    for class_id in range(num_classes):
        mask = labels == class_id
        supports[class_id] = float(mask.sum().detach().item())
        if bool(mask.any()):
            recall = (predictions[mask] == class_id).float().mean()
            errors[class_id] = 1.0 - recall.to(device=logits.device, dtype=logits.dtype)
    return errors, supports


def _sourceaware_transport_loss(
    *,
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_logits: torch.Tensor,
    num_classes: int,
    config: OTLossConfig,
    source_sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute one source-specific WJDOT coupling and class-wise transport costs."""

    source_labels = source_labels.long()
    feature_cost = torch.cdist(source_features, target_features, p=2).pow(2)
    label_cost = -F.log_softmax(target_logits.float(), dim=1)[:, source_labels].transpose(0, 1)
    label_cost = label_cost.to(dtype=feature_cost.dtype)
    if bool(config.normalize_costs):
        feature_plan = normalize_cost(feature_cost)
        label_plan = normalize_cost(label_cost)
    else:
        feature_plan = feature_cost
        label_plan = label_cost
    plan_cost = float(config.feature_weight) * feature_plan + float(config.label_weight) * label_plan
    gamma = solve_coupling(
        plan_cost,
        source_weights=source_sample_weights,
        solver=config.solver,
        sinkhorn_reg=config.sinkhorn_reg,
        sinkhorn_num_iter=config.sinkhorn_num_iter,
    )
    gamma_detached = gamma.detach()
    row_mass = gamma_detached.sum(dim=1)
    weighted_cost_by_row = (gamma_detached * plan_cost).sum(dim=1)
    class_mass = gamma_detached.new_zeros(num_classes)
    class_cost_sum = gamma_detached.new_zeros(num_classes)
    source_index = source_labels.detach().clamp(0, num_classes - 1)
    class_mass.index_add_(0, source_index, row_mass)
    class_cost_sum.index_add_(0, source_index, weighted_cost_by_row)
    class_loss = class_cost_sum / class_mass.clamp_min(1e-8)
    source_present = torch.bincount(source_index, minlength=num_classes).to(
        device=class_loss.device,
    )[:num_classes] > 0
    class_loss = torch.where(source_present, class_loss, torch.zeros_like(class_loss))
    loss = (gamma_detached * plan_cost).sum()
    metrics = {
        "loss_ot": float(loss.detach().item()),
        "ot_feature_cost": float((gamma_detached * feature_plan).sum().detach().item()),
        "ot_label_cost": float((gamma_detached * label_plan).sum().detach().item()),
        "ot_gamma_max": float(gamma_detached.max().detach().item()),
        "ot_transport_mass": float(gamma_detached.sum().detach().item()),
    }
    for class_id in range(num_classes):
        metrics[f"ot_transport_mass_class_{class_id:02d}"] = float(class_mass[class_id].detach().item())
        metrics[f"ot_cost_class_{class_id:02d}"] = float(class_loss[class_id].detach().item())
    return loss, class_loss, class_mass, gamma_detached, metrics


class WJDOTMethod(SingleSourceMethodBase):
    """Weighted JDOT with optional source class balancing and fixed minibatch coupling."""

    method_name = "wjdot"
    supports_multi_source = True

    def __init__(
        self,
        *,
        adaptation_weight: float = 1.0,
        source_ce_weight: float = 1.0,
        feature_weight: float = 0.1,
        label_weight: float = 1.0,
        prototype_weight: float = 0.0,
        prototype_weight_in_coupling: float | None = None,
        prototype_weight_residual: float | None = None,
        prototype_in_coupling: bool | str = True,
        prototype_mode: str = "legacy_pairwise",
        prototype_distance: str = "normalized_l2",
        prototype_cost_clip: str | float | None = None,
        ot_class_entropy_gate: bool = False,
        min_class_transport_mass: float = 1e-4,
        class_mass_weight: str = "sqrt",
        class_mass_weight_min: float = 0.0,
        class_mass_weight_max: float = 1.0,
        relative_margin: float = 0.05,
        relative_margin_temperature: float = 0.05,
        relative_cost_min: float = 0.0,
        relative_cost_max: float = 3.0,
        normalize_costs: bool = True,
        transport_solver: str = "sinkhorn",
        sinkhorn_reg: float = 0.05,
        sinkhorn_num_iter_max: int = 100,
        use_source_class_balance: bool = True,
        use_temporal_prototypes: bool = False,
        confidence_curriculum: bool = False,
        pseudo_weight: float = 0.15,
        consistency_weight: float = 0.05,
        class_balance_clip_min: float = 0.0,
        class_balance_clip_max: float = 0.0,
        target_label_assist_weight: float = 0.0,
        target_label_assist_start_step: int | None = None,
        target_label_assist_warmup_steps: int = 1,
        target_label_assist_class_balance: bool = True,
        tau_start: float = 0.95,
        tau_end: float = 0.70,
        tau_steps: int = 1000,
        alignment_start_step: int = 0,
        alignment_ramp_steps: int = 1,
        pseudo_start_step: int | None = None,
        prototype_start_step: int | None = None,
        prototype_warmup_steps: int = 1,
        prototype_confidence_threshold: float | None = None,
        ot_feature_normalize: bool = True,
        ot_source_stop_gradient: bool = True,
        embedding_norm_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.adaptation_weight = float(adaptation_weight)
        self.source_ce_weight = float(source_ce_weight)
        residual_prototype_weight = (
            float(prototype_weight)
            if prototype_weight_residual is None
            else float(prototype_weight_residual)
        )
        resolved_prototype_mode = str(prototype_mode)
        resolved_prototype_in_coupling = prototype_in_coupling
        if resolved_prototype_mode.strip().lower() in {"tp_residual_safe", "tp_barycentric"}:
            resolved_prototype_in_coupling = False
        self.ot_config = OTLossConfig(
            feature_weight=float(feature_weight),
            label_weight=float(label_weight),
            prototype_weight=residual_prototype_weight,
            prototype_coupling_weight=(
                None if prototype_weight_in_coupling is None else float(prototype_weight_in_coupling)
            ),
            prototype_in_coupling=resolved_prototype_in_coupling,
            prototype_mode=resolved_prototype_mode,
            prototype_distance=str(prototype_distance),
            prototype_cost_clip=prototype_cost_clip,
            ot_class_entropy_gate=bool(ot_class_entropy_gate),
            min_class_transport_mass=float(min_class_transport_mass),
            class_mass_weight=str(class_mass_weight),
            class_mass_weight_min=float(class_mass_weight_min),
            class_mass_weight_max=float(class_mass_weight_max),
            relative_margin=float(relative_margin),
            relative_margin_temperature=float(relative_margin_temperature),
            relative_cost_min=float(relative_cost_min),
            relative_cost_max=float(relative_cost_max),
            normalize_costs=bool(normalize_costs),
            solver=str(transport_solver),
            sinkhorn_reg=float(sinkhorn_reg),
            sinkhorn_num_iter=int(sinkhorn_num_iter_max),
        )
        self.use_source_class_balance = bool(use_source_class_balance)
        self.use_temporal_prototypes = bool(use_temporal_prototypes)
        self.confidence_curriculum = bool(confidence_curriculum)
        self.pseudo_weight = float(pseudo_weight)
        self.consistency_weight = float(consistency_weight)
        self.class_balance_clip_min = float(class_balance_clip_min)
        self.class_balance_clip_max = float(class_balance_clip_max)
        self.target_label_assist_weight = float(target_label_assist_weight)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.tau_steps = max(int(tau_steps), 1)
        self.alignment_start_step = max(int(alignment_start_step), 0)
        self.alignment_ramp_steps = max(int(alignment_ramp_steps), 1)
        self.pseudo_start_step = self.alignment_start_step if pseudo_start_step is None else max(int(pseudo_start_step), 0)
        self.target_label_assist_start_step = (
            self.alignment_start_step
            if target_label_assist_start_step is None
            else max(int(target_label_assist_start_step), 0)
        )
        self.target_label_assist_warmup_steps = max(int(target_label_assist_warmup_steps), 1)
        self.target_label_assist_class_balance = bool(target_label_assist_class_balance)
        self.prototype_start_step = (
            self.alignment_start_step if prototype_start_step is None else max(int(prototype_start_step), 0)
        )
        self.prototype_warmup_steps = max(int(prototype_warmup_steps), 1)
        self.prototype_confidence_threshold = (
            None
            if prototype_confidence_threshold is None
            else min(max(float(prototype_confidence_threshold), 0.0), 1.0)
        )
        self.ot_feature_normalize = bool(ot_feature_normalize)
        self.ot_source_stop_gradient = bool(ot_source_stop_gradient)
        self.embedding_norm_weight = float(embedding_norm_weight)
        self.global_step = 0
        self._last_source_weights: torch.Tensor | None = None
        self._last_class_source_weights: torch.Tensor | None = None

    def current_tau(self) -> float:
        pseudo_step = max(self.global_step - self.pseudo_start_step, 0)
        progress = min(pseudo_step / float(self.tau_steps), 1.0)
        return self.tau_start + progress * (self.tau_end - self.tau_start)

    def current_alignment_weight(self) -> float:
        if self.global_step <= self.alignment_start_step:
            return 0.0
        progress = min((self.global_step - self.alignment_start_step) / float(self.alignment_ramp_steps), 1.0)
        return self.adaptation_weight * progress

    def current_target_label_assist_weight(self) -> float:
        base_weight = self.target_label_assist_weight
        if base_weight <= 0 or self.global_step <= self.target_label_assist_start_step:
            return 0.0
        progress = min(
            (self.global_step - self.target_label_assist_start_step)
            / float(self.target_label_assist_warmup_steps),
            1.0,
        )
        return base_weight * progress

    def current_prototype_weight(self) -> float:
        base_weight = float(self.ot_config.prototype_weight)
        if base_weight <= 0 or self.global_step <= self.prototype_start_step:
            return 0.0
        progress = min((self.global_step - self.prototype_start_step) / float(self.prototype_warmup_steps), 1.0)
        return base_weight * progress

    def current_prototype_coupling_weight(self) -> float:
        base_weight = self.ot_config.prototype_coupling_weight
        if base_weight is None:
            base_weight = self.ot_config.prototype_weight
        base_weight = float(base_weight)
        if base_weight <= 0 or self.global_step <= self.prototype_start_step:
            return 0.0
        progress = min((self.global_step - self.prototype_start_step) / float(self.prototype_warmup_steps), 1.0)
        return base_weight * progress

    def _prepare_ot_features(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = torch.nan_to_num(source_features, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target_features, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.ot_feature_normalize:
            source = F.normalize(source, p=2, dim=1, eps=1e-6)
            target = F.normalize(target, p=2, dim=1, eps=1e-6)
        if self.ot_source_stop_gradient:
            source = source.detach()
        return source, target

    def _embedding_norm_loss(self, *feature_batches: torch.Tensor) -> torch.Tensor:
        if self.embedding_norm_weight <= 0:
            return feature_batches[0].new_zeros(())
        losses = [features.float().pow(2).mean() for features in feature_batches if features.numel() > 0]
        if not losses:
            return feature_batches[0].new_zeros(())
        return torch.stack(losses).mean()

    def _target_regularizers(
        self,
        target_x: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        probabilities = F.softmax(target_logits.detach().float(), dim=1)
        confidence, pseudo_labels = probabilities.max(dim=1)
        tau = self.current_tau()
        confident_mask = confidence > tau
        zero = target_logits.new_zeros(())
        metrics = {
            "tau": float(tau),
            "pseudo_acceptance": float(confident_mask.float().mean().detach().item()),
            "target_confidence": float(confidence.mean().detach().item()),
        }
        if not self.confidence_curriculum or self.global_step <= self.pseudo_start_step:
            return zero, metrics, None

        pseudo_loss = zero
        if bool(confident_mask.any()):
            class_weights = target_pseudo_class_weights(
                pseudo_labels,
                confident_mask,
                num_classes=target_logits.shape[1],
            ).to(device=target_logits.device, dtype=target_logits.dtype)
            if self.class_balance_clip_min > 0 or self.class_balance_clip_max > 0:
                if self.class_balance_clip_min > 0:
                    class_weights = class_weights.clamp_min(self.class_balance_clip_min)
                if self.class_balance_clip_max > 0:
                    class_weights = class_weights.clamp_max(self.class_balance_clip_max)
            pseudo_loss = F.cross_entropy(
                target_logits[confident_mask],
                pseudo_labels[confident_mask],
                weight=class_weights,
            )

        logits_augmented, _ = self.forward(augment_signal(target_x))
        consistency_loss = F.kl_div(
            F.log_softmax(logits_augmented.float(), dim=1),
            probabilities,
            reduction="none",
        ).sum(dim=1)
        consistency_loss = (consistency_loss * confidence).mean()
        loss = self.pseudo_weight * pseudo_loss + self.consistency_weight * consistency_loss
        metrics.update(
            {
                "loss_pseudo": float(pseudo_loss.detach().item()),
                "loss_consistency": float(consistency_loss.detach().item()),
            }
        )
        return loss, metrics, confident_mask.to(dtype=target_logits.dtype)

    def _target_label_assist_loss(
        self,
        target_logits: torch.Tensor,
        target_y: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weight = self.current_target_label_assist_weight()
        valid_mask = target_y >= 0
        zero = target_logits.new_zeros(())
        metrics = {
            "lambda_target_label_assist": float(weight),
            "target_label_assist_count": float(valid_mask.sum().detach().item()),
            "loss_target_label_assist": 0.0,
            "acc_target_label_assist": 0.0,
        }
        if weight <= 0 or not bool(valid_mask.any()):
            return zero, metrics

        valid_logits = target_logits[valid_mask]
        valid_labels = target_y[valid_mask].long()
        sample_weights = None
        if self.target_label_assist_class_balance:
            class_weights = inverse_sqrt_class_weights(
                valid_labels,
                num_classes=int(target_logits.shape[1]),
            ).to(device=valid_logits.device, dtype=valid_logits.dtype)
            if self.class_balance_clip_min > 0:
                class_weights = class_weights.clamp_min(self.class_balance_clip_min)
            if self.class_balance_clip_max > 0:
                class_weights = class_weights.clamp_max(self.class_balance_clip_max)
            sample_weights = class_weights[valid_labels]

        loss_ce = _weighted_ce(valid_logits, valid_labels, sample_weights)
        metrics.update(
            {
                "loss_target_label_assist": float(loss_ce.detach().item()),
                "acc_target_label_assist": accuracy_from_logits(valid_logits, valid_labels),
            }
        )
        return weight * loss_ce, metrics

    def _prototype_gate(
        self,
        target_logits: torch.Tensor,
        prototype_weight: float,
        fallback_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        metrics = {"lambda_prototype": float(prototype_weight)}
        threshold = self.prototype_confidence_threshold
        if prototype_weight <= 0:
            metrics["prototype_acceptance"] = 0.0
            return None, metrics

        if not self.confidence_curriculum:
            metrics.update(
                {
                    "prototype_acceptance": 1.0,
                    "prototype_gate_source": 0.0,
                }
            )
            return None, metrics

        if threshold is None and fallback_gate is not None:
            gate = fallback_gate.to(device=target_logits.device, dtype=target_logits.dtype)
            metrics.update(
                {
                    "prototype_acceptance": float(gate.float().mean().detach().item()),
                    "prototype_gate_source": 1.0,
                }
            )
            return gate, metrics

        if prototype_weight <= 0 or threshold is None:
            metrics["prototype_acceptance"] = 1.0
            return None, metrics

        probabilities = F.softmax(target_logits.detach().float(), dim=1)
        confidence = probabilities.max(dim=1).values
        mask = confidence >= threshold
        metrics.update(
            {
                "prototype_acceptance": float(mask.float().mean().detach().item()),
                "prototype_confidence_threshold": float(threshold),
            }
        )
        return mask.to(dtype=target_logits.dtype), metrics

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        self.global_step += 1
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, target_y = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)
        num_classes = int(logits_target.shape[1])
        current_alignment_weight = self.current_alignment_weight()

        class_weights = inverse_sqrt_class_weights(source_y, num_classes=num_classes).to(
            device=source_y.device,
            dtype=features_source.dtype,
        )
        source_sample_weights = class_weights[source_y.long()] if self.use_source_class_balance else None
        loss_cls = _weighted_ce(logits_source, source_y, source_sample_weights)
        loss_norm = self._embedding_norm_loss(features_source, features_target)
        target_label_assist, target_label_metrics = self._target_label_assist_loss(logits_target, target_y)

        if current_alignment_weight <= 0:
            loss_total = self.source_ce_weight * loss_cls + target_label_assist + self.embedding_norm_weight * loss_norm
            metrics = {
                "loss_total": float(loss_total.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_alignment": 0.0,
                "loss_embedding_norm": float(loss_norm.detach().item()),
                "lambda_alignment": 0.0,
                "lambda_prototype": 0.0,
                "acc_source": accuracy_from_logits(logits_source, source_y),
            }
            metrics.update(target_label_metrics)
            return MethodStepOutput(loss=loss_total, metrics=metrics)

        ot_source_features, ot_target_features = self._prepare_ot_features(features_source, features_target)
        current_prototype_weight = self.current_prototype_weight()
        current_prototype_coupling_weight = self.current_prototype_coupling_weight()
        ot_config = replace(
            self.ot_config,
            prototype_weight=current_prototype_weight,
            prototype_coupling_weight=current_prototype_coupling_weight,
        )

        prototypes = None
        if (
            max(current_prototype_weight, current_prototype_coupling_weight) > 0
            and (self.use_temporal_prototypes or self.ot_config.prototype_weight > 0)
        ):
            prototypes, _ = compute_class_prototypes(
                ot_source_features,
                source_y,
                num_classes=num_classes,
                sample_weights=source_sample_weights,
            )

        target_regularizer, target_metrics, label_gate = self._target_regularizers(target_x, logits_target)
        prototype_gate, prototype_metrics = self._prototype_gate(
            logits_target,
            max(current_prototype_weight, current_prototype_coupling_weight),
            fallback_gate=label_gate,
        )
        loss_ot, ot_metrics = jdot_transport_loss(
            source_features=ot_source_features,
            source_labels=source_y,
            target_features=ot_target_features,
            target_logits=logits_target,
            num_classes=num_classes,
            config=ot_config,
            source_sample_weights=source_sample_weights,
            source_prototypes=prototypes,
            label_gate=label_gate,
            prototype_gate=prototype_gate,
        )
        loss_total = (
            self.source_ce_weight * loss_cls
            + current_alignment_weight * loss_ot
            + target_regularizer
            + target_label_assist
            + self.embedding_norm_weight * loss_norm
        )
        metrics = {
            "loss_total": float(loss_total.detach().item()),
            "loss_cls": float(loss_cls.detach().item()),
            "loss_alignment": float(loss_ot.detach().item()),
            "loss_embedding_norm": float(loss_norm.detach().item()),
            "lambda_alignment": float(current_alignment_weight),
            "acc_source": accuracy_from_logits(logits_source, source_y),
        }
        metrics.update(ot_metrics)
        metrics.update(target_metrics)
        metrics.update(prototype_metrics)
        metrics.update(target_label_metrics)
        metrics["lambda_prototype_residual"] = float(current_prototype_weight)
        metrics["lambda_prototype_coupling"] = float(current_prototype_coupling_weight)
        return MethodStepOutput(loss=loss_total, metrics=metrics)

    def reliability_snapshot(self) -> dict[str, torch.Tensor]:
        snapshot: dict[str, torch.Tensor] = {}
        if self._last_source_weights is not None:
            snapshot["source_weights"] = self._last_source_weights.detach().cpu()
        if self._last_class_source_weights is not None:
            snapshot["class_source_weights"] = self._last_class_source_weights.detach().cpu()
        return snapshot


class PooledWJDOTMethod(WJDOTMethod):
    """Explicit name for the legacy pooled-source WJDOT baseline."""

    method_name = "pooled_wjdot"


class SourceAwareWJDOTSharedHeadMethod(WJDOTMethod):
    """Source-aware WJDOT with shared encoder/head and per-source OT plans."""

    method_name = "sourceaware_wjdot_shared_head"

    def __init__(
        self,
        *,
        num_sources: int = 1,
        source_alpha_mode: str = "uniform",
        source_alpha_temperature: float = 1.0,
        source_ce_reduction: str = "mean",
        class_transport_normalize: bool = True,
        sample_outlier_downweight: bool = False,
        sample_weight_min: float = 0.3,
        sample_weight_max: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_sources = max(int(num_sources), 1)
        self.source_alpha_mode = str(source_alpha_mode).strip().lower()
        self.source_alpha_temperature = max(float(source_alpha_temperature), 1e-8)
        self.source_ce_reduction = str(source_ce_reduction).strip().lower()
        self.class_transport_normalize = bool(class_transport_normalize)
        self.sample_outlier_downweight = bool(sample_outlier_downweight)
        self.sample_weight_min = float(sample_weight_min)
        self.sample_weight_max = float(sample_weight_max)
        self._last_transport_mass_matrix: torch.Tensor | None = None
        self._last_per_class_ot_cost_matrix: torch.Tensor | None = None
        self._last_source_ot_plans: list[torch.Tensor] = []
        self._last_source_prediction_histogram: torch.Tensor | None = None
        self.store_source_ot_plans = True

    def _source_head_logits(
        self,
        features: torch.Tensor,
        source_index: int,
    ) -> torch.Tensor:
        del source_index
        return self.classifier(features)

    def _target_head_logits(
        self,
        features: torch.Tensor,
        source_index: int,
    ) -> torch.Tensor:
        del source_index
        return self.classifier(features)

    def _source_expert_probabilities_from_features(self, features: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(self.classifier(features).float(), dim=1)
        return probabilities.unsqueeze(0).repeat(self.num_sources, 1, 1)

    def source_expert_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self._source_expert_probabilities_from_features(features)

    def _source_alpha_from_losses(self, losses: list[torch.Tensor], *, device, dtype) -> torch.Tensor:
        count = max(len(losses), 1)
        if not losses or self.source_alpha_mode in {"uniform", "equal", "none"}:
            return torch.full((count,), 1.0 / float(count), device=device, dtype=dtype)
        values = torch.stack([loss.detach().to(device=device, dtype=dtype) for loss in losses])
        if self.source_alpha_mode in {"softmax_loss", "ot_loss", "softmax"}:
            normalized = values / values.mean().clamp_min(1e-8)
            return F.softmax(-normalized / self.source_alpha_temperature, dim=0).detach()
        return torch.full((count,), 1.0 / float(count), device=device, dtype=dtype)

    def _source_ce_loss(self, losses: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(losses)
        if self.source_ce_reduction == "sum":
            return stacked.sum()
        return stacked.mean()

    def _sample_weights_for_source(
        self,
        *,
        features: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        weights = class_weights[labels.long()]
        if self.sample_outlier_downweight:
            outlier = source_outlier_weights(features.detach(), labels, prototypes.detach()).to(
                device=features.device,
                dtype=features.dtype,
            )
            weights = weights * outlier.clamp(self.sample_weight_min, self.sample_weight_max)
        return weights

    def _compute_sourceaware_terms(
        self,
        source_batches,
        target_batch,
    ) -> dict[str, Any]:
        self.global_step += 1
        target_x, target_y = target_batch
        features_target = self.encoder(target_x)
        fused_target_logits = torch.log(
            self._source_expert_probabilities_from_features(features_target).mean(dim=0).clamp_min(1e-8)
        )
        num_classes = int(fused_target_logits.shape[1])
        current_alignment_weight = self.current_alignment_weight()

        source_logits: list[torch.Tensor] = []
        source_features: list[torch.Tensor] = []
        source_ot_features: list[torch.Tensor] = []
        source_target_logits: list[torch.Tensor] = []
        source_target_probs: list[torch.Tensor] = []
        source_labels: list[torch.Tensor] = []
        source_prototypes: list[torch.Tensor] = []
        source_present: list[torch.Tensor] = []
        source_ce_losses: list[torch.Tensor] = []
        source_ot_losses: list[torch.Tensor] = []
        source_class_losses: list[torch.Tensor] = []
        source_class_mass: list[torch.Tensor] = []
        source_eval_errors: list[torch.Tensor] = []
        source_eval_supports: list[torch.Tensor] = []
        source_transport_weights: list[torch.Tensor] = []
        gamma_plans: list[torch.Tensor] = []
        metrics: dict[str, float] = {}

        for source_index, (source_x, source_y) in enumerate(source_batches):
            features_source = self.encoder(source_x)
            logits_source = self._source_head_logits(features_source, source_index)
            target_logits_k = self._target_head_logits(features_target, source_index)
            target_probs_k = F.softmax(target_logits_k.detach().float(), dim=1)
            class_weights = inverse_sqrt_class_weights(source_y, num_classes=num_classes).to(
                device=source_y.device,
                dtype=features_source.dtype,
            )
            source_sample_weights = class_weights[source_y.long()] if self.use_source_class_balance else None
            loss_ce_k = _weighted_ce(logits_source, source_y, source_sample_weights)
            ot_source_features, ot_target_features = self._prepare_ot_features(features_source, features_target)
            prototypes, present = compute_class_prototypes(
                ot_source_features.detach(),
                source_y,
                num_classes=num_classes,
                sample_weights=source_sample_weights,
            )
            transport_weights = None
            if self.use_source_class_balance or self.sample_outlier_downweight:
                transport_weights = self._sample_weights_for_source(
                    features=ot_source_features,
                    labels=source_y,
                    class_weights=class_weights,
                    prototypes=prototypes,
                )
            source_logits.append(logits_source)
            source_features.append(features_source)
            source_ot_features.append(ot_source_features)
            source_target_logits.append(target_logits_k)
            source_target_probs.append(target_probs_k)
            source_labels.append(source_y)
            source_prototypes.append(prototypes)
            source_present.append(present)
            source_ce_losses.append(loss_ce_k)
            source_transport_weights.append(
                torch.ones_like(source_y, dtype=features_source.dtype)
                if transport_weights is None
                else transport_weights.detach()
            )
            error_k, support_k = _source_class_recall_error(
                logits_source,
                source_y,
                num_classes=num_classes,
            )
            source_eval_errors.append(error_k)
            source_eval_supports.append(support_k)

            if current_alignment_weight > 0:
                ot_loss_k, class_loss_k, class_mass_k, gamma_k, ot_metrics = _sourceaware_transport_loss(
                    source_features=ot_source_features,
                    source_labels=source_y,
                    target_features=ot_target_features,
                    target_logits=target_logits_k,
                    num_classes=num_classes,
                    config=self.ot_config,
                    source_sample_weights=transport_weights,
                )
            else:
                ot_loss_k = features_target.new_zeros(())
                class_loss_k = features_target.new_zeros(num_classes)
                class_mass_k = features_target.new_zeros(num_classes)
                gamma_k = features_target.new_zeros((int(source_y.shape[0]), int(target_x.shape[0])))
                ot_metrics = {
                    "loss_ot": 0.0,
                    "ot_transport_mass": 0.0,
                    **{f"ot_transport_mass_class_{class_id:02d}": 0.0 for class_id in range(num_classes)},
                    **{f"ot_cost_class_{class_id:02d}": 0.0 for class_id in range(num_classes)},
                }
            source_ot_losses.append(ot_loss_k)
            source_class_losses.append(class_loss_k)
            source_class_mass.append(class_mass_k)
            if self.store_source_ot_plans:
                gamma_plans.append(gamma_k.detach().cpu())
            metrics[f"loss_ce_source_{source_index}"] = float(loss_ce_k.detach().item())
            metrics[f"loss_ot_source_{source_index}"] = float(ot_loss_k.detach().item())
            metrics[f"transport_mass_source_{source_index}"] = float(ot_metrics.get("ot_transport_mass", 0.0))
            for class_id in range(num_classes):
                metrics[f"loss_ot_source_{source_index}_class_{class_id:02d}"] = float(
                    class_loss_k[class_id].detach().item()
                )
                metrics[f"transport_mass_source_{source_index}_class_{class_id:02d}"] = float(
                    class_mass_k[class_id].detach().item()
                )

        loss_cls = self._source_ce_loss(source_ce_losses)
        loss_norm = self._embedding_norm_loss(*source_features, features_target)
        target_label_assist, target_label_metrics = self._target_label_assist_loss(fused_target_logits, target_y)
        source_alpha = self._source_alpha_from_losses(
            source_ot_losses,
            device=features_target.device,
            dtype=features_target.dtype,
        )
        transport_mass_matrix = torch.stack(source_class_mass, dim=0).detach()
        per_class_ot_cost_matrix = torch.stack(source_class_losses, dim=0).detach()
        source_present_matrix = torch.stack(
            [present.to(device=features_target.device, dtype=torch.bool) for present in source_present],
            dim=0,
        )
        source_target_prob_tensor = torch.stack(source_target_probs, dim=0)
        prediction_histogram = features_target.new_zeros((len(source_batches), num_classes))
        for source_index, probs in enumerate(source_target_prob_tensor):
            counts = torch.bincount(probs.argmax(dim=1), minlength=num_classes).to(
                device=features_target.device,
                dtype=features_target.dtype,
            )
            prediction_histogram[source_index] = counts

        metrics.update(
            {
                "loss_cls": float(loss_cls.detach().item()),
                "loss_embedding_norm": float(loss_norm.detach().item()),
                "lambda_alignment": float(current_alignment_weight),
                "acc_source": accuracy_from_logits(torch.cat(source_logits, dim=0), torch.cat(source_labels, dim=0)),
                "sourceaware_source_count": float(len(source_batches)),
            }
        )
        for source_index, alpha_value in enumerate(source_alpha):
            metrics[f"alpha_source_{source_index}"] = float(alpha_value.detach().item())
            metrics[f"loss_source_weight_{source_index}"] = float(alpha_value.detach().item())
        metrics.update(target_label_metrics)

        return {
            "features_target": features_target,
            "target_logits": fused_target_logits,
            "target_y": target_y,
            "source_logits": source_logits,
            "source_features": source_features,
            "source_ot_features": source_ot_features,
            "source_labels": source_labels,
            "source_prototypes": source_prototypes,
            "source_present": source_present_matrix,
            "source_target_probs": source_target_prob_tensor,
            "source_eval_errors": torch.stack(source_eval_errors, dim=0),
            "source_eval_supports": torch.stack(source_eval_supports, dim=0),
            "source_ce_losses": source_ce_losses,
            "source_ot_losses": source_ot_losses,
            "source_class_losses": torch.stack(source_class_losses, dim=0),
            "source_class_mass": transport_mass_matrix,
            "source_alpha": source_alpha,
            "loss_cls": loss_cls,
            "loss_norm": loss_norm,
            "target_label_assist": target_label_assist,
            "current_alignment_weight": current_alignment_weight,
            "metrics": metrics,
            "gamma_plans": gamma_plans,
            "per_class_ot_cost_matrix": per_class_ot_cost_matrix,
            "transport_mass_matrix": transport_mass_matrix,
            "prediction_histogram": prediction_histogram.detach(),
        }

    def _aggregate_transport_loss(self, terms: dict[str, Any]) -> torch.Tensor:
        losses = terms["source_ot_losses"]
        source_alpha = terms["source_alpha"]
        if not losses:
            return terms["features_target"].new_zeros(())
        stacked = torch.stack(losses)
        return (source_alpha.to(device=stacked.device, dtype=stacked.dtype) * stacked).sum()

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        terms = self._compute_sourceaware_terms(source_batches, target_batch)
        loss_ot = self._aggregate_transport_loss(terms)
        loss_total = (
            self.source_ce_weight * terms["loss_cls"]
            + float(terms["current_alignment_weight"]) * loss_ot
            + terms["target_label_assist"]
            + self.embedding_norm_weight * terms["loss_norm"]
        )
        metrics = dict(terms["metrics"])
        metrics.update(
            {
                "loss_total": float(loss_total.detach().item()),
                "loss_alignment": float(loss_ot.detach().item()),
                "sourceaware_class_transport_training": 0.0,
            }
        )
        self._last_source_weights = terms["source_alpha"].detach()
        self._last_class_source_weights = terms["source_alpha"].view(-1, 1).repeat(
            1,
            int(terms["source_class_losses"].shape[1]),
        ).detach()
        self._last_transport_mass_matrix = terms["transport_mass_matrix"].detach()
        self._last_per_class_ot_cost_matrix = terms["per_class_ot_cost_matrix"].detach()
        self._last_source_ot_plans = terms["gamma_plans"]
        self._last_source_prediction_histogram = terms["prediction_histogram"].detach()
        return MethodStepOutput(loss=loss_total, metrics=metrics)

    def reliability_snapshot(self) -> dict[str, torch.Tensor]:
        snapshot = super().reliability_snapshot()
        if self._last_transport_mass_matrix is not None:
            snapshot["transport_mass_matrix"] = self._last_transport_mass_matrix.detach().cpu()
        if self._last_per_class_ot_cost_matrix is not None:
            snapshot["per_class_ot_cost_matrix"] = self._last_per_class_ot_cost_matrix.detach().cpu()
        if self._last_source_prediction_histogram is not None:
            snapshot["source_prediction_histogram"] = self._last_source_prediction_histogram.detach().cpu()
        return snapshot


class SourceAwareWJDOTMultiHeadMethod(SourceAwareWJDOTSharedHeadMethod):
    """Source-aware WJDOT with shared encoder and one classifier head per source."""

    method_name = "sourceaware_wjdot_multi_head"

    def __init__(self, *, num_sources: int = 1, **kwargs: Any) -> None:
        hidden_dim = int(kwargs.get("classifier_hidden_dim", 128))
        dropout = float(kwargs.get("dropout", 0.1))
        num_classes = int(kwargs.get("num_classes"))
        super().__init__(num_sources=num_sources, **kwargs)
        self.extra_source_classifiers = nn.ModuleList(
            [
                ClassifierHead(
                    in_features=self.encoder.out_dim,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(max(self.num_sources - 1, 0))
            ]
        )

    def _head_for_source(self, source_index: int) -> nn.Module:
        if source_index <= 0 or len(self.extra_source_classifiers) == 0:
            return self.classifier
        index = min(source_index - 1, len(self.extra_source_classifiers) - 1)
        return self.extra_source_classifiers[index]

    def _source_head_logits(
        self,
        features: torch.Tensor,
        source_index: int,
    ) -> torch.Tensor:
        return self._head_for_source(source_index)(features)

    def _target_head_logits(
        self,
        features: torch.Tensor,
        source_index: int,
    ) -> torch.Tensor:
        return self._head_for_source(source_index)(features)

    def _source_expert_probabilities_from_features(self, features: torch.Tensor) -> torch.Tensor:
        probabilities = [
            F.softmax(self._head_for_source(source_index)(features).float(), dim=1)
            for source_index in range(self.num_sources)
        ]
        return torch.stack(probabilities, dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        probabilities = self._source_expert_probabilities_from_features(features)
        fused = probabilities.mean(dim=0)
        logits = torch.log(fused.clamp_min(1e-8))
        return logits, features


class SACCsrWJDOTTrainMethod(SourceAwareWJDOTMultiHeadMethod):
    """SA-CCSR-WJDOT: class-source reliability influences transport training."""

    method_name = "sa_ccsr_wjdot_train"

    def __init__(
        self,
        *,
        reliability_start_ratio: float = 0.30,
        reliability_ramp_ratio: float = 0.20,
        reliability_total_steps: int = 1000,
        reliability_start_step: int | None = None,
        reliability_ramp_steps: int | None = None,
        class_temperature: float | None = None,
        T_class: float | None = None,
        top_m_per_class: int | None = None,
        floor: float | None = None,
        w_proto: float = 0.35,
        w_ot: float = 0.35,
        w_entropy: float = 0.20,
        w_source_error: float = 0.10,
        tau_proto: float = 0.85,
        min_proto_samples: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.store_source_ot_plans = False
        total_steps = max(int(reliability_total_steps), 1)
        self.reliability_start_step = (
            max(int(reliability_start_step), 0)
            if reliability_start_step is not None
            else max(int(round(float(reliability_start_ratio) * total_steps)), 0)
        )
        self.reliability_ramp_steps = (
            max(int(reliability_ramp_steps), 1)
            if reliability_ramp_steps is not None
            else max(int(round(float(reliability_ramp_ratio) * total_steps)), 1)
        )
        default_temperature = 1.0 if self.num_sources <= 2 else 0.5
        self.class_temperature = max(float(T_class if T_class is not None else (class_temperature if class_temperature is not None else default_temperature)), 1e-8)
        self.top_m_per_class = (
            0 if self.num_sources <= 2 else 3
        ) if top_m_per_class is None else max(int(top_m_per_class), 0)
        self.class_floor = (
            0.05 if self.num_sources <= 2 else 0.03
        ) if floor is None else max(float(floor), 0.0)
        self.reliability_component_weights = {
            "D_proto": float(w_proto),
            "D_ot": float(w_ot),
            "H_pred": float(w_entropy),
            "E_src": float(w_source_error),
        }
        self.tau_proto = float(tau_proto)
        self.min_proto_samples = max(int(min_proto_samples), 1)
        self._last_reliability_matrix: torch.Tensor | None = None
        self._last_reliability_components: dict[str, torch.Tensor] = {}

    def _reliability_ramp(self) -> float:
        if self.global_step <= self.reliability_start_step:
            return 0.0
        progress = (self.global_step - self.reliability_start_step) / float(self.reliability_ramp_steps)
        return max(0.0, min(float(progress), 1.0))

    def _target_prototypes_from_terms(self, terms: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        target_features = terms["features_target"].detach()
        target_features = F.normalize(target_features, p=2, dim=1, eps=1e-6)
        source_probs = terms["source_target_probs"].detach()
        mean_probs = source_probs.mean(dim=0)
        confidence, provisional = mean_probs.max(dim=1)
        num_classes = int(mean_probs.shape[1])
        prototypes = target_features.new_zeros((num_classes, target_features.shape[1]))
        present = torch.zeros(num_classes, device=target_features.device, dtype=torch.bool)
        for class_id in range(num_classes):
            high_conf = (provisional == class_id) & (confidence >= self.tau_proto)
            if int(high_conf.sum().detach().item()) >= self.min_proto_samples:
                prototypes[class_id] = F.normalize(
                    target_features[high_conf].mean(dim=0, keepdim=True),
                    p=2,
                    dim=1,
                    eps=1e-6,
                )[0]
                present[class_id] = True
                continue
            barycenters = []
            for source_index in range(source_probs.shape[0]):
                weights = source_probs[source_index, :, class_id]
                mass = weights.sum()
                if float(mass.detach().item()) <= 1e-8:
                    continue
                barycenter = (target_features * weights.view(-1, 1)).sum(dim=0, keepdim=True) / mass.clamp_min(1e-8)
                barycenters.append(F.normalize(barycenter, p=2, dim=1, eps=1e-6)[0])
            if barycenters:
                prototypes[class_id] = F.normalize(
                    torch.stack(barycenters, dim=0).mean(dim=0, keepdim=True),
                    p=2,
                    dim=1,
                    eps=1e-6,
                )[0]
                present[class_id] = True
        return prototypes, present

    def _compute_class_alpha(self, terms: dict[str, Any]) -> torch.Tensor:
        source_count = int(terms["source_class_losses"].shape[0])
        num_classes = int(terms["source_class_losses"].shape[1])
        uniform = terms["features_target"].new_full((source_count, num_classes), 1.0 / float(max(source_count, 1)))
        ramp = self._reliability_ramp()
        if ramp <= 0:
            self._last_reliability_matrix = uniform.new_zeros(uniform.shape)
            self._last_reliability_components = {"alpha_uniform": uniform.detach()}
            return uniform

        target_prototypes, target_present = self._target_prototypes_from_terms(terms)
        d_proto = uniform.new_ones((source_count, num_classes))
        for source_index, prototypes in enumerate(terms["source_prototypes"]):
            normalized_source = F.normalize(prototypes.detach(), p=2, dim=1, eps=1e-6)
            distances = (normalized_source - target_prototypes).pow(2).sum(dim=1)
            valid = terms["source_present"][source_index] & target_present
            d_proto[source_index] = torch.where(valid, distances, torch.full_like(distances, 1e6))
        d_ot = torch.where(
            terms["source_present"],
            terms["source_class_losses"].detach(),
            torch.full_like(terms["source_class_losses"].detach(), 1e6),
        )
        source_probs = terms["source_target_probs"].detach()
        mean_pred = source_probs.mean(dim=0).argmax(dim=1)
        h_pred = uniform.new_ones((source_count, num_classes))
        for source_index in range(source_count):
            entropy = _normalized_entropy(source_probs[source_index])
            for class_id in range(num_classes):
                mask = mean_pred == class_id
                if bool(mask.any()):
                    h_pred[source_index, class_id] = entropy[mask].mean()
        e_src = torch.where(
            terms["source_present"],
            terms["source_eval_errors"].detach(),
            torch.ones_like(terms["source_eval_errors"].detach()),
        )
        present = terms["source_present"]
        d_proto_norm = _minmax_normalize_by_class(d_proto, present)
        d_ot_norm = _minmax_normalize_by_class(d_ot, present)
        h_pred_norm = _minmax_normalize_by_class(h_pred, present)
        e_src_norm = _minmax_normalize_by_class(e_src, present)
        reliability = (
            self.reliability_component_weights["D_proto"] * d_proto_norm
            + self.reliability_component_weights["D_ot"] * d_ot_norm
            + self.reliability_component_weights["H_pred"] * h_pred_norm
            + self.reliability_component_weights["E_src"] * e_src_norm
        )
        alpha = _softmax_source_floor(
            reliability,
            temperature=self.class_temperature,
            floor=self.class_floor,
            top_m=self.top_m_per_class,
        )
        mixed = (1.0 - ramp) * uniform + ramp * alpha
        mixed = mixed / mixed.sum(dim=0, keepdim=True).clamp_min(1e-8)
        self._last_reliability_matrix = reliability.detach()
        self._last_reliability_components = {
            "D_proto": d_proto.detach(),
            "D_ot": d_ot.detach(),
            "H_pred": h_pred.detach(),
            "E_src": e_src.detach(),
            "D_proto_norm": d_proto_norm.detach(),
            "D_ot_norm": d_ot_norm.detach(),
            "H_pred_norm": h_pred_norm.detach(),
            "E_src_norm": e_src_norm.detach(),
            "R": reliability.detach(),
            "alpha": mixed.detach(),
        }
        return mixed

    def _aggregate_class_transport_loss(self, terms: dict[str, Any], alpha: torch.Tensor) -> torch.Tensor:
        class_losses = terms["source_class_losses"]
        present = terms["source_present"].to(device=class_losses.device, dtype=class_losses.dtype)
        weights = alpha.to(device=class_losses.device, dtype=class_losses.dtype) * present
        value = (weights * class_losses).sum()
        if self.class_transport_normalize:
            active_classes = present.sum(dim=0).clamp_max(1.0).sum().clamp_min(1.0)
            value = value / active_classes
        return value

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        terms = self._compute_sourceaware_terms(source_batches, target_batch)
        class_alpha = self._compute_class_alpha(terms)
        loss_ot = self._aggregate_class_transport_loss(terms, class_alpha)
        loss_total = (
            self.source_ce_weight * terms["loss_cls"]
            + float(terms["current_alignment_weight"]) * loss_ot
            + terms["target_label_assist"]
            + self.embedding_norm_weight * terms["loss_norm"]
        )
        metrics = dict(terms["metrics"])
        metrics.update(
            {
                "loss_total": float(loss_total.detach().item()),
                "loss_alignment": float(loss_ot.detach().item()),
                "sourceaware_class_transport_training": 1.0,
                "reliability_ramp": float(self._reliability_ramp()),
                "class_alpha_min": float(class_alpha.min().detach().item()),
                "class_alpha_max": float(class_alpha.max().detach().item()),
                "class_alpha_std": float(class_alpha.std(unbiased=False).detach().item()),
            }
        )
        for source_index in range(class_alpha.shape[0]):
            metrics[f"alpha_source_{source_index}"] = float(class_alpha[source_index].mean().detach().item())
            for class_id in range(class_alpha.shape[1]):
                metrics[f"alpha_source_{source_index}_class_{class_id:02d}"] = float(
                    class_alpha[source_index, class_id].detach().item()
                )
        self._last_source_weights = class_alpha.mean(dim=1).detach()
        self._last_class_source_weights = class_alpha.detach()
        self._last_transport_mass_matrix = terms["transport_mass_matrix"].detach()
        self._last_per_class_ot_cost_matrix = terms["per_class_ot_cost_matrix"].detach()
        self._last_source_ot_plans = terms["gamma_plans"]
        self._last_source_prediction_histogram = terms["prediction_histogram"].detach()
        return MethodStepOutput(loss=loss_total, metrics=metrics)

    def reliability_snapshot(self) -> dict[str, torch.Tensor]:
        snapshot = super().reliability_snapshot()
        if self._last_reliability_matrix is not None:
            snapshot["reliability_matrix"] = self._last_reliability_matrix.detach().cpu()
        for key, value in self._last_reliability_components.items():
            snapshot[f"reliability_{key}"] = value.detach().cpu()
        return snapshot


class CACCSRWJDOTMethod(SourceAwareWJDOTSharedHeadMethod):
    """CoDATS-augmented CCSR-WJDOT with a frozen teacher anchor."""

    method_name = "ca_ccsr_wjdot"

    def __init__(
        self,
        *,
        reliability_start_ratio: float = 0.30,
        reliability_ramp_ratio: float = 0.20,
        reliability_total_steps: int = 1000,
        reliability_start_step: int | None = None,
        reliability_ramp_steps: int | None = None,
        class_temperature: float | None = None,
        T_class: float | None = None,
        top_m_per_class: int | None = None,
        floor: float | None = None,
        w_proto: float = 0.30,
        w_ot: float = 0.35,
        w_entropy: float = 0.20,
        w_source_error: float = 0.15,
        tau_proto: float = 0.85,
        min_proto_samples: int = 3,
        lambda_adv: float = 0.5,
        lambda_ot: float = 0.10,
        lambda_ccsr: float = 0.10,
        lambda_teacher: float = 0.05,
        teacher_temperature: float = 1.0,
        teacher_anchor_mode: str = "kl",
        teacher_requires_checkpoint: bool = True,
        teacher_start_step: int = 0,
        teacher_ramp_steps: int = 1,
        domain_adaptation_schedule: str = "warm_start",
        domain_adaptation_max_steps: int = 1000,
        domain_adaptation_schedule_alpha: float = 10.0,
        grl_lambda: float = 1.0,
        grl_warm_start: bool = True,
        grl_max_iters: int = 1000,
        domain_hidden_dim: int | None = None,
        domain_num_hidden_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.store_source_ot_plans = False
        num_classes = int(kwargs.get("num_classes"))
        hidden_dim = int(kwargs.get("classifier_hidden_dim", 128))
        dropout = float(kwargs.get("dropout", 0.1))
        self.classifier = CoDATSClassifierHead(
            in_features=self.encoder.out_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.domain_alignment_scheduler = AdaptationWeightScheduler(
            base_weight=float(lambda_adv),
            schedule=str(domain_adaptation_schedule),
            max_steps=int(domain_adaptation_max_steps),
            alpha=float(domain_adaptation_schedule_alpha),
        )
        if grl_warm_start:
            self.grl = WarmStartGradientReverseLayer(
                alpha=float(domain_adaptation_schedule_alpha),
                lo=0.0,
                hi=float(grl_lambda),
                max_iters=int(grl_max_iters),
                auto_step=True,
            )
        else:
            self.grl = GradientReverseLayer(lambda_=float(grl_lambda))
        self.discriminator = DomainDiscriminator(
            self.encoder.out_dim,
            hidden_dim=domain_hidden_dim or max(128, self.encoder.out_dim),
            dropout=dropout,
            num_hidden_layers=int(domain_num_hidden_layers),
        )

        total_steps = max(int(reliability_total_steps), 1)
        self.reliability_start_step = (
            max(int(reliability_start_step), 0)
            if reliability_start_step is not None
            else max(int(round(float(reliability_start_ratio) * total_steps)), 0)
        )
        self.reliability_ramp_steps = (
            max(int(reliability_ramp_steps), 1)
            if reliability_ramp_steps is not None
            else max(int(round(float(reliability_ramp_ratio) * total_steps)), 1)
        )
        default_temperature = 1.0 if self.num_sources <= 2 else 0.5
        self.class_temperature = max(
            float(T_class if T_class is not None else (class_temperature if class_temperature is not None else default_temperature)),
            1e-8,
        )
        self.top_m_per_class = (
            0 if self.num_sources <= 2 else 3
        ) if top_m_per_class is None else max(int(top_m_per_class), 0)
        self.class_floor = (
            0.05 if self.num_sources <= 2 else 0.03
        ) if floor is None else max(float(floor), 0.0)
        self.reliability_component_weights = {
            "D_proto": float(w_proto),
            "D_ot": float(w_ot),
            "H_pred": float(w_entropy),
            "E_src": float(w_source_error),
        }
        self.tau_proto = float(tau_proto)
        self.min_proto_samples = max(int(min_proto_samples), 1)
        self.lambda_ot = float(lambda_ot)
        self.lambda_ccsr = float(lambda_ccsr)
        self.lambda_teacher = float(lambda_teacher)
        self.teacher_temperature = max(float(teacher_temperature), 1e-6)
        self.teacher_anchor_mode = str(teacher_anchor_mode).strip().lower()
        self.teacher_requires_checkpoint = bool(teacher_requires_checkpoint)
        self.teacher_start_step = max(int(teacher_start_step), 0)
        self.teacher_ramp_steps = max(int(teacher_ramp_steps), 1)
        self.teacher_checkpoint_loaded = False
        self._last_reliability_matrix: torch.Tensor | None = None
        self._last_reliability_components: dict[str, torch.Tensor] = {}

        self.teacher_encoder = deepcopy(self.encoder)
        self.teacher_classifier = deepcopy(self.classifier)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher_encoder.eval()
        self.teacher_classifier.eval()
        for module in (self.teacher_encoder, self.teacher_classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def reset_teacher_from_student(self) -> None:
        self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        self.teacher_classifier.load_state_dict(self.classifier.state_dict())
        self._freeze_teacher()

    @staticmethod
    def _compatible_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        current = module.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                compatible[key] = value
        return compatible

    def load_teacher_checkpoint_state(self, state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
        compatible = self._compatible_state_dict(self, state_dict)
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        del unexpected
        self.reset_teacher_from_student()
        self.teacher_checkpoint_loaded = bool(compatible)
        return {
            "teacher_checkpoint_loaded": float(self.teacher_checkpoint_loaded),
            "teacher_checkpoint_loaded_tensors": float(len(compatible)),
            "teacher_checkpoint_missing_tensors": float(len(missing)),
        }

    def teacher_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.teacher_encoder.eval()
        self.teacher_classifier.eval()
        features = self.teacher_encoder(x)
        logits = self.teacher_classifier(features)
        return logits, features

    def student_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features

    def _reliability_ramp(self) -> float:
        if self.global_step <= self.reliability_start_step:
            return 0.0
        progress = (self.global_step - self.reliability_start_step) / float(self.reliability_ramp_steps)
        return max(0.0, min(float(progress), 1.0))

    def _scheduled_teacher_weight(self) -> float:
        if (
            self.lambda_teacher <= 0
            or self.global_step <= self.teacher_start_step
            or (self.teacher_requires_checkpoint and not self.teacher_checkpoint_loaded)
        ):
            return 0.0
        progress = min((self.global_step - self.teacher_start_step) / float(self.teacher_ramp_steps), 1.0)
        return self.lambda_teacher * progress

    def _teacher_available_for_anchor(self) -> bool:
        return (not self.teacher_requires_checkpoint) or self.teacher_checkpoint_loaded

    def _target_prototypes_from_terms(self, terms: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        target_features = F.normalize(terms["features_target"].detach(), p=2, dim=1, eps=1e-6)
        student_probs = terms["source_target_probs"].detach().mean(dim=0)
        teacher_probs = terms.get("teacher_target_probs")
        if teacher_probs is not None:
            mean_probs = 0.5 * (student_probs + teacher_probs.detach().to(device=student_probs.device, dtype=student_probs.dtype))
        else:
            mean_probs = student_probs
        confidence, provisional = mean_probs.max(dim=1)
        num_classes = int(mean_probs.shape[1])
        prototypes = target_features.new_zeros((num_classes, target_features.shape[1]))
        present = torch.zeros(num_classes, device=target_features.device, dtype=torch.bool)
        for class_id in range(num_classes):
            high_conf = (provisional == class_id) & (confidence >= self.tau_proto)
            if int(high_conf.sum().detach().item()) >= self.min_proto_samples:
                prototypes[class_id] = F.normalize(
                    target_features[high_conf].mean(dim=0, keepdim=True),
                    p=2,
                    dim=1,
                    eps=1e-6,
                )[0]
                present[class_id] = True
                continue

            barycenters = []
            if terms.get("gamma_plans"):
                for source_index, gamma_cpu in enumerate(terms["gamma_plans"]):
                    gamma = gamma_cpu.to(device=target_features.device, dtype=target_features.dtype)
                    labels = terms["source_labels"][source_index].detach().long()
                    if gamma.numel() == 0 or gamma.shape[0] != labels.shape[0]:
                        continue
                    class_rows = labels == class_id
                    if not bool(class_rows.any()):
                        continue
                    weights = gamma[class_rows].sum(dim=0)
                    mass = weights.sum()
                    if float(mass.detach().item()) <= 1e-8:
                        continue
                    barycenter = (target_features * weights.view(-1, 1)).sum(dim=0, keepdim=True) / mass.clamp_min(1e-8)
                    barycenters.append(F.normalize(barycenter, p=2, dim=1, eps=1e-6)[0])
            else:
                source_probs = terms["source_target_probs"].detach()
                for source_index in range(source_probs.shape[0]):
                    weights = source_probs[source_index, :, class_id]
                    mass = weights.sum()
                    if float(mass.detach().item()) <= 1e-8:
                        continue
                    barycenter = (target_features * weights.view(-1, 1)).sum(dim=0, keepdim=True) / mass.clamp_min(1e-8)
                    barycenters.append(F.normalize(barycenter, p=2, dim=1, eps=1e-6)[0])
            if barycenters:
                prototypes[class_id] = F.normalize(
                    torch.stack(barycenters, dim=0).mean(dim=0, keepdim=True),
                    p=2,
                    dim=1,
                    eps=1e-6,
                )[0]
                present[class_id] = True
        return prototypes, present

    def _compute_class_alpha(self, terms: dict[str, Any]) -> torch.Tensor:
        source_count = int(terms["source_class_losses"].shape[0])
        num_classes = int(terms["source_class_losses"].shape[1])
        uniform = terms["features_target"].new_full((source_count, num_classes), 1.0 / float(max(source_count, 1)))
        ramp = self._reliability_ramp()
        if ramp <= 0:
            self._last_reliability_matrix = uniform.new_zeros(uniform.shape)
            self._last_reliability_components = {"alpha_uniform": uniform.detach()}
            return uniform

        target_prototypes, target_present = self._target_prototypes_from_terms(terms)
        d_proto = uniform.new_ones((source_count, num_classes))
        for source_index, prototypes in enumerate(terms["source_prototypes"]):
            normalized_source = F.normalize(prototypes.detach(), p=2, dim=1, eps=1e-6)
            distances = (normalized_source - target_prototypes).pow(2).sum(dim=1)
            valid = terms["source_present"][source_index] & target_present
            d_proto[source_index] = torch.where(valid, distances, torch.full_like(distances, 1e6))
        d_ot = torch.where(
            terms["source_present"],
            terms["source_class_losses"].detach(),
            torch.full_like(terms["source_class_losses"].detach(), 1e6),
        )
        source_probs = terms["source_target_probs"].detach()
        teacher_probs = terms.get("teacher_target_probs")
        if teacher_probs is not None:
            provisional_probs = 0.5 * (
                source_probs.mean(dim=0)
                + teacher_probs.detach().to(device=source_probs.device, dtype=source_probs.dtype)
            )
        else:
            provisional_probs = source_probs.mean(dim=0)
        mean_pred = provisional_probs.argmax(dim=1)
        h_pred = uniform.new_ones((source_count, num_classes))
        for source_index in range(source_count):
            entropy = _normalized_entropy(source_probs[source_index])
            for class_id in range(num_classes):
                mask = mean_pred == class_id
                if bool(mask.any()):
                    h_pred[source_index, class_id] = entropy[mask].mean()
        e_src = torch.where(
            terms["source_present"],
            terms["source_eval_errors"].detach(),
            torch.ones_like(terms["source_eval_errors"].detach()),
        )
        present = terms["source_present"]
        d_proto_norm = _minmax_normalize_by_class(d_proto, present)
        d_ot_norm = _minmax_normalize_by_class(d_ot, present)
        h_pred_norm = _minmax_normalize_by_class(h_pred, present)
        e_src_norm = _minmax_normalize_by_class(e_src, present)
        reliability = (
            self.reliability_component_weights["D_proto"] * d_proto_norm
            + self.reliability_component_weights["D_ot"] * d_ot_norm
            + self.reliability_component_weights["H_pred"] * h_pred_norm
            + self.reliability_component_weights["E_src"] * e_src_norm
        )
        alpha = _softmax_source_floor(
            reliability,
            temperature=self.class_temperature,
            floor=self.class_floor,
            top_m=self.top_m_per_class,
        )
        mixed = (1.0 - ramp) * uniform + ramp * alpha
        mixed = mixed / mixed.sum(dim=0, keepdim=True).clamp_min(1e-8)
        self._last_reliability_matrix = reliability.detach()
        self._last_reliability_components = {
            "D_proto": d_proto.detach(),
            "D_ot": d_ot.detach(),
            "H_pred": h_pred.detach(),
            "E_src": e_src.detach(),
            "D_proto_norm": d_proto_norm.detach(),
            "D_ot_norm": d_ot_norm.detach(),
            "H_pred_norm": h_pred_norm.detach(),
            "E_src_norm": e_src_norm.detach(),
            "R": reliability.detach(),
            "alpha": mixed.detach(),
        }
        return mixed

    def _aggregate_class_transport_loss(self, terms: dict[str, Any], alpha: torch.Tensor) -> torch.Tensor:
        class_losses = terms["source_class_losses"]
        present = terms["source_present"].to(device=class_losses.device, dtype=class_losses.dtype)
        weights = alpha.to(device=class_losses.device, dtype=class_losses.dtype) * present
        value = (weights * class_losses).sum()
        if self.class_transport_normalize:
            active_classes = present.sum(dim=0).clamp_max(1.0).sum().clamp_min(1.0)
            value = value / active_classes
        return value

    def _teacher_anchor_loss(
        self,
        target_x: torch.Tensor,
        student_logits: torch.Tensor,
        student_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        weight = self._scheduled_teacher_weight()
        zero = student_logits.new_zeros(())
        if not self._teacher_available_for_anchor():
            return zero, {
                "lambda_teacher": 0.0,
                "loss_teacher_anchor": 0.0,
                "teacher_student_agreement": 0.0,
                "teacher_checkpoint_loaded": float(self.teacher_checkpoint_loaded),
                "teacher_anchor_enabled": 0.0,
            }, None
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher_logits_and_features(target_x)
        metrics = {
            "lambda_teacher": float(weight),
            "loss_teacher_anchor": 0.0,
            "teacher_student_agreement": 0.0,
            "teacher_checkpoint_loaded": float(self.teacher_checkpoint_loaded),
            "teacher_anchor_enabled": 1.0,
        }
        teacher_probs = F.softmax(teacher_logits.detach().float(), dim=1).to(dtype=student_logits.dtype)
        student_probs = F.softmax(student_logits.detach().float(), dim=1).to(dtype=student_logits.dtype)
        if teacher_probs.numel() > 0:
            metrics["teacher_student_agreement"] = float(
                (teacher_probs.argmax(dim=1) == student_probs.argmax(dim=1)).float().mean().detach().item()
            )
        if weight <= 0:
            return zero, metrics, teacher_probs.detach()

        temperature = self.teacher_temperature
        mode = self.teacher_anchor_mode
        losses: list[torch.Tensor] = []
        if mode in {"kl", "ce", "logit_kl", "kl_mse"}:
            teacher_soft = F.softmax(teacher_logits.detach().float() / temperature, dim=1)
            student_log = F.log_softmax(student_logits.float() / temperature, dim=1)
            losses.append(F.kl_div(student_log, teacher_soft, reduction="batchmean") * (temperature ** 2))
        if mode in {"mse", "embedding_mse", "kl_mse"}:
            losses.append(
                F.mse_loss(
                    F.normalize(student_features.float(), p=2, dim=1, eps=1e-6),
                    F.normalize(teacher_features.detach().float(), p=2, dim=1, eps=1e-6),
                )
            )
        if not losses:
            teacher_soft = F.softmax(teacher_logits.detach().float() / temperature, dim=1)
            student_log = F.log_softmax(student_logits.float() / temperature, dim=1)
            losses.append(F.kl_div(student_log, teacher_soft, reduction="batchmean") * (temperature ** 2))
        loss = torch.stack(losses).mean()
        metrics["loss_teacher_anchor"] = float(loss.detach().item())
        return weight * loss, metrics, teacher_probs.detach()

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        target_x, _ = target_batch
        terms = self._compute_sourceaware_terms(source_batches, target_batch)
        loss_teacher, teacher_metrics, teacher_target_probs = self._teacher_anchor_loss(
            target_x,
            terms["target_logits"],
            terms["features_target"],
        )
        if teacher_target_probs is not None:
            terms["teacher_target_probs"] = teacher_target_probs
        class_alpha = self._compute_class_alpha(terms)
        loss_ot = self._aggregate_transport_loss(terms)
        loss_ccsr = self._aggregate_class_transport_loss(terms, class_alpha)
        source_features = torch.cat(terms["source_features"], dim=0)
        loss_adv, domain_acc = domain_adversarial_loss(
            source_features,
            terms["features_target"],
            discriminator=self.discriminator,
            grl=self.grl,
        )
        lambda_adv = self.domain_alignment_scheduler.step()
        ot_ramp = float(terms["current_alignment_weight"])
        ccsr_ramp = self._reliability_ramp()
        loss_total = (
            self.source_ce_weight * terms["loss_cls"]
            + lambda_adv * loss_adv
            + self.lambda_ot * ot_ramp * loss_ot
            + self.lambda_ccsr * ccsr_ramp * loss_ccsr
            + loss_teacher
            + terms["target_label_assist"]
            + self.embedding_norm_weight * terms["loss_norm"]
        )
        metrics = dict(terms["metrics"])
        metrics.update(
            {
                "loss_total": float(loss_total.detach().item()),
                "loss_alignment": float((loss_ot + loss_ccsr).detach().item()),
                "loss_ot": float(loss_ot.detach().item()),
                "loss_ccsr": float(loss_ccsr.detach().item()),
                "loss_domain_adv": float(loss_adv.detach().item()),
                "lambda_adv": float(lambda_adv),
                "lambda_ot": float(self.lambda_ot * ot_ramp),
                "lambda_ccsr": float(self.lambda_ccsr * ccsr_ramp),
                "sourceaware_class_transport_training": 1.0,
                "reliability_ramp": float(ccsr_ramp),
                "class_alpha_min": float(class_alpha.min().detach().item()),
                "class_alpha_max": float(class_alpha.max().detach().item()),
                "class_alpha_std": float(class_alpha.std(unbiased=False).detach().item()),
                "acc_domain": float(domain_acc),
                "grl_coeff": float(getattr(self.grl, "last_coeff", 1.0)),
            }
        )
        metrics.update(teacher_metrics)
        for source_index in range(class_alpha.shape[0]):
            metrics[f"alpha_source_{source_index}"] = float(class_alpha[source_index].mean().detach().item())
            for class_id in range(class_alpha.shape[1]):
                metrics[f"alpha_source_{source_index}_class_{class_id:02d}"] = float(
                    class_alpha[source_index, class_id].detach().item()
                )
        self._last_source_weights = class_alpha.mean(dim=1).detach()
        self._last_class_source_weights = class_alpha.detach()
        self._last_transport_mass_matrix = terms["transport_mass_matrix"].detach()
        self._last_per_class_ot_cost_matrix = terms["per_class_ot_cost_matrix"].detach()
        self._last_source_ot_plans = terms["gamma_plans"]
        self._last_source_prediction_histogram = terms["prediction_histogram"].detach()
        return MethodStepOutput(loss=loss_total, metrics=metrics)

    def reliability_snapshot(self) -> dict[str, torch.Tensor]:
        snapshot = super().reliability_snapshot()
        if self._last_reliability_matrix is not None:
            snapshot["reliability_matrix"] = self._last_reliability_matrix.detach().cpu()
        for key, value in self._last_reliability_components.items():
            snapshot[f"reliability_{key}"] = value.detach().cpu()
        return snapshot


class CCSRWJDOTFusionMethod(WJDOTMethod):
    """WJDOT training with final-stage CCSR prediction fusion."""

    method_name = "ccsr_wjdot_fusion"


class JDOTMethod(WJDOTMethod):
    """Single-source JDOT baseline without source-domain reweighting."""

    method_name = "jdot"
    supports_multi_source = False

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("use_source_class_balance", False)
        super().__init__(**kwargs)


class TPJDOTMethod(JDOTMethod):
    """Temporal-Prototype JDOT for the single-source innovation line."""

    method_name = "tp_jdot"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["use_temporal_prototypes"] = True
        kwargs.setdefault("prototype_weight", 0.04)
        kwargs.setdefault("prototype_mode", "tp_barycentric")
        kwargs.setdefault("prototype_in_coupling", False)
        kwargs.setdefault("ot_class_entropy_gate", True)
        super().__init__(**kwargs)


class CBTPJDOTMethod(TPJDOTMethod):
    """Confidence-Balanced Temporal-Prototype JDOT."""

    method_name = "cbtp_jdot"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["confidence_curriculum"] = True
        base_tp_mode = kwargs.pop("base_tp_mode", None)
        if base_tp_mode is not None:
            kwargs.setdefault("prototype_mode", str(base_tp_mode))
        super().__init__(**kwargs)


class TPWJDOTMethod(WJDOTMethod):
    """Temporal-Prototype WJDOT."""

    method_name = "tp_wjdot"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["use_temporal_prototypes"] = True
        kwargs.setdefault("prototype_weight", 0.04)
        kwargs.setdefault("prototype_mode", "tp_barycentric")
        kwargs.setdefault("prototype_in_coupling", False)
        kwargs.setdefault("ot_class_entropy_gate", True)
        super().__init__(**kwargs)


class CBTPWJDOTMethod(TPWJDOTMethod):
    """Confidence-Balanced Temporal-Prototype WJDOT."""

    method_name = "cbtp_wjdot"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["confidence_curriculum"] = True
        base_tp_mode = kwargs.pop("base_tp_mode", None)
        if base_tp_mode is not None:
            kwargs.setdefault("prototype_mode", str(base_tp_mode))
        super().__init__(**kwargs)


class MSCBTPWJDOTMethod(CBTPWJDOTMethod):
    """Multi-source CBTP-WJDOT with source/class/sample reliability weights."""

    method_name = "ms_cbtp_wjdot"

    def __init__(
        self,
        *,
        source_weight_temperature: float = 1.0,
        class_weight_temperature: float = 1.0,
        negative_gate_threshold: float = 0.05,
        negative_gate_floor: float = 0.1,
        ms_weighting_mode: str = "global_alpha",
        class_alpha_only: bool = False,
        class_alpha_top_m_sources: int = 0,
        class_alpha_renormalize_top_m: bool = True,
        class_unbalanced_transport: bool = False,
        unbalanced_sinkhorn_reg_m: float = 1.0,
        target_label_calibration: bool = False,
        target_label_assisted_source_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source_weight_temperature = max(float(source_weight_temperature), 1e-6)
        self.class_weight_temperature = max(float(class_weight_temperature), 1e-6)
        self.negative_gate_threshold = float(negative_gate_threshold)
        self.negative_gate_floor = float(negative_gate_floor)
        mode = str(ms_weighting_mode).strip().lower()
        if bool(class_alpha_only):
            mode = "class_alpha"
        if bool(class_unbalanced_transport):
            mode = "class_alpha_unbalanced"
        if mode in {"current", "legacy", "global", "source_alpha"}:
            mode = "global_alpha"
        if mode not in {"global_alpha", "class_alpha", "class_alpha_unbalanced"}:
            raise ValueError(f"Unsupported MS-CBTP weighting mode: {ms_weighting_mode}")
        self.ms_weighting_mode = mode
        self.class_alpha_top_m_sources = max(int(class_alpha_top_m_sources), 0)
        self.class_alpha_renormalize_top_m = bool(class_alpha_renormalize_top_m)
        self.class_unbalanced_transport = mode == "class_alpha_unbalanced"
        self.unbalanced_sinkhorn_reg_m = max(float(unbalanced_sinkhorn_reg_m), 1e-6)
        self.target_label_calibration = bool(target_label_calibration or target_label_assisted_source_weights)
        self.target_label_assisted_source_weights = self.target_label_calibration
        self._last_transport_mass_matrix: torch.Tensor | None = None
        self._last_per_class_ot_cost_matrix: torch.Tensor | None = None
        self._last_per_class_proto_distance_matrix: torch.Tensor | None = None

    def _source_alpha(self, source_features: list[torch.Tensor], target_features: torch.Tensor) -> torch.Tensor:
        distances = [
            torch.cdist(features.detach(), target_features.detach(), p=2).pow(2).mean()
            for features in source_features
        ]
        return F.softmax(-torch.stack(distances) / self.source_weight_temperature, dim=0).detach()

    def _class_alpha(
        self,
        source_prototypes: list[torch.Tensor],
        source_present: list[torch.Tensor],
        target_features: torch.Tensor,
        target_logits: torch.Tensor,
        target_y: torch.Tensor | None,
        source_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        probabilities = F.softmax(target_logits.detach().float(), dim=1)
        confidence, pseudo_labels = probabilities.max(dim=1)
        num_classes = int(target_logits.shape[1])
        source_count = len(source_prototypes)
        label_assist_mask = (
            target_y is not None
            and self.target_label_calibration
            and bool((target_y >= 0).any())
        )
        if label_assist_mask:
            valid_target = target_y >= 0
            target_prototypes, target_present = compute_class_prototypes(
                target_features[valid_target].detach(),
                target_y[valid_target].long(),
                num_classes=num_classes,
            )
        else:
            confident_mask = confidence > self.current_tau()
            if bool(confident_mask.any()):
                target_prototypes, target_present = compute_class_prototypes(
                    target_features[confident_mask].detach(),
                    pseudo_labels[confident_mask],
                    num_classes=num_classes,
                    sample_weights=confidence[confident_mask],
                )
            else:
                target_prototypes = target_features.new_zeros((num_classes, target_features.shape[1]))
                target_present = torch.zeros(num_classes, device=target_features.device, dtype=torch.bool)

        class_alpha = target_features.new_zeros((source_count, num_classes))
        proto_distances = target_features.new_full((source_count, num_classes), float("nan"))
        for class_id in range(num_classes):
            valid_sources = [
                source_index
                for source_index in range(source_count)
                if bool(source_present[source_index][class_id]) and bool(target_present[class_id])
            ]
            if not valid_sources:
                class_alpha[:, class_id] = source_alpha.to(device=target_features.device, dtype=target_features.dtype)
                continue
            distances = torch.stack(
                [
                    (source_prototypes[source_index][class_id] - target_prototypes[class_id]).pow(2).sum()
                    for source_index in valid_sources
                ]
            )
            valid_alpha = F.softmax(-distances / self.class_weight_temperature, dim=0)
            for local_index, source_index in enumerate(valid_sources):
                class_alpha[source_index, class_id] = valid_alpha[local_index]
                proto_distances[source_index, class_id] = distances[local_index]
        class_alpha = self._apply_class_top_m(class_alpha, source_present)
        return class_alpha.detach(), proto_distances.detach(), bool(label_assist_mask)

    def _apply_class_top_m(
        self,
        class_alpha: torch.Tensor,
        source_present: list[torch.Tensor],
    ) -> torch.Tensor:
        top_m = self.class_alpha_top_m_sources
        source_count, num_classes = int(class_alpha.shape[0]), int(class_alpha.shape[1])
        if top_m <= 0 or top_m >= source_count:
            return class_alpha
        present = torch.stack(
            [mask.to(device=class_alpha.device, dtype=torch.bool) for mask in source_present],
            dim=0,
        )
        filtered = class_alpha.new_zeros(class_alpha.shape)
        for class_id in range(num_classes):
            valid = present[:, class_id] & (class_alpha[:, class_id] > 0)
            if not bool(valid.any()):
                filtered[:, class_id] = class_alpha[:, class_id]
                continue
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            values = class_alpha[valid_indices, class_id]
            keep_count = min(top_m, int(values.numel()))
            keep_local = torch.topk(values, k=keep_count, largest=True).indices
            keep_indices = valid_indices[keep_local]
            filtered[keep_indices, class_id] = class_alpha[keep_indices, class_id]
            if self.class_alpha_renormalize_top_m:
                total = filtered[:, class_id].sum()
                if bool(torch.isfinite(total).item()) and float(total.item()) > 0:
                    filtered[:, class_id] = filtered[:, class_id] / total.clamp_min(1e-8)
        return filtered

    def _ms_source_weights(
        self,
        source_features: list[torch.Tensor],
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.ms_weighting_mode == "global_alpha":
            return self._source_alpha(source_features, target_features)
        count = max(len(source_features), 1)
        return target_features.new_full((count,), 1.0 / float(count))

    def _source_class_sample_weights(
        self,
        *,
        source_index: int,
        labels: torch.Tensor,
        base_class_weights: torch.Tensor,
        outlier_weight: torch.Tensor,
        class_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = outlier_weight.dtype
        if self.ms_weighting_mode == "global_alpha":
            class_gate = torch.where(
                class_alpha[source_index, labels.long()] < self.negative_gate_threshold,
                torch.full_like(class_alpha[source_index, labels.long()], self.negative_gate_floor),
                torch.ones_like(class_alpha[source_index, labels.long()]),
            )
            pair_weights = class_gate.to(device=labels.device, dtype=dtype)
        else:
            pair_weights = class_alpha[source_index, labels.long()].to(device=labels.device, dtype=dtype).clamp_min(0.0)
        sample_weights = (
            base_class_weights[labels.long()]
            * outlier_weight.to(device=labels.device, dtype=dtype)
            * pair_weights
        )
        return sample_weights, pair_weights

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        if len(source_batches) == 1:
            output = super().compute_loss(source_batches, target_batch)
            output.metrics["ms_cbtp_single_source_degenerate_to_cbtp"] = 1.0
            return output

        self.global_step += 1
        target_x, target_y = target_batch
        logits_target, features_target = self.forward(target_x)
        num_classes = int(logits_target.shape[1])
        current_alignment_weight = self.current_alignment_weight()

        source_logits: list[torch.Tensor] = []
        source_features: list[torch.Tensor] = []
        source_ot_features: list[torch.Tensor] = []
        source_labels: list[torch.Tensor] = []
        source_prototypes: list[torch.Tensor] = []
        source_present: list[torch.Tensor] = []
        source_class_weights: list[torch.Tensor] = []
        outlier_weights: list[torch.Tensor] = []

        for source_x, source_y in source_batches:
            logits_source, features_source = self.forward(source_x)
            ot_source_features, _ = self._prepare_ot_features(features_source, features_target)
            class_weights = inverse_sqrt_class_weights(source_y, num_classes=num_classes).to(
                device=source_y.device,
                dtype=features_source.dtype,
            )
            prototypes, present = compute_class_prototypes(
                ot_source_features,
                source_y,
                num_classes=num_classes,
                sample_weights=class_weights[source_y.long()],
            )
            source_logits.append(logits_source)
            source_features.append(features_source)
            source_ot_features.append(ot_source_features)
            source_labels.append(source_y)
            source_prototypes.append(prototypes)
            source_present.append(present)
            source_class_weights.append(class_weights)
            outlier_weights.append(source_outlier_weights(ot_source_features.detach(), source_y, prototypes.detach()))

        loss_cls = features_target.new_zeros(())
        for source_index, y_source in enumerate(source_labels):
            sample_weights = source_class_weights[source_index][y_source.long()]
            loss_cls = loss_cls + _weighted_ce(source_logits[source_index], y_source, sample_weights)
        loss_cls = loss_cls / float(max(len(source_labels), 1))
        loss_norm = self._embedding_norm_loss(*source_features, features_target)
        target_label_assist, target_label_metrics = self._target_label_assist_loss(logits_target, target_y)
        if current_alignment_weight <= 0:
            loss_total = self.source_ce_weight * loss_cls + target_label_assist + self.embedding_norm_weight * loss_norm
            metrics = {
                "loss_total": float(loss_total.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_alignment": 0.0,
                "loss_embedding_norm": float(loss_norm.detach().item()),
                "lambda_alignment": 0.0,
                "lambda_prototype": 0.0,
                "acc_source": accuracy_from_logits(torch.cat(source_logits, dim=0), torch.cat(source_labels, dim=0)),
            }
            metrics.update(target_label_metrics)
            return MethodStepOutput(loss=loss_total, metrics=metrics)

        _, ot_target_features = self._prepare_ot_features(features_target.detach(), features_target)
        current_prototype_weight = self.current_prototype_weight()
        current_prototype_coupling_weight = self.current_prototype_coupling_weight()
        ot_config = replace(
            self.ot_config,
            prototype_weight=current_prototype_weight,
            prototype_coupling_weight=current_prototype_coupling_weight,
        )
        target_regularizer, target_metrics, label_gate = self._target_regularizers(target_x, logits_target)
        prototype_gate, prototype_metrics = self._prototype_gate(
            logits_target,
            max(current_prototype_weight, current_prototype_coupling_weight),
            fallback_gate=label_gate,
        )

        source_alpha = self._ms_source_weights(source_ot_features, ot_target_features)
        class_alpha, proto_distance_matrix, target_label_calibration_used = self._class_alpha(
            source_prototypes,
            source_present,
            ot_target_features,
            logits_target,
            target_y,
            source_alpha,
        )
        loss_ot = features_target.new_zeros(())
        metrics: dict[str, float] = {}
        transport_mass_matrix = features_target.new_zeros((len(source_labels), num_classes))
        per_class_ot_cost_matrix = features_target.new_zeros((len(source_labels), num_classes))
        aggregation_weights = source_alpha.to(device=features_target.device, dtype=features_target.dtype)
        for source_index, y_source in enumerate(source_labels):
            sample_weights, pair_weights = self._source_class_sample_weights(
                source_index=source_index,
                labels=y_source,
                base_class_weights=source_class_weights[source_index],
                outlier_weight=outlier_weights[source_index],
                class_alpha=class_alpha,
            )
            if (
                self.ms_weighting_mode == "class_alpha"
                and not self.class_unbalanced_transport
                and float(sample_weights.detach().sum().item()) <= 0
            ):
                ot_loss = features_target.new_zeros(())
                ot_metrics = {
                    "loss_ot": 0.0,
                    "ot_transport_mass": 0.0,
                    **{f"ot_transport_mass_class_{class_id:02d}": 0.0 for class_id in range(num_classes)},
                    **{f"ot_cost_class_{class_id:02d}": 0.0 for class_id in range(num_classes)},
                }
            else:
                source_ot_config = replace(
                    ot_config,
                    unbalanced_transport=self.class_unbalanced_transport,
                    unbalanced_reg_m=self.unbalanced_sinkhorn_reg_m,
                )
                ot_loss, ot_metrics = jdot_transport_loss(
                    source_features=source_ot_features[source_index],
                    source_labels=y_source,
                    target_features=ot_target_features,
                    target_logits=logits_target,
                    num_classes=num_classes,
                    config=source_ot_config,
                    source_sample_weights=sample_weights,
                    source_prototypes=source_prototypes[source_index],
                    label_gate=label_gate,
                    prototype_gate=prototype_gate,
                )
            for class_id in range(num_classes):
                transport_mass_matrix[source_index, class_id] = float(
                    ot_metrics.get(f"ot_transport_mass_class_{class_id:02d}", 0.0)
                )
                per_class_ot_cost_matrix[source_index, class_id] = float(
                    ot_metrics.get(f"ot_cost_class_{class_id:02d}", 0.0)
                )
            source_weight = aggregation_weights[source_index].to(
                device=features_target.device,
                dtype=features_target.dtype,
            )
            loss_ot = loss_ot + source_weight * ot_loss
            source_alpha_stat = class_alpha[source_index].mean()
            if self.ms_weighting_mode == "global_alpha":
                source_alpha_stat = source_alpha[source_index]
            metrics[f"alpha_source_{source_index}"] = float(source_alpha_stat.detach().item())
            metrics[f"loss_source_weight_{source_index}"] = float(source_weight.detach().item())
            metrics[f"loss_ot_source_{source_index}"] = float(ot_metrics["loss_ot"])
            metrics[f"transport_mass_source_{source_index}"] = float(ot_metrics.get("ot_transport_mass", 0.0))
            metrics[f"class_pair_weight_mean_source_{source_index}"] = float(pair_weights.detach().mean().item())
            metrics[f"class_pair_weight_nonzero_source_{source_index}"] = float(
                (pair_weights.detach() > 0).float().mean().item()
            )

        finite_proto = proto_distance_matrix[torch.isfinite(proto_distance_matrix)]
        proto_distance_min = float(finite_proto.min().detach().item()) if bool(finite_proto.numel() > 0) else 0.0
        proto_distance_max = float(finite_proto.max().detach().item()) if bool(finite_proto.numel() > 0) else 0.0
        loss_total = (
            self.source_ce_weight * loss_cls
            + current_alignment_weight * loss_ot
            + target_regularizer
            + target_label_assist
            + self.embedding_norm_weight * loss_norm
        )
        metrics.update(
            {
                "loss_total": float(loss_total.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_alignment": float(loss_ot.detach().item()),
                "loss_embedding_norm": float(loss_norm.detach().item()),
                "lambda_alignment": float(current_alignment_weight),
                "acc_source": accuracy_from_logits(torch.cat(source_logits, dim=0), torch.cat(source_labels, dim=0)),
                "ms_weighting_mode_id": float(
                    {
                        "global_alpha": 0,
                        "class_alpha": 1,
                        "class_alpha_unbalanced": 2,
                    }[self.ms_weighting_mode]
                ),
                "target_label_calibration_used": float(target_label_calibration_used),
                "class_alpha_min": float(class_alpha.min().detach().item()),
                "class_alpha_max": float(class_alpha.max().detach().item()),
                "class_alpha_std": float(class_alpha.std(unbiased=False).detach().item()),
                "class_proto_distance_min": proto_distance_min,
                "class_proto_distance_max": proto_distance_max,
                "transport_mass_total": float(transport_mass_matrix.sum().detach().item()),
            }
        )
        for class_id in range(num_classes):
            metrics[f"transport_mass_class_{class_id:02d}"] = float(
                transport_mass_matrix[:, class_id].sum().detach().item()
            )
            active_distances = proto_distance_matrix[:, class_id]
            active_distances = active_distances[torch.isfinite(active_distances)]
            metrics[f"proto_distance_class_{class_id:02d}"] = (
                float(active_distances.mean().detach().item()) if bool(active_distances.numel() > 0) else 0.0
            )
        metrics.update(target_metrics)
        metrics.update(prototype_metrics)
        metrics.update(target_label_metrics)
        metrics["lambda_prototype_residual"] = float(current_prototype_weight)
        metrics["lambda_prototype_coupling"] = float(current_prototype_coupling_weight)
        self._last_source_weights = (
            class_alpha.mean(dim=1).detach()
            if self.ms_weighting_mode != "global_alpha"
            else source_alpha.detach()
        )
        self._last_class_source_weights = class_alpha.detach()
        self._last_transport_mass_matrix = transport_mass_matrix.detach()
        self._last_per_class_ot_cost_matrix = per_class_ot_cost_matrix.detach()
        self._last_per_class_proto_distance_matrix = proto_distance_matrix.detach()
        return MethodStepOutput(loss=loss_total, metrics=metrics)

    def reliability_snapshot(self) -> dict[str, torch.Tensor]:
        snapshot = super().reliability_snapshot()
        if self._last_transport_mass_matrix is not None:
            snapshot["transport_mass_matrix"] = self._last_transport_mass_matrix.detach().cpu()
        if self._last_per_class_ot_cost_matrix is not None:
            snapshot["per_class_ot_cost_matrix"] = self._last_per_class_ot_cost_matrix.detach().cpu()
        if self._last_per_class_proto_distance_matrix is not None:
            snapshot["per_class_proto_distance_matrix"] = self._last_per_class_proto_distance_matrix.detach().cpu()
        return snapshot
