"""WJDOT family methods integrated into the existing benchmark trainer."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from src.tep_ot.augment import augment_signal
from src.tep_ot.ot_losses import (
    OTLossConfig,
    compute_class_prototypes,
    inverse_sqrt_class_weights,
    jdot_transport_loss,
    source_outlier_weights,
    target_pseudo_class_weights,
)

from .base import MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


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
