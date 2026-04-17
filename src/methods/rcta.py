"""Reliability-aware class temporal adaptation."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.losses import (
    ConditionalDomainAdversarialLoss,
    DomainDiscriminator,
    GradientReverseLayer,
    MinimumClassConfusionLoss,
    WarmStartGradientReverseLayer,
    deepjdot_loss,
    domain_adversarial_loss,
)

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


_EPSILON = 1e-6


def _zero(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((), device=device, dtype=dtype)


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, dim=1, eps=_EPSILON)


class _TemporalAugmenter:
    """Conservative augmentations for fixed-length process windows."""

    def __init__(
        self,
        *,
        weak_jitter_std: float = 0.01,
        weak_scaling_std: float = 0.01,
        strong_jitter_std: float = 0.02,
        strong_scaling_std: float = 0.02,
        strong_time_mask_ratio: float = 0.1,
        strong_channel_dropout_prob: float = 0.1,
    ) -> None:
        self.weak_jitter_std = max(float(weak_jitter_std), 0.0)
        self.weak_scaling_std = max(float(weak_scaling_std), 0.0)
        self.strong_jitter_std = max(float(strong_jitter_std), 0.0)
        self.strong_scaling_std = max(float(strong_scaling_std), 0.0)
        self.strong_time_mask_ratio = min(max(float(strong_time_mask_ratio), 0.0), 1.0)
        self.strong_channel_dropout_prob = min(max(float(strong_channel_dropout_prob), 0.0), 1.0)

    def _apply_jitter(self, x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        return x + torch.randn_like(x) * std

    def _apply_scaling(self, x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        scale = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) * std
        return x * scale

    def _apply_time_mask(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
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

    def _apply_channel_dropout(self, x: torch.Tensor, probability: float) -> torch.Tensor:
        if probability <= 0:
            return x
        keep_mask = (
            torch.rand(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype) >= probability
        ).to(x.dtype)
        return x * keep_mask

    def weak(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_jitter(x, self.weak_jitter_std)
        return self._apply_scaling(x, self.weak_scaling_std)

    def strong(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_jitter(x, self.strong_jitter_std)
        x = self._apply_scaling(x, self.strong_scaling_std)
        x = self._apply_time_mask(x, self.strong_time_mask_ratio)
        return self._apply_channel_dropout(x, self.strong_channel_dropout_prob)

    def augment_pair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weak(x), self.strong(x)


class _ReliabilityGate:
    """Apply a score floor and per-class top-ratio curriculum."""

    def __init__(
        self,
        *,
        score_floor: float,
        accept_ratio_start: float,
        accept_ratio_end: float,
        curriculum_steps: int,
    ) -> None:
        self.score_floor = float(score_floor)
        self.accept_ratio_start = min(max(float(accept_ratio_start), 0.0), 1.0)
        self.accept_ratio_end = min(max(float(accept_ratio_end), 0.0), 1.0)
        self.curriculum_steps = max(int(curriculum_steps), 1)

    def curriculum_ratio(self, step_num: int) -> float:
        progress = min(max(float(step_num), 0.0) / float(self.curriculum_steps), 1.0)
        return self.accept_ratio_start + (self.accept_ratio_end - self.accept_ratio_start) * progress

    def select(
        self,
        scores: torch.Tensor,
        pseudo_labels: torch.Tensor,
        step_num: int,
        *,
        score_floor_override: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        selected = torch.zeros_like(scores, dtype=torch.bool)
        ratio = self.curriculum_ratio(step_num)
        if ratio <= 0:
            return selected, ratio

        score_floor = self.score_floor if score_floor_override is None else float(score_floor_override)
        floor_mask = scores >= score_floor
        for class_id in torch.unique(pseudo_labels):
            class_indices = torch.nonzero((pseudo_labels == class_id) & floor_mask, as_tuple=False).squeeze(1)
            if class_indices.numel() == 0:
                continue
            keep_count = max(1, int(math.ceil(class_indices.numel() * ratio)))
            keep_count = min(keep_count, int(class_indices.numel()))
            class_scores = scores[class_indices]
            top_positions = torch.topk(class_scores, k=keep_count, largest=True).indices
            selected[class_indices[top_positions]] = True
        return selected, ratio


class _ReliabilityPartition:
    """Split target samples into reliable, semi-reliable, and unreliable groups."""

    def __init__(self, *, reliable_floor: float, semi_reliable_floor: float) -> None:
        self.reliable_floor = float(reliable_floor)
        self.semi_reliable_floor = float(semi_reliable_floor)

    def split(self, scores: torch.Tensor, gate_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reliable = gate_mask & (scores >= self.reliable_floor)
        semi_reliable = gate_mask & ~reliable & (scores >= self.semi_reliable_floor)
        unreliable = ~(reliable | semi_reliable)
        return reliable, semi_reliable, unreliable

    def soft_weights(self, scores: torch.Tensor) -> torch.Tensor:
        reliable = torch.ones_like(scores)
        semi_reliable = torch.clamp((scores - self.semi_reliable_floor) / max(self.reliable_floor - self.semi_reliable_floor, _EPSILON), 0.0, 1.0)
        weights = torch.where(scores >= self.reliable_floor, reliable, semi_reliable)
        return weights.clamp(0.0, 1.0)


class _CDANAligner(nn.Module):
    """CDAN alignment branch used inside RCTA."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        adaptation_weight: float,
        adaptation_schedule: str,
        adaptation_max_steps: int,
        adaptation_schedule_alpha: float,
        grl_lambda: float,
        grl_warm_start: bool,
        grl_max_iters: int,
        randomized: bool,
        randomized_dim: int,
        entropy_conditioning: bool,
        domain_hidden_dim: int | None,
        domain_num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        conditioned_dim = randomized_dim if randomized else feature_dim * num_classes
        if grl_warm_start:
            grl = WarmStartGradientReverseLayer(
                alpha=adaptation_schedule_alpha,
                lo=0.0,
                hi=grl_lambda,
                max_iters=grl_max_iters,
                auto_step=True,
            )
        else:
            grl = GradientReverseLayer(lambda_=grl_lambda)
        self.loss = ConditionalDomainAdversarialLoss(
            domain_discriminator=DomainDiscriminator(
                conditioned_dim,
                hidden_dim=domain_hidden_dim or max(256, feature_dim),
                dropout=dropout,
                num_hidden_layers=domain_num_hidden_layers,
            ),
            feature_dim=feature_dim,
            num_classes=num_classes,
            entropy_conditioning=entropy_conditioning,
            randomized=randomized,
            randomized_dim=randomized_dim,
            grl=grl,
        )

    def forward(
        self,
        *,
        logits_source: torch.Tensor,
        features_source: torch.Tensor,
        logits_target: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_alignment, domain_acc = self.loss(
            logits_source,
            features_source,
            logits_target,
            features_target,
        )
        current_weight = self.scheduler.step()
        return loss_alignment, {
            "lambda_alignment": current_weight,
            "acc_domain": domain_acc,
            "grl_coeff": float(getattr(self.loss.grl, "last_coeff", 1.0)),
        }


class _DeepJDOTAligner(nn.Module):
    """DeepJDOT alignment branch used inside RCTA."""

    def __init__(
        self,
        *,
        adaptation_weight: float,
        adaptation_schedule: str,
        adaptation_max_steps: int,
        adaptation_schedule_alpha: float,
        reg_dist: float,
        reg_cl: float,
        normalize_feature_cost: bool,
    ) -> None:
        super().__init__()
        self.scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        self.reg_dist = float(reg_dist)
        self.reg_cl = float(reg_cl)
        self.normalize_feature_cost = bool(normalize_feature_cost)

    def forward(
        self,
        *,
        source_labels: torch.Tensor,
        logits_target: torch.Tensor,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_alignment = deepjdot_loss(
            source_labels,
            logits_target,
            features_source,
            features_target,
            reg_dist=self.reg_dist,
            reg_cl=self.reg_cl,
            normalize_feature_cost=self.normalize_feature_cost,
        )
        current_weight = self.scheduler.step()
        return loss_alignment, {
            "lambda_alignment": current_weight,
        }


class _DANNAligner(nn.Module):
    """DANN alignment branch used inside RCTA."""

    def __init__(
        self,
        *,
        feature_dim: int,
        adaptation_weight: float,
        adaptation_schedule: str,
        adaptation_max_steps: int,
        adaptation_schedule_alpha: float,
        grl_lambda: float,
        grl_warm_start: bool,
        grl_max_iters: int,
        domain_hidden_dim: int | None,
        domain_num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        if grl_warm_start:
            self.grl = WarmStartGradientReverseLayer(
                alpha=adaptation_schedule_alpha,
                lo=0.0,
                hi=grl_lambda,
                max_iters=grl_max_iters,
                auto_step=True,
            )
        else:
            self.grl = GradientReverseLayer(lambda_=grl_lambda)
        self.discriminator = DomainDiscriminator(
            feature_dim,
            hidden_dim=domain_hidden_dim or max(128, feature_dim),
            dropout=dropout,
            num_hidden_layers=domain_num_hidden_layers,
        )

    def forward(
        self,
        *,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_alignment, domain_acc = domain_adversarial_loss(
            features_source,
            features_target,
            discriminator=self.discriminator,
            grl=self.grl,
        )
        current_weight = self.scheduler.step()
        return loss_alignment, {
            "lambda_alignment": current_weight,
            "acc_domain": domain_acc,
            "grl_coeff": float(getattr(self.grl, "last_coeff", 1.0)),
        }


class RCTAMethod(SingleSourceMethodBase):
    """Reliable temporal class adaptation for fixed windows."""

    method_name = "rcta"

    def __init__(
        self,
        *,
        num_classes: int,
        base_align: str = "cdan",
        use_mcc: bool = True,
        mcc_weight: float = 0.1,
        mcc_temperature: float = 2.0,
        teacher_ema_decay: float = 0.99,
        teacher_temperature: float = 1.5,
        reliability_weights: dict[str, float] | None = None,
        gate_score_floor: float = 0.55,
        gate_score_floor_start: float | None = None,
        gate_score_floor_end: float | None = None,
        gate_score_floor_schedule_steps: int = 1000,
        gate_accept_ratio_start: float = 0.2,
        gate_accept_ratio_end: float = 0.7,
        gate_curriculum_steps: int = 1000,
        reliable_score_floor: float | None = None,
        semi_reliable_score_floor: float | None = None,
        semi_reliable_consistency_weight: float = 0.5,
        unreliable_entropy_weight: float = 0.05,
        pseudo_label_weight: float = 0.2,
        pseudo_warmup_steps: int = 0,
        pseudo_use_reliability_weighting: bool = True,
        pseudo_confidence_power: float = 1.0,
        prototype_weight: float = 0.1,
        prototype_start_step: int = 0,
        prototype_warmup_steps: int | None = None,
        prototype_separation_weight: float = 0.1,
        consistency_weight: float = 0.1,
        consistency_start_step: int = 0,
        consistency_warmup_steps: int | None = None,
        consistency_gate_only: bool = False,
        consistency_reliability_power: float = 1.0,
        alignment_start_step: int = 0,
        alignment_use_reliable_only: bool = True,
        prototype_momentum: float = 0.9,
        prototype_separation_margin: float = 0.2,
        augment_kwargs: dict[str, Any] | None = None,
        cdan_kwargs: dict[str, Any] | None = None,
        dann_kwargs: dict[str, Any] | None = None,
        deepjdot_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout, **kwargs)
        self.base_align = str(base_align).strip().lower()
        if self.base_align not in {"cdan", "dann", "deepjdot"}:
            raise KeyError(f"Unsupported RCTA base_align: {base_align}")

        self.teacher_ema_decay = min(max(float(teacher_ema_decay), 0.0), 0.9999)
        self.teacher_temperature = max(float(teacher_temperature), _EPSILON)
        self.pseudo_label_weight = float(pseudo_label_weight)
        self.pseudo_warmup_steps = max(int(pseudo_warmup_steps), 0)
        self.pseudo_use_reliability_weighting = bool(pseudo_use_reliability_weighting)
        self.pseudo_confidence_power = max(float(pseudo_confidence_power), 0.0)
        self.prototype_weight = float(prototype_weight)
        self.prototype_start_step = max(int(prototype_start_step), 0)
        self.prototype_warmup_steps = (
            self.pseudo_warmup_steps if prototype_warmup_steps is None else max(int(prototype_warmup_steps), 0)
        )
        self.prototype_separation_weight = float(prototype_separation_weight)
        self.consistency_weight = float(consistency_weight)
        self.consistency_start_step = max(int(consistency_start_step), 0)
        self.consistency_warmup_steps = (
            self.pseudo_warmup_steps if consistency_warmup_steps is None else max(int(consistency_warmup_steps), 0)
        )
        self.consistency_gate_only = bool(consistency_gate_only)
        self.consistency_reliability_power = max(float(consistency_reliability_power), 0.0)
        self.alignment_start_step = max(int(alignment_start_step), 0)
        self.alignment_use_reliable_only = bool(alignment_use_reliable_only)
        self.prototype_momentum = min(max(float(prototype_momentum), 0.0), 0.9999)
        self.prototype_separation_margin = float(prototype_separation_margin)

        self.gate_score_floor_start = float(gate_score_floor if gate_score_floor_start is None else gate_score_floor_start)
        self.gate_score_floor_end = float(gate_score_floor if gate_score_floor_end is None else gate_score_floor_end)
        self.gate_score_floor_schedule_steps = max(int(gate_score_floor_schedule_steps), 1)
        self.reliable_score_floor = float(self.gate_score_floor_end if reliable_score_floor is None else reliable_score_floor)
        self.semi_reliable_score_floor = float(
            self.gate_score_floor_start if semi_reliable_score_floor is None else semi_reliable_score_floor
        )
        self.semi_reliable_consistency_weight = min(max(float(semi_reliable_consistency_weight), 0.0), 1.0)
        self.unreliable_entropy_weight = max(float(unreliable_entropy_weight), 0.0)

        weights = {"cal_conf": 1.0, "inv_entropy": 1.0, "consistency": 1.0, "proto_sim": 1.0}
        if isinstance(reliability_weights, dict):
            for key in weights:
                if key in reliability_weights:
                    weights[key] = max(float(reliability_weights[key]), 0.0)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            total_weight = float(len(weights))
            weights = {key: 1.0 for key in weights}
        self.reliability_weights = {key: value / total_weight for key, value in weights.items()}

        self.augmenter = _TemporalAugmenter(**(augment_kwargs or {}))
        self.gate = _ReliabilityGate(
            score_floor=gate_score_floor,
            accept_ratio_start=gate_accept_ratio_start,
            accept_ratio_end=gate_accept_ratio_end,
            curriculum_steps=gate_curriculum_steps,
        )
        self.reliability_partition = _ReliabilityPartition(
            reliable_floor=self.reliable_score_floor,
            semi_reliable_floor=self.semi_reliable_score_floor,
        )

        self.teacher_encoder = deepcopy(self.encoder)
        self.teacher_classifier = deepcopy(self.classifier)
        self._set_teacher_requires_grad(False)
        self.teacher_encoder.eval()
        self.teacher_classifier.eval()

        feature_dim = self.encoder.out_dim
        self.register_buffer("source_prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("target_prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("source_prototype_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("target_prototype_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("step_num", torch.zeros((), dtype=torch.long))

        if self.base_align == "cdan":
            self.aligner = _CDANAligner(feature_dim=feature_dim, num_classes=num_classes, dropout=dropout, **(cdan_kwargs or {}))
        elif self.base_align == "dann":
            self.aligner = _DANNAligner(feature_dim=feature_dim, dropout=dropout, **(dann_kwargs or {}))
        else:
            self.aligner = _DeepJDOTAligner(**(deepjdot_kwargs or {}))

        self.mcc_weight = float(mcc_weight) if use_mcc else 0.0
        self.mcc = MinimumClassConfusionLoss(mcc_temperature) if self.mcc_weight > 0 else None

        self._cached_source_features: torch.Tensor | None = None
        self._cached_source_labels: torch.Tensor | None = None
        self._cached_target_features: torch.Tensor | None = None
        self._cached_target_labels: torch.Tensor | None = None

    def _alignment_branch_metrics(self, align_metrics: dict[str, float]) -> dict[str, float]:
        return {key: float(value) for key, value in align_metrics.items() if key != "lambda_alignment"}

    def _alignment_weight(self, current_step: int, align_metrics: dict[str, float]) -> float:
        if current_step < self.alignment_start_step:
            return 0.0
        return float(align_metrics["lambda_alignment"])

    def _should_use_reliable_alignment(self, current_step: int, gate_mask: torch.Tensor) -> bool:
        return self.alignment_use_reliable_only and current_step >= self.alignment_start_step and gate_mask.any()

    def _select_alignment_inputs(
        self,
        *,
        current_step: int,
        gate_mask: torch.Tensor,
        features_target_align: torch.Tensor,
        logits_target_align: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._should_use_reliable_alignment(current_step, gate_mask):
            return features_target_align[gate_mask], logits_target_align[gate_mask]
        return features_target_align, logits_target_align

    def _compute_alignment_loss(
        self,
        *,
        current_step: int,
        source_y: torch.Tensor,
        logits_source: torch.Tensor,
        features_source: torch.Tensor,
        alignment_target_logits: torch.Tensor,
        alignment_target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if current_step < self.alignment_start_step:
            return _zero(logits_source.device, logits_source.dtype), {"lambda_alignment": 0.0}
        if self.base_align == "cdan":
            return self.aligner(
                logits_source=logits_source,
                features_source=features_source,
                logits_target=alignment_target_logits,
                features_target=alignment_target_features,
            )
        if self.base_align == "dann":
            return self.aligner(features_source=features_source, features_target=alignment_target_features)
        return self.aligner(
            source_labels=source_y,
            logits_target=alignment_target_logits,
            features_source=features_source,
            features_target=alignment_target_features,
        )

    def _compute_consistency_terms(
        self,
        *,
        student_probabilities: torch.Tensor,
        teacher_probabilities: torch.Tensor,
        reliability_score: torch.Tensor,
        gate_mask: torch.Tensor,
        semi_reliable_mask: torch.Tensor,
        unreliable_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        consistency_values = ((student_probabilities - teacher_probabilities) ** 2).mean(dim=1)
        if self.consistency_gate_only:
            consistency_weights = gate_mask.to(dtype=consistency_values.dtype)
        else:
            consistency_weights = reliability_score.detach()
        if self.consistency_reliability_power != 1.0:
            consistency_weights = consistency_weights.clamp_min(_EPSILON).pow(self.consistency_reliability_power)
        loss_consistency = self._weighted_mean(consistency_values, consistency_weights)
        semi_reliable_consistency = (
            self._weighted_mean(consistency_values[semi_reliable_mask], reliability_score[semi_reliable_mask].detach())
            if semi_reliable_mask.any()
            else _zero(student_probabilities.device, student_probabilities.dtype)
        )
        unreliable_entropy = (
            self._entropy_penalty(student_probabilities[unreliable_mask])
            if unreliable_mask.any()
            else _zero(student_probabilities.device, student_probabilities.dtype)
        )
        return loss_consistency, semi_reliable_consistency, unreliable_entropy

    def train(self, mode: bool = True) -> "RCTAMethod":
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
        return torch.softmax(logits / self.teacher_temperature, dim=1)

    def _current_reference_prototypes(self, source_batch_prototypes: torch.Tensor, source_batch_active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference = source_batch_prototypes.clone()
        active = source_batch_active.clone()
        source_active = self.source_prototype_counts > 0
        if source_active.any():
            reference[source_active] = self.source_prototypes[source_active]
            active = active | source_active
        target_active = self.target_prototype_counts > 0
        overlap = target_active & active
        if overlap.any():
            reference[overlap] = _normalize_rows(reference[overlap] + self.target_prototypes[overlap])
        target_only = target_active & ~active
        if target_only.any():
            reference[target_only] = self.target_prototypes[target_only]
            active = active | target_only
        if active.any():
            reference[active] = _normalize_rows(reference[active])
        return reference, active

    def _batch_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = torch.zeros_like(self.source_prototypes)
        active = torch.zeros_like(self.source_prototype_counts, dtype=torch.bool)
        for class_id in torch.unique(labels):
            class_index = int(class_id.item())
            class_features = features[labels == class_id]
            if class_features.numel() == 0:
                continue
            prototypes[class_index] = F.normalize(class_features.mean(dim=0, keepdim=True), dim=1, eps=_EPSILON)[0]
            active[class_index] = True
        return prototypes, active

    def _normalized_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        entropy = -(probabilities * torch.log(probabilities.clamp_min(_EPSILON))).sum(dim=1)
        return entropy / math.log(probabilities.shape[1])

    def _prototype_similarity(self, teacher_features: torch.Tensor, pseudo_labels: torch.Tensor, reference_prototypes: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        similarity = torch.full((teacher_features.shape[0],), 0.5, device=teacher_features.device, dtype=teacher_features.dtype)
        valid = active_mask[pseudo_labels]
        if valid.any():
            prototype_slice = reference_prototypes[pseudo_labels[valid]].to(dtype=teacher_features.dtype)
            cosine = F.cosine_similarity(teacher_features[valid], prototype_slice, dim=1, eps=_EPSILON)
            similarity[valid] = ((cosine + 1.0) * 0.5).to(dtype=teacher_features.dtype)
        return similarity

    def _prototype_distance(self, features: torch.Tensor, labels: torch.Tensor, reference_prototypes: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        if features.numel() == 0:
            return _zero(reference_prototypes.device, reference_prototypes.dtype)
        if active_mask is not None:
            valid = active_mask[labels]
            if not valid.any():
                return _zero(reference_prototypes.device, reference_prototypes.dtype)
            features = features[valid]
            labels = labels[valid]
        prototype_slice = reference_prototypes[labels].detach().to(dtype=features.dtype)
        distances = 1.0 - F.cosine_similarity(features, prototype_slice, dim=1, eps=_EPSILON)
        return distances.mean()

    def _entropy_penalty(self, probabilities: torch.Tensor) -> torch.Tensor:
        entropy = self._normalized_entropy(probabilities)
        return entropy.mean()

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        total_weight = weights.sum().clamp_min(_EPSILON)
        return (values * weights).sum() / total_weight

    def _scheduled_gate_score_floor(self, step_num: int) -> float:
        progress = min(max(float(step_num), 0.0) / float(self.gate_score_floor_schedule_steps), 1.0)
        return self.gate_score_floor_start + (self.gate_score_floor_end - self.gate_score_floor_start) * progress

    def _ramp_weight(self, base_weight: float, step_num: int, start_step: int, warmup_steps: int) -> float:
        if base_weight <= 0:
            return 0.0
        if step_num < start_step:
            return 0.0
        if warmup_steps <= 0:
            return base_weight
        progress = min(max(float(step_num - start_step), 0.0) / float(warmup_steps), 1.0)
        return base_weight * progress

    def _prototype_attraction_loss(self, features: torch.Tensor, labels: torch.Tensor, reference_prototypes: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        if features.numel() == 0:
            return _zero(reference_prototypes.device, reference_prototypes.dtype)
        if active_mask is not None:
            valid = active_mask[labels]
            if not valid.any():
                return _zero(reference_prototypes.device, reference_prototypes.dtype)
            features = features[valid]
            labels = labels[valid]
        prototypes = reference_prototypes[labels].detach().to(dtype=features.dtype)
        cosine = F.cosine_similarity(features, prototypes, dim=1, eps=_EPSILON)
        return (1.0 - cosine).mean()

    def _prototype_separation_loss(self, reference_prototypes: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        active_prototypes = reference_prototypes[active_mask]
        if active_prototypes.shape[0] < 2:
            return _zero(reference_prototypes.device, reference_prototypes.dtype)
        normalized = _normalize_rows(active_prototypes)
        similarity = normalized @ normalized.t()
        penalty = torch.relu(similarity - self.prototype_separation_margin)
        upper = torch.triu(penalty, diagonal=1)
        valid = upper[upper > 0]
        if valid.numel() == 0:
            return _zero(reference_prototypes.device, reference_prototypes.dtype)
        return valid.mean()

    def _update_prototype_bank(self, features: torch.Tensor | None, labels: torch.Tensor | None, *, prototypes: torch.Tensor, counts: torch.Tensor) -> None:
        if features is None or labels is None or features.numel() == 0 or labels.numel() == 0:
            return
        for class_id in torch.unique(labels):
            class_index = int(class_id.item())
            class_features = features[labels == class_id]
            if class_features.numel() == 0:
                continue
            centroid = F.normalize(class_features.mean(dim=0, keepdim=True), dim=1, eps=_EPSILON)[0]
            if int(counts[class_index].item()) == 0:
                prototypes[class_index] = centroid
            else:
                prototypes[class_index] = F.normalize(self.prototype_momentum * prototypes[class_index] + (1.0 - self.prototype_momentum) * centroid, dim=0, eps=_EPSILON)
            counts[class_index] += 1

    def _ema_update_teacher(self) -> None:
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
                teacher_parameter.data.mul_(self.teacher_ema_decay).add_(student_parameter.data, alpha=1.0 - self.teacher_ema_decay)
            for teacher_parameter, student_parameter in zip(self.teacher_classifier.parameters(), self.classifier.parameters()):
                teacher_parameter.data.mul_(self.teacher_ema_decay).add_(student_parameter.data, alpha=1.0 - self.teacher_ema_decay)
            for teacher_buffer, student_buffer in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
                teacher_buffer.copy_(student_buffer)
            for teacher_buffer, student_buffer in zip(self.teacher_classifier.buffers(), self.classifier.buffers()):
                teacher_buffer.copy_(student_buffer)

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        weak_target, strong_target = self.augmenter.augment_pair(target_x)
        logits_target_align, features_target_align = self.forward(weak_target)
        logits_target_strong, _ = self.forward(strong_target)
        teacher_logits, teacher_features = self._teacher_forward(weak_target)
        teacher_probabilities = self._teacher_probabilities(teacher_logits).detach()
        pseudo_labels = teacher_probabilities.argmax(dim=1)
        source_batch_prototypes, source_batch_active = self._batch_prototypes(features_source.detach(), source_y)
        reference_prototypes, prototype_active = self._current_reference_prototypes(source_batch_prototypes, source_batch_active)
        student_probabilities = torch.softmax(logits_target_strong, dim=1)
        cal_conf = teacher_probabilities.max(dim=1).values.to(dtype=logits_source.dtype)
        inv_entropy = (1.0 - self._normalized_entropy(teacher_probabilities)).to(dtype=logits_source.dtype)
        consistency_score = student_probabilities.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)
        proto_similarity = self._prototype_similarity(teacher_features.detach(), pseudo_labels, reference_prototypes.detach(), prototype_active).to(dtype=logits_source.dtype)
        reliability_score = (self.reliability_weights["cal_conf"] * cal_conf + self.reliability_weights["inv_entropy"] * inv_entropy + self.reliability_weights["consistency"] * consistency_score.detach() + self.reliability_weights["proto_sim"] * proto_similarity).clamp(0.0, 1.0)
        if self.pseudo_confidence_power != 1.0:
            reliability_score = reliability_score.clamp_min(_EPSILON).pow(self.pseudo_confidence_power).clamp(0.0, 1.0)
        current_step = int(self.step_num.item())
        scheduled_gate_floor = self._scheduled_gate_score_floor(current_step)
        gate_mask, curriculum_ratio = self.gate.select(reliability_score.detach(), pseudo_labels.detach(), current_step, score_floor_override=scheduled_gate_floor)
        loss_cls = F.cross_entropy(logits_source, source_y)
        alignment_target_features, alignment_target_logits = self._select_alignment_inputs(
            current_step=current_step,
            gate_mask=gate_mask,
            features_target_align=features_target_align,
            logits_target_align=logits_target_align,
        )
        loss_alignment, align_metrics = self._compute_alignment_loss(
            current_step=current_step,
            source_y=source_y,
            logits_source=logits_source,
            features_source=features_source,
            alignment_target_logits=alignment_target_logits,
            alignment_target_features=alignment_target_features,
        )
        loss_mcc = self.mcc(alignment_target_logits) if self.mcc is not None and alignment_target_logits.numel() > 0 else _zero(logits_target_align.device, logits_target_align.dtype)
        if gate_mask.any():
            gated_logits_target_strong = logits_target_strong[gate_mask]
            gated_pseudo_labels = pseudo_labels[gate_mask]
            if self.pseudo_use_reliability_weighting:
                pseudo_ce_values = F.cross_entropy(gated_logits_target_strong, gated_pseudo_labels, reduction="none")
                pseudo_weights = reliability_score[gate_mask].detach()
                loss_pseudo = self._weighted_mean(pseudo_ce_values, pseudo_weights)
            else:
                pseudo_ce_values = F.cross_entropy(gated_logits_target_strong, gated_pseudo_labels, reduction="none")
                loss_pseudo = pseudo_ce_values.mean()
            gated_target_proto_features = features_target_align[gate_mask]
        else:
            loss_pseudo = _zero(logits_target_align.device, logits_target_align.dtype)
            gated_pseudo_labels = pseudo_labels[:0]
            gated_target_proto_features = features_target_align[:0]
        loss_source_proto = self._prototype_attraction_loss(features_source, source_y, reference_prototypes, prototype_active)
        reliable_mask, semi_reliable_mask, unreliable_mask = self.reliability_partition.split(
            reliability_score.detach(), gate_mask
        )
        loss_target_proto = self._prototype_attraction_loss(
            gated_target_proto_features, gated_pseudo_labels, reference_prototypes, prototype_active
        )
        loss_proto_separation = self._prototype_separation_loss(reference_prototypes, prototype_active)
        soft_target_weights = self.reliability_partition.soft_weights(reliability_score.detach())
        loss_soft_target_proto = self._weighted_mean(
            1.0 - F.cosine_similarity(
                features_target_align,
                reference_prototypes[pseudo_labels].detach(),
                dim=1,
                eps=_EPSILON,
            ),
            soft_target_weights,
        )
        loss_prototype = loss_source_proto + loss_target_proto + 0.5 * loss_soft_target_proto + self.prototype_separation_weight * loss_proto_separation
        loss_consistency, semi_reliable_consistency, unreliable_entropy = self._compute_consistency_terms(
            student_probabilities=student_probabilities,
            teacher_probabilities=teacher_probabilities,
            reliability_score=reliability_score,
            gate_mask=gate_mask,
            semi_reliable_mask=semi_reliable_mask,
            unreliable_mask=unreliable_mask,
        )
        lambda_pseudo = self._ramp_weight(self.pseudo_label_weight, current_step, 0, self.pseudo_warmup_steps)
        lambda_prototype = self._ramp_weight(self.prototype_weight, current_step, self.prototype_start_step, self.prototype_warmup_steps)
        lambda_consistency = self._ramp_weight(self.consistency_weight, current_step, self.consistency_start_step, self.consistency_warmup_steps)
        lambda_alignment = self._alignment_weight(current_step, align_metrics)
        total_loss = loss_cls + lambda_alignment * loss_alignment + self.mcc_weight * loss_mcc + lambda_pseudo * loss_pseudo + lambda_prototype * loss_prototype + lambda_consistency * loss_consistency + self.semi_reliable_consistency_weight * semi_reliable_consistency + self.unreliable_entropy_weight * unreliable_entropy
        self._cached_source_features = features_source.detach().clone()
        self._cached_source_labels = source_y.detach().clone()
        self._cached_target_features = teacher_features.detach().clone()
        self._cached_target_labels = pseudo_labels.detach().clone()
        kept_count = int(gate_mask.sum().item())
        batch_size_target = max(int(target_x.shape[0]), 1)
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_alignment": float(loss_alignment.item()),
            "loss_mcc": float(loss_mcc.item()),
            "loss_pseudo": float(loss_pseudo.item()),
            "loss_prototype": float(loss_prototype.item()),
            "loss_soft_target_proto": float(loss_soft_target_proto.item()),
            "loss_consistency": float(loss_consistency.item()),
            "loss_semi_reliable_consistency": float(semi_reliable_consistency.item()),
            "loss_unreliable_entropy": float(unreliable_entropy.item()),
            "loss_prototype_separation": float(loss_proto_separation.item()),
            "lambda_alignment": float(lambda_alignment),
            "lambda_mcc": float(self.mcc_weight),
            "lambda_pseudo": float(lambda_pseudo),
            "lambda_prototype": float(lambda_prototype),
            "lambda_consistency": float(lambda_consistency),
            "acc_source": accuracy_from_logits(logits_source, source_y),
            "gate_mean_score": float(reliability_score.mean().item()),
            "pseudo_kept_mean_reliability": float(reliability_score[gate_mask].mean().item()) if gate_mask.any() else 0.0,
            "gate_accept_ratio": float(kept_count / batch_size_target),
            "gate_curriculum_ratio": float(curriculum_ratio),
            "gate_score_floor": float(scheduled_gate_floor),
            "pseudo_label_kept": float(kept_count),
            "reliable_count": float(reliable_mask.sum().item()),
            "semi_reliable_count": float(semi_reliable_mask.sum().item()),
            "unreliable_count": float(unreliable_mask.sum().item()),
        }
        metrics.update({key: float(value) for key, value in self._alignment_branch_metrics(align_metrics).items() if key != "lambda_alignment"})
        return MethodStepOutput(loss=total_loss, metrics=metrics)

    def after_optimizer_step(self) -> dict[str, float]:
        self._ema_update_teacher()
        self._update_prototype_bank(self._cached_source_features, self._cached_source_labels, prototypes=self.source_prototypes, counts=self.source_prototype_counts)
        self._update_prototype_bank(self._cached_target_features, self._cached_target_labels, prototypes=self.target_prototypes, counts=self.target_prototype_counts)
        self.step_num += 1
        self._cached_source_features = None
        self._cached_source_labels = None
        self._cached_target_features = None
        self._cached_target_labels = None
        return {
            "teacher_ema_decay": float(self.teacher_ema_decay),
            "source_prototype_active_classes": float((self.source_prototype_counts > 0).sum().item()),
            "target_prototype_active_classes": float((self.target_prototype_counts > 0).sum().item()),
        }
