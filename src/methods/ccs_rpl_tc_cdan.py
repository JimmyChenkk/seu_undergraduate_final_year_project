"""Class-conditional structure preserving RPL-TC-CDAN."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import MethodStepOutput
from .rpl_tc_cdan import RPLTCCDANMethod
from .tc_cdan import _EPSILON, _zero_like_loss


class CCSRPLTCCDANMethod(RPLTCCDANMethod):
    """CCS-RPL-TC-CDAN with EMA source/target class prototypes."""

    method_name = "ccs_rpl_tc_cdan"

    def __init__(
        self,
        *,
        prototype_weight: float = 0.05,
        prototype_start_step: int = 1200,
        prototype_warmup_steps: int = 800,
        prototype_momentum: float = 0.95,
        prototype_min_target_per_class: int = 1,
        target_prototype_blend: float = 0.35,
        class_separation_weight: float = 0.04,
        class_separation_margin: float = 0.20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prototype_weight = float(prototype_weight)
        self.prototype_start_step = max(int(prototype_start_step), 0)
        self.prototype_warmup_steps = max(int(prototype_warmup_steps), 0)
        self.prototype_momentum = min(max(float(prototype_momentum), 0.0), 0.9999)
        self.prototype_min_target_per_class = max(int(prototype_min_target_per_class), 1)
        self.target_prototype_blend = min(max(float(target_prototype_blend), 0.0), 1.0)
        self.class_separation_weight = float(class_separation_weight)
        self.class_separation_margin = float(class_separation_margin)

        feature_dim = int(self.encoder.out_dim)
        self.register_buffer("source_prototypes", torch.zeros(self.num_classes, feature_dim))
        self.register_buffer("target_prototypes", torch.zeros(self.num_classes, feature_dim))
        self.register_buffer("source_prototype_counts", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("target_prototype_counts", torch.zeros(self.num_classes, dtype=torch.long))
        self._cached_source_features: torch.Tensor | None = None
        self._cached_source_labels: torch.Tensor | None = None
        self._cached_target_features: torch.Tensor | None = None
        self._cached_target_labels: torch.Tensor | None = None

    def _batch_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = features.float()
        prototypes = torch.zeros_like(self.source_prototypes, dtype=torch.float32, device=features.device)
        active = torch.zeros(self.num_classes, dtype=torch.bool, device=features.device)
        for class_id in torch.unique(labels.detach().long()):
            class_index = int(class_id.item())
            if class_index < 0 or class_index >= self.num_classes:
                continue
            class_features = features[labels == class_id]
            if class_features.numel() == 0:
                continue
            prototypes[class_index] = F.normalize(class_features.mean(dim=0), dim=0, eps=_EPSILON)
            active[class_index] = True
        return prototypes, active

    def _reference_prototypes(
        self,
        features_source: torch.Tensor,
        source_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_device = features_source.device
        batch_prototypes, batch_active = self._batch_prototypes(features_source.detach().float(), source_y)
        reference = batch_prototypes.clone()
        active = batch_active.clone()

        source_active = (self.source_prototype_counts > 0).to(device=feature_device)
        if source_active.any():
            reference[source_active] = self.source_prototypes[source_active].to(
                device=feature_device,
                dtype=reference.dtype,
            )
            active = active | source_active

        target_active = (self.target_prototype_counts > 0).to(device=feature_device)
        if self.target_prototype_blend > 0 and target_active.any():
            target_bank = self.target_prototypes.to(device=feature_device, dtype=reference.dtype)
            overlap = active & target_active
            if overlap.any():
                reference[overlap] = F.normalize(
                    (1.0 - self.target_prototype_blend) * reference[overlap]
                    + self.target_prototype_blend * target_bank[overlap],
                    dim=1,
                    eps=_EPSILON,
                )
            target_only = target_active & ~active
            if target_only.any():
                reference[target_only] = target_bank[target_only]
                active = active | target_only

        if active.any():
            reference[active] = F.normalize(reference[active], dim=1, eps=_EPSILON)
        return reference, active

    def _prototype_attraction_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        reference: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if features.numel() == 0 or labels.numel() == 0:
            return _zero_like_loss(reference)
        valid = active[labels]
        if not valid.any():
            return _zero_like_loss(reference)
        features = features[valid]
        labels = labels[valid]
        prototypes = reference[labels].detach().to(device=features.device, dtype=features.dtype)
        return (1.0 - F.cosine_similarity(features, prototypes, dim=1, eps=_EPSILON)).mean()

    def _class_separation_loss(self, reference: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        active_reference = reference[active]
        if active_reference.shape[0] < 2:
            return _zero_like_loss(reference)
        normalized = F.normalize(active_reference, dim=1, eps=_EPSILON)
        similarity = normalized @ normalized.t()
        penalty = torch.relu(similarity - self.class_separation_margin)
        upper = torch.triu(penalty, diagonal=1)
        valid = upper[upper > 0]
        if valid.numel() == 0:
            return _zero_like_loss(reference)
        return valid.mean()

    def _inter_class_distance(self, reference: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        active_reference = reference[active]
        if active_reference.shape[0] < 2:
            return _zero_like_loss(reference)
        normalized = F.normalize(active_reference, dim=1, eps=_EPSILON)
        distance = 1.0 - normalized @ normalized.t()
        upper = torch.triu(distance, diagonal=1)
        valid = upper[upper > 0]
        if valid.numel() == 0:
            return _zero_like_loss(reference)
        return valid.mean()

    def _reliable_target_histogram(self, labels: torch.Tensor) -> dict[str, float]:
        counts = torch.bincount(labels.detach().long(), minlength=self.num_classes)
        return {
            f"reliable_target_per_class_{class_index:02d}": float(counts[class_index].item())
            for class_index in range(self.num_classes)
        }

    def _compute_ccs_terms(self, components: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        pseudo_state = components["pseudo_state"]
        selected_mask = pseudo_state["selected_mask"]
        pseudo_labels = pseudo_state["pseudo_labels"]
        features_target = components["features_target_weak"]
        reference, active = self._reference_prototypes(components["features_source"], components["source_y"])
        current_step = int(components["current_step"])

        if selected_mask.any():
            selected_features = features_target[selected_mask]
            selected_labels = pseudo_labels[selected_mask]
        else:
            selected_features = features_target[:0]
            selected_labels = pseudo_labels[:0]

        loss_prototype = self._prototype_attraction_loss(selected_features, selected_labels, reference, active)
        loss_separation = self._class_separation_loss(reference, active)
        lambda_proto = self._ramp_weight(
            self.prototype_weight,
            current_step,
            self.prototype_start_step,
            self.prototype_warmup_steps,
        )
        lambda_sep = self._ramp_weight(
            self.class_separation_weight,
            current_step,
            self.prototype_start_step,
            self.prototype_warmup_steps,
        )

        inter_distance = self._inter_class_distance(reference, active)
        ratio = float((inter_distance / (loss_prototype.detach() + _EPSILON)).item()) if inter_distance.numel() else 0.0
        update_labels = selected_labels.detach()
        metrics = {
            "loss_prototype": float(loss_prototype.item()),
            "prototype_loss": float(loss_prototype.item()),
            "loss_class_separation": float(loss_separation.item()),
            "class_separation_loss": float(loss_separation.item()),
            "lambda_proto": float(lambda_proto),
            "lambda_sep": float(lambda_sep),
            "class_inter_intra_ratio": ratio,
            "prototype_update_count": float(update_labels.numel()),
            "source_prototype_active_classes": float((self.source_prototype_counts > 0).sum().item()),
            "target_prototype_active_classes": float((self.target_prototype_counts > 0).sum().item()),
        }
        metrics.update(self._reliable_target_histogram(update_labels))

        self._cached_source_features = components["features_source"].detach().clone()
        self._cached_source_labels = components["source_y"].detach().clone()
        self._cached_target_features = selected_features.detach().clone()
        self._cached_target_labels = update_labels.clone()
        return lambda_proto * loss_prototype, lambda_sep * loss_separation, metrics

    def _update_bank(
        self,
        features: torch.Tensor | None,
        labels: torch.Tensor | None,
        *,
        prototypes: torch.Tensor,
        counts: torch.Tensor,
        min_per_class: int = 1,
    ) -> int:
        if features is None or labels is None or features.numel() == 0 or labels.numel() == 0:
            return 0
        features = features.float()
        update_count = 0
        for class_id in torch.unique(labels.detach().long()):
            class_index = int(class_id.item())
            if class_index < 0 or class_index >= self.num_classes:
                continue
            class_features = features[labels == class_id]
            if int(class_features.shape[0]) < min_per_class:
                continue
            centroid = F.normalize(class_features.mean(dim=0), dim=0, eps=_EPSILON).to(
                device=prototypes.device,
                dtype=prototypes.dtype,
            )
            if int(counts[class_index].item()) == 0:
                prototypes[class_index] = centroid
            else:
                prototypes[class_index] = F.normalize(
                    self.prototype_momentum * prototypes[class_index]
                    + (1.0 - self.prototype_momentum) * centroid,
                    dim=0,
                    eps=_EPSILON,
                )
            counts[class_index] += 1
            update_count += int(class_features.shape[0])
        return update_count

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        components = self._compute_rpl_components(source_batches, target_batch)
        weighted_proto, weighted_sep, ccs_metrics = self._compute_ccs_terms(components)
        total_loss = components["rpl_loss"] + weighted_proto + weighted_sep
        metrics = dict(components["metrics"])
        metrics.update(ccs_metrics)
        metrics["loss_total"] = float(total_loss.item())
        return MethodStepOutput(loss=total_loss, metrics=metrics)

    def after_optimizer_step(self) -> dict[str, float]:
        self._ema_update_teacher()
        source_updates = self._update_bank(
            self._cached_source_features,
            self._cached_source_labels,
            prototypes=self.source_prototypes,
            counts=self.source_prototype_counts,
            min_per_class=1,
        )
        target_updates = self._update_bank(
            self._cached_target_features,
            self._cached_target_labels,
            prototypes=self.target_prototypes,
            counts=self.target_prototype_counts,
            min_per_class=self.prototype_min_target_per_class,
        )
        self.step_num += 1
        self._cached_source_features = None
        self._cached_source_labels = None
        self._cached_target_features = None
        self._cached_target_labels = None
        return {
            "teacher_ema_decay": float(self.teacher_ema_decay),
            "prototype_source_update_count": float(source_updates),
            "prototype_target_update_count": float(target_updates),
            "source_prototype_active_classes": float((self.source_prototype_counts > 0).sum().item()),
            "target_prototype_active_classes": float((self.target_prototype_counts > 0).sum().item()),
        }
