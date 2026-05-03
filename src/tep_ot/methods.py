"""Training objectives for the minimal TEP DA experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from .augment import augment_signal
from .model import TEClassifier
from .ot_losses import (
    OTLossConfig,
    compute_class_prototypes,
    coral_loss,
    inverse_sqrt_class_weights,
    jdot_transport_loss,
    mmd_loss,
    source_outlier_weights,
    target_pseudo_class_weights,
)


@dataclass
class MethodOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().detach().item())


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels.long(), reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coeff * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class BaseMethod(nn.Module):
    method_name = "base"
    is_ot_family = False

    def __init__(self, model: TEClassifier, *, num_classes: int = 29) -> None:
        super().__init__()
        self.model = model
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict_logits(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(x)

    def merge_source_batches(self, source_batches: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(source_batches) == 1:
            return source_batches[0]
        return (
            torch.cat([batch[0] for batch in source_batches], dim=0),
            torch.cat([batch[1] for batch in source_batches], dim=0),
        )

    def source_supervised_loss(self, source_batches: list[tuple[torch.Tensor, torch.Tensor]]) -> MethodOutput:
        x_source, y_source = self.merge_source_batches(source_batches)
        logits_source, _ = self.forward(x_source)
        loss = F.cross_entropy(logits_source, y_source.long())
        return MethodOutput(
            loss=loss,
            metrics={
                "loss_total": float(loss.detach().item()),
                "loss_cls": float(loss.detach().item()),
                "acc_source": accuracy_from_logits(logits_source, y_source),
            },
        )

    def compute_loss(
        self,
        source_batches: list[tuple[torch.Tensor, torch.Tensor]],
        target_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> MethodOutput:
        raise NotImplementedError

    def reliability_snapshot(self) -> dict[str, Any]:
        return {}


class SourceOnlyMethod(BaseMethod):
    method_name = "source_only"

    def compute_loss(self, source_batches, target_batch) -> MethodOutput:
        del target_batch
        return self.source_supervised_loss(source_batches)


class TargetOnlyMethod(BaseMethod):
    method_name = "target_only"

    def compute_loss(self, source_batches, target_batch) -> MethodOutput:
        del source_batches
        target_x, target_y = target_batch
        if bool((target_y < 0).any()):
            raise RuntimeError("Target-only requires labeled target-train data; hidden labels were supplied.")
        logits_target, _ = self.forward(target_x)
        loss = F.cross_entropy(logits_target, target_y.long())
        return MethodOutput(
            loss=loss,
            metrics={
                "loss_total": float(loss.detach().item()),
                "loss_cls": float(loss.detach().item()),
                "acc_source": accuracy_from_logits(logits_target, target_y),
            },
        )


class MMDMethod(BaseMethod):
    method_name = "mmd"

    def __init__(self, model: TEClassifier, *, adaptation_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.adaptation_weight = float(adaptation_weight)

    def compute_loss(self, source_batches, target_batch) -> MethodOutput:
        x_source, y_source = self.merge_source_batches(source_batches)
        x_target, _ = target_batch
        logits_source, features_source = self.forward(x_source)
        _, features_target = self.forward(x_target)
        loss_cls = F.cross_entropy(logits_source, y_source.long())
        loss_align = mmd_loss(features_source, features_target)
        loss = loss_cls + self.adaptation_weight * loss_align
        return MethodOutput(
            loss=loss,
            metrics={
                "loss_total": float(loss.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_alignment": float(loss_align.detach().item()),
                "acc_source": accuracy_from_logits(logits_source, y_source),
            },
        )


class CORALMethod(MMDMethod):
    method_name = "coral"

    def compute_loss(self, source_batches, target_batch) -> MethodOutput:
        x_source, y_source = self.merge_source_batches(source_batches)
        x_target, _ = target_batch
        logits_source, features_source = self.forward(x_source)
        _, features_target = self.forward(x_target)
        loss_cls = F.cross_entropy(logits_source, y_source.long())
        loss_align = coral_loss(features_source, features_target)
        loss = loss_cls + self.adaptation_weight * loss_align
        return MethodOutput(
            loss=loss,
            metrics={
                "loss_total": float(loss.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_alignment": float(loss_align.detach().item()),
                "acc_source": accuracy_from_logits(logits_source, y_source),
            },
        )


class DANNMethod(BaseMethod):
    method_name = "dann"

    def __init__(
        self,
        model: TEClassifier,
        *,
        adaptation_weight: float = 0.5,
        grl_coeff: float = 1.0,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.adaptation_weight = float(adaptation_weight)
        self.grl_coeff = float(grl_coeff)
        self.discriminator = DomainDiscriminator(self.model.encoder.out_dim, dropout=dropout)

    def compute_loss(self, source_batches, target_batch) -> MethodOutput:
        x_source, y_source = self.merge_source_batches(source_batches)
        x_target, _ = target_batch
        logits_source, features_source = self.forward(x_source)
        _, features_target = self.forward(x_target)
        loss_cls = F.cross_entropy(logits_source, y_source.long())

        features = torch.cat([features_source, features_target], dim=0)
        domain_logits = self.discriminator(GradientReverse.apply(features, self.grl_coeff))
        domain_labels = torch.cat(
            [
                torch.ones(features_source.shape[0], device=features.device),
                torch.zeros(features_target.shape[0], device=features.device),
            ]
        )
        loss_domain = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
        domain_acc = float(((torch.sigmoid(domain_logits) >= 0.5).float() == domain_labels).float().mean().item())
        loss = loss_cls + self.adaptation_weight * loss_domain
        return MethodOutput(
            loss=loss,
            metrics={
                "loss_total": float(loss.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_alignment": float(loss_domain.detach().item()),
                "acc_source": accuracy_from_logits(logits_source, y_source),
                "acc_domain": domain_acc,
            },
        )


class JDOTMethod(BaseMethod):
    method_name = "jdot"
    is_ot_family = True

    def __init__(
        self,
        model: TEClassifier,
        *,
        adaptation_weight: float = 1.0,
        source_ce_weight: float = 1.0,
        ot_config: OTLossConfig | None = None,
        use_source_class_balance: bool = False,
        use_temporal_prototypes: bool = False,
        confidence_curriculum: bool = False,
        pseudo_weight: float = 0.15,
        consistency_weight: float = 0.05,
        tau_start: float = 0.95,
        tau_end: float = 0.70,
        tau_steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.adaptation_weight = float(adaptation_weight)
        self.source_ce_weight = float(source_ce_weight)
        self.ot_config = ot_config or OTLossConfig()
        self.use_source_class_balance = bool(use_source_class_balance)
        self.use_temporal_prototypes = bool(use_temporal_prototypes)
        self.confidence_curriculum = bool(confidence_curriculum)
        self.pseudo_weight = float(pseudo_weight)
        self.consistency_weight = float(consistency_weight)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.tau_steps = max(int(tau_steps), 1)
        self.global_step = 0

    def current_tau(self) -> float:
        progress = min(self.global_step / float(self.tau_steps), 1.0)
        return self.tau_start + progress * (self.tau_end - self.tau_start)

    def _target_regularizers(self, target_x: torch.Tensor, target_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        probs = F.softmax(target_logits.detach().float(), dim=1)
        confidence, pseudo = probs.max(dim=1)
        tau = self.current_tau()
        mask = confidence > tau
        zero = target_logits.new_zeros(())
        metrics = {
            "tau": float(tau),
            "pseudo_acceptance": float(mask.float().mean().detach().item()),
            "target_confidence": float(confidence.mean().detach().item()),
        }
        if not self.confidence_curriculum:
            return zero, metrics, torch.ones_like(confidence)

        pseudo_loss = zero
        if bool(mask.any()):
            class_weights = target_pseudo_class_weights(pseudo, mask, num_classes=self.num_classes).to(
                device=target_logits.device,
                dtype=target_logits.dtype,
            )
            pseudo_loss = F.cross_entropy(target_logits[mask], pseudo[mask], weight=class_weights)

        augmented = augment_signal(target_x)
        logits_aug, _ = self.forward(augmented)
        consistency = F.kl_div(
            F.log_softmax(logits_aug.float(), dim=1),
            probs,
            reduction="none",
        ).sum(dim=1)
        consistency_loss = (consistency * confidence).mean()
        loss = self.pseudo_weight * pseudo_loss + self.consistency_weight * consistency_loss
        metrics.update(
            {
                "loss_pseudo": float(pseudo_loss.detach().item()),
                "loss_consistency": float(consistency_loss.detach().item()),
            }
        )
        return loss, metrics, mask.to(dtype=target_logits.dtype)

    def compute_loss(self, source_batches, target_batch) -> MethodOutput:
        self.global_step += 1
        x_source, y_source = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(x_source)
        logits_target, features_target = self.forward(target_x)

        class_weights = inverse_sqrt_class_weights(y_source, num_classes=self.num_classes).to(
            device=y_source.device,
            dtype=features_source.dtype,
        )
        source_sample_weights = class_weights[y_source.long()] if self.use_source_class_balance else None
        loss_cls = weighted_ce(logits_source, y_source, source_sample_weights)

        prototypes = None
        if self.use_temporal_prototypes or self.ot_config.prototype_weight > 0:
            prototypes, _ = compute_class_prototypes(
                features_source,
                y_source,
                num_classes=self.num_classes,
                sample_weights=source_sample_weights,
            )

        target_reg, target_metrics, label_gate = self._target_regularizers(target_x, logits_target)
        if not self.confidence_curriculum:
            label_gate = None
        loss_ot, ot_metrics = jdot_transport_loss(
            source_features=features_source,
            source_labels=y_source,
            target_features=features_target,
            target_logits=logits_target,
            num_classes=self.num_classes,
            config=self.ot_config,
            source_sample_weights=source_sample_weights,
            source_prototypes=prototypes,
            label_gate=label_gate,
        )
        loss = self.source_ce_weight * loss_cls + self.adaptation_weight * loss_ot + target_reg
        metrics = {
            "loss_total": float(loss.detach().item()),
            "loss_cls": float(loss_cls.detach().item()),
            "loss_alignment": float(loss_ot.detach().item()),
            "acc_source": accuracy_from_logits(logits_source, y_source),
        }
        metrics.update(ot_metrics)
        metrics.update(target_metrics)
        return MethodOutput(loss=loss, metrics=metrics)


class WJDOTMethod(JDOTMethod):
    method_name = "wjdot"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_source_class_balance"] = True
        super().__init__(*args, **kwargs)


class TPJDOTMethod(JDOTMethod):
    method_name = "tp_jdot"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_temporal_prototypes"] = True
        kwargs.setdefault("use_source_class_balance", False)
        super().__init__(*args, **kwargs)


class CBTPJDOTMethod(TPJDOTMethod):
    method_name = "cbtp_jdot"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["confidence_curriculum"] = True
        super().__init__(*args, **kwargs)


class TPWJDOTMethod(WJDOTMethod):
    method_name = "tp_wjdot"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_temporal_prototypes"] = True
        super().__init__(*args, **kwargs)


class CBTPWJDOTMethod(TPWJDOTMethod):
    method_name = "cbtp_wjdot"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["confidence_curriculum"] = True
        super().__init__(*args, **kwargs)


class MSCBTPWJDOTMethod(CBTPWJDOTMethod):
    method_name = "ms_cbtp_wjdot"

    def __init__(
        self,
        *args: Any,
        source_weight_temperature: float = 1.0,
        class_weight_temperature: float = 1.0,
        negative_gate_threshold: float = 0.05,
        negative_gate_floor: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_weight_temperature = max(float(source_weight_temperature), 1e-6)
        self.class_weight_temperature = max(float(class_weight_temperature), 1e-6)
        self.negative_gate_threshold = float(negative_gate_threshold)
        self.negative_gate_floor = float(negative_gate_floor)
        self._last_source_weights: np.ndarray | None = None
        self._last_class_source_weights: np.ndarray | None = None

    def _source_alpha(self, source_features: list[torch.Tensor], target_features: torch.Tensor) -> torch.Tensor:
        scores = []
        for features in source_features:
            distance = torch.cdist(features.detach(), target_features.detach(), p=2).pow(2).mean()
            scores.append(distance)
        score_tensor = torch.stack(scores)
        return F.softmax(-score_tensor / self.source_weight_temperature, dim=0).detach()

    def _class_alpha(
        self,
        source_prototypes: list[torch.Tensor],
        source_present: list[torch.Tensor],
        target_features: torch.Tensor,
        target_logits: torch.Tensor,
        source_alpha: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(target_logits.detach().float(), dim=1)
        confidence, pseudo = probs.max(dim=1)
        mask = confidence > self.current_tau()
        k_sources = len(source_prototypes)
        if bool(mask.any()):
            target_prototypes, target_present = compute_class_prototypes(
                target_features[mask].detach(),
                pseudo[mask],
                num_classes=self.num_classes,
                sample_weights=confidence[mask],
            )
        else:
            target_prototypes = target_features.new_zeros((self.num_classes, target_features.shape[1]))
            target_present = torch.zeros(self.num_classes, device=target_features.device, dtype=torch.bool)

        class_weights = target_features.new_zeros((k_sources, self.num_classes))
        for class_id in range(self.num_classes):
            valid_sources = [
                source_index
                for source_index in range(k_sources)
                if bool(source_present[source_index][class_id]) and bool(target_present[class_id])
            ]
            if not valid_sources:
                class_weights[:, class_id] = source_alpha.to(device=target_features.device, dtype=target_features.dtype)
                continue
            distances = []
            for source_index in valid_sources:
                gap = source_prototypes[source_index][class_id] - target_prototypes[class_id]
                distances.append(gap.pow(2).sum())
            values = torch.stack(distances)
            alpha_valid = F.softmax(-values / self.class_weight_temperature, dim=0)
            for local_index, source_index in enumerate(valid_sources):
                class_weights[source_index, class_id] = alpha_valid[local_index]
        return class_weights.detach()

    def compute_loss(self, source_batches, target_batch) -> MethodOutput:
        if len(source_batches) == 1:
            output = super().compute_loss(source_batches, target_batch)
            output.metrics["ms_cbtp_single_source_degenerate_to_cbtp"] = 1.0
            return output

        self.global_step += 1
        target_x, _ = target_batch
        logits_target, features_target = self.forward(target_x)
        target_reg, target_metrics, label_gate = self._target_regularizers(target_x, logits_target)

        source_logits = []
        source_features = []
        source_labels = []
        source_prototypes = []
        source_present = []
        source_class_weights = []
        source_outliers = []
        for source_x, source_y in source_batches:
            logits_source, features_source = self.forward(source_x)
            class_weights = inverse_sqrt_class_weights(source_y, num_classes=self.num_classes).to(
                device=source_y.device,
                dtype=features_source.dtype,
            )
            prototypes, present = compute_class_prototypes(
                features_source,
                source_y,
                num_classes=self.num_classes,
                sample_weights=class_weights[source_y.long()],
            )
            source_logits.append(logits_source)
            source_features.append(features_source)
            source_labels.append(source_y)
            source_prototypes.append(prototypes)
            source_present.append(present)
            source_class_weights.append(class_weights)
            source_outliers.append(source_outlier_weights(features_source.detach(), source_y, prototypes.detach()))

        source_alpha = self._source_alpha(source_features, features_target)
        class_alpha = self._class_alpha(
            source_prototypes,
            source_present,
            features_target,
            logits_target,
            source_alpha,
        )
        k_sources = len(source_batches)

        loss_cls = features_target.new_zeros(())
        loss_ot = features_target.new_zeros(())
        metrics: dict[str, float] = {}
        for source_index in range(k_sources):
            y_source = source_labels[source_index].long()
            gate = torch.where(
                class_alpha[source_index, y_source] < self.negative_gate_threshold,
                torch.full_like(class_alpha[source_index, y_source], self.negative_gate_floor),
                torch.ones_like(class_alpha[source_index, y_source]),
            )
            sample_weights = (
                source_class_weights[source_index][y_source]
                * source_outliers[source_index].to(device=y_source.device, dtype=features_target.dtype)
                * gate.to(device=y_source.device, dtype=features_target.dtype)
            )
            cls = weighted_ce(source_logits[source_index], y_source, sample_weights)
            ot_loss, ot_metrics = jdot_transport_loss(
                source_features=source_features[source_index],
                source_labels=y_source,
                target_features=features_target,
                target_logits=logits_target,
                num_classes=self.num_classes,
                config=self.ot_config,
                source_sample_weights=sample_weights,
                source_prototypes=source_prototypes[source_index],
                label_gate=label_gate,
            )
            alpha = source_alpha[source_index].to(device=features_target.device, dtype=features_target.dtype)
            loss_cls = loss_cls + alpha * cls
            loss_ot = loss_ot + alpha * ot_loss
            metrics[f"loss_ot_source_{source_index}"] = float(ot_metrics["loss_ot"])
            metrics[f"alpha_source_{source_index}"] = float(source_alpha[source_index].detach().item())

        loss = self.source_ce_weight * loss_cls + self.adaptation_weight * loss_ot + target_reg
        merged_logits = torch.cat(source_logits, dim=0)
        merged_labels = torch.cat(source_labels, dim=0)
        metrics.update(
            {
                "loss_total": float(loss.detach().item()),
                "loss_cls": float(loss_cls.detach().item()),
                "loss_alignment": float(loss_ot.detach().item()),
                "acc_source": accuracy_from_logits(merged_logits, merged_labels),
                "class_alpha_min": float(class_alpha.min().detach().item()),
                "class_alpha_max": float(class_alpha.max().detach().item()),
            }
        )
        metrics.update(target_metrics)
        self._last_source_weights = source_alpha.detach().cpu().numpy()
        self._last_class_source_weights = class_alpha.detach().cpu().numpy()
        return MethodOutput(loss=loss, metrics=metrics)

    def reliability_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        if self._last_source_weights is not None:
            snapshot["source_weights"] = self._last_source_weights.copy()
        if self._last_class_source_weights is not None:
            snapshot["class_source_weights"] = self._last_class_source_weights.copy()
        return snapshot


def build_method(
    method_name: str,
    *,
    num_classes: int,
    input_shape: tuple[int, int],
    embedding_dim: int = 128,
    dropout: float = 0.1,
    classifier_hidden_dim: int = 128,
    adaptation_weight: float = 1.0,
    ot_solver: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
    prototype_weight: float | None = None,
    prototype_in_coupling: bool | str = True,
    tau_steps: int = 1000,
) -> BaseMethod:
    normalized = method_name.strip().lower()
    if normalized == "otda":
        normalized = "jdot"
    model = TEClassifier(
        num_classes=num_classes,
        in_channels=int(input_shape[0]),
        input_length=int(input_shape[1]),
        embedding_dim=embedding_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout=dropout,
        temporal_pooling=True,
    )
    shared = {"num_classes": num_classes}
    if normalized == "source_only":
        return SourceOnlyMethod(model, **shared)
    if normalized == "target_only":
        return TargetOnlyMethod(model, **shared)
    if normalized == "mmd":
        return MMDMethod(model, adaptation_weight=adaptation_weight, **shared)
    if normalized == "coral":
        return CORALMethod(model, adaptation_weight=adaptation_weight, **shared)
    if normalized == "dann":
        return DANNMethod(model, adaptation_weight=adaptation_weight, dropout=dropout, **shared)

    proto = 0.0
    if normalized in {"tp_jdot", "tp_wjdot"}:
        proto = 0.04 if prototype_weight is None else float(prototype_weight)
    elif normalized in {"cbtp_jdot", "cbtp_wjdot", "ms_cbtp_wjdot"}:
        proto = 0.2 if prototype_weight is None else float(prototype_weight)
    elif prototype_weight is not None:
        proto = float(prototype_weight)
    safe_tp_family = normalized in {"tp_jdot", "cbtp_jdot", "tp_wjdot", "cbtp_wjdot", "ms_cbtp_wjdot"}
    ot_config = OTLossConfig(
        feature_weight=0.1,
        label_weight=1.0,
        prototype_weight=proto,
        prototype_in_coupling=False if safe_tp_family else prototype_in_coupling,
        prototype_mode="tp_barycentric" if safe_tp_family else "legacy_pairwise",
        prototype_cost_clip="p90" if safe_tp_family else None,
        ot_class_entropy_gate=safe_tp_family,
        solver=ot_solver,
        sinkhorn_reg=sinkhorn_reg,
    )
    ot_kwargs = {
        "adaptation_weight": adaptation_weight,
        "ot_config": ot_config,
        "tau_steps": tau_steps,
        **shared,
    }
    if normalized == "jdot":
        return JDOTMethod(model, **ot_kwargs)
    if normalized == "tp_jdot":
        return TPJDOTMethod(model, **ot_kwargs)
    if normalized == "cbtp_jdot":
        return CBTPJDOTMethod(model, **ot_kwargs)
    if normalized == "wjdot":
        return WJDOTMethod(model, **ot_kwargs)
    if normalized == "tp_wjdot":
        return TPWJDOTMethod(model, **ot_kwargs)
    if normalized == "cbtp_wjdot":
        return CBTPWJDOTMethod(model, **ot_kwargs)
    if normalized == "ms_cbtp_wjdot":
        return MSCBTPWJDOTMethod(model, **ot_kwargs)
    raise KeyError(f"Unsupported method: {method_name}")
