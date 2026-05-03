"""Temporal-consistency CDAN for cross-condition TEP diagnosis."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch
import torch.nn.functional as F

from src.losses import (
    ConditionalDomainAdversarialLoss,
    DomainDiscriminator,
    GradientReverseLayer,
    WarmStartGradientReverseLayer,
)

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


_EPSILON = 1e-6


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    entropy = -(probabilities * torch.log(probabilities.clamp_min(_EPSILON))).sum(dim=1)
    return entropy / math.log(max(int(probabilities.shape[1]), 2))


class TemporalAugmenter:
    """Conservative weak/strong augmentations for 34 x 600 process windows."""

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

    def _jitter(self, x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        return x + torch.randn_like(x) * std

    def _scaling(self, x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        scale = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) * std
        return x * scale

    def _time_mask(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
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

    def _channel_dropout(self, x: torch.Tensor, probability: float) -> torch.Tensor:
        if probability <= 0:
            return x
        keep_mask = (
            torch.rand(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype) >= probability
        ).to(dtype=x.dtype)
        return x * keep_mask

    def weak(self, x: torch.Tensor) -> torch.Tensor:
        x = self._jitter(x, self.weak_jitter_std)
        return self._scaling(x, self.weak_scaling_std)

    def strong(self, x: torch.Tensor) -> torch.Tensor:
        x = self._jitter(x, self.strong_jitter_std)
        x = self._scaling(x, self.strong_scaling_std)
        x = self._time_mask(x, self.strong_time_mask_ratio)
        return self._channel_dropout(x, self.strong_channel_dropout_prob)

    def pair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weak(x), self.strong(x)


class TCCDANMethod(SingleSourceMethodBase):
    """TC-CDAN: CDAN with EMA teacher and weak/strong temporal consistency."""

    method_name = "tc_cdan"

    def __init__(
        self,
        *,
        num_classes: int,
        adaptation_weight: float = 0.25,
        adaptation_schedule: str = "warm_start",
        adaptation_max_steps: int = 1200,
        adaptation_schedule_alpha: float = 10.0,
        grl_lambda: float = 1.0,
        grl_warm_start: bool = True,
        grl_max_iters: int = 1200,
        randomized: bool = True,
        randomized_dim: int = 256,
        entropy_conditioning: bool = True,
        domain_hidden_dim: int | None = None,
        domain_num_hidden_layers: int = 2,
        teacher_ema_decay: float = 0.995,
        teacher_temperature: float = 1.0,
        consistency_weight: float = 0.05,
        consistency_start_step: int = 0,
        consistency_warmup_steps: int = 1000,
        consistency_loss: str = "kl",
        augment_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout, **kwargs)
        self.num_classes = int(num_classes)
        self.teacher_ema_decay = min(max(float(teacher_ema_decay), 0.0), 0.9999)
        self.teacher_temperature = max(float(teacher_temperature), _EPSILON)
        self.consistency_weight = float(consistency_weight)
        self.consistency_start_step = max(int(consistency_start_step), 0)
        self.consistency_warmup_steps = max(int(consistency_warmup_steps), 0)
        self.consistency_loss = str(consistency_loss).strip().lower()
        if self.consistency_loss not in {"kl", "mse"}:
            raise KeyError(f"Unsupported consistency loss: {consistency_loss}")

        self.domain_scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        conditioned_dim = randomized_dim if randomized else self.encoder.out_dim * num_classes
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
        self.domain_adv = ConditionalDomainAdversarialLoss(
            domain_discriminator=DomainDiscriminator(
                conditioned_dim,
                hidden_dim=domain_hidden_dim or max(256, self.encoder.out_dim),
                dropout=dropout,
                num_hidden_layers=domain_num_hidden_layers,
            ),
            feature_dim=self.encoder.out_dim,
            num_classes=num_classes,
            entropy_conditioning=entropy_conditioning,
            randomized=randomized,
            randomized_dim=randomized_dim,
            grl=grl,
        )
        self.augmenter = TemporalAugmenter(**(augment_kwargs or {}))
        self.teacher_encoder = deepcopy(self.encoder)
        self.teacher_classifier = deepcopy(self.classifier)
        self._set_teacher_requires_grad(False)
        self.teacher_encoder.eval()
        self.teacher_classifier.eval()
        self.register_buffer("step_num", torch.zeros((), dtype=torch.long))

    def train(self, mode: bool = True) -> "TCCDANMethod":
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

    def _ramp_weight(self, base_weight: float, current_step: int, start_step: int, warmup_steps: int) -> float:
        if base_weight <= 0 or current_step < start_step:
            return 0.0
        if warmup_steps <= 0:
            return float(base_weight)
        progress = min(max(float(current_step - start_step), 0.0) / float(warmup_steps), 1.0)
        return float(base_weight) * progress

    def _compute_consistency(
        self,
        *,
        logits_student_strong: torch.Tensor,
        probabilities_student_strong: torch.Tensor,
        probabilities_teacher_weak: torch.Tensor,
    ) -> torch.Tensor:
        if self.consistency_loss == "mse":
            return ((probabilities_student_strong - probabilities_teacher_weak.detach()) ** 2).mean()
        log_probabilities_student = F.log_softmax(logits_student_strong, dim=1)
        return F.kl_div(log_probabilities_student, probabilities_teacher_weak.detach(), reduction="batchmean")

    def _compute_tc_components(self, source_batches, target_batch) -> dict[str, Any]:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        current_step = int(self.step_num.item())

        logits_source, features_source = self.forward(source_x)
        weak_target, strong_target = self.augmenter.pair(target_x)
        logits_target_weak, features_target_weak = self.forward(weak_target)
        logits_target_strong, features_target_strong = self.forward(strong_target)
        teacher_logits_weak, teacher_features_weak = self._teacher_forward(weak_target)
        teacher_probabilities = self._teacher_probabilities(teacher_logits_weak).detach()
        student_probabilities = torch.softmax(logits_target_strong, dim=1)

        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_domain, domain_accuracy = self.domain_adv(
            logits_source,
            features_source,
            logits_target_weak,
            features_target_weak,
        )
        lambda_domain = self.domain_scheduler.step()
        loss_consistency = self._compute_consistency(
            logits_student_strong=logits_target_strong,
            probabilities_student_strong=student_probabilities,
            probabilities_teacher_weak=teacher_probabilities,
        )
        lambda_consistency = self._ramp_weight(
            self.consistency_weight,
            current_step,
            self.consistency_start_step,
            self.consistency_warmup_steps,
        )
        base_loss = loss_cls + lambda_domain * loss_domain + lambda_consistency * loss_consistency

        teacher_predictions = teacher_probabilities.argmax(dim=1)
        student_predictions = student_probabilities.argmax(dim=1)
        confidence = teacher_probabilities.max(dim=1).values
        entropy = normalized_entropy(teacher_probabilities)
        agreement = (teacher_predictions == student_predictions).float()
        metrics = {
            "loss_total": float(base_loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_alignment": float(loss_domain.item()),
            "loss_domain": float(loss_domain.item()),
            "loss_consistency": float(loss_consistency.item()),
            "consistency_loss": float(loss_consistency.item()),
            "lambda_alignment": float(lambda_domain),
            "lambda_domain": float(lambda_domain),
            "lambda_consistency": float(lambda_consistency),
            "acc_source": accuracy_from_logits(logits_source, source_y),
            "acc_domain": float(domain_accuracy),
            "domain_accuracy": float(domain_accuracy),
            "mean_target_confidence": float(confidence.mean().item()),
            "mean_target_entropy": float(entropy.mean().item()),
            "teacher_student_agreement": float(agreement.mean().item()),
            "grl_coeff": float(getattr(self.domain_adv.grl, "last_coeff", 1.0)),
        }
        return {
            "base_loss": base_loss,
            "loss_cls": loss_cls,
            "loss_domain": loss_domain,
            "loss_consistency": loss_consistency,
            "lambda_domain": lambda_domain,
            "lambda_consistency": lambda_consistency,
            "metrics": metrics,
            "source_y": source_y,
            "logits_source": logits_source,
            "features_source": features_source,
            "target_x": target_x,
            "logits_target_weak": logits_target_weak,
            "features_target_weak": features_target_weak,
            "logits_target_strong": logits_target_strong,
            "features_target_strong": features_target_strong,
            "teacher_logits_weak": teacher_logits_weak,
            "teacher_features_weak": teacher_features_weak,
            "teacher_probabilities": teacher_probabilities,
            "student_probabilities": student_probabilities,
            "teacher_predictions": teacher_predictions,
            "student_predictions": student_predictions,
            "target_confidence": confidence,
            "target_entropy": entropy,
            "agreement": agreement,
            "current_step": current_step,
        }

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        components = self._compute_tc_components(source_batches, target_batch)
        return MethodStepOutput(loss=components["base_loss"], metrics=components["metrics"])

    def _ema_update_teacher(self) -> None:
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
                teacher_parameter.data.mul_(self.teacher_ema_decay).add_(
                    student_parameter.data,
                    alpha=1.0 - self.teacher_ema_decay,
                )
            for teacher_parameter, student_parameter in zip(
                self.teacher_classifier.parameters(),
                self.classifier.parameters(),
            ):
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
        self.step_num += 1
        return {
            "teacher_ema_decay": float(self.teacher_ema_decay),
        }
