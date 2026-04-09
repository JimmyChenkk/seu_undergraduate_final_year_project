"""Deep CORAL implementation."""

from __future__ import annotations

import torch.nn.functional as F

from src.losses import coral_loss

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class CORALMethod(SingleSourceMethodBase):
    """Moment alignment baseline."""

    method_name = "coral"

    def __init__(
        self,
        *,
        adaptation_weight: float = 0.5,
        adaptation_schedule: str = "constant",
        adaptation_max_steps: int = 1000,
        adaptation_schedule_alpha: float = 10.0,
        align_mean: bool = True,
        normalize_covariance: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.align_mean = align_mean
        self.normalize_covariance = normalize_covariance
        self.alignment_scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        _, features_target = self.forward(target_x)
        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_alignment = coral_loss(
            features_source,
            features_target,
            align_mean=self.align_mean,
            normalize_covariance=self.normalize_covariance,
        )
        current_weight = self.alignment_scheduler.step()
        loss_total = loss_cls + current_weight * loss_alignment
        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_alignment.item()),
                "lambda_alignment": current_weight,
                "acc_source": accuracy_from_logits(logits_source, source_y),
            },
        )
