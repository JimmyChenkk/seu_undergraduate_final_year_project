"""DSAN baseline with label-aware LMMD alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses import LocalMaximumMeanDiscrepancyLoss

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class DSANMethod(SingleSourceMethodBase):
    """Deep subdomain adaptation network using class-conditional LMMD."""

    method_name = "dsan"

    def __init__(
        self,
        *,
        num_classes: int,
        adaptation_weight: float = 0.5,
        adaptation_schedule: str = "warm_start",
        adaptation_max_steps: int = 1000,
        adaptation_schedule_alpha: float = 10.0,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        self.alignment_scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        self.lmmd = LocalMaximumMeanDiscrepancyLoss(
            num_classes=num_classes,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            fix_sigma=fix_sigma,
        )

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)

        loss_cls = F.cross_entropy(logits_source, source_y)
        target_probabilities = torch.softmax(logits_target, dim=1)
        loss_alignment = self.lmmd(
            features_source,
            features_target,
            source_y,
            target_probabilities,
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
