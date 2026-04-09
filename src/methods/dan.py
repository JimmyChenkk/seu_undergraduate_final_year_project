"""MK-MMD-based DAN implementation."""

from __future__ import annotations

import torch.nn.functional as F

from src.losses import GaussianKernel, MultipleKernelMaximumMeanDiscrepancy

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class DANMethod(SingleSourceMethodBase):
    """Deep adaptation network using MK-MMD."""

    method_name = "dan"

    def __init__(
        self,
        *,
        adaptation_weight: float = 0.5,
        adaptation_schedule: str = "constant",
        adaptation_max_steps: int = 1000,
        adaptation_schedule_alpha: float = 10.0,
        kernel_scales: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0),
        linear_mmd: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alignment_scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        self.mkmmd = MultipleKernelMaximumMeanDiscrepancy(
            [GaussianKernel(alpha=float(scale)) for scale in kernel_scales],
            linear=linear_mmd,
        )

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        _, features_target = self.forward(target_x)
        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_alignment = self.mkmmd(features_source, features_target)
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
