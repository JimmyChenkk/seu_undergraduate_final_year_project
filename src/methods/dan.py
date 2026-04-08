"""MK-MMD-based DAN implementation."""

from __future__ import annotations

import torch.nn.functional as F

from src.losses import multiple_kernel_mmd

from .base import MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class DANMethod(SingleSourceMethodBase):
    """Deep adaptation network using MK-MMD."""

    method_name = "dan"

    def __init__(self, *, adaptation_weight: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.adaptation_weight = adaptation_weight

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        _, features_target = self.forward(target_x)
        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_alignment = multiple_kernel_mmd(features_source, features_target)
        loss_total = loss_cls + self.adaptation_weight * loss_alignment
        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_alignment.item()),
                "acc_source": accuracy_from_logits(logits_source, source_y),
            },
        )
