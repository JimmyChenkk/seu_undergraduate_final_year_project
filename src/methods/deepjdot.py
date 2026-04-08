"""DeepJDOT implementation using minibatch OT on GPU-friendly torch tensors."""

from __future__ import annotations

import torch.nn.functional as F

from src.losses import deepjdot_loss

from .base import MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class DeepJDOTMethod(SingleSourceMethodBase):
    """Deep joint distribution optimal transport with a shared encoder."""

    method_name = "deepjdot"

    def __init__(
        self,
        *,
        adaptation_weight: float = 1.0,
        reg_dist: float = 0.1,
        reg_cl: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.adaptation_weight = adaptation_weight
        self.reg_dist = reg_dist
        self.reg_cl = reg_cl

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)

        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_alignment = deepjdot_loss(
            source_y,
            logits_target,
            features_source,
            features_target,
            reg_dist=self.reg_dist,
            reg_cl=self.reg_cl,
        )
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
