"""Target-only baseline."""

from __future__ import annotations

import torch.nn.functional as F

from .base import MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class TargetOnlyMethod(SingleSourceMethodBase):
    """Standard supervised target-only training."""

    method_name = "target_only"

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        del source_batches
        target_x, target_y = target_batch
        logits_target, _ = self.forward(target_x)
        classification_loss = F.cross_entropy(logits_target, target_y)
        return MethodStepOutput(
            loss=classification_loss,
            metrics={
                "loss_total": float(classification_loss.item()),
                "loss_cls": float(classification_loss.item()),
                "acc_source": accuracy_from_logits(logits_target, target_y),
            },
        )
