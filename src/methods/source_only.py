"""Source-only baseline."""

from __future__ import annotations

import torch.nn.functional as F

from .base import MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class SourceOnlyMethod(SingleSourceMethodBase):
    """Standard supervised source-only training."""

    method_name = "source_only"

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        del target_batch
        source_x, source_y = self.merge_source_batches(source_batches)
        logits_source, _ = self.forward(source_x)
        classification_loss = F.cross_entropy(logits_source, source_y)
        return MethodStepOutput(
            loss=classification_loss,
            metrics={
                "loss_total": float(classification_loss.item()),
                "loss_cls": float(classification_loss.item()),
                "acc_source": accuracy_from_logits(logits_source, source_y),
            },
        )
