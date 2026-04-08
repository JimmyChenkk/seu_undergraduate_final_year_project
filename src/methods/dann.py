"""DANN implementation."""

from __future__ import annotations

import torch.nn.functional as F

from src.losses import DomainDiscriminator, GradientReverseLayer, domain_adversarial_loss

from .base import MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class DANNMethod(SingleSourceMethodBase):
    """Adversarial single-source UDA baseline."""

    method_name = "dann"

    def __init__(
        self,
        *,
        adaptation_weight: float = 0.5,
        grl_lambda: float = 1.0,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(dropout=dropout, **kwargs)
        self.adaptation_weight = adaptation_weight
        self.grl = GradientReverseLayer(lambda_=grl_lambda)
        self.discriminator = DomainDiscriminator(self.encoder.out_dim, dropout=dropout)

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        _, features_target = self.forward(target_x)
        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_adv, domain_acc = domain_adversarial_loss(
            features_source,
            features_target,
            discriminator=self.discriminator,
            grl=self.grl,
        )
        loss_total = loss_cls + self.adaptation_weight * loss_adv
        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_adv.item()),
                "acc_source": accuracy_from_logits(logits_source, source_y),
                "acc_domain": domain_acc,
            },
        )
