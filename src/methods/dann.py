"""DANN implementation."""

from __future__ import annotations

import torch.nn.functional as F

from src.losses import (
    DomainDiscriminator,
    GradientReverseLayer,
    WarmStartGradientReverseLayer,
    domain_adversarial_loss,
)

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class DANNMethod(SingleSourceMethodBase):
    """Adversarial single-source UDA baseline."""

    method_name = "dann"

    def __init__(
        self,
        *,
        adaptation_weight: float = 0.5,
        adaptation_schedule: str = "constant",
        adaptation_max_steps: int = 1000,
        adaptation_schedule_alpha: float = 10.0,
        grl_lambda: float = 1.0,
        grl_warm_start: bool = True,
        grl_max_iters: int = 1000,
        domain_hidden_dim: int | None = None,
        domain_num_hidden_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(dropout=dropout, **kwargs)
        self.alignment_scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        if grl_warm_start:
            self.grl = WarmStartGradientReverseLayer(
                alpha=adaptation_schedule_alpha,
                lo=0.0,
                hi=grl_lambda,
                max_iters=grl_max_iters,
                auto_step=True,
            )
        else:
            self.grl = GradientReverseLayer(lambda_=grl_lambda)
        self.discriminator = DomainDiscriminator(
            self.encoder.out_dim,
            hidden_dim=domain_hidden_dim or max(128, self.encoder.out_dim),
            dropout=dropout,
            num_hidden_layers=domain_num_hidden_layers,
        )

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
        current_weight = self.alignment_scheduler.step()
        loss_total = loss_cls + current_weight * loss_adv
        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_adv.item()),
                "lambda_alignment": current_weight,
                "acc_source": accuracy_from_logits(logits_source, source_y),
                "acc_domain": domain_acc,
                "grl_coeff": float(getattr(self.grl, "last_coeff", 1.0)),
            },
        )
