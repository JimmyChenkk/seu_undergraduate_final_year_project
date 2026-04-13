"""CDAN implementation adapted for the TEP benchmark scaffold."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses import (
    ConditionalDomainAdversarialLoss,
    DomainDiscriminator,
    GradientReverseLayer,
    MinimumClassConfusionLoss,
    WarmStartGradientReverseLayer,
)

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class CDANMethod(SingleSourceMethodBase):
    """Conditional adversarial domain adaptation."""

    method_name = "cdan"

    def __init__(
        self,
        *,
        num_classes: int,
        adaptation_weight: float = 0.5,
        adaptation_schedule: str = "constant",
        adaptation_max_steps: int = 1000,
        adaptation_schedule_alpha: float = 10.0,
        grl_lambda: float = 1.0,
        grl_warm_start: bool = True,
        grl_max_iters: int = 1000,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        in_channels: int = 34,
        randomized: bool = False,
        randomized_dim: int = 1024,
        entropy_conditioning: bool = True,
        mcc_weight: float = 0.0,
        mcc_temperature: float = 2.0,
        domain_hidden_dim: int | None = None,
        domain_num_hidden_layers: int = 2,
        input_length: int = 600,
        backbone_name: str = "fcn",
        backbone_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            input_length=input_length,
            dropout=dropout,
            classifier_hidden_dim=classifier_hidden_dim,
            backbone_name=backbone_name,
            backbone_kwargs=backbone_kwargs,
        )
        self.alignment_scheduler = AdaptationWeightScheduler(
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
        self.mcc_weight = float(mcc_weight)
        self.mcc = MinimumClassConfusionLoss(mcc_temperature) if self.mcc_weight > 0 else None

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)

        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_domain, domain_acc = self.domain_adv(
            logits_source,
            features_source,
            logits_target,
            features_target,
        )
        loss_mcc = (
            self.mcc(logits_target)
            if self.mcc is not None
            else torch.zeros((), device=logits_source.device, dtype=logits_source.dtype)
        )
        current_weight = self.alignment_scheduler.step()
        loss_total = loss_cls + current_weight * loss_domain + self.mcc_weight * loss_mcc

        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_domain.item()),
                "loss_mcc": float(loss_mcc.item()),
                "lambda_alignment": current_weight,
                "lambda_mcc": self.mcc_weight,
                "acc_source": accuracy_from_logits(logits_source, source_y),
                "acc_domain": domain_acc,
                "grl_coeff": float(getattr(self.domain_adv.grl, "last_coeff", 1.0)),
            },
        )
