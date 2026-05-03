"""CoDATS baseline for time-series domain adaptation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.losses import (
    DomainDiscriminator,
    GradientReverseLayer,
    WarmStartGradientReverseLayer,
    domain_adversarial_loss,
)

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class CoDATSClassifierHead(nn.Module):
    """Two-hidden-layer classifier head used by the CoDATS reference code."""

    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 500,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.main = nn.Linear(in_features, num_classes)
            return

        dropout_layer = nn.Identity() if dropout <= 0 else nn.Dropout(dropout)
        self.main = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class CoDATSMethod(SingleSourceMethodBase):
    """CoDATS: DANN-style adversarial alignment with the CoDATS classifier."""

    method_name = "codats"

    def __init__(
        self,
        *,
        num_classes: int,
        adaptation_weight: float = 0.5,
        adaptation_schedule: str = "warm_start",
        adaptation_max_steps: int = 1000,
        adaptation_schedule_alpha: float = 10.0,
        grl_lambda: float = 1.0,
        grl_warm_start: bool = True,
        grl_max_iters: int = 1000,
        domain_hidden_dim: int | None = None,
        domain_num_hidden_layers: int = 2,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 500,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            dropout=dropout,
            classifier_hidden_dim=classifier_hidden_dim,
            **kwargs,
        )
        self.classifier = CoDATSClassifierHead(
            in_features=self.encoder.out_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
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
