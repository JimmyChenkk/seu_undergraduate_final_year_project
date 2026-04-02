"""CDAN implementation adapted for the TEP benchmark scaffold."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.losses import DomainDiscriminator, GradientReverseLayer

from .base import MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class RandomizedMultiLinearMap(nn.Module):
    """Lower-dimensional randomized conditioning map used for stable CDAN training."""

    def __init__(self, feature_dim: int, num_classes: int, output_dim: int = 1024) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("rf", torch.randn(feature_dim, output_dim))
        self.register_buffer("rg", torch.randn(num_classes, output_dim))

    def forward(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        projected_features = features @ self.rf
        projected_probabilities = probabilities @ self.rg
        return (projected_features * projected_probabilities) / math.sqrt(float(self.output_dim))


class CDANMethod(SingleSourceMethodBase):
    """Conditional adversarial domain adaptation."""

    method_name = "cdan"

    def __init__(
        self,
        *,
        num_classes: int,
        adaptation_weight: float = 0.5,
        grl_lambda: float = 1.0,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        in_channels: int = 34,
        randomized_dim: int = 1024,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
            classifier_hidden_dim=classifier_hidden_dim,
        )
        self.adaptation_weight = adaptation_weight
        self.grl = GradientReverseLayer(lambda_=grl_lambda)
        conditioned_dim = randomized_dim
        self.discriminator = DomainDiscriminator(
            conditioned_dim,
            hidden_dim=max(256, self.encoder.out_dim),
            dropout=dropout,
        )
        self.map = RandomizedMultiLinearMap(self.encoder.out_dim, num_classes, output_dim=randomized_dim)

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = source_batches[0]
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)

        probabilities_source = F.softmax(logits_source, dim=1).detach()
        probabilities_target = F.softmax(logits_target, dim=1).detach()

        conditioned_source = self.map(features_source, probabilities_source)
        conditioned_target = self.map(features_target, probabilities_target)

        adv_source = self.grl(conditioned_source)
        adv_target = self.grl(conditioned_target)
        logits_domain_source = self.discriminator(adv_source)
        logits_domain_target = self.discriminator(adv_target)

        labels_source = torch.ones_like(logits_domain_source)
        labels_target = torch.zeros_like(logits_domain_target)
        loss_domain_source = F.binary_cross_entropy_with_logits(logits_domain_source, labels_source)
        loss_domain_target = F.binary_cross_entropy_with_logits(logits_domain_target, labels_target)
        loss_domain = 0.5 * (loss_domain_source + loss_domain_target)

        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_total = loss_cls + self.adaptation_weight * loss_domain

        with torch.no_grad():
            predictions_source = (torch.sigmoid(logits_domain_source) >= 0.5).float()
            predictions_target = (torch.sigmoid(logits_domain_target) >= 0.5).float()
            correct = (predictions_source == labels_source).float().sum() + (
                predictions_target == labels_target
            ).float().sum()
            total = labels_source.numel() + labels_target.numel()
            domain_acc = float(correct / max(total, 1))

        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_domain.item()),
                "acc_source": accuracy_from_logits(logits_source, source_y),
                "acc_domain": domain_acc,
            },
        )
