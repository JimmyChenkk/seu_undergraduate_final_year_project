"""Lightweight multi-source MFSAN-style model."""

from __future__ import annotations

import itertools

import torch
from torch import nn
import torch.nn.functional as F

from src.backbones import FullyConvolutionalEncoder
from src.losses import multiple_kernel_mmd

from .base import MethodStepOutput, accuracy_from_logits


class MFSANMethod(nn.Module):
    """A compact multi-source variant inspired by MFSAN."""

    method_name = "mfsan"
    supports_multi_source = True

    def __init__(
        self,
        *,
        num_classes: int,
        num_sources: int,
        in_channels: int = 34,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        mmd_weight: float = 0.5,
        discrepancy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = FullyConvolutionalEncoder(in_channels=in_channels, dropout=dropout)
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.encoder.out_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for _ in range(num_sources)
            ]
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_sources)]
        )
        self.mmd_weight = mmd_weight
        self.discrepancy_weight = discrepancy_weight

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        logits = [classifier(projector(encoded)) for projector, classifier in zip(self.projectors, self.classifiers)]
        probabilities = [F.softmax(item, dim=1) for item in logits]
        return torch.stack(probabilities, dim=0).mean(dim=0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        target_x, _ = target_batch
        encoded_target = self.encoder(target_x)

        classification_losses = []
        source_accuracies = []
        mmd_losses = []
        target_probabilities = []

        for (source_x, source_y), projector, classifier in zip(
            source_batches, self.projectors, self.classifiers
        ):
            encoded_source = self.encoder(source_x)
            projected_source = projector(encoded_source)
            projected_target = projector(encoded_target)
            logits_source = classifier(projected_source)
            target_logits = classifier(projected_target)
            classification_losses.append(F.cross_entropy(logits_source, source_y))
            source_accuracies.append(accuracy_from_logits(logits_source, source_y))
            mmd_losses.append(multiple_kernel_mmd(projected_source, projected_target))
            target_probabilities.append(F.softmax(target_logits, dim=1))

        loss_cls = torch.stack(classification_losses).mean()
        loss_mmd = torch.stack(mmd_losses).mean()

        discrepancy_losses = []
        for left, right in itertools.combinations(target_probabilities, 2):
            discrepancy_losses.append(torch.abs(left - right).mean())
        if discrepancy_losses:
            loss_discrepancy = torch.stack(discrepancy_losses).mean()
        else:
            loss_discrepancy = torch.tensor(0.0, device=loss_cls.device)

        loss_total = loss_cls + self.mmd_weight * loss_mmd + self.discrepancy_weight * loss_discrepancy
        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_mmd.item()),
                "loss_discrepancy": float(loss_discrepancy.item()),
                "acc_source": float(sum(source_accuracies) / max(len(source_accuracies), 1)),
            },
        )
