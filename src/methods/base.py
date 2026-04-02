"""Shared method abstractions for TE benchmark reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.backbones import ClassifierHead, FullyConvolutionalEncoder


@dataclass
class MethodStepOutput:
    """One optimization step output."""

    loss: torch.Tensor
    metrics: dict[str, float]


class SingleSourceMethodBase(nn.Module):
    """Shared encoder/classifier stack for single-source methods."""

    supports_multi_source = False

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 34,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = FullyConvolutionalEncoder(in_channels=in_channels, dropout=dropout)
        self.classifier = ClassifierHead(
            in_features=self.encoder.out_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.forward(x)
        return features


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute minibatch accuracy."""

    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())
