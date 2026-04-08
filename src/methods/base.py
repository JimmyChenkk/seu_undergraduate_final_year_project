"""Shared method abstractions for TE benchmark reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.backbones import ClassifierHead, build_backbone


@dataclass
class MethodStepOutput:
    """One optimization step output."""

    loss: torch.Tensor
    metrics: dict[str, float]


class SingleSourceMethodBase(nn.Module):
    """Shared encoder/classifier stack for single-source methods."""

    supports_multi_source = True

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 34,
        input_length: int = 600,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        backbone_name: str = "fcn",
        backbone_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.encoder = build_backbone(
            name=backbone_name,
            in_channels=in_channels,
            input_length=input_length,
            dropout=dropout,
            backbone_kwargs=backbone_kwargs,
        )
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

    def merge_source_batches(self, source_batches) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge one or more source-domain minibatches into one supervised batch."""

        if len(source_batches) == 1:
            return source_batches[0]

        source_x = torch.cat([x_batch for x_batch, _ in source_batches], dim=0)
        source_y = torch.cat([y_batch for _, y_batch in source_batches], dim=0)
        return source_x, source_y


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute minibatch accuracy."""

    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())
