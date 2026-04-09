"""Shared method abstractions for TE benchmark reproduction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from src.backbones import ClassifierHead, build_backbone


@dataclass
class MethodStepOutput:
    """One optimization step output."""

    loss: torch.Tensor
    metrics: dict[str, float]


class AdaptationWeightScheduler:
    """Warm-start scheduler for domain-alignment losses."""

    def __init__(
        self,
        *,
        base_weight: float,
        schedule: str = "constant",
        max_steps: int = 1000,
        alpha: float = 10.0,
    ) -> None:
        self.base_weight = float(base_weight)
        self.schedule = str(schedule).strip().lower()
        self.max_steps = max(int(max_steps), 1)
        self.alpha = float(alpha)
        self.step_num = 0
        self.last_weight = self.base_weight if self.schedule == "constant" else 0.0

    def _factor(self) -> float:
        if self.schedule in {"constant", "none"}:
            return 1.0

        progress = min(self.step_num / float(self.max_steps), 1.0)
        if self.schedule in {"linear", "ramp"}:
            return progress
        if self.schedule in {"warm_start", "sigmoid", "dann"}:
            return 2.0 / (1.0 + math.exp(-self.alpha * progress)) - 1.0
        raise KeyError(f"Unsupported adaptation schedule: {self.schedule}")

    def step(self) -> float:
        self.step_num += 1
        self.last_weight = self.base_weight * self._factor()
        return self.last_weight


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
