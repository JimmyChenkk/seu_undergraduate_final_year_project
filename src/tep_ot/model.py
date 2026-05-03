"""FCN backbone with temporal multi-segment prototype embeddings."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class TemporalFCNEncoder(nn.Module):
    """Three-block FCN encoder for input tensors shaped ``(B, 34, 600)``.

    The embedding combines global average pooling with four fixed temporal
    windows: first 200, middle 200, last 200, and last 300 time steps.
    """

    def __init__(
        self,
        *,
        in_channels: int = 34,
        input_length: int = 600,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        temporal_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.temporal_pooling = bool(temporal_pooling)
        self.features = nn.Sequential(
            ConvBlock(in_channels, 128, kernel_size=9, dropout=dropout),
            ConvBlock(128, 256, kernel_size=5, dropout=dropout),
            ConvBlock(256, 128, kernel_size=3, dropout=0.0),
        )
        pooled_parts = 5 if self.temporal_pooling else 1
        self.projection = nn.Sequential(
            nn.Linear(128 * pooled_parts, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.out_dim = int(embedding_dim)

    def _pool_segment(self, features: torch.Tensor, start: int, end: int) -> torch.Tensor:
        length = int(features.shape[-1])
        start = max(0, min(int(start), length - 1))
        end = max(start + 1, min(int(end), length))
        return features[..., start:end].mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        full = features.mean(dim=-1)
        if not self.temporal_pooling:
            return self.projection(full)

        length = int(features.shape[-1])
        first_end = min(200, length)
        middle_start = min(max(length // 2 - 100, 0), length - 1)
        middle_end = min(middle_start + 200, length)
        last_200_start = max(length - 200, 0)
        last_300_start = max(length - 300, 0)
        pooled = torch.cat(
            [
                full,
                self._pool_segment(features, 0, first_end),
                self._pool_segment(features, middle_start, middle_end),
                self._pool_segment(features, last_200_start, length),
                self._pool_segment(features, last_300_start, length),
            ],
            dim=1,
        )
        return self.projection(pooled)


class TEClassifier(nn.Module):
    """Backbone plus linear classifier for 29 TEP classes."""

    def __init__(
        self,
        *,
        num_classes: int = 29,
        in_channels: int = 34,
        input_length: int = 600,
        embedding_dim: int = 128,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        temporal_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = TemporalFCNEncoder(
            in_channels=in_channels,
            input_length=input_length,
            embedding_dim=embedding_dim,
            dropout=dropout,
            temporal_pooling=temporal_pooling,
        )
        if classifier_hidden_dim <= 0:
            self.classifier = nn.Linear(self.encoder.out_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder.out_dim, classifier_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(classifier_hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.forward(x)
        return features
