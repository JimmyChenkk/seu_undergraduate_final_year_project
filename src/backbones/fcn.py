"""Compact 1D backbones adapted from the public TEP benchmark code."""

from __future__ import annotations

import torch
from torch import nn


class FullyConvolutionalEncoder(nn.Module):
    """Simple FCN encoder for TEP signals shaped as ``(N, C, T)``."""

    def __init__(
        self,
        *,
        in_channels: int = 34,
        instance_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        norm = nn.InstanceNorm1d if instance_norm else nn.BatchNorm1d
        dropout_layer = nn.Identity() if dropout <= 0 else nn.Dropout(dropout)
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=9, padding="same"),
            norm(128),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Conv1d(128, 256, kernel_size=5, padding="same"),
            norm(256),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            norm(128),
            nn.ReLU(inplace=True),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).mean(dim=-1)


class TemporalPrototypeFCNEncoder(nn.Module):
    """FCN encoder with global and multi-segment temporal pooling.

    Besides the global GAP representation, this encoder pools the first 200,
    middle 200, last 200, and last 300 time steps from the convolutional feature
    map, then projects their concatenation to a compact embedding. It is meant
    for TP-WJDOT-style prototype costs while keeping the existing benchmark
    backbone family intact.
    """

    def __init__(
        self,
        *,
        in_channels: int = 34,
        instance_norm: bool = True,
        dropout: float = 0.1,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        norm = nn.InstanceNorm1d if instance_norm else nn.BatchNorm1d
        dropout_layer = nn.Identity() if dropout <= 0 else nn.Dropout(dropout)
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=9, padding="same"),
            norm(128),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Conv1d(128, 256, kernel_size=5, padding="same"),
            norm(256),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            norm(128),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * 5, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            dropout_layer,
        )
        self.out_dim = int(embedding_dim)

    @staticmethod
    def _pool_segment(features: torch.Tensor, start: int, end: int) -> torch.Tensor:
        length = int(features.shape[-1])
        start = max(0, min(int(start), length - 1))
        end = max(start + 1, min(int(end), length))
        return features[..., start:end].mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.main(x)
        length = int(features.shape[-1])
        middle_start = min(max(length // 2 - 100, 0), length - 1)
        pooled = torch.cat(
            [
                features.mean(dim=-1),
                self._pool_segment(features, 0, min(200, length)),
                self._pool_segment(features, middle_start, min(middle_start + 200, length)),
                self._pool_segment(features, max(length - 200, 0), length),
                self._pool_segment(features, max(length - 300, 0), length),
            ],
            dim=1,
        )
        return self.projection(pooled)


class ClassifierHead(nn.Module):
    """Small classification head shared by most methods."""

    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.main = nn.Linear(in_features, num_classes)
        else:
            dropout_layer = nn.Identity() if dropout <= 0 else nn.Dropout(dropout)
            self.main = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                dropout_layer,
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
