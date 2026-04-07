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
