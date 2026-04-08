"""Transformer-based backbone adapted from the public TEP notebook prototype."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Batch-first sinusoidal positional encoding."""

    def __init__(self, emb_size: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2, dtype=torch.float32) * (-math.log(10000.0) / emb_size)
        )
        encoding = torch.zeros(max_len, emb_size, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("pos_embedding", encoding.unsqueeze(0), persistent=False)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        sequence_length = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_embedding[:, :sequence_length, :])


class TransformerTimeSeriesEncoder(nn.Module):
    """Time-series transformer encoder that accepts ``(N, C, T)`` or ``(N, T, C)`` inputs."""

    def __init__(
        self,
        *,
        in_channels: int = 34,
        seq_len: int = 600,
        head_dim: int = 16,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_layers: int = 6,
        reduction: str = "last",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.reduction = str(reduction).strip().lower()
        if self.reduction not in {"last", "avg"}:
            raise ValueError(f"Unsupported transformer reduction: {reduction}")

        d_model = int(head_dim) * int(nhead)
        if d_model <= 0:
            raise ValueError("Transformer d_model must be positive.")

        self.input_projection = nn.Linear(in_channels, d_model)
        self.pos_encoding = PositionalEncoding(emb_size=d_model, dropout=dropout, max_len=seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        if x.shape[1] == self.in_channels:
            x = x.transpose(1, 2)
        elif x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, got shape {tuple(x.shape)}"
            )

        embeddings = self.pos_encoding(self.input_projection(x))
        hidden = self.transformer(embeddings)
        if self.reduction == "avg":
            return hidden.mean(dim=1)
        return hidden[:, -1, :]
