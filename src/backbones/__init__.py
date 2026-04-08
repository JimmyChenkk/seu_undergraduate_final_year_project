"""Backbone registry for TE domain adaptation experiments."""

from __future__ import annotations

from .fcn import ClassifierHead, FullyConvolutionalEncoder
from .transformer import PositionalEncoding, TransformerTimeSeriesEncoder


def build_backbone(
    *,
    name: str,
    in_channels: int,
    input_length: int,
    dropout: float,
    backbone_kwargs: dict | None = None,
):
    """Instantiate a configured encoder backbone."""

    config = dict(backbone_kwargs or {})
    normalized = str(name).strip().lower()
    if normalized == "fcn":
        return FullyConvolutionalEncoder(
            in_channels=in_channels,
            instance_norm=bool(config.get("instance_norm", True)),
            dropout=dropout,
        )
    if normalized == "transformer":
        return TransformerTimeSeriesEncoder(
            in_channels=in_channels,
            seq_len=int(config.get("seq_len", input_length)),
            head_dim=int(config.get("head_dim", 16)),
            nhead=int(config.get("nhead", 8)),
            dim_feedforward=int(config.get("dim_feedforward", 256)),
            dropout=dropout,
            num_layers=int(config.get("num_layers", 6)),
            reduction=str(config.get("reduction", "last")),
        )
    raise KeyError(f"Unsupported backbone: {name}")


__all__ = [
    "ClassifierHead",
    "FullyConvolutionalEncoder",
    "PositionalEncoding",
    "TransformerTimeSeriesEncoder",
    "build_backbone",
]
