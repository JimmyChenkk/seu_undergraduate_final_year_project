"""Backbone registry for TE domain adaptation experiments."""

from __future__ import annotations

from .fcn import ClassifierHead, FullyConvolutionalEncoder


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
    raise KeyError(f"Unsupported backbone: {name}")


__all__ = [
    "ClassifierHead",
    "FullyConvolutionalEncoder",
    "build_backbone",
]
