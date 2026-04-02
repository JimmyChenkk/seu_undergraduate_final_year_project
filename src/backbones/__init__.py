"""Backbone registry for TE domain adaptation experiments."""

from .fcn import ClassifierHead, FullyConvolutionalEncoder

__all__ = ["ClassifierHead", "FullyConvolutionalEncoder"]
