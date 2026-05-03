"""Minimal TEP OT/domain-adaptation experiment framework."""

from .data import ExperimentData, TEPDomainLoader
from .model import TEClassifier, TemporalFCNEncoder
from .train import run_experiment

__all__ = [
    "ExperimentData",
    "TEPDomainLoader",
    "TEClassifier",
    "TemporalFCNEncoder",
    "run_experiment",
]
