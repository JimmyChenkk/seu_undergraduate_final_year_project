"""Method registry for the TE benchmark reproduction package."""

from __future__ import annotations

from .cdan import CDANMethod
from .coral import CORALMethod
from .dan import DANMethod
from .dann import DANNMethod
from .jdot import run_jdot_experiment
from .mfsan import MFSANMethod
from .source_only import SourceOnlyMethod


def build_method(method_config, *, num_classes: int, in_channels: int, num_sources: int):
    """Instantiate one configured method."""

    method_name = str(method_config["method_name"]).lower()
    optimization = method_config.get("optimization", {})
    backbone = method_config.get("backbone", {})
    loss = method_config.get("loss", {})
    dropout = float(backbone.get("dropout", 0.1))
    classifier_hidden_dim = int(backbone.get("classifier_hidden_dim", 128))

    shared_kwargs = {
        "num_classes": num_classes,
        "in_channels": in_channels,
        "dropout": dropout,
        "classifier_hidden_dim": classifier_hidden_dim,
    }
    if method_name == "source_only":
        return SourceOnlyMethod(**shared_kwargs)
    if method_name == "cdan":
        return CDANMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            grl_lambda=float(loss.get("grl_lambda", 1.0)),
            **shared_kwargs,
        )
    if method_name == "coral":
        return CORALMethod(adaptation_weight=float(loss.get("adaptation_weight", 0.5)), **shared_kwargs)
    if method_name == "dan":
        return DANMethod(adaptation_weight=float(loss.get("adaptation_weight", 0.5)), **shared_kwargs)
    if method_name == "dann":
        return DANNMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            grl_lambda=float(loss.get("grl_lambda", 1.0)),
            **shared_kwargs,
        )
    if method_name == "mfsan":
        return MFSANMethod(
            num_classes=num_classes,
            num_sources=num_sources,
            in_channels=in_channels,
            dropout=dropout,
            hidden_dim=int(backbone.get("hidden_dim", 128)),
            mmd_weight=float(loss.get("adaptation_weight", 0.5)),
            discrepancy_weight=float(loss.get("discrepancy_weight", 0.1)),
        )
    raise KeyError(f"Unsupported method: {method_name}")


__all__ = [
    "CDANMethod",
    "CORALMethod",
    "DANMethod",
    "DANNMethod",
    "MFSANMethod",
    "SourceOnlyMethod",
    "build_method",
    "run_jdot_experiment",
]
