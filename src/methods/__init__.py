"""Method registry for the TE benchmark reproduction package."""

from __future__ import annotations

from .cdan import CDANMethod
from .coral import CORALMethod
from .dan import DANMethod
from .deepjdot import DeepJDOTMethod
from .dann import DANNMethod
from .source_only import SourceOnlyMethod


def build_method(method_config, *, num_classes: int, in_channels: int, input_length: int, num_sources: int):
    """Instantiate one configured method."""

    method_name = str(method_config["method_name"]).lower()
    optimization = method_config.get("optimization", {})
    backbone = method_config.get("backbone", {})
    loss = method_config.get("loss", {})
    backbone_name = str(backbone.get("name", "fcn"))
    dropout = float(backbone.get("dropout", 0.1))
    classifier_hidden_dim = int(backbone.get("classifier_hidden_dim", 128))
    backbone_kwargs = {
        key: value
        for key, value in backbone.items()
        if key not in {"name", "dropout", "classifier_hidden_dim"}
    }

    shared_kwargs = {
        "num_classes": num_classes,
        "in_channels": in_channels,
        "input_length": input_length,
        "dropout": dropout,
        "classifier_hidden_dim": classifier_hidden_dim,
        "backbone_name": backbone_name,
        "backbone_kwargs": backbone_kwargs,
    }
    if method_name == "source_only":
        return SourceOnlyMethod(**shared_kwargs)
    if method_name == "cdan":
        return CDANMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            adaptation_schedule=str(loss.get("adaptation_schedule", "constant")),
            adaptation_max_steps=int(loss.get("adaptation_max_steps", 1000)),
            adaptation_schedule_alpha=float(loss.get("adaptation_schedule_alpha", 10.0)),
            grl_lambda=float(loss.get("grl_lambda", 1.0)),
            grl_warm_start=bool(loss.get("grl_warm_start", True)),
            grl_max_iters=int(loss.get("grl_max_iters", 1000)),
            randomized=bool(loss.get("randomized", False)),
            randomized_dim=int(loss.get("randomized_dim", 1024)),
            entropy_conditioning=bool(loss.get("entropy_conditioning", True)),
            domain_hidden_dim=(
                None if loss.get("domain_hidden_dim") is None else int(loss.get("domain_hidden_dim"))
            ),
            domain_num_hidden_layers=int(loss.get("domain_num_hidden_layers", 2)),
            **shared_kwargs,
        )
    if method_name == "coral":
        return CORALMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            adaptation_schedule=str(loss.get("adaptation_schedule", "constant")),
            adaptation_max_steps=int(loss.get("adaptation_max_steps", 1000)),
            adaptation_schedule_alpha=float(loss.get("adaptation_schedule_alpha", 10.0)),
            align_mean=bool(loss.get("align_mean", True)),
            normalize_covariance=bool(loss.get("normalize_covariance", True)),
            **shared_kwargs,
        )
    if method_name == "dan":
        kernel_scales = tuple(float(value) for value in loss.get("kernel_scales", [0.125, 0.25, 0.5, 1.0, 2.0]))
        return DANMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            adaptation_schedule=str(loss.get("adaptation_schedule", "constant")),
            adaptation_max_steps=int(loss.get("adaptation_max_steps", 1000)),
            adaptation_schedule_alpha=float(loss.get("adaptation_schedule_alpha", 10.0)),
            kernel_scales=kernel_scales,
            linear_mmd=bool(loss.get("linear_mmd", True)),
            **shared_kwargs,
        )
    if method_name == "deepjdot":
        return DeepJDOTMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 1.0)),
            adaptation_schedule=str(loss.get("adaptation_schedule", "constant")),
            adaptation_max_steps=int(loss.get("adaptation_max_steps", 1000)),
            adaptation_schedule_alpha=float(loss.get("adaptation_schedule_alpha", 10.0)),
            reg_dist=float(loss.get("reg_dist", 0.1)),
            reg_cl=float(loss.get("reg_cl", 1.0)),
            normalize_feature_cost=bool(loss.get("normalize_feature_cost", True)),
            **shared_kwargs,
        )
    if method_name == "dann":
        return DANNMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            adaptation_schedule=str(loss.get("adaptation_schedule", "constant")),
            adaptation_max_steps=int(loss.get("adaptation_max_steps", 1000)),
            adaptation_schedule_alpha=float(loss.get("adaptation_schedule_alpha", 10.0)),
            grl_lambda=float(loss.get("grl_lambda", 1.0)),
            grl_warm_start=bool(loss.get("grl_warm_start", True)),
            grl_max_iters=int(loss.get("grl_max_iters", 1000)),
            domain_hidden_dim=(
                None if loss.get("domain_hidden_dim") is None else int(loss.get("domain_hidden_dim"))
            ),
            domain_num_hidden_layers=int(loss.get("domain_num_hidden_layers", 2)),
            **shared_kwargs,
        )
    raise KeyError(f"Unsupported method: {method_name}")


__all__ = [
    "CDANMethod",
    "CORALMethod",
    "DANMethod",
    "DeepJDOTMethod",
    "DANNMethod",
    "SourceOnlyMethod",
    "build_method",
]
