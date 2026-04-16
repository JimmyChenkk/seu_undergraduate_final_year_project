"""Method registry for the TE benchmark reproduction package."""

from __future__ import annotations

from .cdan import CDANMethod
from .coral import CORALMethod
from .dan import DANMethod
from .deepjdot import DeepJDOTMethod
from .dann import DANNMethod
from .rcta import RCTAMethod
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
            mcc_weight=float(loss.get("mcc_weight", 0.0)),
            mcc_temperature=float(loss.get("mcc_temperature", 2.0)),
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
            align_mean=bool(loss.get("align_mean", False)),
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
            adaptation_schedule=str(loss.get("adaptation_schedule", "warm_start")),
            adaptation_max_steps=int(loss.get("adaptation_max_steps", 1000)),
            adaptation_schedule_alpha=float(loss.get("adaptation_schedule_alpha", 10.0)),
            reg_dist=float(loss.get("reg_dist", 0.1)),
            reg_cl=float(loss.get("reg_cl", 1.0)),
            normalize_feature_cost=bool(loss.get("normalize_feature_cost", True)),
            transport_solver=str(loss.get("transport_solver", "sinkhorn")),
            sinkhorn_reg=float(loss.get("sinkhorn_reg", 0.05)),
            sinkhorn_num_iter_max=int(loss.get("sinkhorn_num_iter_max", 100)),
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
    if method_name == "rcta":
        cdan_loss = loss.get("cdan", {})
        dann_loss = loss.get("dann", {})
        deepjdot_loss = loss.get("deepjdot", {})
        augment_loss = loss.get("augment", {})
        return RCTAMethod(
            base_align=str(loss.get("base_align", "cdan")),
            use_mcc=bool(loss.get("use_mcc", True)),
            mcc_weight=float(loss.get("mcc_weight", 0.1)),
            mcc_temperature=float(loss.get("mcc_temperature", 2.0)),
            teacher_ema_decay=float(loss.get("teacher_ema_decay", 0.99)),
            teacher_temperature=float(loss.get("teacher_temperature", 1.5)),
            reliability_weights=loss.get("reliability_weights", {}),
            gate_score_floor=float(loss.get("gate_score_floor", 0.55)),
            gate_score_floor_start=(
                None if loss.get("gate_score_floor_start") is None else float(loss.get("gate_score_floor_start"))
            ),
            gate_score_floor_end=(
                None if loss.get("gate_score_floor_end") is None else float(loss.get("gate_score_floor_end"))
            ),
            gate_score_floor_schedule_steps=int(loss.get("gate_score_floor_schedule_steps", 1000)),
            gate_accept_ratio_start=float(loss.get("gate_accept_ratio_start", 0.2)),
            gate_accept_ratio_end=float(loss.get("gate_accept_ratio_end", 0.7)),
            gate_curriculum_steps=int(loss.get("gate_curriculum_steps", 1000)),
            pseudo_label_weight=float(loss.get("pseudo_label_weight", 0.2)),
            pseudo_warmup_steps=int(loss.get("pseudo_warmup_steps", 0)),
            prototype_weight=float(loss.get("prototype_weight", 0.1)),
            prototype_start_step=int(loss.get("prototype_start_step", 0)),
            prototype_warmup_steps=(
                None if loss.get("prototype_warmup_steps") is None else int(loss.get("prototype_warmup_steps"))
            ),
            prototype_separation_weight=float(loss.get("prototype_separation_weight", 0.1)),
            consistency_weight=float(loss.get("consistency_weight", 0.1)),
            consistency_start_step=int(loss.get("consistency_start_step", 0)),
            consistency_warmup_steps=(
                None if loss.get("consistency_warmup_steps") is None else int(loss.get("consistency_warmup_steps"))
            ),
            prototype_momentum=float(loss.get("prototype_momentum", 0.9)),
            prototype_separation_margin=float(loss.get("prototype_separation_margin", 0.2)),
            augment_kwargs={
                "weak_jitter_std": float(augment_loss.get("weak_jitter_std", 0.01)),
                "weak_scaling_std": float(augment_loss.get("weak_scaling_std", 0.01)),
                "strong_jitter_std": float(augment_loss.get("strong_jitter_std", 0.02)),
                "strong_scaling_std": float(augment_loss.get("strong_scaling_std", 0.02)),
                "strong_time_mask_ratio": float(augment_loss.get("strong_time_mask_ratio", 0.1)),
                "strong_channel_dropout_prob": float(augment_loss.get("strong_channel_dropout_prob", 0.1)),
            },
            cdan_kwargs={
                "adaptation_weight": float(cdan_loss.get("adaptation_weight", 0.2)),
                "adaptation_schedule": str(cdan_loss.get("adaptation_schedule", "warm_start")),
                "adaptation_max_steps": int(cdan_loss.get("adaptation_max_steps", 1000)),
                "adaptation_schedule_alpha": float(cdan_loss.get("adaptation_schedule_alpha", 10.0)),
                "grl_lambda": float(cdan_loss.get("grl_lambda", 1.0)),
                "grl_warm_start": bool(cdan_loss.get("grl_warm_start", True)),
                "grl_max_iters": int(cdan_loss.get("grl_max_iters", 1000)),
                "randomized": bool(cdan_loss.get("randomized", True)),
                "randomized_dim": int(cdan_loss.get("randomized_dim", 256)),
                "entropy_conditioning": bool(cdan_loss.get("entropy_conditioning", True)),
                "domain_hidden_dim": (
                    None if cdan_loss.get("domain_hidden_dim") is None else int(cdan_loss.get("domain_hidden_dim"))
                ),
                "domain_num_hidden_layers": int(cdan_loss.get("domain_num_hidden_layers", 2)),
            },
            dann_kwargs={
                "adaptation_weight": float(dann_loss.get("adaptation_weight", 0.5)),
                "adaptation_schedule": str(dann_loss.get("adaptation_schedule", "warm_start")),
                "adaptation_max_steps": int(dann_loss.get("adaptation_max_steps", 1000)),
                "adaptation_schedule_alpha": float(dann_loss.get("adaptation_schedule_alpha", 10.0)),
                "grl_lambda": float(dann_loss.get("grl_lambda", 1.0)),
                "grl_warm_start": bool(dann_loss.get("grl_warm_start", True)),
                "grl_max_iters": int(dann_loss.get("grl_max_iters", 1000)),
                "domain_hidden_dim": (
                    None if dann_loss.get("domain_hidden_dim") is None else int(dann_loss.get("domain_hidden_dim"))
                ),
                "domain_num_hidden_layers": int(dann_loss.get("domain_num_hidden_layers", 2)),
            },
            deepjdot_kwargs={
                "adaptation_weight": float(deepjdot_loss.get("adaptation_weight", 1.0)),
                "adaptation_schedule": str(deepjdot_loss.get("adaptation_schedule", "constant")),
                "adaptation_max_steps": int(deepjdot_loss.get("adaptation_max_steps", 1000)),
                "adaptation_schedule_alpha": float(deepjdot_loss.get("adaptation_schedule_alpha", 10.0)),
                "reg_dist": float(deepjdot_loss.get("reg_dist", 0.1)),
                "reg_cl": float(deepjdot_loss.get("reg_cl", 1.0)),
                "normalize_feature_cost": bool(deepjdot_loss.get("normalize_feature_cost", True)),
            },
            **shared_kwargs,
        )
    raise KeyError(f"Unsupported method: {method_name}")


__all__ = [
    "CDANMethod",
    "CORALMethod",
    "DANMethod",
    "DeepJDOTMethod",
    "DANNMethod",
    "RCTAMethod",
    "SourceOnlyMethod",
    "build_method",
]
