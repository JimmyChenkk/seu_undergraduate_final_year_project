"""Method registry for the TE benchmark reproduction package."""

from __future__ import annotations

from .cdan import CDANMethod
from .cdan_ts import CDANTSMethod
from .ccs_rpl_tc_cdan import CCSRPLTCCDANMethod
from .codats import CoDATSMethod
from .coral import CORALMethod
from .dan import DANMethod
from .deepjdot import (
    CBTPDeepJDOTMethod,
    CBTPUDeepJDOTMethod,
    DeepJDOTMethod,
    TPDeepJDOTMethod,
    TPUDeepJDOTMethod,
    UDeepJDOTMethod,
)
from .dann import DANNMethod
from .dsan import DSANMethod
from .raincoat import RAINCOATMethod
from .rcta import RCTAMethod
from .rpl_tc_cdan import RPLTCCDANMethod
from .source_only import SourceOnlyMethod
from .target_only import TargetOnlyMethod
from .tc_cdan import TCCDANMethod
from .wjdot import (
    CBTPJDOTMethod,
    CBTPWJDOTMethod,
    CACCSRWJDOTMethod,
    CCSRWJDOTFusionMethod,
    JDOTMethod,
    MSCBTPWJDOTMethod,
    PooledWJDOTMethod,
    SACCsrWJDOTTrainMethod,
    SourceAwareWJDOTMultiHeadMethod,
    SourceAwareWJDOTSharedHeadMethod,
    TPJDOTMethod,
    TPWJDOTMethod,
    WJDOTMethod,
)


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
    if method_name in {"target_only", "target_ref", "target_supervised_reference", "target_oracle_matched"}:
        return TargetOnlyMethod(**shared_kwargs)
    if method_name in {"cdan", "cdan_ts"}:
        method_cls = CDANTSMethod if method_name == "cdan_ts" else CDANMethod
        return method_cls(
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
    if method_name == "codats":
        return CoDATSMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            adaptation_schedule=str(loss.get("adaptation_schedule", "warm_start")),
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
    if method_name == "dsan":
        fix_sigma = loss.get("fix_sigma")
        return DSANMethod(
            adaptation_weight=float(loss.get("adaptation_weight", 0.5)),
            adaptation_schedule=str(loss.get("adaptation_schedule", "warm_start")),
            adaptation_max_steps=int(loss.get("adaptation_max_steps", 1000)),
            adaptation_schedule_alpha=float(loss.get("adaptation_schedule_alpha", 10.0)),
            kernel_mul=float(loss.get("kernel_mul", 2.0)),
            kernel_num=int(loss.get("kernel_num", 5)),
            fix_sigma=None if fix_sigma is None else float(fix_sigma),
            **shared_kwargs,
        )
    if method_name == "raincoat":
        raincoat_loss = loss.get("raincoat", loss)
        return RAINCOATMethod(
            fourier_modes=int(raincoat_loss.get("fourier_modes", backbone.get("fourier_modes", 64))),
            mid_channels=int(raincoat_loss.get("mid_channels", backbone.get("mid_channels", 64))),
            final_out_channels=int(
                raincoat_loss.get("final_out_channels", backbone.get("final_out_channels", 128))
            ),
            kernel_size=int(raincoat_loss.get("kernel_size", backbone.get("kernel_size", 5))),
            stride=int(raincoat_loss.get("stride", backbone.get("stride", 1))),
            features_len=int(raincoat_loss.get("features_len", backbone.get("features_len", 1))),
            classifier_temperature=float(raincoat_loss.get("classifier_temperature", 0.1)),
            sinkhorn_weight=float(raincoat_loss.get("sinkhorn_weight", 0.5)),
            reconstruction_weight=float(raincoat_loss.get("reconstruction_weight", 1e-4)),
            sinkhorn_epsilon=float(raincoat_loss.get("sinkhorn_epsilon", 0.05)),
            sinkhorn_max_iter=int(raincoat_loss.get("sinkhorn_max_iter", 50)),
            sinkhorn_threshold=float(raincoat_loss.get("sinkhorn_threshold", 1e-3)),
            reconstruction_reduction=str(raincoat_loss.get("reconstruction_reduction", "mean")),
            **shared_kwargs,
        )
    if method_name in {
        "deepjdot",
        "u_deepjdot",
        "tp_deepjdot",
        "cbtp_deepjdot",
        "tpu_deepjdot",
        "cbtpu_deepjdot",
    }:
        method_cls = {
            "deepjdot": DeepJDOTMethod,
            "u_deepjdot": UDeepJDOTMethod,
            "tp_deepjdot": TPDeepJDOTMethod,
            "cbtp_deepjdot": CBTPDeepJDOTMethod,
            "tpu_deepjdot": TPUDeepJDOTMethod,
            "cbtpu_deepjdot": CBTPUDeepJDOTMethod,
        }[method_name]
        augment_loss = loss.get("augment", {})
        deepjdot_kwargs = {
            "adaptation_weight": float(loss.get("adaptation_weight", 1.0)),
            "adaptation_schedule": str(loss.get("adaptation_schedule", "constant")),
            "adaptation_max_steps": int(loss.get("adaptation_max_steps", 1000)),
            "adaptation_schedule_alpha": float(loss.get("adaptation_schedule_alpha", 10.0)),
            "reg_dist": float(loss.get("reg_dist", 0.1)),
            "reg_cl": float(loss.get("reg_cl", 1.0)),
            "normalize_feature_cost": bool(loss.get("normalize_feature_cost", False)),
            "transport_solver": str(loss.get("transport_solver", "emd")),
            "sinkhorn_reg": float(loss.get("sinkhorn_reg", 0.05)),
            "sinkhorn_num_iter_max": int(loss.get("sinkhorn_num_iter_max", 100)),
            "uot_tau_s": float(loss.get("uot_tau_s", loss.get("unbalanced_tau_s", 1.0))),
            "uot_tau_t": float(loss.get("uot_tau_t", loss.get("unbalanced_tau_t", 1.0))),
            **shared_kwargs,
        }
        if method_name in {"u_deepjdot", "tpu_deepjdot", "cbtpu_deepjdot"}:
            deepjdot_kwargs.update(
                {
                    "unbalanced_transport": bool(loss.get("unbalanced_transport", True)),
                    "use_ce_cost_for_plan": bool(
                        loss.get("use_ce_cost_for_plan", method_name in {"tpu_deepjdot", "cbtpu_deepjdot"})
                    ),
                }
            )
        if method_name in {"tp_deepjdot", "cbtp_deepjdot"}:
            deepjdot_kwargs.update(
                {
                    "prototype_weight": float(loss.get("prototype_weight", 0.02)),
                    "prototype_source_weight": float(loss.get("prototype_source_weight", 0.25)),
                    "prototype_target_weight": float(loss.get("prototype_target_weight", 1.0)),
                    "prototype_start_step": int(loss.get("prototype_start_step", 1200)),
                    "prototype_warmup_steps": int(loss.get("prototype_warmup_steps", 1600)),
                    "prototype_distance": str(loss.get("prototype_distance", "normalized_l2")),
                    "prototype_confidence_threshold": float(
                        loss.get("prototype_confidence_threshold", 0.0)
                    ),
                    "prototype_target_confidence_power": float(
                        loss.get("prototype_target_confidence_power", 1.0)
                    ),
                    "prototype_probability_temperature": float(
                        loss.get("prototype_probability_temperature", 1.0)
                    ),
                    "prototype_class_balance": bool(loss.get("prototype_class_balance", False)),
                    "prototype_class_balance_clip_min": float(
                        loss.get("prototype_class_balance_clip_min", 0.0)
                    ),
                    "prototype_class_balance_clip_max": float(
                        loss.get("prototype_class_balance_clip_max", 0.0)
                    ),
                }
            )
        if method_name in {"tpu_deepjdot", "cbtpu_deepjdot"}:
            deepjdot_kwargs.update(
                {
                    "prototype_cost_weight": float(loss.get("prototype_cost_weight", 0.015)),
                    "prototype_start_step": int(loss.get("prototype_start_step", 500)),
                    "prototype_warmup_steps": int(loss.get("prototype_warmup_steps", 800)),
                    "prototype_distance": str(loss.get("prototype_distance", "normalized_l2")),
                    "prototype_momentum": float(loss.get("prototype_momentum", 0.95)),
                    "prototype_relative_min": float(loss.get("prototype_relative_min", -1.0)),
                    "prototype_relative_max": float(loss.get("prototype_relative_max", 3.0)),
                    "temporal_cost_weight": float(loss.get("temporal_cost_weight", 0.01)),
                    "temporal_start_step": (
                        None if loss.get("temporal_start_step") is None else int(loss.get("temporal_start_step"))
                    ),
                    "temporal_warmup_steps": int(loss.get("temporal_warmup_steps", 800)),
                    "supcon_weight": float(loss.get("supcon_weight", 0.04)),
                    "supcon_temperature": float(loss.get("supcon_temperature", 0.10)),
                    "source_warmup_steps": int(loss.get("source_warmup_steps", 500)),
                    "supcon_warmup_only": bool(loss.get("supcon_warmup_only", True)),
                    "alignment_start_step": (
                        None if loss.get("alignment_start_step") is None else int(loss.get("alignment_start_step"))
                    ),
                }
            )
        if method_name == "cbtp_deepjdot":
            deepjdot_kwargs.update(
                {
                    "pseudo_weight": float(loss.get("pseudo_weight", 0.015)),
                    "pseudo_start_step": (
                        None if loss.get("pseudo_start_step") is None else int(loss.get("pseudo_start_step"))
                    ),
                    "pseudo_warmup_steps": int(loss.get("pseudo_warmup_steps", 1000)),
                    "tau_start": float(loss.get("tau_start", 0.97)),
                    "tau_end": float(loss.get("tau_end", 0.92)),
                    "tau_steps": int(loss.get("tau_steps", 1200)),
                    "class_balance_clip_min": float(loss.get("class_balance_clip_min", 0.5)),
                    "class_balance_clip_max": float(loss.get("class_balance_clip_max", 2.5)),
                    "pseudo_confidence_power": float(loss.get("pseudo_confidence_power", 1.0)),
                    "pseudo_min_acceptance": float(loss.get("pseudo_min_acceptance", 0.0)),
                    "pseudo_min_classes": int(loss.get("pseudo_min_classes", 1)),
                    "target_im_weight": float(loss.get("target_im_weight", 0.0)),
                    "target_im_start_step": (
                        None
                        if loss.get("target_im_start_step") is None
                        else int(loss.get("target_im_start_step"))
                    ),
                    "target_im_warmup_steps": int(loss.get("target_im_warmup_steps", 1000)),
                    "target_im_temperature": float(loss.get("target_im_temperature", 1.0)),
                    "target_im_entropy_weight": float(loss.get("target_im_entropy_weight", 1.0)),
                    "target_im_diversity_weight": float(loss.get("target_im_diversity_weight", 1.0)),
                }
            )
        if method_name == "cbtpu_deepjdot":
            deepjdot_kwargs.update(
                {
                    "teacher_ema_decay": float(loss.get("teacher_ema_decay", 0.995)),
                    "teacher_temperature": float(loss.get("teacher_temperature", 1.0)),
                    "proto_temperature": float(loss.get("proto_temperature", 0.20)),
                    "q_ot_power": float(loss.get("q_ot_power", 1.0)),
                    "q_cls_power": float(loss.get("q_cls_power", 1.0)),
                    "q_proto_power": float(loss.get("q_proto_power", 1.0)),
                    "pseudo_weight": float(loss.get("pseudo_weight", 0.015)),
                    "pseudo_start_step": int(loss.get("pseudo_start_step", 800)),
                    "pseudo_warmup_steps": int(loss.get("pseudo_warmup_steps", 1000)),
                    "tau_start": float(loss.get("tau_start", 0.95)),
                    "tau_end": float(loss.get("tau_end", 0.85)),
                    "tau_steps": int(loss.get("tau_steps", 1500)),
                    "pseudo_max_acceptance": float(loss.get("pseudo_max_acceptance", 0.0)),
                    "q_ot_entropy_threshold": float(loss.get("q_ot_entropy_threshold", 0.70)),
                    "js_threshold": float(loss.get("js_threshold", 0.08)),
                    "consistency_weight": float(loss.get("consistency_weight", 0.02)),
                    "consistency_start_step": int(loss.get("consistency_start_step", 800)),
                    "consistency_warmup_steps": int(loss.get("consistency_warmup_steps", 1000)),
                    "consistency_mid_tau": float(loss.get("consistency_mid_tau", 0.55)),
                    "logit_adjustment_eta": float(loss.get("logit_adjustment_eta", 0.10)),
                    "infomax_weight": float(loss.get("infomax_weight", 0.0)),
                    "infomax_start_ratio": float(loss.get("infomax_start_ratio", 0.70)),
                    "infomax_start_step": (
                        None if loss.get("infomax_start_step") is None else int(loss.get("infomax_start_step"))
                    ),
                    "infomax_warmup_steps": int(loss.get("infomax_warmup_steps", 500)),
                    "augment_kwargs": {
                        "weak_jitter_std": float(augment_loss.get("weak_jitter_std", 0.006)),
                        "weak_scaling_std": float(augment_loss.get("weak_scaling_std", 0.006)),
                        "strong_jitter_std": float(augment_loss.get("strong_jitter_std", 0.014)),
                        "strong_scaling_std": float(augment_loss.get("strong_scaling_std", 0.014)),
                        "strong_time_mask_ratio": float(augment_loss.get("strong_time_mask_ratio", 0.06)),
                        "strong_channel_dropout_prob": float(
                            augment_loss.get("strong_channel_dropout_prob", 0.05)
                        ),
                    },
                }
            )
        return method_cls(**deepjdot_kwargs)
    if method_name in {
        "jdot",
        "otda",
        "tp_jdot",
        "cbtp_jdot",
        "wjdot",
        "pooled_wjdot",
        "sourceaware_wjdot_shared_head",
        "sourceaware_wjdot_multi_head",
        "sa_ccsr_wjdot_train",
        "ca_ccsr_wjdot",
        "tp_wjdot",
        "cbtp_wjdot",
        "ms_cbtp_wjdot",
        "ccsr_wjdot_fusion",
    }:
        method_cls = {
            "jdot": JDOTMethod,
            "otda": JDOTMethod,
            "tp_jdot": TPJDOTMethod,
            "cbtp_jdot": CBTPJDOTMethod,
            "wjdot": WJDOTMethod,
            "pooled_wjdot": PooledWJDOTMethod,
            "sourceaware_wjdot_shared_head": SourceAwareWJDOTSharedHeadMethod,
            "sourceaware_wjdot_multi_head": SourceAwareWJDOTMultiHeadMethod,
            "sa_ccsr_wjdot_train": SACCsrWJDOTTrainMethod,
            "ca_ccsr_wjdot": CACCSRWJDOTMethod,
            "tp_wjdot": TPWJDOTMethod,
            "cbtp_wjdot": CBTPWJDOTMethod,
            "ms_cbtp_wjdot": MSCBTPWJDOTMethod,
            "ccsr_wjdot_fusion": CCSRWJDOTFusionMethod,
        }[method_name]
        if method_name in {"tp_jdot", "tp_wjdot"}:
            default_prototype_weight = 0.04
        elif method_name in {"cbtp_jdot", "cbtp_wjdot", "ms_cbtp_wjdot"}:
            default_prototype_weight = 0.2
        else:
            default_prototype_weight = 0.0
        source_balance_default = method_name in {
            "wjdot",
            "pooled_wjdot",
            "sourceaware_wjdot_shared_head",
            "sourceaware_wjdot_multi_head",
            "sa_ccsr_wjdot_train",
            "ca_ccsr_wjdot",
            "tp_wjdot",
            "cbtp_wjdot",
            "ms_cbtp_wjdot",
            "ccsr_wjdot_fusion",
        }
        wjdot_kwargs = {
            "adaptation_weight": float(loss.get("adaptation_weight", 1.0)),
            "source_ce_weight": float(loss.get("source_ce_weight", 1.0)),
            "feature_weight": float(loss.get("feature_weight", 0.1)),
            "label_weight": float(loss.get("label_weight", 1.0)),
            "prototype_weight": float(loss.get("prototype_weight", default_prototype_weight)),
            "prototype_weight_in_coupling": (
                None
                if loss.get("prototype_weight_in_coupling") is None
                else float(loss.get("prototype_weight_in_coupling"))
            ),
            "prototype_weight_residual": (
                None
                if loss.get("prototype_weight_residual") is None
                else float(loss.get("prototype_weight_residual"))
            ),
            "prototype_in_coupling": loss.get("prototype_in_coupling", True),
            "prototype_mode": str(loss.get("prototype_mode", loss.get("base_tp_mode", "legacy_pairwise"))),
            "prototype_distance": str(loss.get("prototype_distance", "normalized_l2")),
            "prototype_cost_clip": loss.get("prototype_cost_clip"),
            "ot_class_entropy_gate": bool(loss.get("ot_class_entropy_gate", False)),
            "min_class_transport_mass": float(loss.get("min_class_transport_mass", 1e-4)),
            "class_mass_weight": str(loss.get("class_mass_weight", "sqrt")),
            "class_mass_weight_min": float(loss.get("class_mass_weight_min", 0.0)),
            "class_mass_weight_max": float(loss.get("class_mass_weight_max", 1.0)),
            "relative_margin": float(loss.get("margin", loss.get("relative_margin", 0.05))),
            "relative_margin_temperature": float(loss.get("relative_margin_temperature", 0.05)),
            "relative_cost_min": float(loss.get("relative_cost_min", 0.0)),
            "relative_cost_max": float(loss.get("relative_cost_max", 3.0)),
            "normalize_costs": bool(loss.get("normalize_costs", True)),
            "transport_solver": str(loss.get("transport_solver", "sinkhorn")),
            "sinkhorn_reg": float(loss.get("sinkhorn_reg", 0.05)),
            "sinkhorn_num_iter_max": int(loss.get("sinkhorn_num_iter_max", 100)),
            "use_source_class_balance": bool(
                loss.get("use_source_class_balance", source_balance_default)
            ),
            "pseudo_weight": float(loss.get("pseudo_weight", 0.15)),
            "consistency_weight": float(loss.get("consistency_weight", 0.05)),
            "class_balance_clip_min": float(loss.get("class_balance_clip_min", 0.0)),
            "class_balance_clip_max": float(loss.get("class_balance_clip_max", 0.0)),
            "target_label_assist_weight": float(loss.get("target_label_assist_weight", 0.0)),
            "target_label_assist_start_step": (
                None
                if loss.get("target_label_assist_start_step") is None
                else int(loss.get("target_label_assist_start_step"))
            ),
            "target_label_assist_warmup_steps": int(loss.get("target_label_assist_warmup_steps", 1)),
            "target_label_assist_class_balance": bool(loss.get("target_label_assist_class_balance", True)),
            "tau_start": float(loss.get("tau_start", 0.95)),
            "tau_end": float(loss.get("tau_end", 0.70)),
            "tau_steps": int(loss.get("tau_steps", 1000)),
            "alignment_start_step": int(loss.get("alignment_start_step", 0)),
            "alignment_ramp_steps": int(loss.get("alignment_ramp_steps", 1)),
            "pseudo_start_step": (
                None if loss.get("pseudo_start_step") is None else int(loss.get("pseudo_start_step"))
            ),
            "prototype_start_step": (
                None if loss.get("prototype_start_step") is None else int(loss.get("prototype_start_step"))
            ),
            "prototype_warmup_steps": int(loss.get("prototype_warmup_steps", 1)),
            "prototype_confidence_threshold": (
                None
                if loss.get("prototype_confidence_threshold") is None
                else float(loss.get("prototype_confidence_threshold"))
            ),
            "ot_feature_normalize": bool(loss.get("ot_feature_normalize", True)),
            "ot_source_stop_gradient": bool(loss.get("ot_source_stop_gradient", True)),
            "embedding_norm_weight": float(loss.get("embedding_norm_weight", 0.0)),
            **shared_kwargs,
        }
        if method_name in {"tp_jdot", "cbtp_jdot", "tp_wjdot", "cbtp_wjdot", "ms_cbtp_wjdot"}:
            wjdot_kwargs["prototype_mode"] = str(
                loss.get("prototype_mode", loss.get("base_tp_mode", "tp_barycentric"))
            )
            wjdot_kwargs["prototype_in_coupling"] = loss.get("prototype_in_coupling", False)
            wjdot_kwargs["ot_class_entropy_gate"] = bool(loss.get("ot_class_entropy_gate", True))
        if method_name in {"cbtp_jdot", "cbtp_wjdot", "ms_cbtp_wjdot"}:
            wjdot_kwargs["base_tp_mode"] = str(loss.get("base_tp_mode", wjdot_kwargs["prototype_mode"]))
        if method_name == "ms_cbtp_wjdot":
            wjdot_kwargs.update(
                {
                    "source_weight_temperature": float(loss.get("source_weight_temperature", 1.0)),
                    "class_weight_temperature": float(loss.get("class_weight_temperature", 1.0)),
                    "negative_gate_threshold": float(loss.get("negative_gate_threshold", 0.05)),
                    "negative_gate_floor": float(loss.get("negative_gate_floor", 0.1)),
                    "ms_weighting_mode": str(loss.get("ms_weighting_mode", "global_alpha")),
                    "class_alpha_only": bool(loss.get("class_alpha_only", False)),
                    "class_alpha_top_m_sources": int(loss.get("class_alpha_top_m_sources", 0)),
                    "class_alpha_renormalize_top_m": bool(
                        loss.get("class_alpha_renormalize_top_m", True)
                    ),
                    "class_unbalanced_transport": bool(loss.get("class_unbalanced_transport", False)),
                    "unbalanced_sinkhorn_reg_m": float(loss.get("unbalanced_sinkhorn_reg_m", 1.0)),
                    "target_label_calibration": bool(loss.get("target_label_calibration", False)),
                    "target_label_assisted_source_weights": bool(
                        loss.get("target_label_assisted_source_weights", False)
                    ),
                }
            )
        if method_name in {
            "sourceaware_wjdot_shared_head",
            "sourceaware_wjdot_multi_head",
            "sa_ccsr_wjdot_train",
            "ca_ccsr_wjdot",
        }:
            wjdot_kwargs.update(
                {
                    "num_sources": int(num_sources),
                    "source_alpha_mode": str(loss.get("source_alpha_mode", "uniform")),
                    "source_alpha_temperature": float(loss.get("source_alpha_temperature", 1.0)),
                    "source_ce_reduction": str(loss.get("source_ce_reduction", "mean")),
                    "class_transport_normalize": bool(loss.get("class_transport_normalize", True)),
                    "sample_outlier_downweight": bool(loss.get("sample_outlier_downweight", False)),
                    "sample_weight_min": float(loss.get("sample_weight_min", 0.3)),
                    "sample_weight_max": float(loss.get("sample_weight_max", 1.0)),
                }
            )
        if method_name in {"sa_ccsr_wjdot_train", "ca_ccsr_wjdot"}:
            wjdot_kwargs.update(
                {
                    "reliability_start_ratio": float(loss.get("reliability_start_ratio", 0.30)),
                    "reliability_ramp_ratio": float(loss.get("reliability_ramp_ratio", 0.20)),
                    "reliability_total_steps": int(loss.get("reliability_total_steps", 1000)),
                    "reliability_start_step": (
                        None
                        if loss.get("reliability_start_step") is None
                        else int(loss.get("reliability_start_step"))
                    ),
                    "reliability_ramp_steps": (
                        None
                        if loss.get("reliability_ramp_steps") is None
                        else int(loss.get("reliability_ramp_steps"))
                    ),
                    "class_temperature": (
                        None
                        if loss.get("class_temperature") is None
                        else float(loss.get("class_temperature"))
                    ),
                    "T_class": None if loss.get("T_class") is None else float(loss.get("T_class")),
                    "top_m_per_class": (
                        None if loss.get("top_m_per_class") is None else int(loss.get("top_m_per_class"))
                    ),
                    "floor": None if loss.get("floor") is None else float(loss.get("floor")),
                    "w_proto": float(loss.get("w_proto", 0.30 if method_name == "ca_ccsr_wjdot" else 0.35)),
                    "w_ot": float(loss.get("w_ot", 0.35)),
                    "w_entropy": float(loss.get("w_entropy", 0.20)),
                    "w_source_error": float(
                        loss.get("w_source_error", 0.15 if method_name == "ca_ccsr_wjdot" else 0.10)
                    ),
                    "tau_proto": float(loss.get("tau_proto", 0.85)),
                    "min_proto_samples": int(loss.get("min_proto_samples", 3)),
                }
            )
        if method_name == "ca_ccsr_wjdot":
            wjdot_kwargs.update(
                {
                    "lambda_adv": float(loss.get("lambda_adv", loss.get("codats_adaptation_weight", 0.5))),
                    "lambda_ot": float(loss.get("lambda_ot", 0.10)),
                    "lambda_ccsr": float(loss.get("lambda_ccsr", 0.10)),
                    "lambda_teacher": float(loss.get("lambda_teacher", 0.05)),
                    "teacher_temperature": float(loss.get("teacher_temperature", 1.0)),
                    "teacher_anchor_mode": str(loss.get("teacher_anchor_mode", "kl")),
                    "teacher_requires_checkpoint": bool(loss.get("teacher_requires_checkpoint", True)),
                    "teacher_start_step": int(loss.get("teacher_start_step", 0)),
                    "teacher_ramp_steps": int(loss.get("teacher_ramp_steps", 1)),
                    "domain_adaptation_schedule": str(loss.get("domain_adaptation_schedule", "warm_start")),
                    "domain_adaptation_max_steps": int(loss.get("domain_adaptation_max_steps", 1000)),
                    "domain_adaptation_schedule_alpha": float(
                        loss.get("domain_adaptation_schedule_alpha", 10.0)
                    ),
                    "grl_lambda": float(loss.get("grl_lambda", 1.0)),
                    "grl_warm_start": bool(loss.get("grl_warm_start", True)),
                    "grl_max_iters": int(loss.get("grl_max_iters", 1000)),
                    "domain_hidden_dim": (
                        None if loss.get("domain_hidden_dim") is None else int(loss.get("domain_hidden_dim"))
                    ),
                    "domain_num_hidden_layers": int(loss.get("domain_num_hidden_layers", 2)),
                }
            )
        return method_cls(**wjdot_kwargs)
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
    if method_name in {"tc_cdan", "rpl_tc_cdan", "ccs_rpl_tc_cdan"}:
        augment_loss = loss.get("augment", {})
        tc_kwargs = {
            "adaptation_weight": float(loss.get("adaptation_weight", 0.25)),
            "adaptation_schedule": str(loss.get("adaptation_schedule", "warm_start")),
            "adaptation_max_steps": int(loss.get("adaptation_max_steps", 1200)),
            "adaptation_schedule_alpha": float(loss.get("adaptation_schedule_alpha", 10.0)),
            "grl_lambda": float(loss.get("grl_lambda", 1.0)),
            "grl_warm_start": bool(loss.get("grl_warm_start", True)),
            "grl_max_iters": int(loss.get("grl_max_iters", 1200)),
            "randomized": bool(loss.get("randomized", True)),
            "randomized_dim": int(loss.get("randomized_dim", 256)),
            "entropy_conditioning": bool(loss.get("entropy_conditioning", True)),
            "domain_hidden_dim": (
                None if loss.get("domain_hidden_dim") is None else int(loss.get("domain_hidden_dim"))
            ),
            "domain_num_hidden_layers": int(loss.get("domain_num_hidden_layers", 2)),
            "teacher_ema_decay": float(loss.get("teacher_ema_decay", 0.995)),
            "teacher_temperature": float(loss.get("teacher_temperature", 1.0)),
            "consistency_weight": float(loss.get("consistency_weight", 0.05)),
            "consistency_start_step": int(loss.get("consistency_start_step", 0)),
            "consistency_warmup_steps": int(loss.get("consistency_warmup_steps", 1000)),
            "consistency_loss": str(loss.get("consistency_loss", "kl")),
            "augment_kwargs": {
                "weak_jitter_std": float(augment_loss.get("weak_jitter_std", 0.006)),
                "weak_scaling_std": float(augment_loss.get("weak_scaling_std", 0.006)),
                "strong_jitter_std": float(augment_loss.get("strong_jitter_std", 0.014)),
                "strong_scaling_std": float(augment_loss.get("strong_scaling_std", 0.014)),
                "strong_time_mask_ratio": float(augment_loss.get("strong_time_mask_ratio", 0.06)),
                "strong_channel_dropout_prob": float(augment_loss.get("strong_channel_dropout_prob", 0.05)),
            },
        }
        if method_name == "tc_cdan":
            return TCCDANMethod(**tc_kwargs, **shared_kwargs)
        rpl_kwargs = {
            "pseudo_weight": float(loss.get("pseudo_weight", 0.08)),
            "pseudo_start_step": int(loss.get("pseudo_start_step", 800)),
            "pseudo_warmup_steps": int(loss.get("pseudo_warmup_steps", 600)),
            "pseudo_confidence_threshold": float(loss.get("pseudo_confidence_threshold", 0.75)),
            "pseudo_entropy_threshold": float(loss.get("pseudo_entropy_threshold", 0.55)),
            "pseudo_max_per_class": int(loss.get("pseudo_max_per_class", 4)),
            "pseudo_use_reliability_weighting": bool(loss.get("pseudo_use_reliability_weighting", True)),
            "reliability_weights": loss.get("reliability_weights", {}),
        }
        if method_name == "rpl_tc_cdan":
            return RPLTCCDANMethod(**rpl_kwargs, **tc_kwargs, **shared_kwargs)
        return CCSRPLTCCDANMethod(
            prototype_weight=float(loss.get("prototype_weight", 0.05)),
            prototype_start_step=int(loss.get("prototype_start_step", 1200)),
            prototype_warmup_steps=int(loss.get("prototype_warmup_steps", 800)),
            prototype_momentum=float(loss.get("prototype_momentum", 0.95)),
            prototype_min_target_per_class=int(loss.get("prototype_min_target_per_class", 1)),
            target_prototype_blend=float(loss.get("target_prototype_blend", 0.35)),
            class_separation_weight=float(loss.get("class_separation_weight", 0.04)),
            class_separation_margin=float(loss.get("class_separation_margin", 0.20)),
            **rpl_kwargs,
            **tc_kwargs,
            **shared_kwargs,
        )
    if method_name == "rcta":
        cdan_loss = loss.get("cdan", {})
        dann_loss = loss.get("dann", {})
        dan_loss = loss.get("dan", {})
        deepjdot_loss = loss.get("deepjdot", {})
        augment_loss = loss.get("augment", {})
        training_context = method_config.get("training_context", {})
        kernel_scales = tuple(float(value) for value in dan_loss.get("kernel_scales", [0.125, 0.25, 0.5, 1.0, 2.0]))
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
            gate_balance_mode=str(loss.get("gate_balance_mode", "per_class_ratio")),
            gate_max_class_fraction=float(loss.get("gate_max_class_fraction", 1.0)),
            reliable_score_floor=(
                None if loss.get("reliable_score_floor") is None else float(loss.get("reliable_score_floor"))
            ),
            semi_reliable_score_floor=(
                None if loss.get("semi_reliable_score_floor") is None else float(loss.get("semi_reliable_score_floor"))
            ),
            semi_reliable_consistency_weight=float(loss.get("semi_reliable_consistency_weight", 0.5)),
            unreliable_entropy_weight=float(loss.get("unreliable_entropy_weight", 0.05)),
            pseudo_label_weight=float(loss.get("pseudo_label_weight", 0.2)),
            pseudo_warmup_steps=int(loss.get("pseudo_warmup_steps", 0)),
            pseudo_use_reliability_weighting=bool(loss.get("pseudo_use_reliability_weighting", True)),
            pseudo_confidence_power=float(loss.get("pseudo_confidence_power", 1.0)),
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
            consistency_gate_only=bool(loss.get("consistency_gate_only", False)),
            consistency_reliability_power=float(loss.get("consistency_reliability_power", 1.0)),
            alignment_start_step=int(loss.get("alignment_start_step", 0)),
            alignment_use_reliable_only=bool(loss.get("alignment_use_reliable_only", True)),
            prototype_momentum=float(loss.get("prototype_momentum", 0.9)),
            prototype_separation_margin=float(loss.get("prototype_separation_margin", 0.2)),
            target_prototype_update_mode=str(loss.get("target_prototype_update_mode", "all")),
            target_prototype_blend_start=float(loss.get("target_prototype_blend_start", 1.0)),
            target_prototype_blend_end=float(loss.get("target_prototype_blend_end", 1.0)),
            target_prototype_blend_schedule_steps=int(loss.get("target_prototype_blend_schedule_steps", 1)),
            multi_source_weighting=bool(loss.get("multi_source_weighting", False)),
            source_weight_temperature=float(loss.get("source_weight_temperature", 4.0)),
            source_weight_momentum=float(loss.get("source_weight_momentum", 0.0)),
            source_weight_floor=float(loss.get("source_weight_floor", 0.0)),
            source_weight_proto=float(loss.get("source_weight_proto", 1.0)),
            source_weight_confidence=float(loss.get("source_weight_confidence", 0.0)),
            source_weight_coverage=float(loss.get("source_weight_coverage", 0.0)),
            hybrid_aligners=[str(item) for item in loss.get("hybrid_aligners", ["dann", "cdan", "dan"])],
            track_detailed_metrics=bool(training_context.get("track_detailed_metrics", False) or num_sources > 1),
            hybrid_alignment_weights={
                str(key): float(value)
                for key, value in (loss.get("hybrid_alignment_weights") or {}).items()
            },
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
            dan_kwargs={
                "adaptation_weight": float(dan_loss.get("adaptation_weight", 0.1)),
                "adaptation_schedule": str(dan_loss.get("adaptation_schedule", "warm_start")),
                "adaptation_max_steps": int(dan_loss.get("adaptation_max_steps", 1000)),
                "adaptation_schedule_alpha": float(dan_loss.get("adaptation_schedule_alpha", 10.0)),
                "kernel_scales": kernel_scales,
                "linear_mmd": bool(dan_loss.get("linear_mmd", True)),
            },
            deepjdot_kwargs={
                "adaptation_weight": float(deepjdot_loss.get("adaptation_weight", 1.0)),
                "adaptation_schedule": str(deepjdot_loss.get("adaptation_schedule", "constant")),
                "adaptation_max_steps": int(deepjdot_loss.get("adaptation_max_steps", 1000)),
                "adaptation_schedule_alpha": float(deepjdot_loss.get("adaptation_schedule_alpha", 10.0)),
                "reg_dist": float(deepjdot_loss.get("reg_dist", 0.1)),
                "reg_cl": float(deepjdot_loss.get("reg_cl", 1.0)),
                "normalize_feature_cost": bool(deepjdot_loss.get("normalize_feature_cost", False)),
                "transport_solver": str(deepjdot_loss.get("transport_solver", "emd")),
                "sinkhorn_reg": float(deepjdot_loss.get("sinkhorn_reg", 0.05)),
                "sinkhorn_num_iter_max": int(deepjdot_loss.get("sinkhorn_num_iter_max", 100)),
            },
            **shared_kwargs,
        )
    raise KeyError(f"Unsupported method: {method_name}")


__all__ = [
    "CDANMethod",
    "CDANTSMethod",
    "CCSRPLTCCDANMethod",
    "CBTPDeepJDOTMethod",
    "CBTPUDeepJDOTMethod",
    "CoDATSMethod",
    "CORALMethod",
    "DANMethod",
    "DeepJDOTMethod",
    "DANNMethod",
    "DSANMethod",
    "RAINCOATMethod",
    "RCTAMethod",
    "RPLTCCDANMethod",
    "SourceOnlyMethod",
    "TargetOnlyMethod",
    "TPDeepJDOTMethod",
    "TPUDeepJDOTMethod",
    "TCCDANMethod",
    "UDeepJDOTMethod",
    "WJDOTMethod",
    "TPWJDOTMethod",
    "CBTPWJDOTMethod",
    "CACCSRWJDOTMethod",
    "MSCBTPWJDOTMethod",
    "build_method",
]
