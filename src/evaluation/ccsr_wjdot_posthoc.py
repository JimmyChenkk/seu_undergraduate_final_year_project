"""Post-hoc CCSR-WJDOT export from an existing WJDOT checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import random

from src.evaluation.review import build_run_review, save_review
from src.utils.fold_policy import canonicalize_fold_choice
from src.utils.random_seed import resolve_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply CCSR-WJDOT fusion to an already trained WJDOT run."
    )
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--base-method-config", type=Path, required=True)
    parser.add_argument("--ccsr-method-config", type=Path, required=True)
    parser.add_argument(
        "--base-experiment-config",
        type=Path,
        required=True,
        help="Experiment config used for the base WJDOT training run.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="Experiment config to use for the CCSR post-hoc output run.",
    )
    parser.add_argument("--base-run-root", type=Path, required=True)
    parser.add_argument("--batch-root-name", type=str, default=None)
    return parser.parse_args()


def _prepare_experiment_payload(
    *,
    experiment_path: Path,
    method_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.trainers.train_benchmark import (
        apply_method_overrides,
        apply_method_runtime_defaults,
        load_yaml,
    )

    experiment_payload = load_yaml(experiment_path)
    method_payload = load_yaml(method_path)
    experiment_payload = apply_method_runtime_defaults(experiment_payload, method_payload)
    experiment_payload, method_payload = apply_method_overrides(
        experiment_payload,
        method_payload,
    )
    return experiment_payload, method_payload


def _prepare_data_context(
    *,
    data_payload: dict[str, Any],
    experiment_payload: dict[str, Any],
    method_payload: dict[str, Any],
) -> dict[str, Any]:
    from src.datasets.te_da_dataset import TEDADatasetConfig
    from src.datasets.te_torch_dataset import prepare_benchmark_data
    from src.trainers.train_benchmark import (
        _resolve_run_fold_names,
        build_run_scene_label,
        build_scenario_id,
        build_setting,
        set_seed,
    )

    data_config = TEDADatasetConfig.from_dict(data_payload)
    protocol_payload = deepcopy(data_payload.get("protocol", {}))
    protocol_payload.update(experiment_payload.get("protocol_override", {}))

    rng_seed, seed_mode = resolve_seed(experiment_payload.get("seed"))
    seed_mode = str(experiment_payload.get("seed_mode", seed_mode))
    experiment_payload["seed"] = rng_seed
    experiment_payload["seed_mode"] = seed_mode
    rng = random.Random(rng_seed)
    fold_selection = _resolve_run_fold_names(
        protocol_payload=protocol_payload,
        default_fold=data_config.preferred_fold,
        rng=rng,
    )
    selected_fold = str(fold_selection["selected_fold"])
    source_fold = str(fold_selection["source_fold"])
    target_fold = str(fold_selection["target_fold"])
    random_fold_enabled = bool(fold_selection["random_fold_enabled"])
    fold_strategy = str(fold_selection["fold_strategy"])
    random_per_scene = bool(fold_selection["random_per_scene"])
    random_per_run = bool(fold_selection["random_per_run"])

    data_config.random_fold_enabled = random_fold_enabled
    data_config.fold_strategy = fold_strategy
    data_config.random_per_scene = random_per_scene
    data_config.random_per_run = random_per_run
    set_seed(rng_seed)

    setting = build_setting(data_config, data_payload, experiment_payload)
    source_domain_ids = [reference.domain.name for reference in setting.source_domains]
    raw_source_folds_by_domain = protocol_payload.get("source_folds_by_domain") or {}
    source_folds_by_domain = {
        domain_name: canonicalize_fold_choice(
            raw_source_folds_by_domain.get(domain_name, source_fold)
        )
        for domain_name in source_domain_ids
    }
    if raw_source_folds_by_domain:
        source_fold = (
            next(iter(source_folds_by_domain.values()))
            if len(set(source_folds_by_domain.values())) == 1
            else "+".join(source_folds_by_domain[domain_name] for domain_name in source_domain_ids)
        )

    data_config.source_fold = source_fold
    data_config.source_folds_by_domain = dict(source_folds_by_domain)
    data_config.target_fold = target_fold

    runtime_payload = experiment_payload.get("runtime", {})
    batch_size = int(method_payload.get("optimization", {}).get("batch_size", 32))
    prepared_data = prepare_benchmark_data(
        config=data_config,
        setting=setting,
        batch_size=batch_size,
        num_workers=int(runtime_payload.get("num_workers", 0)),
        pin_memory=bool(runtime_payload.get("pin_memory", False)),
        persistent_workers=bool(runtime_payload.get("persistent_workers", False)),
        fold_name=selected_fold if not random_fold_enabled else None,
    )
    scenario_id = build_scenario_id(source_domain_ids, prepared_data.target_split.domain_id)
    scene_label = build_run_scene_label(source_domain_ids, prepared_data.target_split.domain_id)
    return {
        "prepared_data": prepared_data,
        "scenario_id": scenario_id,
        "scene_label": scene_label,
        "selected_fold": selected_fold,
        "source_fold": source_fold,
        "source_folds_by_domain": source_folds_by_domain,
        "target_fold": target_fold,
        "random_fold_enabled": random_fold_enabled,
        "fold_strategy": fold_strategy,
        "random_per_scene": random_per_scene,
        "random_per_run": random_per_run,
        "seed": rng_seed,
        "seed_mode": seed_mode,
    }


def _load_base_result(base_run_root: Path) -> dict[str, Any]:
    result_path = base_run_root / "tables" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Base WJDOT result.json not found: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _base_checkpoint_path(base_run_root: Path, base_payload: dict[str, Any]) -> Path:
    result = base_payload.get("result", {})
    checkpoint_value = result.get("checkpoint_path") if isinstance(result, dict) else None
    checkpoint_path = Path(str(checkpoint_value)) if checkpoint_value else base_run_root / "checkpoints" / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base WJDOT checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _copy_base_metric(result: dict[str, Any], key: str) -> Any:
    return result.get(key)


def _build_posthoc_result(
    *,
    base_payload: dict[str, Any],
    base_run_root: Path,
    base_checkpoint_path: Path,
    ccsr_summary: dict[str, Any],
    method_name: str,
    method_display_name: str,
    device_name: str,
    runtime_payload: dict[str, Any],
    timing: dict[str, float],
) -> dict[str, Any]:
    import torch

    base_result = base_payload.get("result", {})
    if not isinstance(base_result, dict):
        base_result = {}

    result = {
        "method_name": method_display_name,
        "method_base_name": method_name,
        "posthoc": True,
        "posthoc_base_method": base_payload.get("method_base_name", base_payload.get("method_name")),
        "posthoc_base_run_root": str(base_run_root),
        "posthoc_base_metrics_path": str(base_run_root / "tables" / "result.json"),
        "posthoc_base_checkpoint_path": str(base_checkpoint_path),
        "history": deepcopy(base_result.get("history", [])),
        "source_train_acc_by_domain": deepcopy(base_result.get("source_train_acc_by_domain", {})),
        "source_eval_acc_by_domain": deepcopy(base_result.get("source_eval_acc_by_domain", {})),
        "source_train_acc": _copy_base_metric(base_result, "source_train_acc"),
        "source_eval_acc": _copy_base_metric(base_result, "source_eval_acc"),
        "best_source_train_acc": _copy_base_metric(base_result, "best_source_train_acc"),
        "final_source_train_acc": _copy_base_metric(base_result, "final_source_train_acc"),
        "best_source_eval_acc": _copy_base_metric(base_result, "best_source_eval_acc"),
        "final_source_eval_acc": _copy_base_metric(base_result, "final_source_eval_acc"),
        "best_target_eval_acc": _copy_base_metric(base_result, "best_target_eval_acc"),
        "final_target_eval_acc": _copy_base_metric(base_result, "final_target_eval_acc"),
        "selected_source_train_acc": _copy_base_metric(base_result, "selected_source_train_acc"),
        "selected_source_eval_acc": _copy_base_metric(base_result, "selected_source_eval_acc"),
        "selected_epoch": _copy_base_metric(base_result, "selected_epoch"),
        "model_selection": _copy_base_metric(base_result, "model_selection"),
        "model_selection_weights": deepcopy(base_result.get("model_selection_weights")),
        "model_selection_params": deepcopy(base_result.get("model_selection_params")),
        "selected_model_selection_score": _copy_base_metric(base_result, "selected_model_selection_score"),
        "model_selection_best_score": _copy_base_metric(base_result, "model_selection_best_score"),
        "epochs_requested": _copy_base_metric(base_result, "epochs_requested"),
        "epochs_completed": _copy_base_metric(base_result, "epochs_completed"),
        "early_stopped": _copy_base_metric(base_result, "early_stopped"),
        "early_stopping_metric": _copy_base_metric(base_result, "early_stopping_metric"),
        "early_stopping_weights": deepcopy(base_result.get("early_stopping_weights")),
        "early_stopping_params": deepcopy(base_result.get("early_stopping_params")),
        "early_stopping_patience": _copy_base_metric(base_result, "early_stopping_patience"),
        "early_stopping_min_delta": _copy_base_metric(base_result, "early_stopping_min_delta"),
        "early_stopping_min_epochs": _copy_base_metric(base_result, "early_stopping_min_epochs"),
        "early_stopping_best_score": _copy_base_metric(base_result, "early_stopping_best_score"),
        "selected_target_train_mean_confidence": _copy_base_metric(
            base_result,
            "selected_target_train_mean_confidence",
        ),
        "selected_target_train_mean_entropy": _copy_base_metric(
            base_result,
            "selected_target_train_mean_entropy",
        ),
        "cache_key": _copy_base_metric(base_result, "cache_key"),
        "cache_hit": bool(base_result.get("cache_hit", False)),
        "device_used": device_name,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "pin_memory": bool(runtime_payload.get("pin_memory", False)),
        "non_blocking_transfers": False,
    }
    result.update(ccsr_summary)
    result["target_eval_acc"] = float(ccsr_summary["target_eval_acc"])
    result["selected_target_eval_acc"] = float(ccsr_summary["target_eval_acc"])
    result["target_eval_macro_f1"] = float(ccsr_summary["target_eval_macro_f1"])
    result["target_eval_balanced_acc"] = float(ccsr_summary["target_eval_balanced_acc"])
    result["target_confusion_matrix"] = ccsr_summary["target_confusion_matrix"]
    result["timing"] = {key: round(float(value), 6) for key, value in timing.items()}
    return result


def run_posthoc(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from src.evaluation.ccsr_wjdot_fusion import export_ccsr_wjdot_fusion_artifacts
    from src.methods import build_method
    from src.trainers.train_benchmark import (
        _export_run_figures,
        _refresh_batch_outputs,
        build_run_paths,
        build_terminal_summary,
        configure_torch_runtime,
        load_yaml,
        save_json,
    )

    total_start = perf_counter()
    data_payload = load_yaml(args.data_config)
    base_experiment, base_method = _prepare_experiment_payload(
        experiment_path=args.base_experiment_config,
        method_path=args.base_method_config,
    )
    ccsr_experiment, ccsr_method = _prepare_experiment_payload(
        experiment_path=args.experiment_config,
        method_path=args.ccsr_method_config,
    )

    data_start = perf_counter()
    context = _prepare_data_context(
        data_payload=data_payload,
        experiment_payload=ccsr_experiment,
        method_payload=ccsr_method,
    )
    prepared_data = context["prepared_data"]
    data_prepare_seconds = perf_counter() - data_start

    runtime_payload = ccsr_experiment.get("runtime", {})
    device_name = str(ccsr_experiment.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    configure_torch_runtime(runtime_payload, device_name)
    device = torch.device(device_name)
    amp_enabled = False

    model = build_method(
        base_method,
        num_classes=29,
        in_channels=int(prepared_data.target_split.input_shape[0]),
        input_length=int(prepared_data.target_split.input_shape[-1]),
        num_sources=len(prepared_data.source_splits),
    )
    base_run_root = args.base_run_root
    base_payload = _load_base_result(base_run_root)
    checkpoint_path = _base_checkpoint_path(base_run_root, base_payload)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    method_name = str(ccsr_method.get("method_name", "ccsr_wjdot_fusion")).lower()
    method_display_name = str(ccsr_method.get("method_display_name", method_name)).lower()
    backbone_name = str(ccsr_method.get("backbone", {}).get("name", "fcn"))
    run_paths = build_run_paths(
        experiment_config=ccsr_experiment,
        method_name=method_display_name,
        scenario_id=str(context["scene_label"]),
        backbone_name=backbone_name,
        fold_name=str(context["selected_fold"]),
        source_fold_name=str(context["source_fold"]),
        target_fold_name=str(context["target_fold"]),
        batch_root_name=args.batch_root_name,
    )

    ccsr_start = perf_counter()
    ccsr_summary = export_ccsr_wjdot_fusion_artifacts(
        model=model,
        prepared_data=prepared_data,
        device=device,
        analysis_path=run_paths["artifacts_dir"] / "ccsr_analysis.npz",
        tables_dir=run_paths["tables_dir"],
        figures_dir=run_paths["figures_dir"],
        scenario_id=str(context["scenario_id"]),
        method_name=method_display_name,
        ccsr_config=ccsr_method.get("loss", {}),
        max_batches=runtime_payload.get("analysis_max_batches"),
        non_blocking=False,
        amp_enabled=amp_enabled,
    )
    timing = {
        "data_prepare_seconds": data_prepare_seconds,
        "ccsr_fusion_seconds": perf_counter() - ccsr_start,
        "train_step_seconds": 0.0,
        "train_steps_completed": 0.0,
        "validation_calls": 0.0,
    }
    timing["total_run_seconds"] = perf_counter() - total_start
    timing["total_with_data_prepare_seconds"] = timing["total_run_seconds"]

    method_result = _build_posthoc_result(
        base_payload=base_payload,
        base_run_root=base_run_root,
        base_checkpoint_path=checkpoint_path,
        ccsr_summary=ccsr_summary,
        method_name=method_name,
        method_display_name=method_display_name,
        device_name=str(device),
        runtime_payload=runtime_payload,
        timing=timing,
    )

    result_payload = {
        "experiment_name": ccsr_experiment.get("experiment_name"),
        "method_name": method_display_name,
        "method_base_name": method_name,
        "setting": prepared_data.setting.setting_name,
        "source_domains": [split.domain_id for split in prepared_data.source_splits],
        "target_domain": prepared_data.target_split.domain_id,
        "scenario_id": context["scenario_id"],
        "scene_label": context["scene_label"],
        "backbone_name": backbone_name,
        "fold_name": context["selected_fold"],
        "selected_fold": context["selected_fold"],
        "source_fold": context["source_fold"],
        "source_folds_by_domain": dict(context["source_folds_by_domain"]),
        "target_fold": context["target_fold"],
        "random_fold_enabled": context["random_fold_enabled"],
        "fold_strategy": context["fold_strategy"],
        "random_per_scene": context["random_per_scene"],
        "random_per_run": context["random_per_run"],
        "seed": context["seed"],
        "seed_mode": context["seed_mode"],
        "timestamp": run_paths["timestamp"],
        "batch_root": str(run_paths["batch_root"]) if run_paths["batch_root"] else None,
        "run_root": str(run_paths["run_root"]),
        "result": method_result,
    }
    figure_paths = _export_run_figures(result_payload, run_paths)
    review_payload = build_run_review(result_payload, figure_paths=figure_paths)
    result_payload["figure_paths"] = figure_paths
    result_payload["metrics_path"] = str(run_paths["metrics_path"])
    result_payload["review_path"] = str(run_paths["review_path"])
    save_json(run_paths["metrics_path"], result_payload)
    save_review(run_paths["review_path"], review_payload)
    if run_paths["batch_root"] is not None and bool(runtime_payload.get("refresh_batch_outputs", True)):
        _refresh_batch_outputs(run_paths["batch_root"])
    print(json.dumps(build_terminal_summary(result_payload), indent=2, ensure_ascii=False))
    return result_payload


def main() -> None:
    run_posthoc(parse_args())


if __name__ == "__main__":
    main()
