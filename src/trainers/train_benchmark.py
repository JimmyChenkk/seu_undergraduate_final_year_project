"""Train benchmark-oriented TEP baselines and DA methods."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING, Any

from src.evaluation.review import build_run_review, save_review
from src.trainers.selection_metrics import resolve_selection_metric
from src.utils.run_layout import build_run_layout

if TYPE_CHECKING:
    from src.datasets.te_da_dataset import TEDADatasetConfig


class TrainingDependencyError(RuntimeError):
    """Raised when the benchmark training stack is not installed."""


def _import_numpy():
    if importlib.util.find_spec("numpy") is None:
        raise TrainingDependencyError(
            "Missing training dependency: numpy. Install it manually in tep_env, for example with "
            + "`pip install -r requirements-benchmark.txt`."
        )

    import numpy as np

    return np


def _import_yaml():
    if importlib.util.find_spec("yaml") is None:
        raise TrainingDependencyError(
            "Missing training dependency: yaml. Install it manually in tep_env, for example with "
            + "`pip install -r requirements-benchmark.txt`."
        )

    import yaml

    return yaml


def load_yaml(path: Path) -> dict[str, Any]:
    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_runtime_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge runtime config while replacing metric-config leaf dicts wholesale."""

    replace_keys = {
        "selection_weights",
        "selection_params",
        "early_stopping_weights",
        "early_stopping_params",
    }
    merged = deepcopy(base)
    for key, value in override.items():
        if key in replace_keys and isinstance(value, dict):
            merged[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_method_overrides(
    experiment_payload: dict[str, Any],
    method_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply optional experiment-level per-method overrides."""

    overrides = experiment_payload.get("method_overrides", {})
    if not isinstance(overrides, dict):
        return experiment_payload, method_payload

    method_name = str(method_payload.get("method_name", "")).strip().lower()
    merged_experiment = deepcopy(experiment_payload)
    merged_method = deepcopy(method_payload)
    for key in ("*", "all", method_name):
        override_payload = overrides.get(key)
        if not isinstance(override_payload, dict):
            continue
        for section_name, section_value in override_payload.items():
            if not isinstance(section_value, dict):
                if section_name in {"runtime", "tracking", "protocol_override"}:
                    merged_experiment[section_name] = deepcopy(section_value)
                else:
                    merged_method[section_name] = deepcopy(section_value)
                continue
            if section_name in {"runtime", "tracking", "protocol_override"}:
                if section_name == "runtime":
                    merged_experiment[section_name] = merge_runtime_config(
                        merged_experiment.get(section_name, {}),
                        section_value,
                    )
                else:
                    merged_experiment[section_name] = deep_merge_dict(
                        merged_experiment.get(section_name, {}),
                        section_value,
                    )
            else:
                merged_method[section_name] = deep_merge_dict(
                    merged_method.get(section_name, {}),
                    section_value,
                )
    return merged_experiment, merged_method


def apply_method_runtime_defaults(
    experiment_payload: dict[str, Any],
    method_payload: dict[str, Any],
) -> dict[str, Any]:
    """Apply optional method-owned defaults for experiment runtime settings.

    Method configs may declare ``runtime_defaults`` for method-specific
    model-selection or early-stopping behavior. Experiment configs still own the
    final runtime policy, so explicit experiment-level values override these
    defaults, and ``method_overrides`` can still override both.
    """

    runtime_defaults = method_payload.get("runtime_defaults", {})
    if not isinstance(runtime_defaults, dict):
        return deepcopy(experiment_payload)

    merged_experiment = deepcopy(experiment_payload)
    merged_experiment["runtime"] = merge_runtime_config(
        runtime_defaults,
        merged_experiment.get("runtime", {}),
    )
    return merged_experiment


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_terminal_summary(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Keep CLI output compact while the full payload stays in ``result.json``."""

    result = result_payload.get("result", {})
    return {
        "experiment_name": result_payload.get("experiment_name"),
        "method_name": result_payload.get("method_name"),
        "scenario_id": result_payload.get("scenario_id"),
        "run_root": result_payload.get("run_root"),
        "metrics_path": result_payload.get("metrics_path"),
        "review_path": result_payload.get("review_path"),
        "cache_hit": result_payload.get("cache_hit"),
        "device": {
            "device_used": result.get("device_used"),
            "cudnn_enabled": result.get("cudnn_enabled"),
            "cudnn_benchmark": result.get("cudnn_benchmark"),
            "pin_memory": result.get("pin_memory"),
            "non_blocking_transfers": result.get("non_blocking_transfers"),
        },
        "metrics": {
            "source_train_acc": result.get("source_train_acc"),
            "source_eval_acc": result.get("source_eval_acc"),
            "target_eval_acc": result.get("target_eval_acc"),
            "target_eval_balanced_acc": result.get("target_eval_balanced_acc"),
            "selected_epoch": result.get("selected_epoch"),
            "epochs_completed": result.get("epochs_completed"),
            "early_stopped": result.get("early_stopped"),
        },
        "selection": {
            "model_selection": result.get("model_selection"),
            "model_selection_weights": result.get("model_selection_weights"),
            "model_selection_params": result.get("model_selection_params"),
            "selected_model_selection_score": result.get("selected_model_selection_score"),
            "selected_target_train_mean_confidence": result.get("selected_target_train_mean_confidence"),
            "selected_target_train_mean_entropy": result.get("selected_target_train_mean_entropy"),
            "early_stopping_metric": result.get("early_stopping_metric"),
            "early_stopping_weights": result.get("early_stopping_weights"),
            "early_stopping_params": result.get("early_stopping_params"),
            "early_stopping_best_score": result.get("early_stopping_best_score"),
        },
        "figure_paths": result_payload.get("figure_paths"),
    }


def set_seed(seed: int) -> None:
    np = _import_numpy()
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(runtime: dict[str, Any], device_name: str) -> None:
    """Apply conservative torch backend defaults for the selected device."""

    if not device_name.startswith("cuda"):
        return

    import torch

    if bool(runtime.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        return

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime.get("allow_tf32", True))
    torch.set_float32_matmul_precision(str(runtime.get("matmul_precision", "high")))


def _should_show_progress(runtime: dict[str, Any]) -> bool:
    """Enable terminal progress output by default only for interactive runs."""

    show_progress = runtime.get("show_progress")
    if show_progress is None:
        return sys.stderr.isatty()
    return bool(show_progress)


def _progress_update_interval(runtime: dict[str, Any], steps_per_epoch: int) -> int:
    """Limit how often we redraw the terminal progress line."""

    configured = runtime.get("progress_update_interval")
    if configured is not None:
        return max(int(configured), 1)
    return max(steps_per_epoch // 20, 1)


def _render_progress_bar(current: int, total: int, *, width: int = 24) -> str:
    """Render a compact ASCII progress bar."""

    if total <= 0:
        return "-" * width
    filled = min(width, int(width * current / total))
    return "#" * filled + "-" * (width - filled)


def _mean_metric(metrics: dict[str, list[float]], key: str) -> float | None:
    values = metrics.get(key, [])
    if not values:
        return None
    return float(sum(values) / len(values))


def _normalize_metric_weights(weights: Any) -> dict[str, float]:
    if not isinstance(weights, dict):
        return {}
    return {str(key): float(value) for key, value in weights.items()}


def _normalize_metric_params(params: Any) -> dict[str, float]:
    if not isinstance(params, dict):
        return {}
    return {str(key): float(value) for key, value in params.items()}


def _resolve_metric_score(
    summary: dict[str, float],
    metric_name: str,
    *,
    weights: dict[str, float] | None = None,
    params: dict[str, float] | None = None,
) -> float | None:
    """Resolve one score-to-maximize from epoch summary metrics."""

    return resolve_selection_metric(
        summary,
        metric_name,
        weights=weights,
        params=params,
    )


def _emit_early_stop_notice(
    *,
    method_name: str,
    scenario_id: str,
    epoch_index: int,
    epochs: int,
    metric_name: str,
    best_score: float,
    patience: int,
) -> None:
    print(
        (
            f"[{method_name}][{scenario_id}] early stop at epoch {epoch_index + 1}/{epochs} "
            f"after patience={patience} on {metric_name} "
            f"(best={best_score:.4f})"
        ),
        file=sys.stderr,
        flush=True,
    )


def _emit_step_progress(
    *,
    method_name: str,
    scenario_id: str,
    epoch_index: int,
    epochs: int,
    step_index: int,
    steps_per_epoch: int,
    epoch_metrics: dict[str, list[float]],
) -> None:
    """Refresh one in-place training progress line on stderr."""

    current = step_index + 1
    bar = _render_progress_bar(current, steps_per_epoch)
    loss_value = _mean_metric(epoch_metrics, "loss_total")
    source_acc = _mean_metric(epoch_metrics, "acc_source")
    loss_text = "n/a" if loss_value is None else f"{loss_value:.4f}"
    acc_text = "n/a" if source_acc is None else f"{source_acc:.3f}"
    line = (
        f"\r[{method_name}][{scenario_id}] "
        f"epoch {epoch_index + 1}/{epochs} "
        f"[{bar}] {current}/{steps_per_epoch} "
        f"loss={loss_text} acc={acc_text}"
    )
    print(line, end="", file=sys.stderr, flush=True)


def _emit_epoch_summary(
    *,
    method_name: str,
    scenario_id: str,
    epoch_index: int,
    epochs: int,
    summary: dict[str, float],
) -> None:
    """Print a concise epoch summary after validation metrics are ready."""

    def _fmt(value: float | None, digits: int) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    print(file=sys.stderr)
    print(
        (
            f"[{method_name}][{scenario_id}] "
            f"epoch {epoch_index + 1}/{epochs} done "
            f"loss={_fmt(summary.get('loss_total'), 4)} "
            f"src_train={_fmt(summary.get('acc_source_train'), 3)} "
            f"src_eval={_fmt(summary.get('acc_source_eval'), 3)} "
            f"tgt_eval={_fmt(summary.get('target_eval_acc'), 3)}"
        ),
        file=sys.stderr,
        flush=True,
    )


def ensure_dependencies(
    method_name: str,
    experiment_config: dict[str, Any],
    method_config: dict[str, Any] | None = None,
) -> None:
    """Fail early with a concise message when the training stack is missing."""

    required = ["numpy", "yaml", "torch", "sklearn"]
    if bool(experiment_config.get("runtime", {}).get("save_analysis", True)):
        required.append("matplotlib")
    requires_ot = method_name == "deepjdot"
    if method_name == "rcta":
        base_align = str((method_config or {}).get("loss", {}).get("base_align", "cdan")).strip().lower()
        requires_ot = base_align == "deepjdot"

    if requires_ot:
        required.append("ot")

    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise TrainingDependencyError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install them manually in tep_env, for example with "
            + "`pip install -r requirements-benchmark.txt`."
        )


def build_scenario_id(source_domains: list[str], target_domain: str) -> str:
    """Create a stable scenario identifier such as ``mode4_to_mode1``."""

    source_part = "-".join(source_domains)
    return f"{source_part}_to_{target_domain}"


def build_run_scene_label(source_domains: list[str], target_domain: str) -> str:
    """Create a human-readable scene label such as ``1_to_4`` or ``1_4_to_5``."""

    def _short(domain_name: str) -> str:
        text = str(domain_name).strip().lower()
        if text.startswith("mode"):
            return text.replace("mode", "")
        return text

    source_part = "_".join(_short(domain) for domain in source_domains)
    return f"{source_part}_to_{_short(target_domain)}"


def _normalize_fold_value(fold_value: Any, default: str) -> str:
    text = str(fold_value).strip()
    return text if text else default


def _resolve_random_fold_enabled(protocol_payload: dict[str, Any]) -> bool:
    return bool(protocol_payload.get("random_fold_enabled", False))


def _has_explicit_fold_override(protocol_payload: dict[str, Any], fold_key: str) -> bool:
    if fold_key not in protocol_payload:
        return False
    value = protocol_payload.get(fold_key)
    return value is not None and bool(str(value).strip())


def _canonicalize_fold_name(fold_name: Any) -> str:
    text = str(fold_name).strip()
    if not text:
        return "Fold 1"
    lowered = text.lower()
    if lowered.startswith("fold"):
        suffix = text[4:].strip()
        return f"Fold {suffix}" if suffix else "Fold 1"
    return f"Fold {text}"


def _sample_fold_name(available_folds: list[str], rng: random.Random) -> str:
    if not available_folds:
        raise ValueError("No fold choices available for random fold sampling.")
    return _canonicalize_fold_name(rng.choice(available_folds))


def _resolve_run_fold_names(
    *,
    protocol_payload: dict[str, Any],
    default_fold: str,
    rng: random.Random,
) -> tuple[str, str, str, bool]:
    selected_fold = _canonicalize_fold_name(
        _normalize_fold_value(
            protocol_payload.get("preferred_fold", default_fold),
            default_fold,
        )
    )
    source_fold_default = _canonicalize_fold_name(
        _normalize_fold_value(protocol_payload.get("source_fold", selected_fold), selected_fold)
    )
    target_fold_default = _canonicalize_fold_name(
        _normalize_fold_value(protocol_payload.get("target_fold", selected_fold), selected_fold)
    )
    random_fold_enabled = _resolve_random_fold_enabled(protocol_payload)
    source_fold_choices = [_canonicalize_fold_name(item) for item in protocol_payload.get("source_folds", [1, 2, 3, 4, 5])]
    target_fold_choices = [_canonicalize_fold_name(item) for item in protocol_payload.get("target_folds", [1, 2, 3, 4, 5])]

    if random_fold_enabled and not _has_explicit_fold_override(protocol_payload, "source_fold"):
        source_fold = _sample_fold_name(source_fold_choices, rng)
    else:
        source_fold = source_fold_default

    if random_fold_enabled and not _has_explicit_fold_override(protocol_payload, "target_fold"):
        target_fold = _sample_fold_name(target_fold_choices, rng)
    else:
        target_fold = target_fold_default

    return selected_fold, source_fold, target_fold, random_fold_enabled


def build_run_paths(
    *,
    experiment_config: dict[str, Any],
    method_name: str,
    scenario_id: str,
    backbone_name: str,
    fold_name: str | None = None,
    source_fold_name: str | None = None,
    target_fold_name: str | None = None,
    batch_root_name: str | None = None,
) -> dict[str, Any]:
    """Create per-run output paths under ``runs/``."""

    tracking_config = experiment_config.get("tracking", {})
    layout = build_run_layout(
        output_dir=Path(experiment_config.get("output_dir", "runs")),
        method_name=method_name,
        scenario_id=scenario_id,
        backbone_name=backbone_name,
        fold_name=fold_name,
        source_fold_name=source_fold_name,
        target_fold_name=target_fold_name,
        batch_root_name=batch_root_name or tracking_config.get("batch_root_name"),
    )
    return {
        "timestamp": layout.timestamp,
        "batch_root": layout.batch_root,
        "run_root": layout.run_root,
        "artifacts_dir": layout.artifacts_dir,
        "tables_dir": layout.tables_dir,
        "figures_dir": layout.figures_dir,
        "logs_dir": layout.logs_dir,
        "checkpoints_dir": layout.checkpoints_dir,
        "analysis_path": layout.artifacts_dir / "analysis.npz",
        "metrics_path": layout.tables_dir / "result.json",
        "review_path": layout.tables_dir / "review.json",
    }


def build_setting(
    data_config: "TEDADatasetConfig",
    data_payload: dict[str, Any],
    experiment_payload: dict[str, Any],
):
    """Build single-source or multi-source setting from config files."""

    from src.datasets.te_da_dataset import TEDADatasetInterface

    protocol = deepcopy(data_payload.get("protocol", {}))
    protocol_override = experiment_payload.get("protocol_override", {})
    protocol.update(protocol_override)

    interface = TEDADatasetInterface(data_config)
    setting_name = str(protocol.get("setting", "single_source"))
    source_domains = list(protocol.get("source_domains", []))
    target_domain = str(protocol.get("target_domain"))
    use_target_labels = bool(protocol.get("use_target_labels", False))
    few_shot_target_samples = int(protocol.get("few_shot_target_samples", 0))
    target_label_mode = "few_shot" if use_target_labels and few_shot_target_samples > 0 else (
        "labeled" if use_target_labels else "unlabeled"
    )
    if setting_name == "multi_source":
        return interface.build_multi_source_setting(
            source_domains,
            target_domain,
            target_label_mode=target_label_mode,
            few_shot_target_samples=few_shot_target_samples or None,
        )
    return interface.build_single_source_setting(
        source_domains[0],
        target_domain,
        target_label_mode=target_label_mode,
        few_shot_target_samples=few_shot_target_samples or None,
    )


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _mean_metrics(history_chunk):
    summary = {}
    for key, values in history_chunk.items():
        summary[key] = float(sum(values) / max(len(values), 1))
    return summary


def _max_history_metric(history: list[dict[str, float]], key: str) -> float | None:
    values = [float(item[key]) for item in history if item.get(key) is not None]
    if not values:
        return None
    return float(max(values))


def _latest_history_metric(history: list[dict[str, float]], key: str) -> float | None:
    for item in reversed(history):
        value = item.get(key)
        if value is not None:
            return float(value)
    return None


def _string_array(values: list[str]):
    np = _import_numpy()
    items = [str(value) for value in values]
    if not items:
        return np.empty((0,), dtype="<U1")
    max_length = max(len(item) for item in items)
    return np.asarray(items, dtype=f"<U{max(max_length, 1)}")


def _evaluate_accuracy(
    model,
    loader,
    device,
    *,
    max_batches: int | None = None,
    non_blocking: bool = False,
    amp_enabled: bool = False,
):
    import torch

    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model.predict_logits(x_batch)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == y_batch).sum().item())
            total += int(y_batch.numel())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return correct / max(total, 1)


def _evaluate_unlabeled_target_proxy(
    model,
    loader,
    device,
    *,
    max_batches: int | None = None,
    non_blocking: bool = False,
    amp_enabled: bool = False,
) -> dict[str, float]:
    """Compute label-free target-domain proxy metrics for UDA model selection."""

    import torch

    model.eval()
    total = 0
    confidence_sum = 0.0
    entropy_sum = 0.0
    with torch.inference_mode():
        for batch_index, (x_batch, _) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model.predict_logits(x_batch)
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities.max(dim=1).values
            entropy = -(
                probabilities * torch.log(probabilities.clamp_min(1e-8))
            ).sum(dim=1) / math.log(probabilities.shape[1])
            confidence_sum += float(confidence.sum().item())
            entropy_sum += float(entropy.sum().item())
            total += int(probabilities.shape[0])
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    denominator = max(total, 1)
    return {
        "target_train_mean_confidence": float(confidence_sum / denominator),
        "target_train_mean_entropy": float(entropy_sum / denominator),
    }


def _evaluate_domain_accuracies(
    model,
    loaders,
    domain_ids,
    device,
    *,
    max_batches: int | None = None,
    non_blocking: bool = False,
) -> tuple[dict[str, float], float]:
    per_domain = {
        domain_id: float(
            _evaluate_accuracy(
                model,
                loader,
                device,
                max_batches=max_batches,
                non_blocking=non_blocking,
            )
        )
        for domain_id, loader in zip(domain_ids, loaders)
    }
    mean_accuracy = float(sum(per_domain.values()) / max(len(per_domain), 1))
    return per_domain, mean_accuracy


def _collect_loader_outputs(
    model,
    loader,
    device,
    *,
    domain_name: str,
    max_batches: int | None = None,
    non_blocking: bool = False,
):
    """Collect logits, predictions, labels and embeddings for one loader."""

    np = _import_numpy()
    import torch

    embeddings = []
    logits_list = []
    labels_list = []
    domains = []

    model.eval()
    with torch.inference_mode():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            logits, features = model(x_batch)
            embeddings.append(features.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(y_batch.numpy())
            domains.append(_string_array([domain_name] * len(y_batch)))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    if not embeddings:
        return {
            "embeddings": np.empty((0, 0), dtype=np.float32),
            "logits": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "predictions": np.empty((0,), dtype=np.int64),
            "domains": _string_array([]),
        }

    embeddings_array = np.concatenate(embeddings, axis=0)
    logits_array = np.concatenate(logits_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    predictions_array = logits_array.argmax(axis=1)
    domains_array = np.concatenate(domains, axis=0)
    finite_embeddings = np.isfinite(embeddings_array).all(axis=1)
    finite_logits = np.isfinite(logits_array).all(axis=1)
    finite_mask = finite_embeddings & finite_logits
    if not finite_mask.all():
        invalid_count = int((~finite_mask).sum())
        print(
            f"[trainer] dropping {invalid_count} non-finite samples from collected outputs for {domain_name}"
        )
        embeddings_array = embeddings_array[finite_mask]
        logits_array = logits_array[finite_mask]
        labels_array = labels_array[finite_mask]
        predictions_array = predictions_array[finite_mask]
        domains_array = domains_array[finite_mask]
    return {
        "embeddings": embeddings_array,
        "logits": logits_array,
        "labels": labels_array,
        "predictions": predictions_array,
        "domains": domains_array,
    }


def _evaluate_target_metrics(
    model,
    loader,
    device,
    *,
    domain_name: str,
    max_batches: int | None = None,
    non_blocking: bool = False,
) -> dict[str, Any]:
    """Compute target-side accuracy metrics independently of analysis export."""

    from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

    target_chunk = _collect_loader_outputs(
        model,
        loader,
        device,
        domain_name=domain_name,
        max_batches=max_batches,
        non_blocking=non_blocking,
    )
    labels = target_chunk["labels"]
    predictions = target_chunk["predictions"]
    if len(labels) == 0:
        return {
            "target_eval_acc": 0.0,
            "target_eval_balanced_acc": 0.0,
            "target_confusion_matrix": [],
        }

    target_accuracy = float(accuracy_score(labels, predictions))
    present_labels = sorted(set(int(label) for label in labels.tolist()))
    target_balanced_accuracy = float(
        recall_score(labels, predictions, labels=present_labels, average="macro", zero_division=0)
    )
    target_confusion = confusion_matrix(labels, predictions, labels=list(range(29)))
    return {
        "target_eval_acc": target_accuracy,
        "target_eval_balanced_acc": target_balanced_accuracy,
        "target_confusion_matrix": target_confusion.tolist(),
    }


def export_analysis_artifacts(
    *,
    model,
    prepared_data,
    device,
    analysis_path: Path,
    scenario_id: str,
    method_name: str,
    max_batches: int | None = None,
    non_blocking: bool = False,
) -> dict[str, Any]:
    """Persist embeddings and prediction traces for later figures."""

    np = _import_numpy()
    from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

    ensure_parent(analysis_path)
    source_chunks = []
    for split, loader in zip(prepared_data.source_splits, prepared_data.source_eval_loaders):
        chunk = _collect_loader_outputs(
            model,
            loader,
            device,
            domain_name=split.domain_id,
            max_batches=max_batches,
            non_blocking=non_blocking,
        )
        source_chunks.append(chunk)

    target_chunk = _collect_loader_outputs(
        model,
        prepared_data.target_eval_loader,
        device,
        domain_name=prepared_data.target_split.domain_id,
        max_batches=max_batches,
        non_blocking=non_blocking,
    )

    source_embeddings = np.concatenate([chunk["embeddings"] for chunk in source_chunks], axis=0)
    source_labels = np.concatenate([chunk["labels"] for chunk in source_chunks], axis=0)
    source_predictions = np.concatenate([chunk["predictions"] for chunk in source_chunks], axis=0)
    source_domains = np.concatenate([chunk["domains"] for chunk in source_chunks], axis=0)

    np.savez_compressed(
        analysis_path,
        scenario_id=_string_array([scenario_id]),
        method_name=_string_array([method_name]),
        source_embeddings=source_embeddings,
        source_labels=source_labels,
        source_predictions=source_predictions,
        source_domains=source_domains,
        target_embeddings=target_chunk["embeddings"],
        target_labels=target_chunk["labels"],
        target_predictions=target_chunk["predictions"],
        target_domains=target_chunk["domains"],
        target_logits=target_chunk["logits"],
    )

    target_accuracy = float(accuracy_score(target_chunk["labels"], target_chunk["predictions"]))
    present_labels = sorted(set(int(label) for label in target_chunk["labels"].tolist()))
    target_balanced_accuracy = float(
        recall_score(
            target_chunk["labels"],
            target_chunk["predictions"],
            labels=present_labels,
            average="macro",
            zero_division=0,
        )
    )
    target_confusion = confusion_matrix(
        target_chunk["labels"],
        target_chunk["predictions"],
        labels=np.arange(29),
    )
    return {
        "analysis_path": str(analysis_path),
        "target_eval_acc": target_accuracy,
        "target_eval_balanced_acc": target_balanced_accuracy,
        "target_confusion_matrix": target_confusion.tolist(),
    }


def run_deep_experiment(
    *,
    method_config: dict[str, Any],
    experiment_config: dict[str, Any],
    prepared_data,
    run_paths: dict[str, Path],
    scenario_id: str,
) -> dict[str, Any]:
    """Train one torch-based method."""

    import torch
    from src.methods import build_method

    optimization = method_config.get("optimization", {})
    runtime = experiment_config.get("runtime", {})
    device_name = str(experiment_config.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    configure_torch_runtime(runtime, device_name)
    device = torch.device(device_name)
    pin_memory_enabled = bool(runtime.get("pin_memory", False))
    transfer_non_blocking = (
        bool(runtime.get("non_blocking_transfers", pin_memory_enabled))
        and pin_memory_enabled
        and device.type == "cuda"
    )
    num_workers = int(runtime.get("num_workers", 4 if device.type == "cuda" else 0))
    persistent_workers = bool(runtime.get("persistent_workers", num_workers > 0))
    amp_enabled = bool(runtime.get("amp", device.type == "cuda"))

    model = build_method(
        method_config,
        num_classes=29,
        in_channels=int(prepared_data.target_split.input_shape[0]),
        input_length=int(prepared_data.target_split.input_shape[-1]),
        num_sources=len(prepared_data.source_splits),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimization.get("learning_rate", 1e-3)),
        weight_decay=float(optimization.get("weight_decay", 0.0)),
    )
    max_grad_norm = optimization.get("max_grad_norm")
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)

    history = []
    target_iterator = _cycle(prepared_data.target_train_loader)
    source_iterators = [_cycle(loader) for loader in prepared_data.source_train_loaders]
    steps_per_epoch = max(len(loader) for loader in prepared_data.source_train_loaders)
    if runtime.get("dry_run", False):
        steps_per_epoch = min(steps_per_epoch, 2)
    show_progress = _should_show_progress(runtime)
    progress_update_interval = _progress_update_interval(runtime, steps_per_epoch)
    evaluation_max_batches = runtime.get("eval_max_batches")
    if evaluation_max_batches is None and runtime.get("dry_run", False):
        evaluation_max_batches = 2
    if evaluation_max_batches is not None:
        evaluation_max_batches = int(evaluation_max_batches)
    analysis_max_batches = runtime.get("analysis_max_batches")
    if analysis_max_batches is None and runtime.get("dry_run", False):
        analysis_max_batches = evaluation_max_batches
    if analysis_max_batches is not None:
        analysis_max_batches = int(analysis_max_batches)
    evaluation_interval = max(int(runtime.get("evaluation_interval", 1)), 1)
    selection_interval = max(int(runtime.get("selection_interval", evaluation_interval)), 1)
    early_stopping_interval = max(int(runtime.get("early_stopping_interval", evaluation_interval)), 1)
    final_epoch_evaluation = bool(runtime.get("final_epoch_evaluation", True))
    final_selection_evaluation = bool(runtime.get("final_selection_evaluation", True))

    source_domain_ids = [split.domain_id for split in prepared_data.source_splits]
    final_source_train_by_domain: dict[str, float] = {}
    final_source_eval_by_domain: dict[str, float] = {}
    selected_source_train_by_domain: dict[str, float] = {}
    selected_source_eval_by_domain: dict[str, float] = {}
    selected_source_train_acc = 0.0
    selected_source_eval_acc = 0.0
    selected_target_eval_acc = 0.0
    last_valid_source_train_acc: float | None = None
    last_valid_source_eval_acc: float | None = None
    last_valid_target_eval_acc: float | None = None
    last_valid_target_proxy_metrics: dict[str, float | None] = {
        "target_train_mean_confidence": None,
        "target_train_mean_entropy": None,
    }
    selection_mode = str(runtime.get("model_selection", "best_source_eval")).lower()
    selection_weights = _normalize_metric_weights(runtime.get("selection_weights", {}))
    selection_params = _normalize_metric_params(runtime.get("selection_params", {}))
    best_selection_score = float("-inf")
    selected_epoch = 0
    selected_state_dict: dict[str, torch.Tensor] | None = None
    selected_summary: dict[str, float] | None = None
    early_stopping_patience = runtime.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = max(int(early_stopping_patience), 1)
    early_stopping_min_delta = float(runtime.get("early_stopping_min_delta", 0.0))
    early_stopping_min_epochs = max(int(runtime.get("early_stopping_min_epochs", 0)), 0)
    early_stopping_metric = str(runtime.get("early_stopping_metric", "source_eval")).strip().lower()
    early_stopping_weights = _normalize_metric_weights(
        runtime.get("early_stopping_weights", selection_weights)
    )
    early_stopping_params = _normalize_metric_params(
        runtime.get("early_stopping_params", selection_params)
    )
    early_stopping_best_score = float("-inf")
    early_stopping_bad_epochs = 0
    early_stopped = False
    epochs = int(optimization.get("epochs", 1))
    method_name = str(method_config["method_name"])
    for epoch_index in range(epochs):
        model.train()
        epoch_metrics = defaultdict(list)
        for step_index in range(steps_per_epoch):
            source_batches = [next(iterator) for iterator in source_iterators]
            target_batch = next(target_iterator)

            source_batches = [
                (
                    x_batch.to(device, non_blocking=transfer_non_blocking),
                    y_batch.to(device, non_blocking=transfer_non_blocking),
                )
                for x_batch, y_batch in source_batches
            ]
            target_batch = (
                target_batch[0].to(device, non_blocking=transfer_non_blocking),
                target_batch[1].to(device, non_blocking=transfer_non_blocking),
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                step_output = model.compute_loss(source_batches, target_batch)
            step_output.loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            post_step_metrics = model.after_optimizer_step()

            for key, value in step_output.metrics.items():
                epoch_metrics[key].append(value)
            for key, value in post_step_metrics.items():
                epoch_metrics[key].append(value)
            if show_progress and (
                step_index == 0
                or step_index + 1 == steps_per_epoch
                or (step_index + 1) % progress_update_interval == 0
            ):
                _emit_step_progress(
                    method_name=method_name,
                    scenario_id=scenario_id,
                    epoch_index=epoch_index,
                    epochs=epochs,
                    step_index=step_index,
                    steps_per_epoch=steps_per_epoch,
                    epoch_metrics=epoch_metrics,
                )

        summary = _mean_metrics(epoch_metrics)
        epoch_number = epoch_index + 1
        is_final_epoch = epoch_number == epochs
        should_periodic_eval = epoch_number % evaluation_interval == 0
        should_selection_eval = selection_mode != "final" and epoch_number % selection_interval == 0
        should_early_stopping_eval = (
            early_stopping_patience is not None and epoch_number % early_stopping_interval == 0
        )
        should_force_final_eval = is_final_epoch and final_epoch_evaluation
        did_validate = (
            should_periodic_eval
            or should_selection_eval
            or should_early_stopping_eval
            or should_force_final_eval
        )

        source_train_acc = float(summary.get("acc_source_train", summary.get("acc_source", 0.0)))
        source_eval_acc = last_valid_source_eval_acc
        target_eval_acc = last_valid_target_eval_acc
        target_proxy_metrics = deepcopy(last_valid_target_proxy_metrics)
        if did_validate:
            final_source_train_by_domain, source_train_acc = _evaluate_domain_accuracies(
                model,
                prepared_data.source_train_eval_loaders,
                source_domain_ids,
                device,
                max_batches=evaluation_max_batches,
                non_blocking=transfer_non_blocking,
            )
            final_source_eval_by_domain, source_eval_acc = _evaluate_domain_accuracies(
                model,
                prepared_data.source_eval_loaders,
                source_domain_ids,
                device,
                max_batches=evaluation_max_batches,
                non_blocking=transfer_non_blocking,
            )
            target_eval_acc = _evaluate_accuracy(
                model,
                prepared_data.target_eval_loader,
                device,
                max_batches=evaluation_max_batches,
                non_blocking=transfer_non_blocking,
                amp_enabled=amp_enabled,
            )
            target_proxy_metrics = _evaluate_unlabeled_target_proxy(
                model,
                prepared_data.target_train_loader,
                device,
                max_batches=evaluation_max_batches,
                non_blocking=transfer_non_blocking,
                amp_enabled=amp_enabled,
            )
            last_valid_source_train_acc = float(source_train_acc)
            last_valid_source_eval_acc = float(source_eval_acc)
            last_valid_target_eval_acc = float(target_eval_acc)
            last_valid_target_proxy_metrics = deepcopy(target_proxy_metrics)
        summary["epoch"] = epoch_number
        summary["acc_source_train"] = float(source_train_acc)
        summary["acc_source_eval"] = None if source_eval_acc is None else float(source_eval_acc)
        summary["target_eval_acc"] = None if target_eval_acc is None else float(target_eval_acc)
        summary.update(target_proxy_metrics)
        summary["validation_epoch"] = bool(did_validate)
        history.append(summary)
        if show_progress:
            _emit_epoch_summary(
                method_name=method_name,
                scenario_id=scenario_id,
                epoch_index=epoch_index,
                epochs=epochs,
                summary=summary,
            )

        selection_score = _resolve_metric_score(
            summary,
            selection_mode,
            weights=selection_weights,
            params=selection_params,
        ) if did_validate and (should_selection_eval or should_force_final_eval) else None
        summary["model_selection_metric"] = selection_mode
        if selection_weights:
            summary["model_selection_weights"] = deepcopy(selection_weights)
        if selection_params:
            summary["model_selection_params"] = deepcopy(selection_params)
        if selection_score is not None:
            summary["model_selection_score"] = float(selection_score)
        if selection_mode != "final" and selection_score is not None and selection_score > best_selection_score:
            best_selection_score = selection_score
            selected_epoch = epoch_index + 1
            selected_summary = deepcopy(summary)
            selected_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if early_stopping_patience is not None:
            stop_score = _resolve_metric_score(
                summary,
                early_stopping_metric,
                weights=early_stopping_weights,
                params=early_stopping_params,
            ) if did_validate and (should_early_stopping_eval or should_force_final_eval) else None
            summary["early_stopping_metric"] = early_stopping_metric
            if early_stopping_weights:
                summary["early_stopping_weights"] = deepcopy(early_stopping_weights)
            if early_stopping_params:
                summary["early_stopping_params"] = deepcopy(early_stopping_params)
            if stop_score is not None:
                summary["early_stopping_score"] = float(stop_score)
            if stop_score is not None:
                if stop_score > early_stopping_best_score + early_stopping_min_delta:
                    early_stopping_best_score = stop_score
                    early_stopping_bad_epochs = 0
                else:
                    early_stopping_bad_epochs += 1

                if (
                    epoch_index + 1 >= early_stopping_min_epochs
                    and early_stopping_bad_epochs >= early_stopping_patience
                ):
                    early_stopped = True
                    _emit_early_stop_notice(
                        method_name=method_name,
                        scenario_id=scenario_id,
                        epoch_index=epoch_index,
                        epochs=epochs,
                        metric_name=early_stopping_metric,
                        best_score=early_stopping_best_score,
                        patience=early_stopping_patience,
                    )
                    break

    if selected_summary is None and history:
        selected_summary = deepcopy(history[-1])

    if selection_mode != "final" and selected_state_dict is not None:
        model.load_state_dict(selected_state_dict)
        if final_selection_evaluation:
            selected_source_train_by_domain, selected_source_train_acc = _evaluate_domain_accuracies(
                model,
                prepared_data.source_train_eval_loaders,
                source_domain_ids,
                device,
                max_batches=evaluation_max_batches,
                non_blocking=transfer_non_blocking,
            )
            selected_source_eval_by_domain, selected_source_eval_acc = _evaluate_domain_accuracies(
                model,
                prepared_data.source_eval_loaders,
                source_domain_ids,
                device,
                max_batches=evaluation_max_batches,
                non_blocking=transfer_non_blocking,
            )
            selected_target_eval_acc = _evaluate_accuracy(
                model,
                prepared_data.target_eval_loader,
                device,
                max_batches=evaluation_max_batches,
                non_blocking=transfer_non_blocking,
            )
        else:
            selected_source_train_by_domain = final_source_train_by_domain
            selected_source_eval_by_domain = final_source_eval_by_domain
            selected_source_train_acc = float(history[-1]["acc_source_train"])
            selected_source_eval_acc = _latest_history_metric(history, "acc_source_eval")
            selected_target_eval_acc = _latest_history_metric(history, "target_eval_acc")
    else:
        selected_epoch = len(history)
        selected_source_train_by_domain = final_source_train_by_domain
        selected_source_eval_by_domain = final_source_eval_by_domain
        selected_source_train_acc = float(history[-1]["acc_source_train"])
        selected_source_eval_acc = _latest_history_metric(history, "acc_source_eval")
        selected_target_eval_acc = _latest_history_metric(history, "target_eval_acc")

    selected_target_metrics = _evaluate_target_metrics(
        model,
        prepared_data.target_eval_loader,
        device,
        domain_name=prepared_data.target_split.domain_id,
        max_batches=evaluation_max_batches,
        non_blocking=transfer_non_blocking,
    )
    selected_target_eval_acc = float(selected_target_metrics["target_eval_acc"])

    result = {
        "method_name": str(method_config["method_name"]),
        "history": history,
        "source_train_acc_by_domain": selected_source_train_by_domain,
        "source_eval_acc_by_domain": selected_source_eval_by_domain,
        "source_train_acc": float(selected_source_train_acc),
        "source_eval_acc": None if selected_source_eval_acc is None else float(selected_source_eval_acc),
        "target_eval_acc": None if selected_target_eval_acc is None else float(selected_target_eval_acc),
        "best_source_train_acc": float(max(item["acc_source_train"] for item in history)),
        "final_source_train_acc": float(history[-1]["acc_source_train"]),
        "best_source_eval_acc": _max_history_metric(history, "acc_source_eval"),
        "final_source_eval_acc": _latest_history_metric(history, "acc_source_eval"),
        "best_target_eval_acc": _max_history_metric(history, "target_eval_acc"),
        "final_target_eval_acc": _latest_history_metric(history, "target_eval_acc"),
        "selected_source_train_acc": float(selected_source_train_acc),
        "selected_source_eval_acc": None if selected_source_eval_acc is None else float(selected_source_eval_acc),
        "selected_target_eval_acc": None if selected_target_eval_acc is None else float(selected_target_eval_acc),
        "target_eval_balanced_acc": float(selected_target_metrics["target_eval_balanced_acc"]),
        "target_confusion_matrix": selected_target_metrics["target_confusion_matrix"],
        "selected_target_train_mean_confidence": (
            None
            if selected_summary is None or selected_summary.get("target_train_mean_confidence") is None
            else float(selected_summary["target_train_mean_confidence"])
        ),
        "selected_target_train_mean_entropy": (
            None
            if selected_summary is None or selected_summary.get("target_train_mean_entropy") is None
            else float(selected_summary["target_train_mean_entropy"])
        ),
        "selected_epoch": int(selected_epoch),
        "model_selection": selection_mode,
        "model_selection_weights": deepcopy(selection_weights),
        "model_selection_params": deepcopy(selection_params),
        "selected_model_selection_score": (
            None
            if selected_summary is None or selected_summary.get("model_selection_score") is None
            else float(selected_summary["model_selection_score"])
        ),
        "model_selection_best_score": (
            None if selection_mode == "final" or best_selection_score == float("-inf") else float(best_selection_score)
        ),
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(history)),
        "early_stopped": bool(early_stopped),
        "early_stopping_metric": early_stopping_metric if early_stopping_patience is not None else None,
        "early_stopping_weights": (
            deepcopy(early_stopping_weights) if early_stopping_patience is not None else None
        ),
        "early_stopping_params": (
            deepcopy(early_stopping_params) if early_stopping_patience is not None else None
        ),
        "early_stopping_patience": int(early_stopping_patience) if early_stopping_patience is not None else None,
        "early_stopping_min_delta": float(early_stopping_min_delta) if early_stopping_patience is not None else None,
        "early_stopping_min_epochs": int(early_stopping_min_epochs) if early_stopping_patience is not None else None,
        "early_stopping_best_score": (
            float(early_stopping_best_score) if early_stopping_patience is not None else None
        ),
    }

    if bool(runtime.get("save_checkpoint", False)):
        checkpoint_path = run_paths["checkpoints_dir"] / "model.pt"
        ensure_parent(checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path)
        result["checkpoint_path"] = str(checkpoint_path)

    if bool(runtime.get("save_analysis", True)):
        analysis_summary = export_analysis_artifacts(
            model=model,
            prepared_data=prepared_data,
            device=device,
            analysis_path=run_paths["analysis_path"],
            scenario_id=scenario_id,
            method_name=str(method_config["method_name"]),
            max_batches=analysis_max_batches,
            non_blocking=transfer_non_blocking,
        )
        result.update(analysis_summary)
        result["target_eval_acc"] = float(selected_target_metrics["target_eval_acc"])
        result["selected_target_eval_acc"] = float(selected_target_metrics["target_eval_acc"])
        result["target_eval_balanced_acc"] = float(selected_target_metrics["target_eval_balanced_acc"])
        result["target_confusion_matrix"] = selected_target_metrics["target_confusion_matrix"]

    result["cache_key"] = getattr(prepared_data, "cache_key", None)
    result["cache_hit"] = bool(getattr(prepared_data, "cache_hit", False))
    result["device_used"] = str(device)
    result["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
    result["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    result["pin_memory"] = pin_memory_enabled
    result["non_blocking_transfers"] = bool(transfer_non_blocking)

    return result


def _export_run_figures(result_payload: dict[str, Any], run_paths: dict[str, Any]) -> dict[str, str | None]:
    figure_paths = {
        "tsne_domain": None,
        "tsne_class": None,
        "confusion_matrix": None,
    }
    analysis_path_value = result_payload.get("result", {}).get("analysis_path")
    if not analysis_path_value:
        return figure_paths

    analysis_path = Path(str(analysis_path_value))
    if not analysis_path.exists():
        return figure_paths

    from src.evaluation.report_figures import export_run_review_figures

    export_run_review_figures(analysis_path, run_paths["figures_dir"])
    for key, filename in {
        "tsne_domain": "tsne_domain.png",
        "tsne_class": "tsne_class.png",
        "confusion_matrix": "confusion_matrix.png",
    }.items():
        figure_path = run_paths["figures_dir"] / filename
        figure_paths[key] = str(figure_path) if figure_path.exists() else None
    return figure_paths


def _refresh_batch_outputs(batch_root: Path) -> None:
    from src.evaluation.evaluate import export_comparison_summary
    from src.evaluation.report_figures import export_summary_figures

    summary_dir = export_comparison_summary(batch_root)
    if summary_dir is None:
        return
    export_summary_figures(batch_root, summary_dir.parent / "figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TE benchmark DA methods.")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument(
        "--batch-root-name",
        type=str,
        default=None,
        help="Optional batch parent under runs/, for example 20260402_full_run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_payload = load_yaml(args.data_config)
    method_payload = load_yaml(args.method_config)
    experiment_payload = load_yaml(args.experiment_config)
    experiment_payload = apply_method_runtime_defaults(
        experiment_payload,
        method_payload,
    )
    experiment_payload, method_payload = apply_method_overrides(
        experiment_payload,
        method_payload,
    )
    method_name = str(method_payload.get("method_name", "")).lower()
    ensure_dependencies(method_name, experiment_payload, method_payload)

    from src.datasets.te_da_dataset import TEDADatasetConfig
    from src.datasets.te_torch_dataset import prepare_benchmark_data

    data_config = TEDADatasetConfig.from_dict(data_payload)
    protocol_payload = deepcopy(data_payload.get("protocol", {}))
    protocol_payload.update(experiment_payload.get("protocol_override", {}))
    random_fold_enabled = _resolve_random_fold_enabled(protocol_payload)
    rng_seed = int(experiment_payload.get("seed", 42))
    rng = random.SystemRandom() if random_fold_enabled else random.Random(rng_seed)
    selected_fold, source_fold, target_fold, random_fold_enabled = _resolve_run_fold_names(
        protocol_payload=protocol_payload,
        default_fold=data_config.preferred_fold,
        rng=rng,
    )

    data_config.random_fold_enabled = random_fold_enabled
    set_seed(rng_seed)
    setting = build_setting(data_config, data_payload, experiment_payload)
    batch_size = int(method_payload.get("optimization", {}).get("batch_size", 32))
    runtime_payload = experiment_payload.get("runtime", {})
    num_workers = int(runtime_payload.get("num_workers", 0))
    pin_memory = bool(runtime_payload.get("pin_memory", False))
    persistent_workers = bool(runtime_payload.get("persistent_workers", num_workers > 0))
    backbone_name = str(method_payload.get("backbone", {}).get("name", "fcn"))
    source_domain_ids = [reference.domain.name for reference in setting.source_domains]
    target_domain_id = setting.target_domain.domain.name
    scenario_id = build_scenario_id(source_domain_ids, target_domain_id)
    run_scene_label = build_run_scene_label(source_domain_ids, target_domain_id)
    run_paths = build_run_paths(
        experiment_config=experiment_payload,
        method_name=method_name,
        scenario_id=run_scene_label,
        backbone_name=backbone_name,
        fold_name=selected_fold,
        source_fold_name=source_fold,
        target_fold_name=target_fold,
        batch_root_name=args.batch_root_name,
    )
    data_config.source_fold = source_fold
    data_config.target_fold = target_fold
    prepared_data = prepare_benchmark_data(
        config=data_config,
        setting=setting,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        fold_name=selected_fold if not random_fold_enabled else None,
    )

    method_result = run_deep_experiment(
        method_config=method_payload,
        experiment_config=experiment_payload,
        prepared_data=prepared_data,
        run_paths=run_paths,
        scenario_id=scenario_id,
    )

    result_payload = {
        "experiment_name": experiment_payload.get("experiment_name"),
        "method_name": method_payload.get("method_name"),
        "setting": prepared_data.setting.setting_name,
        "source_domains": [split.domain_id for split in prepared_data.source_splits],
        "target_domain": prepared_data.target_split.domain_id,
        "scenario_id": scenario_id,
        "scene_label": run_scene_label,
        "backbone_name": backbone_name,
        "fold_name": selected_fold,
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
    if run_paths["batch_root"] is not None:
        _refresh_batch_outputs(run_paths["batch_root"])
    print(json.dumps(build_terminal_summary(result_payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except TrainingDependencyError as exc:
        raise SystemExit(str(exc))
