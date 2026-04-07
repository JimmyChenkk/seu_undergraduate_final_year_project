"""Train benchmark-oriented TEP baselines and DA methods."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING, Any

from src.evaluation.review import build_run_review, save_review
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
    # cuDNN autotune can push some drivers into unstable states during long runs,
    # so keep it opt-in instead of enabling it by default.
    torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", False))


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

    print(file=sys.stderr)
    print(
        (
            f"[{method_name}][{scenario_id}] "
            f"epoch {epoch_index + 1}/{epochs} done "
            f"loss={summary.get('loss_total', float('nan')):.4f} "
            f"src_train={summary.get('acc_source_train', float('nan')):.3f} "
            f"src_eval={summary.get('acc_source_eval', float('nan')):.3f} "
            f"tgt_eval={summary.get('target_eval_acc', float('nan')):.3f}"
        ),
        file=sys.stderr,
        flush=True,
    )


def ensure_dependencies(method_name: str, experiment_config: dict[str, Any]) -> None:
    """Fail early with a concise message when the training stack is missing."""

    required = ["numpy", "yaml", "torch"]
    if bool(experiment_config.get("runtime", {}).get("save_analysis", True)):
        required.extend(["sklearn", "matplotlib"])
    if method_name == "deepjdot":
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


def build_run_paths(
    *,
    experiment_config: dict[str, Any],
    method_name: str,
    scenario_id: str,
    backbone_name: str,
    fold_name: str,
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
):
    import torch

    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            logits = model.predict_logits(x_batch)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == y_batch).sum().item())
            total += int(y_batch.numel())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return correct / max(total, 1)


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
    return {
        "embeddings": embeddings_array,
        "logits": logits_array,
        "labels": labels_array,
        "predictions": predictions_array,
        "domains": domains_array,
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
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

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
    target_balanced_accuracy = float(
        balanced_accuracy_score(target_chunk["labels"], target_chunk["predictions"])
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

    model = build_method(
        method_config,
        num_classes=29,
        in_channels=int(prepared_data.target_split.input_shape[0]),
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

    source_domain_ids = [split.domain_id for split in prepared_data.source_splits]
    final_source_train_by_domain: dict[str, float] = {}
    final_source_eval_by_domain: dict[str, float] = {}
    selected_source_train_by_domain: dict[str, float] = {}
    selected_source_eval_by_domain: dict[str, float] = {}
    selected_source_train_acc = 0.0
    selected_source_eval_acc = 0.0
    selected_target_eval_acc = 0.0
    selection_mode = str(runtime.get("model_selection", "best_source_eval")).lower()
    best_selection_score = float("-inf")
    selected_epoch = 0
    selected_state_dict: dict[str, torch.Tensor] | None = None
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
            step_output = model.compute_loss(source_batches, target_batch)
            step_output.loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            for key, value in step_output.metrics.items():
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
        )
        summary = _mean_metrics(epoch_metrics)
        summary["epoch"] = epoch_index + 1
        summary["acc_source_train"] = float(source_train_acc)
        summary["acc_source_eval"] = float(source_eval_acc)
        summary["target_eval_acc"] = float(target_eval_acc)
        history.append(summary)
        if show_progress:
            _emit_epoch_summary(
                method_name=method_name,
                scenario_id=scenario_id,
                epoch_index=epoch_index,
                epochs=epochs,
                summary=summary,
            )

        selection_score = float(source_eval_acc)
        if selection_mode == "final":
            continue
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            selected_epoch = epoch_index + 1
            selected_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if selection_mode != "final" and selected_state_dict is not None:
        model.load_state_dict(selected_state_dict)
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
        selected_epoch = epochs
        selected_source_train_by_domain = final_source_train_by_domain
        selected_source_eval_by_domain = final_source_eval_by_domain
        selected_source_train_acc = float(history[-1]["acc_source_train"])
        selected_source_eval_acc = float(history[-1]["acc_source_eval"])
        selected_target_eval_acc = float(history[-1]["target_eval_acc"])

    result = {
        "method_name": str(method_config["method_name"]),
        "history": history,
        "source_train_acc_by_domain": selected_source_train_by_domain,
        "source_eval_acc_by_domain": selected_source_eval_by_domain,
        "source_train_acc": float(selected_source_train_acc),
        "source_eval_acc": float(selected_source_eval_acc),
        "target_eval_acc": float(selected_target_eval_acc),
        "best_source_train_acc": float(max(item["acc_source_train"] for item in history)),
        "final_source_train_acc": float(history[-1]["acc_source_train"]),
        "best_source_eval_acc": float(max(item["acc_source_eval"] for item in history)),
        "final_source_eval_acc": float(history[-1]["acc_source_eval"]),
        "best_target_eval_acc": float(max(item["target_eval_acc"] for item in history)),
        "final_target_eval_acc": float(history[-1]["target_eval_acc"]),
        "selected_source_train_acc": float(selected_source_train_acc),
        "selected_source_eval_acc": float(selected_source_eval_acc),
        "selected_target_eval_acc": float(selected_target_eval_acc),
        "selected_epoch": int(selected_epoch),
        "model_selection": selection_mode,
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
    figure_paths["tsne_domain"] = str(run_paths["figures_dir"] / "tsne_domain.png")
    figure_paths["tsne_class"] = str(run_paths["figures_dir"] / "tsne_class.png")
    figure_paths["confusion_matrix"] = str(run_paths["figures_dir"] / "confusion_matrix.png")
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
    method_name = str(method_payload.get("method_name", "")).lower()
    ensure_dependencies(method_name, experiment_payload)

    from src.datasets.te_da_dataset import TEDADatasetConfig
    from src.datasets.te_torch_dataset import prepare_benchmark_data

    data_config = TEDADatasetConfig.from_dict(data_payload)
    protocol_payload = deepcopy(data_payload.get("protocol", {}))
    protocol_payload.update(experiment_payload.get("protocol_override", {}))
    selected_fold = str(protocol_payload.get("preferred_fold", data_config.preferred_fold))

    set_seed(int(experiment_payload.get("seed", 42)))
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
    run_paths = build_run_paths(
        experiment_config=experiment_payload,
        method_name=method_name,
        scenario_id=scenario_id,
        backbone_name=backbone_name,
        fold_name=selected_fold,
        batch_root_name=args.batch_root_name,
    )
    prepared_data = prepare_benchmark_data(
        config=data_config,
        setting=setting,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        fold_name=selected_fold,
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
