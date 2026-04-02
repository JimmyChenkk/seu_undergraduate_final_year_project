"""Train benchmark-oriented TEP baselines and DA methods."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml

from src.datasets.te_da_dataset import TEDADatasetConfig, TEDADatasetInterface
from src.utils.run_layout import build_run_layout


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def merge_nested_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries conservatively."""

    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_nested_dict(result[key], value)
        else:
            result[key] = value
    return result


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dependencies(method_name: str) -> None:
    """Fail early with a concise message when the training stack is missing."""

    required = ["torch"]
    if method_name == "jdot":
        required.extend(["sklearn", "ot"])

    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
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
    }


def build_setting(data_config: TEDADatasetConfig, data_payload: dict[str, Any], experiment_payload: dict[str, Any]):
    """Build single-source or multi-source setting from config files."""

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


def _evaluate_target_accuracy(model, loader, device):
    import torch

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model.predict_logits(x_batch)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == y_batch).sum().item())
            total += int(y_batch.numel())
    return correct / max(total, 1)


def _collect_loader_outputs(model, loader, device, *, domain_name: str):
    """Collect logits, predictions, labels and embeddings for one loader."""

    import torch

    embeddings = []
    logits_list = []
    labels_list = []
    domains = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model.predict_logits(x_batch)
            features = model.extract_features(x_batch)
            embeddings.append(features.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(y_batch.numpy())
            domains.append(np.array([domain_name] * len(y_batch), dtype=object))

    if not embeddings:
        return {
            "embeddings": np.empty((0, 0), dtype=np.float32),
            "logits": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "predictions": np.empty((0,), dtype=np.int64),
            "domains": np.empty((0,), dtype=object),
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
) -> dict[str, Any]:
    """Persist embeddings and prediction traces for later figures."""

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    ensure_parent(analysis_path)
    source_chunks = []
    for split, loader in zip(prepared_data.source_splits, prepared_data.source_eval_loaders):
        chunk = _collect_loader_outputs(model, loader, device, domain_name=split.domain_id)
        source_chunks.append(chunk)

    target_chunk = _collect_loader_outputs(
        model,
        prepared_data.target_eval_loader,
        device,
        domain_name=prepared_data.target_split.domain_id,
    )

    source_embeddings = np.concatenate([chunk["embeddings"] for chunk in source_chunks], axis=0)
    source_labels = np.concatenate([chunk["labels"] for chunk in source_chunks], axis=0)
    source_predictions = np.concatenate([chunk["predictions"] for chunk in source_chunks], axis=0)
    source_domains = np.concatenate([chunk["domains"] for chunk in source_chunks], axis=0)

    np.savez_compressed(
        analysis_path,
        scenario_id=np.array([scenario_id], dtype=object),
        method_name=np.array([method_name], dtype=object),
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
    if bool(runtime.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    device = torch.device(device_name)

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

    history = []
    target_iterator = _cycle(prepared_data.target_train_loader)
    source_iterators = [_cycle(loader) for loader in prepared_data.source_train_loaders]
    steps_per_epoch = max(len(loader) for loader in prepared_data.source_train_loaders)
    if runtime.get("dry_run", False):
        steps_per_epoch = min(steps_per_epoch, 2)

    epochs = int(optimization.get("epochs", 1))
    for epoch_index in range(epochs):
        model.train()
        epoch_metrics = defaultdict(list)
        for _ in range(steps_per_epoch):
            source_batches = [next(iterator) for iterator in source_iterators]
            target_batch = next(target_iterator)

            source_batches = [
                (x_batch.to(device), y_batch.to(device))
                for x_batch, y_batch in source_batches
            ]
            target_batch = (target_batch[0].to(device), target_batch[1].to(device))

            optimizer.zero_grad()
            step_output = model.compute_loss(source_batches, target_batch)
            step_output.loss.backward()
            optimizer.step()

            for key, value in step_output.metrics.items():
                epoch_metrics[key].append(value)

        target_eval_acc = _evaluate_target_accuracy(model, prepared_data.target_eval_loader, device)
        summary = _mean_metrics(epoch_metrics)
        summary["epoch"] = epoch_index + 1
        summary["target_eval_acc"] = float(target_eval_acc)
        history.append(summary)

    result = {
        "method_name": str(method_config["method_name"]),
        "history": history,
        "best_target_eval_acc": float(max(item["target_eval_acc"] for item in history)),
        "final_target_eval_acc": float(history[-1]["target_eval_acc"]),
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
        )
        result.update(analysis_summary)

    result["device_used"] = str(device)
    result["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)

    return result


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
    ensure_dependencies(method_name)

    from src.datasets.te_torch_dataset import prepare_benchmark_data

    data_config = TEDADatasetConfig.from_dict(data_payload)
    protocol_payload = deepcopy(data_payload.get("protocol", {}))
    protocol_payload.update(experiment_payload.get("protocol_override", {}))
    selected_fold = str(protocol_payload.get("preferred_fold", data_config.preferred_fold))

    set_seed(int(experiment_payload.get("seed", 42)))
    setting = build_setting(data_config, data_payload, experiment_payload)
    batch_size = int(method_payload.get("optimization", {}).get("batch_size", 32))
    num_workers = int(experiment_payload.get("runtime", {}).get("num_workers", 0))
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
        fold_name=selected_fold,
    )

    if method_name == "jdot":
        from src.methods import run_jdot_experiment
        method_result = run_jdot_experiment(prepared_data, method_payload)
    else:
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
    save_json(run_paths["metrics_path"], result_payload)
    print(json.dumps(result_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
