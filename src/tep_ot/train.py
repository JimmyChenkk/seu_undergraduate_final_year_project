"""Training and evaluation loop for the focused TEP OT experiment runner."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .data import ExperimentData, make_loader
from .methods import BaseMethod, build_method
from .utils import append_csv_row, ensure_dir, fold_to_index, set_random_seed, task_label, timestamp, write_json


OT_METHODS = {
    "jdot",
    "otda",
    "tp_jdot",
    "cbtp_jdot",
    "wjdot",
    "tp_wjdot",
    "cbtp_wjdot",
    "ms_cbtp_wjdot",
}


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
    return {key: _mean(values) for key, values in metrics.items()}


def _move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[0].to(device), batch[1].to(device)


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def collect_outputs(
    model: BaseMethod,
    loader,
    device: torch.device,
    *,
    domain_id: str,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    embeddings = []
    logits = []
    labels = []
    domains = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            x_batch, y_batch = _move_batch(batch, device)
            batch_logits, batch_embeddings = model(x_batch)
            embeddings.append(batch_embeddings.detach().cpu().numpy())
            logits.append(batch_logits.detach().cpu().numpy())
            labels.append(y_batch.detach().cpu().numpy())
            domains.append(np.asarray([domain_id] * int(y_batch.shape[0])))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    if not embeddings:
        return {
            "embeddings": np.empty((0, 0), dtype=np.float32),
            "logits": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "predictions": np.empty((0,), dtype=np.int64),
            "domains": np.empty((0,), dtype="<U1"),
        }
    logits_np = np.concatenate(logits, axis=0)
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
        "logits": logits_np,
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
        "predictions": logits_np.argmax(axis=1).astype(np.int64),
        "domains": np.concatenate(domains, axis=0),
    }


def evaluate_target(
    model: BaseMethod,
    loader,
    device: torch.device,
    *,
    domain_id: str,
    num_classes: int,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    outputs = collect_outputs(model, loader, device, domain_id=domain_id, max_batches=max_batches)
    labels = outputs["labels"]
    predictions = outputs["predictions"]
    if labels.size == 0:
        metrics = {"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0}
    else:
        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        }
    outputs["confusion_matrix"] = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    return metrics, outputs


def save_per_class_recall(path: Path, confusion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "support", "recall"])
        writer.writeheader()
        for class_id, (support, recall) in enumerate(zip(totals, recalls)):
            writer.writerow({"class_id": class_id, "support": int(support), "recall": float(recall)})


def save_confusion_plot(path: Path, confusion: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(confusion, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        ax.set_xticks(range(0, confusion.shape[1], 4))
        ax.set_yticks(range(0, confusion.shape[0], 4))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception:
        return


def save_weight_tables(
    *,
    run_root: Path,
    history: list[dict[str, float]],
    snapshot: dict[str, Any],
    source_domains: list[str],
    num_classes: int,
) -> tuple[str | None, str | None]:
    tables_dir = ensure_dir(run_root / "tables")
    source_curve_path: str | None = None
    class_heatmap_path: str | None = None

    alpha_keys = sorted(key for key in history[-1].keys() if key.startswith("alpha_source_")) if history else []
    if alpha_keys:
        source_curve = tables_dir / "source_weights.csv"
        with source_curve.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["epoch"] + [source_domains[int(key.rsplit("_", 1)[-1])] for key in alpha_keys]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(
                    {
                        "epoch": int(row.get("epoch", 0)),
                        **{
                            source_domains[int(key.rsplit("_", 1)[-1])]: float(row.get(key, 0.0))
                            for key in alpha_keys
                        },
                    }
                )
        source_curve_path = str(source_curve)

    class_weights = snapshot.get("class_source_weights")
    if class_weights is not None:
        class_weights = np.asarray(class_weights)
        class_heatmap = tables_dir / "class_source_weights.csv"
        with class_heatmap.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["class_id"] + source_domains
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for class_id in range(num_classes):
                writer.writerow(
                    {
                        "class_id": class_id,
                        **{
                            source_domains[source_index]: float(class_weights[source_index, class_id])
                            for source_index in range(min(len(source_domains), class_weights.shape[0]))
                        },
                    }
                )
        class_heatmap_path = str(class_heatmap)

    source_weights = snapshot.get("source_weights")
    if source_weights is not None and not alpha_keys:
        source_curve = tables_dir / "source_weights.csv"
        with source_curve.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_domain", "weight"])
            writer.writeheader()
            for domain_id, weight in zip(source_domains, np.asarray(source_weights).tolist()):
                writer.writerow({"source_domain": domain_id, "weight": float(weight)})
        source_curve_path = str(source_curve)

    return source_curve_path, class_heatmap_path


def save_analysis(
    *,
    run_root: Path,
    model: BaseMethod,
    source_eval_loaders: list[Any],
    target_eval_loader: Any,
    data: ExperimentData,
    device: torch.device,
    num_classes: int,
    max_batches: int | None,
    target_outputs: dict[str, np.ndarray],
) -> str:
    source_chunks = [
        collect_outputs(model, loader, device, domain_id=split.domain_id, max_batches=max_batches)
        for loader, split in zip(source_eval_loaders, data.source_splits)
    ]
    source_embeddings = np.concatenate([chunk["embeddings"] for chunk in source_chunks], axis=0)
    source_labels = np.concatenate([chunk["labels"] for chunk in source_chunks], axis=0)
    source_predictions = np.concatenate([chunk["predictions"] for chunk in source_chunks], axis=0)
    source_domains = np.concatenate([chunk["domains"] for chunk in source_chunks], axis=0)
    analysis_path = run_root / "artifacts" / "embeddings.npz"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        analysis_path,
        source_embeddings=source_embeddings,
        source_labels=source_labels,
        source_predictions=source_predictions,
        source_domains=source_domains,
        target_embeddings=target_outputs["embeddings"],
        target_labels=target_outputs["labels"],
        target_predictions=target_outputs["predictions"],
        target_logits=target_outputs["logits"],
        target_domains=np.asarray([data.target_domain] * len(target_outputs["labels"])),
        num_classes=np.asarray([num_classes]),
    )
    return str(analysis_path)


def run_experiment(
    *,
    data: ExperimentData,
    method_name: str,
    fold: int | str,
    seed: int,
    output_dir: str | Path = "runs/tep_ot",
    aggregate_csv: str | Path | None = None,
    device_name: str = "auto",
    epochs: int = 20,
    pretrain_epochs: int | None = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    max_train_batches: int | None = None,
    eval_max_batches: int | None = None,
    embedding_dim: int = 128,
    dropout: float = 0.1,
    adaptation_weight: float = 1.0,
    ot_solver: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
    prototype_weight: float | None = None,
) -> dict[str, Any]:
    """Train one method and persist metrics/artifacts under ``output_dir``."""

    set_random_seed(seed)
    normalized_method = method_name.strip().lower()
    if pretrain_epochs is None:
        pretrain_epochs = 5 if normalized_method in OT_METHODS else 0
    target_reference_methods = {
        "target_only",
        "target_ref",
        "target_supervised_reference",
        "target_oracle_matched",
    }
    if normalized_method in {"source_only", *target_reference_methods}:
        pretrain_epochs = 0

    device_name = "cuda:0" if device_name == "auto" and torch.cuda.is_available() else device_name
    if device_name == "auto":
        device_name = "cpu"
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    task = task_label(data.source_domains, data.target_domain)
    run_name = f"{timestamp()}_{normalized_method}_{task.replace('->', '_to_').replace('+', '-')}_fold{fold_to_index(fold)}_seed{seed}"
    run_root = ensure_dir(Path(output_dir) / run_name)
    tables_dir = ensure_dir(run_root / "tables")
    logs_dir = ensure_dir(run_root / "logs")
    figures_dir = ensure_dir(run_root / "figures")
    checkpoints_dir = ensure_dir(run_root / "checkpoints")

    input_shape = tuple(int(item) for item in data.target_split.train_x.shape[1:])
    method = build_method(
        normalized_method,
        num_classes=data.num_classes,
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        dropout=dropout,
        adaptation_weight=adaptation_weight,
        ot_solver=ot_solver,
        sinkhorn_reg=sinkhorn_reg,
        prototype_weight=prototype_weight,
        tau_steps=max(int(epochs) * max(1, max(len(split.train_x) // max(batch_size, 1) for split in data.source_splits)), 1),
    ).to(device)

    optimizer = torch.optim.Adam(method.parameters(), lr=learning_rate, weight_decay=weight_decay)
    source_loaders = [
        make_loader(split.train_x, split.train_y, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for split in data.source_splits
    ]
    source_eval_loaders = [
        make_loader(split.train_x, split.train_y, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        for split in data.source_splits
    ]
    target_train_loader = make_loader(
        data.target_split.train_x,
        data.target_split.train_y,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        hide_labels=normalized_method not in target_reference_methods,
    )
    target_eval_loader = make_loader(
        data.target_split.eval_x,
        data.target_split.eval_y,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        hide_labels=False,
        drop_last=False,
    )

    config_payload = {
        "method": normalized_method,
        "sources": data.source_domains,
        "target": data.target_domain,
        "task": task,
        "fold": fold_to_index(fold),
        "raw_fold_name": data.target_split.fold_name,
        "seed": seed,
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(device),
        "normalization": "domain-level per-domain standardization",
        "target_eval_labels_policy": "used only after training for final metrics",
        "target_train_labels_hidden": normalized_method not in target_reference_methods,
    }
    write_json(run_root / "config.json", config_payload)

    history: list[dict[str, float]] = []
    source_iterators = [_cycle(loader) for loader in source_loaders]
    target_iterator = _cycle(target_train_loader)

    for pre_epoch in range(pretrain_epochs):
        method.train()
        epoch_metrics: dict[str, list[float]] = defaultdict(list)
        steps = max(len(loader) for loader in source_loaders)
        if max_train_batches is not None:
            steps = min(steps, int(max_train_batches))
        for _ in range(steps):
            source_batches = [_move_batch(next(iterator), device) for iterator in source_iterators]
            optimizer.zero_grad(set_to_none=True)
            output = method.source_supervised_loss(source_batches)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(method.parameters(), 5.0)
            optimizer.step()
            for key, value in output.metrics.items():
                epoch_metrics[key].append(float(value))
        summary = _mean_metrics(epoch_metrics)
        summary["phase"] = "pretrain"  # type: ignore[assignment]
        summary["epoch"] = pre_epoch + 1  # type: ignore[assignment]
        _write_jsonl(logs_dir / "train_log.jsonl", summary)

    for epoch in range(epochs):
        method.train()
        epoch_metrics = defaultdict(list)
        steps = len(target_train_loader) if normalized_method in target_reference_methods else max(len(loader) for loader in source_loaders)
        if max_train_batches is not None:
            steps = min(steps, int(max_train_batches))
        for _ in range(steps):
            source_batches = [_move_batch(next(iterator), device) for iterator in source_iterators]
            target_batch = _move_batch(next(target_iterator), device)
            optimizer.zero_grad(set_to_none=True)
            output = method.compute_loss(source_batches, target_batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(method.parameters(), 5.0)
            optimizer.step()
            for key, value in output.metrics.items():
                epoch_metrics[key].append(float(value))
        summary = _mean_metrics(epoch_metrics)
        summary["phase"] = "train"  # type: ignore[assignment]
        summary["epoch"] = epoch + 1  # type: ignore[assignment]
        history.append({key: value for key, value in summary.items() if isinstance(value, (int, float))})
        _write_jsonl(logs_dir / "train_log.jsonl", summary)

    final_metrics, target_outputs = evaluate_target(
        method,
        target_eval_loader,
        device,
        domain_id=data.target_domain,
        num_classes=data.num_classes,
        max_batches=eval_max_batches,
    )
    confusion = target_outputs["confusion_matrix"]
    confusion_path = tables_dir / "confusion_matrix.npy"
    np.save(confusion_path, confusion)
    confusion_csv_path = tables_dir / "confusion_matrix.csv"
    np.savetxt(confusion_csv_path, confusion, delimiter=",", fmt="%d")
    per_class_recall_path = tables_dir / "per_class_recall.csv"
    save_per_class_recall(per_class_recall_path, confusion)
    save_confusion_plot(figures_dir / "confusion_matrix.png", confusion)
    analysis_path = save_analysis(
        run_root=run_root,
        model=method,
        source_eval_loaders=source_eval_loaders,
        target_eval_loader=target_eval_loader,
        data=data,
        device=device,
        num_classes=data.num_classes,
        max_batches=eval_max_batches,
        target_outputs=target_outputs,
    )

    source_weight_path, class_source_weight_path = save_weight_tables(
        run_root=run_root,
        history=history,
        snapshot=method.reliability_snapshot(),
        source_domains=data.source_domains,
        num_classes=data.num_classes,
    )
    checkpoint_path = checkpoints_dir / "model.pt"
    torch.save({"model_state_dict": method.state_dict(), "config": config_payload}, checkpoint_path)

    result_row = {
        "method": normalized_method,
        "sources": "+".join(data.source_domains),
        "target": data.target_domain,
        "task": task,
        "fold": fold_to_index(fold),
        "seed": seed,
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "balanced_accuracy": final_metrics["balanced_accuracy"],
        "per_class_recall_path": str(per_class_recall_path),
        "confusion_matrix_path": str(confusion_path),
        "analysis_path": analysis_path,
        "source_weight_path": source_weight_path or "",
        "class_source_weight_path": class_source_weight_path or "",
        "checkpoint_path": str(checkpoint_path),
        "run_root": str(run_root),
    }
    append_csv_row(tables_dir / "result.csv", result_row)
    if aggregate_csv is not None:
        append_csv_row(Path(aggregate_csv), result_row)

    write_json(
        tables_dir / "result.json",
        {
            "config": config_payload,
            "metrics": final_metrics,
            "result_row": result_row,
            "history": history,
        },
    )
    return {
        "run_root": str(run_root),
        "metrics": final_metrics,
        "result_row": result_row,
    }
