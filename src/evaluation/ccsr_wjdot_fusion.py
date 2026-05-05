"""Prediction-level CCSR-WJDOT fusion utilities."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


EPS = 1e-8


def _import_numpy():
    import numpy as np

    return np


def _string_array(values: list[str]):
    np = _import_numpy()
    items = [str(value) for value in values]
    if not items:
        return np.empty((0,), dtype="<U1")
    max_length = max(len(item) for item in items)
    return np.asarray(items, dtype=f"<U{max(max_length, 1)}")


def _as_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    ccsr_payload = payload.get("ccsr")
    if isinstance(ccsr_payload, dict):
        merged = dict(payload)
        merged.update(ccsr_payload)
        return merged
    return dict(payload)


def _normalize_rows(values):
    np = _import_numpy()
    array = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, EPS)


def _softmax(logits, axis: int = -1):
    np = _import_numpy()
    safe = np.nan_to_num(logits.astype(np.float64, copy=False), nan=-1e9, posinf=1e9, neginf=-1e9)
    safe = safe - np.max(safe, axis=axis, keepdims=True)
    exp_values = np.exp(safe)
    return (exp_values / np.maximum(exp_values.sum(axis=axis, keepdims=True), EPS)).astype(np.float32)


def _entropy(probabilities):
    np = _import_numpy()
    class_count = max(int(probabilities.shape[1]), 2)
    values = -np.sum(probabilities * np.log(np.maximum(probabilities, EPS)), axis=1)
    return (values / math.log(class_count)).astype(np.float32)


def _class_prototypes(embeddings, labels, *, num_classes: int):
    np = _import_numpy()
    normalized = _normalize_rows(embeddings)
    prototypes = np.zeros((num_classes, normalized.shape[1]), dtype=np.float32)
    present = np.zeros((num_classes,), dtype=bool)
    labels = labels.astype(np.int64, copy=False)
    for class_id in range(num_classes):
        mask = labels == class_id
        if not np.any(mask):
            continue
        mean_value = normalized[mask].mean(axis=0, keepdims=True)
        prototypes[class_id] = _normalize_rows(mean_value)[0]
        present[class_id] = True
    return prototypes, present


def _squared_distances(left, right):
    np = _import_numpy()
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_norm + right_norm - 2.0 * left.dot(right.T)
    return np.maximum(np.nan_to_num(distances, nan=1e6, posinf=1e6, neginf=0.0), 0.0).astype(np.float32)


def _prototype_probabilities(embeddings, prototypes, present, *, temperature: float):
    np = _import_numpy()
    normalized = _normalize_rows(embeddings)
    distances = _squared_distances(normalized, prototypes.astype(np.float32, copy=False))
    if present.shape[0] != distances.shape[1]:
        raise ValueError("Prototype present mask does not match prototype matrix.")
    missing = ~present.astype(bool, copy=False)
    if np.any(missing):
        distances[:, missing] = 1e6
    if not np.any(present):
        probabilities = np.full(distances.shape, 1.0 / max(distances.shape[1], 1), dtype=np.float32)
        return probabilities, distances
    probabilities = _softmax(-distances / max(float(temperature), EPS), axis=1)
    return probabilities, distances


def _collect_outputs(
    *,
    model,
    loader,
    device,
    domain_name: str,
    max_batches: int | None,
    non_blocking: bool,
    amp_enabled: bool,
):
    np = _import_numpy()
    import torch

    embeddings = []
    logits = []
    expert_probabilities = []
    labels = []
    domains = []
    model.eval()
    with torch.inference_mode():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                batch_logits, batch_features = model(x_batch)
                if hasattr(model, "source_expert_probabilities"):
                    batch_expert_probabilities = model.source_expert_probabilities(x_batch)
                else:
                    batch_expert_probabilities = None
            embeddings.append(batch_features.detach().float().cpu().numpy())
            logits.append(batch_logits.detach().float().cpu().numpy())
            if batch_expert_probabilities is not None:
                expert_probabilities.append(batch_expert_probabilities.detach().float().cpu().numpy())
            labels.append(y_batch.detach().cpu().numpy())
            domains.append(_string_array([domain_name] * int(y_batch.shape[0])))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    if not embeddings:
        return {
            "embeddings": np.empty((0, 0), dtype=np.float32),
            "logits": np.empty((0, 0), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "predictions": np.empty((0,), dtype=np.int64),
            "domains": _string_array([]),
            "expert_probabilities": None,
        }
    logits_array = np.concatenate(logits, axis=0).astype(np.float32, copy=False)
    expert_array = None
    if expert_probabilities:
        expert_array = np.concatenate(expert_probabilities, axis=1).astype(np.float32, copy=False)
    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32, copy=False),
        "logits": logits_array,
        "labels": np.concatenate(labels, axis=0).astype(np.int64, copy=False),
        "predictions": logits_array.argmax(axis=1).astype(np.int64, copy=False),
        "domains": np.concatenate(domains, axis=0),
        "expert_probabilities": expert_array,
    }


def _minmax_by_class(values, present):
    np = _import_numpy()
    normalized = np.zeros_like(values, dtype=np.float32)
    for class_id in range(values.shape[1]):
        valid = np.isfinite(values[:, class_id]) & present[:, class_id]
        if not np.any(valid):
            normalized[:, class_id] = 1.0
            continue
        column = values[valid, class_id]
        min_value = float(np.min(column))
        max_value = float(np.max(column))
        if max_value - min_value <= EPS:
            normalized[valid, class_id] = 0.0
        else:
            normalized[valid, class_id] = (column - min_value) / (max_value - min_value)
        normalized[~valid, class_id] = 1.0
    return normalized


def _resolve_alpha(reliability, *, source_count: int, temperature: float, floor: float, top_m: int):
    np = _import_numpy()
    raw = _softmax(-reliability / max(float(temperature), EPS), axis=0)
    alpha = np.zeros_like(raw, dtype=np.float32)
    for class_id in range(raw.shape[1]):
        next_values = np.maximum(raw[:, class_id], float(floor))
        if 0 < top_m < source_count:
            keep = np.argsort(raw[:, class_id])[-int(top_m):]
            filtered = np.full((source_count,), float(floor), dtype=np.float32)
            filtered[keep] = np.maximum(raw[keep, class_id], float(floor))
            next_values = filtered
        alpha[:, class_id] = next_values / max(float(next_values.sum()), EPS)
    return raw.astype(np.float32), alpha.astype(np.float32)


def _target_prototypes_from_predictions(
    *,
    target_embeddings,
    source_target_probabilities,
    tau_proto: float,
    min_samples: int,
):
    np = _import_numpy()
    source_count, _, num_classes = source_target_probabilities.shape
    target_features = _normalize_rows(target_embeddings)
    mean_probabilities = source_target_probabilities.mean(axis=0)
    confidence = mean_probabilities.max(axis=1)
    provisional = mean_probabilities.argmax(axis=1)
    prototypes = np.zeros((num_classes, target_features.shape[1]), dtype=np.float32)
    source = []
    counts = np.zeros((num_classes,), dtype=np.int64)
    for class_id in range(num_classes):
        high_confidence = (provisional == class_id) & (confidence >= float(tau_proto))
        counts[class_id] = int(high_confidence.sum())
        if counts[class_id] >= int(min_samples):
            prototypes[class_id] = _normalize_rows(target_features[high_confidence].mean(axis=0, keepdims=True))[0]
            source.append("high_confidence")
            continue

        barycenters = []
        for source_index in range(source_count):
            weights = source_target_probabilities[source_index, :, class_id].astype(np.float32, copy=False)
            mass = float(weights.sum())
            if mass <= EPS:
                continue
            barycenter = (target_features * weights.reshape(-1, 1)).sum(axis=0, keepdims=True) / mass
            barycenters.append(_normalize_rows(barycenter)[0])
        if barycenters:
            prototypes[class_id] = _normalize_rows(np.asarray(barycenters, dtype=np.float32).mean(axis=0, keepdims=True))[0]
            source.append("probability_proxy")
        elif target_features.shape[0] > 0:
            prototypes[class_id] = _normalize_rows(target_features.mean(axis=0, keepdims=True))[0]
            source.append("target_global_fallback")
        else:
            source.append("empty_target")
    return prototypes, source, counts, mean_probabilities, provisional


def _compute_source_eval_error(
    *,
    source_eval_chunks: list[dict[str, Any]],
    source_prototypes,
    source_present,
    proto_temperature: float,
    num_classes: int,
):
    np = _import_numpy()
    source_count = len(source_eval_chunks)
    errors = np.ones((source_count, num_classes), dtype=np.float32)
    recalls = np.zeros((source_count, num_classes), dtype=np.float32)
    supports = np.zeros((source_count, num_classes), dtype=np.int64)
    for source_index, chunk in enumerate(source_eval_chunks):
        labels = chunk["labels"].astype(np.int64, copy=False)
        if labels.size == 0:
            continue
        probabilities, _ = _prototype_probabilities(
            chunk["embeddings"],
            source_prototypes[source_index],
            source_present[source_index],
            temperature=proto_temperature,
        )
        predictions = probabilities.argmax(axis=1)
        for class_id in range(num_classes):
            mask = labels == class_id
            supports[source_index, class_id] = int(mask.sum())
            if not np.any(mask):
                errors[source_index, class_id] = 1.0
                continue
            recall = float((predictions[mask] == class_id).mean())
            recalls[source_index, class_id] = recall
            errors[source_index, class_id] = 1.0 - recall
    return errors, recalls, supports


def _write_matrix_csv(path: Path, matrix, *, source_domains: list[str], value_name: str = "value") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", *source_domains])
        for class_id in range(matrix.shape[1]):
            writer.writerow([class_id, *[float(matrix[source_index, class_id]) for source_index in range(matrix.shape[0])]])
    return str(path)


def _save_heatmap(
    path: Path,
    matrix,
    *,
    title: str,
    source_domains: list[str],
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str | None:
    try:
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        width = max(8.0, matrix.shape[1] * 0.32)
        height = max(3.2, matrix.shape[0] * 0.55)
        fig, ax = plt.subplots(figsize=(width, height))
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(image, ax=ax, label="score")
        ax.set_title(title)
        ax.set_xlabel("class id")
        ax.set_ylabel("source domain")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels([str(item) for item in range(matrix.shape[1])], rotation=90, fontsize=7)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(source_domains)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _save_component_heatmaps(path: Path, components: dict[str, Any], *, source_domains: list[str]) -> str | None:
    try:
        import matplotlib.pyplot as plt

        names = [
            "D_proto_norm",
            "D_ot_norm",
            "H_pred_norm",
            "E_src_norm",
            "R",
            "alpha",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(14, max(5.5, len(source_domains) * 0.9)))
        for axis, name in zip(axes.ravel(), names):
            matrix = components[name]
            image = axis.imshow(matrix, aspect="auto", cmap="viridis")
            axis.set_title(name)
            axis.set_xlabel("class id")
            axis.set_ylabel("source")
            axis.set_yticks(range(matrix.shape[0]))
            axis.set_yticklabels(source_domains)
            axis.set_xticks(range(matrix.shape[1]))
            axis.set_xticklabels([str(item) for item in range(matrix.shape[1])], rotation=90, fontsize=6)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _save_prediction_histogram(path: Path, predictions, *, num_classes: int, title: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
        np = _import_numpy()

        path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.bincount(predictions.astype(np.int64, copy=False), minlength=num_classes)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.bar(np.arange(num_classes), counts, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("count")
        ax.set_xticks(np.arange(num_classes))
        ax.set_xticklabels([str(item) for item in range(num_classes)], rotation=90, fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _save_global_vs_class_alpha(path: Path, global_alpha, class_alpha, *, source_domains: list[str]) -> str | None:
    try:
        import matplotlib.pyplot as plt
        np = _import_numpy()

        path.parent.mkdir(parents=True, exist_ok=True)
        mean_class_alpha = class_alpha.mean(axis=1)
        x_values = np.arange(len(source_domains))
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(6.0, len(source_domains) * 1.0), 3.8))
        ax.bar(x_values - width / 2.0, global_alpha, width=width, label="global alpha proxy", color="#4C78A8")
        ax.bar(x_values + width / 2.0, mean_class_alpha, width=width, label="mean class alpha", color="#F58518")
        ax.set_xticks(x_values)
        ax.set_xticklabels(source_domains, rotation=15)
        ax.set_ylim(0.0, max(1.0, float(max(global_alpha.max(), mean_class_alpha.max())) * 1.15))
        ax.set_ylabel("weight")
        ax.set_title("Global vs class-conditional source weights")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _save_entropy_rho_distribution(path: Path, entropy_values, rho_values) -> str | None:
    try:
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].hist(entropy_values, bins=30, color="#59A14F", alpha=0.85)
        axes[0].set_title("CCSR entropy")
        axes[0].set_xlabel("normalized entropy")
        axes[0].set_ylabel("count")
        axes[1].hist(rho_values, bins=30, color="#E15759", alpha=0.85)
        axes[1].set_title("Safety rho")
        axes[1].set_xlabel("rho")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _save_confusion_comparison(path: Path, wjdot_matrix, raw_ccsr_matrix, final_matrix) -> str | None:
    try:
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
        for axis, matrix, title in [
            (axes[0], wjdot_matrix, "WJDOT confusion"),
            (axes[1], raw_ccsr_matrix, "Raw CCSR confusion"),
            (axes[2], final_matrix, "Final CCSR-WJDOT confusion"),
        ]:
            image = axis.imshow(matrix, aspect="auto", cmap="Blues")
            axis.set_title(title)
            axis.set_xlabel("predicted class")
            axis.set_ylabel("true class")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def _per_class_recall(labels, predictions, *, num_classes: int):
    np = _import_numpy()
    recalls = np.zeros((num_classes,), dtype=np.float32)
    supports = np.zeros((num_classes,), dtype=np.int64)
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    for class_id in range(num_classes):
        mask = labels == class_id
        supports[class_id] = int(mask.sum())
        if np.any(mask):
            recalls[class_id] = float((predictions[mask] == class_id).mean())
    return recalls, supports


def _top2_margin(probabilities):
    np = _import_numpy()
    if probabilities.shape[1] <= 1:
        return np.ones((probabilities.shape[0],), dtype=np.float32)
    sorted_values = np.sort(probabilities, axis=1)
    return (sorted_values[:, -1] - sorted_values[:, -2]).astype(np.float32)


def _weighted_source_agreement(source_probabilities, alpha, predicted_classes):
    np = _import_numpy()
    source_top1 = source_probabilities.argmax(axis=2)
    agreement = np.zeros((source_probabilities.shape[1],), dtype=np.float32)
    for sample_index, class_id in enumerate(predicted_classes.astype(np.int64, copy=False)):
        weights = alpha[:, class_id]
        mask = source_top1[:, sample_index] == class_id
        agreement[sample_index] = float(weights[mask].sum() / max(float(weights.sum()), EPS))
    return agreement


def _alpha_concentration_by_prediction(alpha, predicted_classes):
    np = _import_numpy()
    source_count = int(alpha.shape[0])
    concentration = np.ones((predicted_classes.shape[0],), dtype=np.float32)
    if source_count <= 1:
        return concentration
    for sample_index, class_id in enumerate(predicted_classes.astype(np.int64, copy=False)):
        class_alpha = alpha[:, class_id]
        entropy = -float(np.sum(class_alpha * np.log(np.maximum(class_alpha, EPS)))) / math.log(source_count)
        concentration[sample_index] = max(0.0, min(1.0, 1.0 - entropy))
    return concentration


def _reliability_support_by_prediction(reliability, alpha, predicted_classes):
    np = _import_numpy()
    support = np.zeros((predicted_classes.shape[0],), dtype=np.float32)
    for sample_index, class_id in enumerate(predicted_classes.astype(np.int64, copy=False)):
        class_reliability = np.clip(reliability[:, class_id], 0.0, 1.0)
        support[sample_index] = float(np.sum(alpha[:, class_id] * (1.0 - class_reliability)))
    return np.clip(support, 0.0, 1.0).astype(np.float32)


def _apply_calibrated_override(
    *,
    p_wjdot,
    p_ccsr,
    source_probabilities,
    alpha,
    reliability,
    lambda_same: float,
    lambda_override: float,
    override_threshold: float,
    score_weights: dict[str, float] | None = None,
):
    np = _import_numpy()
    weights = {
        "entropy_delta": 0.25,
        "margin": 0.25,
        "source_agreement": 0.20,
        "alpha_concentration": 0.15,
        "reliability_support": 0.15,
    }
    if isinstance(score_weights, dict):
        for key, value in score_weights.items():
            if key in weights:
                weights[key] = float(value)
    total_weight = max(float(sum(max(value, 0.0) for value in weights.values())), EPS)
    p_w = p_wjdot / np.maximum(p_wjdot.sum(axis=1, keepdims=True), EPS)
    p_c = p_ccsr / np.maximum(p_ccsr.sum(axis=1, keepdims=True), EPS)
    pred_w = p_w.argmax(axis=1).astype(np.int64)
    pred_c = p_c.argmax(axis=1).astype(np.int64)
    same_top1 = pred_w == pred_c
    entropy_w = _entropy(p_w)
    entropy_c = _entropy(p_c)
    margin_c = _top2_margin(p_c)
    source_agreement = _weighted_source_agreement(source_probabilities, alpha, pred_c)
    alpha_concentration = _alpha_concentration_by_prediction(alpha, pred_c)
    reliability_support = _reliability_support_by_prediction(reliability, alpha, pred_c)
    entropy_delta = np.clip((entropy_w - entropy_c + 1.0) / 2.0, 0.0, 1.0)
    override_score = (
        max(weights["entropy_delta"], 0.0) * entropy_delta
        + max(weights["margin"], 0.0) * margin_c
        + max(weights["source_agreement"], 0.0) * source_agreement
        + max(weights["alpha_concentration"], 0.0) * alpha_concentration
        + max(weights["reliability_support"], 0.0) * reliability_support
    ) / total_weight
    p_final = p_w.copy()
    lambda_same = min(max(float(lambda_same), 0.0), 1.0)
    lambda_override = min(max(float(lambda_override), 0.0), 1.0)
    if np.any(same_top1):
        p_final[same_top1] = (1.0 - lambda_same) * p_w[same_top1] + lambda_same * p_c[same_top1]
    override_gate = (~same_top1) & (override_score > float(override_threshold))
    if np.any(override_gate):
        p_final[override_gate] = (
            (1.0 - lambda_override) * p_w[override_gate]
            + lambda_override * p_c[override_gate]
        )
    p_final = p_final / np.maximum(p_final.sum(axis=1, keepdims=True), EPS)
    return {
        "p_final": p_final.astype(np.float32),
        "override_gate": override_gate.astype(bool),
        "override_score": override_score.astype(np.float32),
        "same_top1": same_top1.astype(bool),
        "entropy_w": entropy_w.astype(np.float32),
        "entropy_c": entropy_c.astype(np.float32),
        "margin_c": margin_c.astype(np.float32),
        "source_agreement": source_agreement.astype(np.float32),
        "alpha_concentration": alpha_concentration.astype(np.float32),
        "reliability_support": reliability_support.astype(np.float32),
    }


def _balanced_accuracy(labels, predictions, *, num_classes: int):
    np = _import_numpy()
    recalls = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if np.any(mask):
            recalls.append(float((predictions[mask] == class_id).mean()))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def _source_meta_calibrate_gate(
    *,
    source_eval_chunks: list[dict[str, Any]],
    source_eval_probabilities,
    reliability,
    source_domains: list[str],
    default_params: dict[str, Any],
    enabled: bool,
):
    np = _import_numpy()
    if not enabled or source_eval_probabilities is None:
        return [], dict(default_params)
    source_count = int(source_eval_probabilities.shape[0])
    if source_count <= 1:
        return [], dict(default_params)
    lambda_same_grid = [0.3, 0.5, 0.7]
    lambda_override_grid = [0.6, 0.8, 0.9]
    threshold_grid = [0.45, 0.55, 0.65]
    temperature_grid = [0.3, 0.5, 0.8, 1.0]
    floor_grid = [0.01, 0.03, 0.05]
    top_m_grid = [source_count] if source_count <= 2 else [2, 3]
    best_params = dict(default_params)
    best_score = float("-inf")
    rows = []
    labels_by_source = [chunk["labels"].astype(np.int64, copy=False) for chunk in source_eval_chunks]
    for temperature in temperature_grid:
        alpha_raw, alpha_candidate = _resolve_alpha(
            reliability,
            source_count=source_count,
            temperature=temperature,
            floor=float(default_params.get("floor", 0.05)),
            top_m=int(default_params.get("top_m_per_class", 0)),
        )
        del alpha_raw
        for top_m in top_m_grid:
            for floor in floor_grid:
                _, alpha_candidate = _resolve_alpha(
                    reliability,
                    source_count=source_count,
                    temperature=temperature,
                    floor=floor,
                    top_m=0 if top_m >= source_count else top_m,
                )
                for lambda_same in lambda_same_grid:
                    for lambda_override in lambda_override_grid:
                        for threshold in threshold_grid:
                            task_scores = []
                            for pseudo_target_index in range(source_count):
                                labels = labels_by_source[pseudo_target_index]
                                if labels.size == 0:
                                    continue
                                expert_probs = source_eval_probabilities[:, pseudo_target_index]
                                keep = [idx for idx in range(source_count) if idx != pseudo_target_index]
                                if not keep:
                                    continue
                                local_probs = expert_probs[keep]
                                local_alpha = alpha_candidate[keep]
                                local_alpha = local_alpha / np.maximum(local_alpha.sum(axis=0, keepdims=True), EPS)
                                p_w = local_probs.mean(axis=0)
                                p_c = np.einsum("kc,knc->nc", local_alpha, local_probs)
                                p_c = p_c / np.maximum(p_c.sum(axis=1, keepdims=True), EPS)
                                local_reliability = reliability[keep]
                                gate = _apply_calibrated_override(
                                    p_wjdot=p_w,
                                    p_ccsr=p_c,
                                    source_probabilities=local_probs,
                                    alpha=local_alpha,
                                    reliability=local_reliability,
                                    lambda_same=lambda_same,
                                    lambda_override=lambda_override,
                                    override_threshold=threshold,
                                )
                                predictions = gate["p_final"].argmax(axis=1)
                                task_scores.append(
                                    _balanced_accuracy(labels, predictions, num_classes=reliability.shape[1])
                                )
                            if not task_scores:
                                continue
                            mean_score = float(np.mean(task_scores))
                            row = {
                                "lambda_same": lambda_same,
                                "lambda_override": lambda_override,
                                "override_threshold": threshold,
                                "T_class": temperature,
                                "top_m_per_class": 0 if top_m >= source_count else top_m,
                                "floor": floor,
                                "mean_balanced_accuracy": mean_score,
                                "pseudo_task_count": len(task_scores),
                            }
                            rows.append(row)
                            if mean_score > best_score:
                                best_score = mean_score
                                best_params = {
                                    **default_params,
                                    "lambda_same": lambda_same,
                                    "lambda_override": lambda_override,
                                    "override_threshold": threshold,
                                    "T_class": temperature,
                                    "top_m_per_class": 0 if top_m >= source_count else top_m,
                                    "floor": floor,
                                    "meta_balanced_accuracy": mean_score,
                                    "meta_source_domains": list(source_domains),
                                }
    return rows, best_params


def export_ccsr_wjdot_fusion_artifacts(
    *,
    model,
    prepared_data,
    device,
    analysis_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    scenario_id: str,
    method_name: str,
    ccsr_config: dict[str, Any] | None = None,
    max_batches: int | None = None,
    non_blocking: bool = False,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    """Run target-label-free CCSR fusion and save final evaluation artifacts.

    Reliability estimation uses source labels and unlabeled target-train samples.
    Target-eval labels are used only after final predictions are fixed.
    """

    np = _import_numpy()
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    config = _as_config(ccsr_config)
    num_classes = int(config.get("num_classes", 29))
    proto_temperature = float(config.get("proto_probability_temperature", 0.20))
    tau_proto = float(config.get("tau_proto", 0.85))
    min_proto_samples = int(config.get("min_proto_samples", 3))
    lambda_safe = float(config.get("lambda_safe", 0.5))
    rho_min = float(config.get("rho_min", 0.1))
    rho_max = float(config.get("rho_max", 0.8))
    rho_mode = str(config.get("rho_mode", "simple")).strip().lower()
    override_enabled = bool(config.get("override_on_source_agreement", True))
    override_agreement_config = config.get("override_agreement_threshold")
    override_entropy_threshold = float(config.get("override_entropy_threshold", 0.99))
    override_alpha_concentration_min = float(config.get("override_alpha_concentration_min", 0.0))
    override_strength = min(max(float(config.get("override_strength", 1.0)), 0.0), 1.0)
    weights = {
        "D_proto": float(config.get("w_proto", config.get("w1", 0.35))),
        "D_ot": float(config.get("w_ot", config.get("w2", 0.35))),
        "H_pred": float(config.get("w_entropy", config.get("w3", 0.20))),
        "E_src": float(config.get("w_source_error", config.get("w4", 0.10))),
    }

    source_domains = [split.domain_id for split in prepared_data.source_splits]
    source_count = len(source_domains)
    override_agreement_default = 1.0 if source_count <= 2 else 0.6
    override_agreement_threshold = (
        float(override_agreement_config)
        if override_agreement_config is not None
        else override_agreement_default
    )
    class_temperature_default = 1.0 if source_count <= 2 else 0.5
    class_temperature = float(config.get("T_class", config.get("class_temperature", class_temperature_default)))
    floor_default = 0.05 if source_count <= 2 else 0.03
    floor = float(config.get("floor", floor_default))
    top_m_default = 0 if source_count <= 2 else 3
    top_m = int(config.get("top_m_per_class", top_m_default))

    source_train_chunks = [
        _collect_outputs(
            model=model,
            loader=loader,
            device=device,
            domain_name=split.domain_id,
            max_batches=max_batches,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
        )
        for split, loader in zip(prepared_data.source_splits, prepared_data.source_train_eval_loaders)
    ]
    source_eval_chunks = [
        _collect_outputs(
            model=model,
            loader=loader,
            device=device,
            domain_name=split.domain_id,
            max_batches=max_batches,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
        )
        for split, loader in zip(prepared_data.source_splits, prepared_data.source_eval_loaders)
    ]
    target_train_chunk = _collect_outputs(
        model=model,
        loader=prepared_data.target_train_loader,
        device=device,
        domain_name=prepared_data.target_split.domain_id,
        max_batches=max_batches,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
    )
    target_eval_chunk = _collect_outputs(
        model=model,
        loader=prepared_data.target_eval_loader,
        device=device,
        domain_name=prepared_data.target_split.domain_id,
        max_batches=max_batches,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
    )
    if target_eval_chunk["logits"].shape[1] > 0:
        num_classes = int(target_eval_chunk["logits"].shape[1])

    source_prototypes = []
    source_present = []
    for chunk in source_train_chunks:
        prototypes, present = _class_prototypes(chunk["embeddings"], chunk["labels"], num_classes=num_classes)
        source_prototypes.append(prototypes)
        source_present.append(present)
    source_prototypes_array = np.stack(source_prototypes, axis=0)
    source_present_array = np.stack(source_present, axis=0)

    target_train_probs_by_source = []
    target_train_dist_by_source = []
    target_eval_probs_by_source = []
    source_eval_error, source_eval_recall, source_eval_support = _compute_source_eval_error(
        source_eval_chunks=source_eval_chunks,
        source_prototypes=source_prototypes_array,
        source_present=source_present_array,
        proto_temperature=proto_temperature,
        num_classes=num_classes,
    )
    for source_index in range(source_count):
        train_probabilities, train_distances = _prototype_probabilities(
            target_train_chunk["embeddings"],
            source_prototypes_array[source_index],
            source_present_array[source_index],
            temperature=proto_temperature,
        )
        eval_probabilities, _ = _prototype_probabilities(
            target_eval_chunk["embeddings"],
            source_prototypes_array[source_index],
            source_present_array[source_index],
            temperature=proto_temperature,
        )
        target_train_probs_by_source.append(train_probabilities)
        target_train_dist_by_source.append(train_distances)
        target_eval_probs_by_source.append(eval_probabilities)
    prototype_target_train_probs = np.stack(target_train_probs_by_source, axis=0)
    target_train_distances = np.stack(target_train_dist_by_source, axis=0)
    prototype_target_eval_probs = np.stack(target_eval_probs_by_source, axis=0)
    target_train_expert_probs = target_train_chunk.get("expert_probabilities")
    target_eval_expert_probs = target_eval_chunk.get("expert_probabilities")
    if (
        target_train_expert_probs is not None
        and target_eval_expert_probs is not None
        and int(target_train_expert_probs.shape[0]) == source_count
        and int(target_eval_expert_probs.shape[0]) == source_count
    ):
        target_train_probs = target_train_expert_probs.astype(np.float32, copy=False)
        target_eval_probs = target_eval_expert_probs.astype(np.float32, copy=False)
        expert_prediction_source = "model_source_experts"
    else:
        target_train_probs = prototype_target_train_probs
        target_eval_probs = prototype_target_eval_probs
        expert_prediction_source = "source_prototype_proxy"

    source_eval_probabilities = None
    try:
        if all(chunk.get("expert_probabilities") is not None for chunk in source_eval_chunks):
            source_eval_probabilities = np.stack(
                [chunk["expert_probabilities"].astype(np.float32, copy=False) for chunk in source_eval_chunks],
                axis=1,
            )
    except Exception:
        source_eval_probabilities = None
    if source_eval_probabilities is not None and int(source_eval_probabilities.shape[0]) == source_count:
        expert_errors = np.ones((source_count, num_classes), dtype=np.float32)
        expert_recalls = np.zeros((source_count, num_classes), dtype=np.float32)
        expert_supports = np.zeros((source_count, num_classes), dtype=np.int64)
        for source_index, chunk in enumerate(source_eval_chunks):
            labels = chunk["labels"].astype(np.int64, copy=False)
            predictions = source_eval_probabilities[source_index, source_index].argmax(axis=1)
            for class_id in range(num_classes):
                mask = labels == class_id
                expert_supports[source_index, class_id] = int(mask.sum())
                if not np.any(mask):
                    continue
                recall = float((predictions[mask] == class_id).mean())
                expert_recalls[source_index, class_id] = recall
                expert_errors[source_index, class_id] = 1.0 - recall
        source_eval_error = expert_errors
        source_eval_recall = expert_recalls
        source_eval_support = expert_supports

    target_prototypes, target_proto_source, high_conf_counts, target_train_mean_probs, provisional_train = (
        _target_prototypes_from_predictions(
            target_embeddings=target_train_chunk["embeddings"],
            source_target_probabilities=target_train_probs,
            tau_proto=tau_proto,
            min_samples=min_proto_samples,
        )
    )
    target_train_features = _normalize_rows(target_train_chunk["embeddings"])

    d_proto = np.ones((source_count, num_classes), dtype=np.float32)
    d_ot = np.ones((source_count, num_classes), dtype=np.float32)
    h_pred = np.ones((source_count, num_classes), dtype=np.float32)
    for source_index in range(source_count):
        proto_distances = _squared_distances(source_prototypes_array[source_index], target_prototypes)
        d_proto[source_index] = np.diag(proto_distances)
        entropy_values = _entropy(target_train_probs[source_index])
        for class_id in range(num_classes):
            weights_for_class = target_train_probs[source_index, :, class_id]
            mass = float(weights_for_class.sum())
            if mass > EPS:
                d_ot[source_index, class_id] = float(
                    (target_train_distances[source_index, :, class_id] * weights_for_class).sum() / mass
                )
            provisional_mask = provisional_train == class_id
            if np.any(provisional_mask):
                h_pred[source_index, class_id] = float(entropy_values[provisional_mask].mean())
            elif mass > EPS:
                h_pred[source_index, class_id] = float((entropy_values * weights_for_class).sum() / mass)
    d_proto = np.where(source_present_array, d_proto, 1e6)
    d_ot = np.where(source_present_array, d_ot, 1e6)
    h_pred = np.where(source_present_array, h_pred, 1.0)
    source_eval_error = np.where(source_present_array, source_eval_error, 1.0)

    d_proto_norm = _minmax_by_class(d_proto, source_present_array)
    d_ot_norm = _minmax_by_class(d_ot, source_present_array)
    h_pred_norm = _minmax_by_class(h_pred, source_present_array)
    e_src_norm = _minmax_by_class(source_eval_error, source_present_array)
    reliability = (
        weights["D_proto"] * d_proto_norm
        + weights["D_ot"] * d_ot_norm
        + weights["H_pred"] * h_pred_norm
        + weights["E_src"] * e_src_norm
    ).astype(np.float32)
    method_key = str(method_name).strip().lower()
    if "raw" in method_key:
        default_prediction_mode = "ccsr_raw"
    elif "calibrated_override" in method_key or "sa_ccsr" in method_key:
        default_prediction_mode = "ccsr_calibrated_override"
    else:
        default_prediction_mode = "ccsr_safe"
    prediction_mode = str(config.get("prediction_mode", default_prediction_mode)).strip().lower()
    if prediction_mode in {"raw", "ccsr"}:
        prediction_mode = "ccsr_raw"
    elif prediction_mode in {"safe", "safe_mix"}:
        prediction_mode = "ccsr_safe"
    elif prediction_mode in {"calibrated", "override", "calibrated_override"}:
        prediction_mode = "ccsr_calibrated_override"

    gate_params = {
        "lambda_same": float(config.get("lambda_same", 0.5)),
        "lambda_override": float(config.get("lambda_override", 0.8)),
        "override_threshold": float(config.get("override_threshold", 0.55)),
        "T_class": class_temperature,
        "top_m_per_class": top_m,
        "floor": floor,
    }
    source_meta_enabled = bool(
        config.get("source_meta_calibration", prediction_mode == "ccsr_calibrated_override")
    )
    meta_rows, selected_gate_params = _source_meta_calibrate_gate(
        source_eval_chunks=source_eval_chunks,
        source_eval_probabilities=source_eval_probabilities,
        reliability=reliability,
        source_domains=source_domains,
        default_params=gate_params,
        enabled=source_meta_enabled,
    )
    class_temperature = float(selected_gate_params.get("T_class", class_temperature))
    floor = float(selected_gate_params.get("floor", floor))
    top_m = int(selected_gate_params.get("top_m_per_class", top_m))
    alpha_raw, alpha = _resolve_alpha(
        reliability,
        source_count=source_count,
        temperature=class_temperature,
        floor=floor,
        top_m=top_m,
    )

    p_wjdot = _softmax(target_eval_chunk["logits"], axis=1)
    p_ccsr = np.einsum("kc,knc->nc", alpha, target_eval_probs).astype(np.float32)
    p_ccsr = p_ccsr / np.maximum(p_ccsr.sum(axis=1, keepdims=True), EPS)
    ccsr_entropy = _entropy(p_ccsr)
    p_ccsr_pred = p_ccsr.argmax(axis=1)
    p_wjdot_pred = p_wjdot.argmax(axis=1)
    same_as_wjdot = (p_ccsr_pred == p_wjdot_pred).astype(np.float32)
    agreement = _weighted_source_agreement(target_eval_probs, alpha, p_ccsr_pred)
    alpha_concentration = _alpha_concentration_by_prediction(alpha, p_ccsr_pred)
    reliability_support = _reliability_support_by_prediction(reliability, alpha, p_ccsr_pred)
    if rho_mode == "full":
        a1 = float(config.get("rho_a1", 1.0))
        a2 = float(config.get("rho_a2", 1.0))
        a3 = float(config.get("rho_a3", 0.5))
        a4 = float(config.get("rho_a4", 1.0))
        bias = float(config.get("rho_bias", 2.0))
        score = (
            a1 * agreement
            + a2 * (1.0 - ccsr_entropy)
            + a3 * alpha_concentration
            + a4 * same_as_wjdot
            - bias
        )
        rho = 1.0 / (1.0 + np.exp(-score))
    else:
        rho = 0.5 * same_as_wjdot + 0.5 * (1.0 - ccsr_entropy)
    rho = np.clip(rho, rho_min, rho_max).astype(np.float32)
    safe_mix = (lambda_safe * rho).reshape(-1, 1).astype(np.float32)
    p_safe = (1.0 - safe_mix) * p_wjdot + safe_mix * p_ccsr
    safe_override_gate = (
        override_enabled
        & (p_ccsr_pred != p_wjdot_pred)
        & (agreement >= override_agreement_threshold)
        & (ccsr_entropy <= override_entropy_threshold)
        & (alpha_concentration >= override_alpha_concentration_min)
    )
    if bool(np.any(safe_override_gate)):
        p_safe[safe_override_gate] = (
            (1.0 - override_strength) * p_safe[safe_override_gate]
            + override_strength * p_ccsr[safe_override_gate]
        )
    p_safe = p_safe / np.maximum(p_safe.sum(axis=1, keepdims=True), EPS)
    calibrated = _apply_calibrated_override(
        p_wjdot=p_wjdot,
        p_ccsr=p_ccsr,
        source_probabilities=target_eval_probs,
        alpha=alpha,
        reliability=reliability,
        lambda_same=float(selected_gate_params.get("lambda_same", gate_params["lambda_same"])),
        lambda_override=float(selected_gate_params.get("lambda_override", gate_params["lambda_override"])),
        override_threshold=float(selected_gate_params.get("override_threshold", gate_params["override_threshold"])),
        score_weights=config.get("override_score_weights"),
    )
    if prediction_mode == "ccsr_raw":
        p_final = p_ccsr
        override_gate = np.zeros((p_ccsr.shape[0],), dtype=bool)
        override_score = np.ones((p_ccsr.shape[0],), dtype=np.float32)
        safe_mix = np.ones((p_ccsr.shape[0], 1), dtype=np.float32)
    elif prediction_mode == "ccsr_calibrated_override":
        p_final = calibrated["p_final"]
        override_gate = calibrated["override_gate"]
        override_score = calibrated["override_score"]
        agreement = calibrated["source_agreement"]
        alpha_concentration = calibrated["alpha_concentration"]
        reliability_support = calibrated["reliability_support"]
    else:
        p_final = p_safe
        override_gate = safe_override_gate
        override_score = rho.astype(np.float32, copy=False)
    final_predictions = p_final.argmax(axis=1).astype(np.int64, copy=False)

    labels = target_eval_chunk["labels"].astype(np.int64, copy=False)
    present_labels = sorted(set(int(label) for label in labels.tolist()))
    target_accuracy = float(accuracy_score(labels, final_predictions)) if labels.size else 0.0
    target_macro_f1 = float(f1_score(labels, final_predictions, average="macro", zero_division=0)) if labels.size else 0.0
    target_balanced_accuracy = (
        float(recall_score(labels, final_predictions, labels=present_labels, average="macro", zero_division=0))
        if labels.size
        else 0.0
    )
    final_confusion = confusion_matrix(labels, final_predictions, labels=np.arange(num_classes)) if labels.size else np.zeros((num_classes, num_classes), dtype=np.int64)
    wjdot_confusion = confusion_matrix(labels, p_wjdot_pred, labels=np.arange(num_classes)) if labels.size else np.zeros((num_classes, num_classes), dtype=np.int64)
    raw_ccsr_confusion = confusion_matrix(labels, p_ccsr_pred, labels=np.arange(num_classes)) if labels.size else np.zeros((num_classes, num_classes), dtype=np.int64)
    wjdot_accuracy = float(accuracy_score(labels, p_wjdot_pred)) if labels.size else 0.0
    wjdot_macro_f1 = float(f1_score(labels, p_wjdot_pred, average="macro", zero_division=0)) if labels.size else 0.0
    wjdot_balanced_accuracy = (
        float(recall_score(labels, p_wjdot_pred, labels=present_labels, average="macro", zero_division=0))
        if labels.size
        else 0.0
    )
    raw_ccsr_accuracy = float(accuracy_score(labels, p_ccsr_pred)) if labels.size else 0.0
    raw_ccsr_macro_f1 = float(f1_score(labels, p_ccsr_pred, average="macro", zero_division=0)) if labels.size else 0.0
    raw_ccsr_balanced_accuracy = (
        float(recall_score(labels, p_ccsr_pred, labels=present_labels, average="macro", zero_division=0))
        if labels.size
        else 0.0
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reliability_components_path = tables_dir / "reliability_components.csv"
    with reliability_components_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_domain",
            "class_id",
            "D_proto",
            "D_ot",
            "H_pred",
            "E_src",
            "D_proto_norm",
            "D_ot_norm",
            "H_pred_norm",
            "E_src_norm",
            "R",
            "alpha_raw",
            "alpha",
            "source_class_present",
            "target_proto_source",
            "target_high_conf_count",
            "source_eval_support",
            "source_eval_recall",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_index, domain_name in enumerate(source_domains):
            for class_id in range(num_classes):
                writer.writerow(
                    {
                        "source_domain": domain_name,
                        "class_id": class_id,
                        "D_proto": float(d_proto[source_index, class_id]),
                        "D_ot": float(d_ot[source_index, class_id]),
                        "H_pred": float(h_pred[source_index, class_id]),
                        "E_src": float(source_eval_error[source_index, class_id]),
                        "D_proto_norm": float(d_proto_norm[source_index, class_id]),
                        "D_ot_norm": float(d_ot_norm[source_index, class_id]),
                        "H_pred_norm": float(h_pred_norm[source_index, class_id]),
                        "E_src_norm": float(e_src_norm[source_index, class_id]),
                        "R": float(reliability[source_index, class_id]),
                        "alpha_raw": float(alpha_raw[source_index, class_id]),
                        "alpha": float(alpha[source_index, class_id]),
                        "source_class_present": int(source_present_array[source_index, class_id]),
                        "target_proto_source": target_proto_source[class_id],
                        "target_high_conf_count": int(high_conf_counts[class_id]),
                        "source_eval_support": int(source_eval_support[source_index, class_id]),
                        "source_eval_recall": float(source_eval_recall[source_index, class_id]),
                    }
                )

    class_source_alpha_path = Path(
        _write_matrix_csv(tables_dir / "class_source_alpha.csv", alpha, source_domains=source_domains)
    )
    class_source_alpha_matrix_path = Path(
        _write_matrix_csv(tables_dir / "class_source_alpha_matrix.csv", alpha, source_domains=source_domains)
    )
    global_score = np.nanmean(np.where(source_present_array, d_proto_norm + d_ot_norm, np.nan), axis=1)
    global_alpha = _softmax(-global_score.reshape(-1, 1) / max(class_temperature, EPS), axis=0).reshape(-1)
    global_vs_class_path = tables_dir / "source_weight_global_vs_class_conditional.csv"
    with global_vs_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_domain", "global_alpha_proxy", *[f"class_{class_id:02d}" for class_id in range(num_classes)]])
        for source_index, domain_name in enumerate(source_domains):
            writer.writerow([domain_name, float(global_alpha[source_index]), *[float(alpha[source_index, class_id]) for class_id in range(num_classes)]])
    global_vs_class_alias_path = tables_dir / "global_source_alpha_vs_class_source_alpha.csv"
    global_vs_class_alias_path.write_text(global_vs_class_path.read_text(encoding="utf-8"), encoding="utf-8")

    wjdot_recall, supports = _per_class_recall(labels, p_wjdot_pred, num_classes=num_classes)
    final_recall, _ = _per_class_recall(labels, final_predictions, num_classes=num_classes)
    recall_gain_path = tables_dir / "per_class_recall_gain.csv"
    with recall_gain_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "support", "recall_wjdot", "recall_ccsr_wjdot_fusion", "recall_gain"])
        for class_id in range(num_classes):
            writer.writerow(
                [
                    class_id,
                    int(supports[class_id]),
                    float(wjdot_recall[class_id]),
                    float(final_recall[class_id]),
                    float(final_recall[class_id] - wjdot_recall[class_id]),
                ]
            )

    disagreement_path = tables_dir / "ccsr_vs_wjdot_prediction_disagreement.csv"
    with disagreement_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "wjdot_pred",
                "ccsr_pred_before_fallback",
                "final_pred",
                "same_ccsr_wjdot",
                "rho",
                "ccsr_entropy",
                "source_agreement",
                "alpha_concentration",
                "override_gate",
                "safe_mix",
            ]
        )
        for sample_index in range(labels.shape[0]):
            writer.writerow(
                [
                    sample_index,
                    int(labels[sample_index]),
                    int(p_wjdot_pred[sample_index]),
                    int(p_ccsr_pred[sample_index]),
                    int(final_predictions[sample_index]),
                    int(same_as_wjdot[sample_index]),
                    float(rho[sample_index]),
                    float(ccsr_entropy[sample_index]),
                    float(agreement[sample_index]),
                    float(alpha_concentration[sample_index]),
                    int(override_gate[sample_index]),
                    float(safe_mix[sample_index, 0]),
                ]
            )

    histogram_path = tables_dir / "target_prediction_histogram.csv"
    with histogram_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "wjdot_count", "ccsr_final_count"])
        wjdot_counts = np.bincount(p_wjdot_pred, minlength=num_classes)
        final_counts = np.bincount(final_predictions, minlength=num_classes)
        for class_id in range(num_classes):
            writer.writerow([class_id, int(wjdot_counts[class_id]), int(final_counts[class_id])])

    per_source_prediction_path = tables_dir / "per_source_prediction.csv"
    with per_source_prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "true_label", "source_domain", "source_pred", "source_confidence"])
        source_predictions = target_eval_probs.argmax(axis=2)
        source_confidence = target_eval_probs.max(axis=2)
        for source_index, domain_name in enumerate(source_domains):
            for sample_index in range(labels.shape[0]):
                writer.writerow(
                    [
                        sample_index,
                        int(labels[sample_index]),
                        domain_name,
                        int(source_predictions[source_index, sample_index]),
                        float(source_confidence[source_index, sample_index]),
                    ]
                )

    per_source_confusion_path = tables_dir / "per_source_confusion_on_target_eval_only_after_training.csv"
    with per_source_confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_domain", "true_class", "pred_class", "count"])
        for source_index, domain_name in enumerate(source_domains):
            matrix = confusion_matrix(labels, source_predictions[source_index], labels=np.arange(num_classes))
            for true_class in range(num_classes):
                for pred_class in range(num_classes):
                    writer.writerow([domain_name, true_class, pred_class, int(matrix[true_class, pred_class])])

    per_source_histogram_path = tables_dir / "target_prediction_histogram_per_source.csv"
    with per_source_histogram_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_domain", "class_id", "count"])
        for source_index, domain_name in enumerate(source_domains):
            counts = np.bincount(source_predictions[source_index], minlength=num_classes)
            for class_id in range(num_classes):
                writer.writerow([domain_name, class_id, int(counts[class_id])])

    per_source_ot_loss_path = tables_dir / "per_source_ot_loss.csv"
    with per_source_ot_loss_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_domain", "mean_class_ot_cost", *[f"class_{class_id:02d}" for class_id in range(num_classes)]])
        for source_index, domain_name in enumerate(source_domains):
            valid = source_present_array[source_index]
            mean_value = float(np.mean(d_ot[source_index, valid])) if np.any(valid) else 0.0
            writer.writerow([domain_name, mean_value, *[float(d_ot[source_index, class_id]) for class_id in range(num_classes)]])

    per_source_alpha_path = tables_dir / "per_source_alpha.csv"
    with per_source_alpha_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_domain", "global_alpha_proxy", "mean_class_alpha"])
        for source_index, domain_name in enumerate(source_domains):
            writer.writerow([domain_name, float(global_alpha[source_index]), float(alpha[source_index].mean())])

    prediction_mode_summary_path = tables_dir / "prediction_mode_summary.csv"
    with prediction_mode_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prediction_mode",
                "expert_prediction_source",
                "raw_vs_wjdot_disagreement_count",
                "final_vs_wjdot_disagreement_count",
                "override_count",
                "lambda_same",
                "lambda_override",
                "override_threshold",
                "T_class",
                "top_m_per_class",
                "floor",
                "source_meta_calibration_enabled",
            ]
        )
        writer.writerow(
            [
                prediction_mode,
                expert_prediction_source,
                int((p_ccsr_pred != p_wjdot_pred).sum()),
                int((final_predictions != p_wjdot_pred).sum()),
                int(override_gate.sum()),
                float(selected_gate_params.get("lambda_same", gate_params["lambda_same"])),
                float(selected_gate_params.get("lambda_override", gate_params["lambda_override"])),
                float(selected_gate_params.get("override_threshold", gate_params["override_threshold"])),
                class_temperature,
                top_m,
                floor,
                int(source_meta_enabled),
            ]
        )

    override_score_path = tables_dir / "override_score_distribution.csv"
    with override_score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "override_score",
                "same_top1",
                "margin_c",
                "source_agreement",
                "alpha_concentration",
                "reliability_support",
                "override_gate",
            ]
        )
        for sample_index in range(labels.shape[0]):
            writer.writerow(
                [
                    sample_index,
                    float(override_score[sample_index]),
                    int(calibrated["same_top1"][sample_index]),
                    float(calibrated["margin_c"][sample_index]),
                    float(agreement[sample_index]),
                    float(alpha_concentration[sample_index]),
                    float(reliability_support[sample_index]),
                    int(override_gate[sample_index]),
                ]
            )

    entropy_compare_path = tables_dir / "entropy_w_vs_entropy_c.csv"
    with entropy_compare_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "true_label", "wjdot_pred", "ccsr_pred", "entropy_w", "entropy_c"])
        for sample_index in range(labels.shape[0]):
            writer.writerow(
                [
                    sample_index,
                    int(labels[sample_index]),
                    int(p_wjdot_pred[sample_index]),
                    int(p_ccsr_pred[sample_index]),
                    float(calibrated["entropy_w"][sample_index]),
                    float(calibrated["entropy_c"][sample_index]),
                ]
            )

    override_cases_path = tables_dir / "override_cases.csv"
    with override_cases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "wjdot_pred",
                "ccsr_pred",
                "final_pred",
                "override_score",
                "override_gate",
                "wjdot_correct",
                "final_correct",
            ]
        )
        for sample_index in range(labels.shape[0]):
            if p_wjdot_pred[sample_index] == p_ccsr_pred[sample_index] and not bool(override_gate[sample_index]):
                continue
            writer.writerow(
                [
                    sample_index,
                    int(labels[sample_index]),
                    int(p_wjdot_pred[sample_index]),
                    int(p_ccsr_pred[sample_index]),
                    int(final_predictions[sample_index]),
                    float(override_score[sample_index]),
                    int(override_gate[sample_index]),
                    int(p_wjdot_pred[sample_index] == labels[sample_index]),
                    int(final_predictions[sample_index] == labels[sample_index]),
                ]
            )

    source_meta_results_path = tables_dir / "source_meta_calibration_results.csv"
    with source_meta_results_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "lambda_same",
            "lambda_override",
            "override_threshold",
            "T_class",
            "top_m_per_class",
            "floor",
            "mean_balanced_accuracy",
            "pseudo_task_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in meta_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    selected_gate_params_path = tables_dir / "selected_gate_params.json"
    selected_gate_params_path.write_text(
        json.dumps(selected_gate_params, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    alpha_heatmap_path = _save_heatmap(
        figures_dir / "class_source_alpha_heatmap.png",
        alpha,
        title="CCSR class-source alpha",
        source_domains=source_domains,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    component_heatmaps_path = _save_component_heatmaps(
        figures_dir / "reliability_component_heatmaps.png",
        {
            "D_proto_norm": d_proto_norm,
            "D_ot_norm": d_ot_norm,
            "H_pred_norm": h_pred_norm,
            "E_src_norm": e_src_norm,
            "R": reliability,
            "alpha": alpha,
        },
        source_domains=source_domains,
    )
    prediction_histogram_figure_path = _save_prediction_histogram(
        figures_dir / "target_prediction_histogram.png",
        final_predictions,
        num_classes=num_classes,
        title="CCSR-WJDOT target prediction histogram",
    )
    global_vs_class_figure_path = _save_global_vs_class_alpha(
        figures_dir / "source_weight_global_vs_class_conditional.png",
        global_alpha,
        alpha,
        source_domains=source_domains,
    )
    global_vs_class_alias_figure_path = None
    if global_vs_class_figure_path is not None:
        alias_path = figures_dir / "global_source_alpha_vs_class_alpha.png"
        alias_path.write_bytes(Path(global_vs_class_figure_path).read_bytes())
        global_vs_class_alias_figure_path = str(alias_path)
    entropy_rho_figure_path = _save_entropy_rho_distribution(
        figures_dir / "target_entropy_rho_distribution.png",
        ccsr_entropy,
        rho,
    )
    confusion_comparison_figure_path = _save_confusion_comparison(
        figures_dir / "wjdot_vs_ccsr_confusion_matrix.png",
        wjdot_confusion,
        raw_ccsr_confusion,
        final_confusion,
    )

    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    source_eval_embeddings = np.concatenate([chunk["embeddings"] for chunk in source_eval_chunks], axis=0)
    source_eval_labels = np.concatenate([chunk["labels"] for chunk in source_eval_chunks], axis=0)
    source_eval_predictions = np.concatenate([chunk["predictions"] for chunk in source_eval_chunks], axis=0)
    source_eval_domains = np.concatenate([chunk["domains"] for chunk in source_eval_chunks], axis=0)
    np.savez_compressed(
        analysis_path,
        scenario_id=_string_array([scenario_id]),
        method_name=_string_array([method_name]),
        source_embeddings=source_eval_embeddings,
        source_labels=source_eval_labels,
        source_predictions=source_eval_predictions,
        source_domains=source_eval_domains,
        target_embeddings=target_eval_chunk["embeddings"],
        target_labels=labels,
        target_predictions=final_predictions,
        target_domains=target_eval_chunk["domains"],
        target_logits=np.log(np.maximum(p_final, EPS)).astype(np.float32),
        target_probabilities=p_final.astype(np.float32),
        target_predictions_wjdot=p_wjdot_pred.astype(np.int64),
        target_predictions_ccsr_raw=p_ccsr_pred.astype(np.int64),
        target_probabilities_wjdot=p_wjdot.astype(np.float32),
        target_probabilities_ccsr=p_ccsr.astype(np.float32),
        ccsr_rho=rho.astype(np.float32),
        ccsr_alpha=alpha.astype(np.float32),
        ccsr_alpha_raw=alpha_raw.astype(np.float32),
    )

    return {
        "analysis_path": str(analysis_path),
        "target_eval_acc": target_accuracy,
        "target_eval_macro_f1": target_macro_f1,
        "target_eval_balanced_acc": target_balanced_accuracy,
        "target_confusion_matrix": final_confusion.tolist(),
        "ccsr_wjdot_fusion": {
            "enabled": True,
            "base_wjdot_target_eval_acc": wjdot_accuracy,
            "base_wjdot_target_eval_macro_f1": wjdot_macro_f1,
            "base_wjdot_target_eval_balanced_acc": wjdot_balanced_accuracy,
            "raw_ccsr_target_eval_acc": raw_ccsr_accuracy,
            "raw_ccsr_target_eval_macro_f1": raw_ccsr_macro_f1,
            "raw_ccsr_target_eval_balanced_acc": raw_ccsr_balanced_accuracy,
            "accuracy_gain_vs_wjdot": float(target_accuracy - wjdot_accuracy),
            "macro_f1_gain_vs_wjdot": float(target_macro_f1 - wjdot_macro_f1),
            "balanced_acc_gain_vs_wjdot": float(target_balanced_accuracy - wjdot_balanced_accuracy),
            "raw_accuracy_gain_vs_wjdot": float(raw_ccsr_accuracy - wjdot_accuracy),
            "raw_disagreement_count": int((p_ccsr_pred != p_wjdot_pred).sum()),
            "final_disagreement_count": int((final_predictions != p_wjdot_pred).sum()),
            "override_count": int(override_gate.sum()),
            "mean_rho": float(rho.mean()) if rho.size else 0.0,
            "mean_ccsr_entropy": float(ccsr_entropy.mean()) if ccsr_entropy.size else 0.0,
            "mean_source_agreement": float(agreement.mean()) if agreement.size else 0.0,
            "mean_same_as_wjdot": float(same_as_wjdot.mean()) if same_as_wjdot.size else 0.0,
            "class_temperature": class_temperature,
            "floor": floor,
            "top_m_per_class": top_m,
            "lambda_safe": lambda_safe,
            "override_on_source_agreement": bool(override_enabled),
            "override_agreement_threshold": override_agreement_threshold,
            "override_entropy_threshold": override_entropy_threshold,
            "override_alpha_concentration_min": override_alpha_concentration_min,
            "override_strength": override_strength,
            "prediction_mode": prediction_mode,
            "expert_prediction_source": expert_prediction_source,
            "lambda_same": float(selected_gate_params.get("lambda_same", gate_params["lambda_same"])),
            "lambda_override": float(selected_gate_params.get("lambda_override", gate_params["lambda_override"])),
            "override_threshold": float(
                selected_gate_params.get("override_threshold", gate_params["override_threshold"])
            ),
            "mean_override_score": float(override_score.mean()) if override_score.size else 0.0,
            "source_meta_calibration_enabled": bool(source_meta_enabled),
        },
        "reliability_components_path": str(reliability_components_path),
        "class_source_alpha_path": str(class_source_alpha_path),
        "class_alpha_matrix_path": str(class_source_alpha_matrix_path),
        "source_weight_global_vs_class_conditional_path": str(global_vs_class_path),
        "global_source_alpha_vs_class_source_alpha_path": str(global_vs_class_alias_path),
        "per_class_recall_gain_path": str(recall_gain_path),
        "ccsr_vs_wjdot_prediction_disagreement_path": str(disagreement_path),
        "target_prediction_histogram_path": str(histogram_path),
        "per_source_prediction_path": str(per_source_prediction_path),
        "per_source_confusion_on_target_eval_only_after_training_path": str(per_source_confusion_path),
        "target_prediction_histogram_per_source_path": str(per_source_histogram_path),
        "per_source_ot_loss_path": str(per_source_ot_loss_path),
        "per_source_alpha_path": str(per_source_alpha_path),
        "prediction_mode_summary_path": str(prediction_mode_summary_path),
        "override_score_distribution_path": str(override_score_path),
        "entropy_w_vs_entropy_c_path": str(entropy_compare_path),
        "override_cases_path": str(override_cases_path),
        "source_meta_calibration_results_path": str(source_meta_results_path),
        "selected_gate_params_path": str(selected_gate_params_path),
        "class_source_alpha_heatmap_path": alpha_heatmap_path,
        "reliability_component_heatmaps_path": component_heatmaps_path,
        "target_prediction_histogram_figure_path": prediction_histogram_figure_path,
        "source_weight_global_vs_class_conditional_figure_path": global_vs_class_figure_path,
        "global_source_alpha_vs_class_alpha_figure_path": global_vs_class_alias_figure_path,
        "target_entropy_rho_distribution_path": entropy_rho_figure_path,
        "wjdot_vs_ccsr_confusion_matrix_path": confusion_comparison_figure_path,
    }
