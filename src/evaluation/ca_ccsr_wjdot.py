"""Final diagnostics for the CA-CCSR-WJDOT main method."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.ccsr_wjdot_fusion import (
    EPS,
    _collect_outputs,
    _entropy,
    _import_numpy,
    _per_class_recall,
    _softmax,
    _string_array,
)


def _as_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    merged = dict(payload)
    fusion = payload.get("teacher_safe_fusion")
    if isinstance(fusion, dict):
        merged.update(fusion)
    return merged


def _collect_teacher_outputs(
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

    if not hasattr(model, "teacher_logits_and_features"):
        return None

    embeddings = []
    logits = []
    labels = []
    domains = []
    model.eval()
    with torch.inference_mode():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                batch_logits, batch_features = model.teacher_logits_and_features(x_batch)
            embeddings.append(batch_features.detach().float().cpu().numpy())
            logits.append(batch_logits.detach().float().cpu().numpy())
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
        }

    logits_array = np.concatenate(logits, axis=0).astype(np.float32, copy=False)
    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32, copy=False),
        "logits": logits_array,
        "labels": np.concatenate(labels, axis=0).astype(np.int64, copy=False),
        "predictions": logits_array.argmax(axis=1).astype(np.int64, copy=False),
        "domains": np.concatenate(domains, axis=0),
    }


def _class_alpha_from_snapshot(*, model, source_count: int, num_classes: int):
    np = _import_numpy()
    uniform = np.full((source_count, num_classes), 1.0 / max(source_count, 1), dtype=np.float32)
    if not hasattr(model, "reliability_snapshot"):
        return uniform, "uniform_no_snapshot"

    try:
        snapshot = model.reliability_snapshot()
    except Exception:
        return uniform, "uniform_snapshot_error"

    value = None
    for key in ("class_source_weights", "reliability_alpha", "reliability_alpha_uniform"):
        if snapshot.get(key) is not None:
            value = snapshot[key]
            break
    if value is None:
        return uniform, "uniform_missing_class_alpha"

    matrix = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != uniform.shape:
        return uniform, "uniform_shape_mismatch"

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.maximum(matrix, 0.0)
    matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), EPS)
    return matrix.astype(np.float32, copy=False), "model_reliability_snapshot"


def _alpha_entropy(alpha):
    np = _import_numpy()
    source_count = int(alpha.shape[0])
    if source_count <= 1:
        return np.zeros((alpha.shape[1],), dtype=np.float32)
    entropy = -np.sum(alpha * np.log(np.maximum(alpha, EPS)), axis=0)
    return (entropy / math.log(source_count)).astype(np.float32)


def _teacher_safe_fusion(student_probs, teacher_probs, config: dict[str, Any]):
    np = _import_numpy()

    fusion_base = str(config.get("fusion_base", "student")).strip().lower()
    lambda_same = float(config.get("lambda_same", config.get("teacher_fusion_lambda_same", 0.15)))
    lambda_override = float(config.get("lambda_override", config.get("teacher_fusion_lambda_override", 0.60)))
    confidence_margin = float(
        config.get("override_confidence_margin", config.get("teacher_override_confidence_margin", 0.10))
    )
    entropy_margin = float(
        config.get("override_entropy_margin", config.get("teacher_override_entropy_margin", 0.05))
    )
    teacher_confidence_floor = float(
        config.get("teacher_confidence_floor", config.get("teacher_override_confidence_floor", 0.0))
    )
    eta_min = max(float(config.get("eta_min", 0.0)), 0.0)
    eta_max = min(max(float(config.get("eta_max", 0.70)), eta_min), 1.0)
    prior_balance_strength = max(float(config.get("prior_balance_strength", 0.0)), 0.0)
    prior_balance_student_mix = min(max(float(config.get("prior_balance_student_mix", 0.50)), 0.0), 1.0)
    prior_balance_min_prior = max(float(config.get("prior_balance_min_prior", 1e-4)), EPS)

    student_pred = student_probs.argmax(axis=1)
    teacher_pred = teacher_probs.argmax(axis=1)
    same_top1 = student_pred == teacher_pred
    student_confidence = student_probs.max(axis=1)
    teacher_confidence = teacher_probs.max(axis=1)
    student_entropy = _entropy(student_probs)
    teacher_entropy = _entropy(teacher_probs)
    confidence_advantage = teacher_confidence - student_confidence
    entropy_advantage = student_entropy - teacher_entropy

    same_eta = lambda_same * np.maximum(1.0 - 0.5 * (student_entropy + teacher_entropy), 0.0)
    if fusion_base in {"prior_balanced", "prior_balance", "balanced_prior"}:
        base_probs = (
            (1.0 - prior_balance_student_mix) * teacher_probs
            + prior_balance_student_mix * student_probs
        )
        observed_prior = np.maximum(base_probs.mean(axis=0, keepdims=True), prior_balance_min_prior)
        if prior_balance_strength > 0.0:
            final_probs = base_probs / np.power(observed_prior, prior_balance_strength)
        else:
            final_probs = base_probs
        confidence_score = np.clip((student_confidence - teacher_confidence) + 0.5, 0.0, 1.0)
        entropy_score = np.clip((teacher_entropy - student_entropy) + 0.5, 0.0, 1.0)
        override_score = (0.5 * confidence_score + 0.5 * entropy_score).astype(np.float32)
        eta = np.full(student_confidence.shape, prior_balance_student_mix, dtype=np.float32)
        override_gate = final_probs.argmax(axis=1) != teacher_pred
    elif fusion_base in {"teacher", "teacher_first", "codats"}:
        student_confidence_advantage = student_confidence - teacher_confidence
        student_entropy_advantage = teacher_entropy - student_entropy
        confidence_score = np.clip(student_confidence_advantage / max(1.0 - confidence_margin, EPS), 0.0, 1.0)
        entropy_score = np.clip(student_entropy_advantage, 0.0, 1.0)
        override_score = (0.5 * confidence_score + 0.5 * entropy_score).astype(np.float32)
        override_gate = (
            (~same_top1)
            & (student_confidence_advantage >= confidence_margin)
            & (student_entropy_advantage >= entropy_margin)
        )
        eta = np.where(same_top1, same_eta, 0.0).astype(np.float32)
        eta = np.where(override_gate, lambda_override * override_score, eta).astype(np.float32)
        eta = np.clip(eta, eta_min, eta_max).astype(np.float32)
        final_probs = (1.0 - eta.reshape(-1, 1)) * teacher_probs + eta.reshape(-1, 1) * student_probs
    else:
        confidence_score = np.clip(confidence_advantage / max(1.0 - confidence_margin, EPS), 0.0, 1.0)
        entropy_score = np.clip(entropy_advantage, 0.0, 1.0)
        override_score = (0.5 * confidence_score + 0.5 * entropy_score).astype(np.float32)
        override_gate = (
            (~same_top1)
            & (teacher_confidence >= teacher_confidence_floor)
            & (confidence_advantage >= confidence_margin)
            & (entropy_advantage >= entropy_margin)
        )
        eta = np.where(same_top1, same_eta, 0.0).astype(np.float32)
        eta = np.where(override_gate, lambda_override * override_score, eta).astype(np.float32)
        eta = np.clip(eta, eta_min, eta_max).astype(np.float32)
        final_probs = (1.0 - eta.reshape(-1, 1)) * student_probs + eta.reshape(-1, 1) * teacher_probs
    final_probs = final_probs / np.maximum(final_probs.sum(axis=1, keepdims=True), EPS)

    return {
        "p_final": final_probs.astype(np.float32, copy=False),
        "fusion_base": fusion_base,
        "eta": eta,
        "override_gate": override_gate.astype(bool, copy=False),
        "override_score": override_score,
        "same_top1": same_top1.astype(bool, copy=False),
        "student_confidence": student_confidence.astype(np.float32, copy=False),
        "teacher_confidence": teacher_confidence.astype(np.float32, copy=False),
        "student_entropy": student_entropy.astype(np.float32, copy=False),
        "teacher_entropy": teacher_entropy.astype(np.float32, copy=False),
        "confidence_advantage": confidence_advantage.astype(np.float32, copy=False),
        "entropy_advantage": entropy_advantage.astype(np.float32, copy=False),
        "lambda_same": lambda_same,
        "lambda_override": lambda_override,
        "override_confidence_margin": confidence_margin,
        "override_entropy_margin": entropy_margin,
        "teacher_confidence_floor": teacher_confidence_floor,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "prior_balance_strength": prior_balance_strength,
        "prior_balance_student_mix": prior_balance_student_mix,
        "prior_balance_min_prior": prior_balance_min_prior,
    }


def _write_recall_gain(path: Path, *, labels, final_predictions, baseline_predictions, baseline_name: str, num_classes: int) -> str:
    final_recall, supports = _per_class_recall(labels, final_predictions, num_classes=num_classes)
    baseline_recall, _ = _per_class_recall(labels, baseline_predictions, num_classes=num_classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "support", f"recall_{baseline_name}", "recall_ca_ccsr_wjdot", "recall_gain"])
        for class_id in range(num_classes):
            writer.writerow(
                [
                    class_id,
                    int(supports[class_id]),
                    float(baseline_recall[class_id]),
                    float(final_recall[class_id]),
                    float(final_recall[class_id] - baseline_recall[class_id]),
                ]
            )
    return str(path)


def export_ca_ccsr_wjdot_artifacts(
    *,
    model,
    prepared_data,
    device,
    analysis_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    scenario_id: str,
    method_name: str,
    ca_config: dict[str, Any] | None = None,
    max_batches: int | None = None,
    non_blocking: bool = False,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    """Export label-free teacher-safe fusion diagnostics and final metrics.

    The fusion gate uses only student/teacher probabilities. Target-eval labels
    are read after final predictions are fixed, solely for metrics and reports.
    """

    del figures_dir

    np = _import_numpy()
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    config = _as_config(ca_config)
    tables_dir.mkdir(parents=True, exist_ok=True)

    target_eval_chunk = _collect_outputs(
        model=model,
        loader=prepared_data.target_eval_loader,
        device=device,
        domain_name=prepared_data.target_split.domain_id,
        max_batches=max_batches,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
    )
    checkpoint_loaded = bool(getattr(model, "teacher_checkpoint_loaded", False))
    allow_unloaded_teacher = bool(config.get("allow_unloaded_teacher_fusion", False))
    teacher_chunk = None
    if checkpoint_loaded or allow_unloaded_teacher:
        teacher_chunk = _collect_teacher_outputs(
            model=model,
            loader=prepared_data.target_eval_loader,
            device=device,
            domain_name=prepared_data.target_split.domain_id,
            max_batches=max_batches,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
        )
    if teacher_chunk is None or teacher_chunk["logits"].shape != target_eval_chunk["logits"].shape:
        teacher_chunk = {
            **target_eval_chunk,
            "logits": target_eval_chunk["logits"].copy(),
            "embeddings": target_eval_chunk["embeddings"].copy(),
            "predictions": target_eval_chunk["predictions"].copy(),
        }
        teacher_available = False
    else:
        teacher_available = True

    num_classes = int(target_eval_chunk["logits"].shape[1]) if target_eval_chunk["logits"].ndim == 2 else int(
        config.get("num_classes", 29)
    )
    source_domains = [split.domain_id for split in prepared_data.source_splits]
    source_count = max(len(source_domains), 1)
    if not source_domains:
        source_domains = [f"source_{index}" for index in range(source_count)]

    student_probs = _softmax(target_eval_chunk["logits"], axis=1)
    teacher_probs = _softmax(teacher_chunk["logits"], axis=1)
    fusion = _teacher_safe_fusion(student_probs, teacher_probs, config)
    final_probabilities = fusion["p_final"]
    student_predictions = student_probs.argmax(axis=1).astype(np.int64, copy=False)
    teacher_predictions = teacher_probs.argmax(axis=1).astype(np.int64, copy=False)
    final_predictions = final_probabilities.argmax(axis=1).astype(np.int64, copy=False)
    labels = target_eval_chunk["labels"].astype(np.int64, copy=False)

    present_labels = sorted(set(int(label) for label in labels.tolist()))
    target_accuracy = float(accuracy_score(labels, final_predictions)) if labels.size else 0.0
    target_macro_f1 = float(f1_score(labels, final_predictions, average="macro", zero_division=0)) if labels.size else 0.0
    target_balanced_accuracy = (
        float(recall_score(labels, final_predictions, labels=present_labels, average="macro", zero_division=0))
        if labels.size
        else 0.0
    )
    student_accuracy = float(accuracy_score(labels, student_predictions)) if labels.size else 0.0
    teacher_accuracy = float(accuracy_score(labels, teacher_predictions)) if labels.size else 0.0
    final_confusion = (
        confusion_matrix(labels, final_predictions, labels=np.arange(num_classes))
        if labels.size
        else np.zeros((num_classes, num_classes), dtype=np.int64)
    )

    alpha, alpha_source = _class_alpha_from_snapshot(
        model=model,
        source_count=source_count,
        num_classes=num_classes,
    )
    alpha_entropy = _alpha_entropy(alpha)
    alpha_entropy_path = tables_dir / "alpha_entropy_per_class.csv"
    with alpha_entropy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_id",
                "alpha_entropy",
                "alpha_concentration",
                "effective_source_count",
                "max_alpha",
                "min_alpha",
                *source_domains,
            ]
        )
        for class_id in range(num_classes):
            effective_sources = math.exp(float(alpha_entropy[class_id]) * math.log(max(source_count, 2)))
            writer.writerow(
                [
                    class_id,
                    float(alpha_entropy[class_id]),
                    float(1.0 - alpha_entropy[class_id]),
                    float(effective_sources),
                    float(alpha[:, class_id].max()),
                    float(alpha[:, class_id].min()),
                    *[float(alpha[source_index, class_id]) for source_index in range(source_count)],
                ]
            )

    eta_distribution_path = tables_dir / "eta_distribution.csv"
    with eta_distribution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "eta",
                "override_score",
                "same_top1",
                "student_confidence",
                "teacher_confidence",
                "student_entropy",
                "teacher_entropy",
                "teacher_confidence_advantage",
                "teacher_entropy_advantage",
                "override_gate",
            ]
        )
        for sample_index in range(labels.shape[0]):
            writer.writerow(
                [
                    sample_index,
                    float(fusion["eta"][sample_index]),
                    float(fusion["override_score"][sample_index]),
                    int(fusion["same_top1"][sample_index]),
                    float(fusion["student_confidence"][sample_index]),
                    float(fusion["teacher_confidence"][sample_index]),
                    float(fusion["student_entropy"][sample_index]),
                    float(fusion["teacher_entropy"][sample_index]),
                    float(fusion["confidence_advantage"][sample_index]),
                    float(fusion["entropy_advantage"][sample_index]),
                    int(fusion["override_gate"][sample_index]),
                ]
            )

    disagreement_path = tables_dir / "teacher_student_disagreement.csv"
    with disagreement_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "student_pred",
                "teacher_pred",
                "final_pred",
                "teacher_student_disagree",
                "eta",
                "student_confidence",
                "teacher_confidence",
                "student_entropy",
                "teacher_entropy",
                "student_correct",
                "teacher_correct",
                "final_correct",
            ]
        )
        for sample_index in range(labels.shape[0]):
            writer.writerow(
                [
                    sample_index,
                    int(labels[sample_index]),
                    int(student_predictions[sample_index]),
                    int(teacher_predictions[sample_index]),
                    int(final_predictions[sample_index]),
                    int(student_predictions[sample_index] != teacher_predictions[sample_index]),
                    float(fusion["eta"][sample_index]),
                    float(fusion["student_confidence"][sample_index]),
                    float(fusion["teacher_confidence"][sample_index]),
                    float(fusion["student_entropy"][sample_index]),
                    float(fusion["teacher_entropy"][sample_index]),
                    int(student_predictions[sample_index] == labels[sample_index]),
                    int(teacher_predictions[sample_index] == labels[sample_index]),
                    int(final_predictions[sample_index] == labels[sample_index]),
                ]
            )

    override_cases_path = tables_dir / "override_cases.csv"
    with override_cases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "student_pred",
                "teacher_pred",
                "final_pred",
                "eta",
                "override_score",
                "override_gate",
                "student_correct",
                "teacher_correct",
                "final_correct",
            ]
        )
        for sample_index in range(labels.shape[0]):
            if (
                student_predictions[sample_index] == teacher_predictions[sample_index]
                and student_predictions[sample_index] == final_predictions[sample_index]
                and not bool(fusion["override_gate"][sample_index])
            ):
                continue
            writer.writerow(
                [
                    sample_index,
                    int(labels[sample_index]),
                    int(student_predictions[sample_index]),
                    int(teacher_predictions[sample_index]),
                    int(final_predictions[sample_index]),
                    float(fusion["eta"][sample_index]),
                    float(fusion["override_score"][sample_index]),
                    int(fusion["override_gate"][sample_index]),
                    int(student_predictions[sample_index] == labels[sample_index]),
                    int(teacher_predictions[sample_index] == labels[sample_index]),
                    int(final_predictions[sample_index] == labels[sample_index]),
                ]
            )

    per_class_gain_vs_codats_path = _write_recall_gain(
        tables_dir / "per_class_recall_gain_vs_codats.csv",
        labels=labels,
        final_predictions=final_predictions,
        baseline_predictions=teacher_predictions,
        baseline_name="codats_teacher",
        num_classes=num_classes,
    )
    per_class_gain_vs_wjdot_path = _write_recall_gain(
        tables_dir / "per_class_recall_gain_vs_wjdot.csv",
        labels=labels,
        final_predictions=final_predictions,
        baseline_predictions=student_predictions,
        baseline_name="student_wjdot",
        num_classes=num_classes,
    )

    summary_path = tables_dir / "teacher_safe_fusion_summary.csv"
    summary_row = {
        "teacher_available": int(teacher_available),
        "teacher_checkpoint_loaded": int(checkpoint_loaded),
        "alpha_source": alpha_source,
        "target_eval_acc": target_accuracy,
        "student_target_eval_acc": student_accuracy,
        "teacher_target_eval_acc": teacher_accuracy,
        "accuracy_gain_vs_wjdot": target_accuracy - student_accuracy,
        "accuracy_gain_vs_codats": target_accuracy - teacher_accuracy,
        "teacher_student_disagreement_count": int((student_predictions != teacher_predictions).sum()),
        "final_vs_student_disagreement_count": int((final_predictions != student_predictions).sum()),
        "override_count": int(fusion["override_gate"].sum()),
        "mean_eta": float(fusion["eta"].mean()) if fusion["eta"].size else 0.0,
        "mean_student_entropy": float(fusion["student_entropy"].mean()) if fusion["student_entropy"].size else 0.0,
        "mean_teacher_entropy": float(fusion["teacher_entropy"].mean()) if fusion["teacher_entropy"].size else 0.0,
        "lambda_same": float(fusion["lambda_same"]),
        "lambda_override": float(fusion["lambda_override"]),
        "override_confidence_margin": float(fusion["override_confidence_margin"]),
        "override_entropy_margin": float(fusion["override_entropy_margin"]),
        "eta_max": float(fusion["eta_max"]),
        "fusion_base": fusion["fusion_base"],
        "prior_balance_strength": float(fusion["prior_balance_strength"]),
        "prior_balance_student_mix": float(fusion["prior_balance_student_mix"]),
    }
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary_row)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)
    (tables_dir / "teacher_safe_fusion_summary.json").write_text(
        json.dumps(summary_row, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

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
    source_embeddings = np.concatenate([chunk["embeddings"] for chunk in source_eval_chunks], axis=0)
    source_labels = np.concatenate([chunk["labels"] for chunk in source_eval_chunks], axis=0)
    source_predictions = np.concatenate([chunk["predictions"] for chunk in source_eval_chunks], axis=0)
    source_domains_array = np.concatenate([chunk["domains"] for chunk in source_eval_chunks], axis=0)

    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        analysis_path,
        scenario_id=_string_array([scenario_id]),
        method_name=_string_array([method_name]),
        source_embeddings=source_embeddings,
        source_labels=source_labels,
        source_predictions=source_predictions,
        source_domains=source_domains_array,
        target_embeddings=target_eval_chunk["embeddings"],
        target_labels=labels,
        target_predictions=final_predictions,
        target_domains=target_eval_chunk["domains"],
        target_logits=np.log(np.maximum(final_probabilities, EPS)).astype(np.float32),
        target_probabilities=final_probabilities.astype(np.float32),
        target_predictions_student=student_predictions.astype(np.int64),
        target_predictions_teacher=teacher_predictions.astype(np.int64),
        target_probabilities_student=student_probs.astype(np.float32),
        target_probabilities_teacher=teacher_probs.astype(np.float32),
        teacher_safe_eta=fusion["eta"].astype(np.float32),
        ccsr_alpha=alpha.astype(np.float32),
        ccsr_alpha_entropy=alpha_entropy.astype(np.float32),
    )

    return {
        "analysis_path": str(analysis_path),
        "target_eval_acc": target_accuracy,
        "target_eval_macro_f1": target_macro_f1,
        "target_eval_balanced_acc": target_balanced_accuracy,
        "target_confusion_matrix": final_confusion.tolist(),
        "ca_ccsr_wjdot": {
            **summary_row,
            "enabled": True,
        },
        "teacher_student_disagreement_path": str(disagreement_path),
        "teacher_safe_fusion_summary_path": str(summary_path),
        "eta_distribution_path": str(eta_distribution_path),
        "override_cases_path": str(override_cases_path),
        "per_class_recall_gain_vs_codats_path": str(per_class_gain_vs_codats_path),
        "per_class_recall_gain_vs_wjdot_path": str(per_class_gain_vs_wjdot_path),
        "alpha_entropy_per_class_path": str(alpha_entropy_path),
    }
