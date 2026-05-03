"""Reliable pseudo-label TC-CDAN for unlabeled target diagnosis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import MethodStepOutput
from .tc_cdan import TCCDANMethod, _EPSILON, _zero_like_loss


class RPLTCCDANMethod(TCCDANMethod):
    """RPL-TC-CDAN: TC-CDAN with conservative reliable pseudo-label learning."""

    method_name = "rpl_tc_cdan"

    def __init__(
        self,
        *,
        pseudo_weight: float = 0.08,
        pseudo_start_step: int = 800,
        pseudo_warmup_steps: int = 600,
        pseudo_confidence_threshold: float = 0.75,
        pseudo_entropy_threshold: float = 0.55,
        pseudo_max_per_class: int = 4,
        pseudo_use_reliability_weighting: bool = True,
        reliability_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pseudo_weight = float(pseudo_weight)
        self.pseudo_start_step = max(int(pseudo_start_step), 0)
        self.pseudo_warmup_steps = max(int(pseudo_warmup_steps), 0)
        self.pseudo_confidence_threshold = min(max(float(pseudo_confidence_threshold), 0.0), 1.0)
        self.pseudo_entropy_threshold = min(max(float(pseudo_entropy_threshold), 0.0), 1.0)
        self.pseudo_max_per_class = max(int(pseudo_max_per_class), 0)
        self.pseudo_use_reliability_weighting = bool(pseudo_use_reliability_weighting)

        raw_weights = {"confidence": 1.0, "inverse_entropy": 1.0, "agreement": 1.0}
        if isinstance(reliability_weights, dict):
            for key in raw_weights:
                if key in reliability_weights:
                    raw_weights[key] = max(float(reliability_weights[key]), 0.0)
        total = sum(raw_weights.values())
        if total <= 0:
            raw_weights = {key: 1.0 for key in raw_weights}
            total = float(len(raw_weights))
        self.reliability_weights = {key: value / total for key, value in raw_weights.items()}

    def _class_balanced_selection(
        self,
        *,
        candidate_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
        reliability_score: torch.Tensor,
    ) -> torch.Tensor:
        selected = torch.zeros_like(candidate_mask, dtype=torch.bool)
        if not candidate_mask.any():
            return selected
        if self.pseudo_max_per_class <= 0:
            return candidate_mask.clone()

        for class_id in torch.unique(pseudo_labels[candidate_mask]):
            class_indices = torch.nonzero(
                candidate_mask & (pseudo_labels == class_id),
                as_tuple=False,
            ).squeeze(1)
            if class_indices.numel() == 0:
                continue
            keep_count = min(self.pseudo_max_per_class, int(class_indices.numel()))
            top_positions = torch.topk(reliability_score[class_indices], k=keep_count, largest=True).indices
            selected[class_indices[top_positions]] = True
        return selected

    def _histogram_metrics(self, prefix: str, labels: torch.Tensor, *, mask: torch.Tensor | None = None) -> dict[str, float]:
        if mask is not None:
            labels = labels[mask]
        counts = torch.bincount(labels.detach().long(), minlength=self.num_classes)
        return {
            f"{prefix}_{class_index:02d}": float(counts[class_index].item())
            for class_index in range(self.num_classes)
        }

    def _compute_pseudo_terms(self, components: dict[str, Any]) -> tuple[torch.Tensor, float, dict[str, float], dict[str, Any]]:
        teacher_probabilities = components["teacher_probabilities"]
        student_probabilities = components["student_probabilities"]
        pseudo_labels = components["teacher_predictions"]
        current_step = int(components["current_step"])
        confidence = components["target_confidence"]
        entropy = components["target_entropy"]
        agreement_binary = components["agreement"]
        agreement_score = student_probabilities.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1).detach()
        inverse_entropy = 1.0 - entropy
        reliability_score = (
            self.reliability_weights["confidence"] * confidence
            + self.reliability_weights["inverse_entropy"] * inverse_entropy
            + self.reliability_weights["agreement"] * agreement_score
        ).clamp(0.0, 1.0)

        pseudo_enabled = current_step >= self.pseudo_start_step
        candidate_mask = (confidence >= self.pseudo_confidence_threshold) & (
            entropy <= self.pseudo_entropy_threshold
        )
        if not pseudo_enabled:
            candidate_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        selected_mask = self._class_balanced_selection(
            candidate_mask=candidate_mask,
            pseudo_labels=pseudo_labels,
            reliability_score=reliability_score,
        )

        logits_target_strong = components["logits_target_strong"]
        if selected_mask.any():
            pseudo_values = F.cross_entropy(
                logits_target_strong[selected_mask],
                pseudo_labels[selected_mask],
                reduction="none",
            )
            if self.pseudo_use_reliability_weighting:
                weights = reliability_score[selected_mask].detach()
                pseudo_loss = (pseudo_values * weights).sum() / weights.sum().clamp_min(_EPSILON)
            else:
                pseudo_loss = pseudo_values.mean()
            pseudo_mean_confidence = float(confidence[selected_mask].mean().item())
            pseudo_mean_entropy = float(entropy[selected_mask].mean().item())
        else:
            pseudo_loss = _zero_like_loss(logits_target_strong)
            pseudo_mean_confidence = 0.0
            pseudo_mean_entropy = 0.0

        lambda_pseudo = self._ramp_weight(
            self.pseudo_weight,
            current_step,
            self.pseudo_start_step,
            self.pseudo_warmup_steps,
        )
        selected_count = int(selected_mask.sum().item())
        batch_size = max(int(pseudo_labels.numel()), 1)
        metrics = {
            "loss_pseudo": float(pseudo_loss.item()),
            "pseudo_loss": float(pseudo_loss.item()),
            "lambda_pseudo": float(lambda_pseudo),
            "pseudo_accept_rate": float(selected_count / batch_size),
            "pseudo_accept_count": float(selected_count),
            "pseudo_mean_confidence": pseudo_mean_confidence,
            "pseudo_mean_entropy": pseudo_mean_entropy,
            "pseudo_candidate_rate": float(candidate_mask.float().mean().item()),
            "pseudo_reliability_mean": float(reliability_score.mean().item()),
            "pseudo_teacher_student_agreement": float(agreement_binary.mean().item()),
        }
        metrics.update(self._histogram_metrics("pseudo_class_histogram", pseudo_labels, mask=selected_mask))
        metrics.update(self._histogram_metrics("target_prediction_class_histogram", pseudo_labels))

        state = {
            "pseudo_labels": pseudo_labels,
            "selected_mask": selected_mask,
            "candidate_mask": candidate_mask,
            "reliability_score": reliability_score,
            "confidence": confidence,
            "entropy": entropy,
        }
        return pseudo_loss, lambda_pseudo, metrics, state

    def _compute_rpl_components(self, source_batches, target_batch) -> dict[str, Any]:
        components = self._compute_tc_components(source_batches, target_batch)
        pseudo_loss, lambda_pseudo, pseudo_metrics, pseudo_state = self._compute_pseudo_terms(components)
        total_loss = components["base_loss"] + lambda_pseudo * pseudo_loss
        metrics = dict(components["metrics"])
        metrics.update(pseudo_metrics)
        metrics["loss_total"] = float(total_loss.item())
        components.update(
            {
                "rpl_loss": total_loss,
                "loss_pseudo": pseudo_loss,
                "lambda_pseudo": lambda_pseudo,
                "pseudo_metrics": pseudo_metrics,
                "pseudo_state": pseudo_state,
                "metrics": metrics,
            }
        )
        return components

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        components = self._compute_rpl_components(source_batches, target_batch)
        return MethodStepOutput(loss=components["rpl_loss"], metrics=components["metrics"])
