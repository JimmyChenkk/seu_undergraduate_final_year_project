"""DeepJDOT implementation using minibatch OT on GPU-friendly torch tensors."""

from __future__ import annotations

import torch.nn.functional as F

from src.losses import deepjdot_loss

from .base import AdaptationWeightScheduler, MethodStepOutput, SingleSourceMethodBase, accuracy_from_logits


class DeepJDOTMethod(SingleSourceMethodBase):
    """Deep joint distribution optimal transport with a shared encoder."""

    method_name = "deepjdot"

    def __init__(
        self,
        *,
        adaptation_weight: float = 1.0,
        adaptation_schedule: str = "warm_start",
        adaptation_max_steps: int = 1000,
        adaptation_schedule_alpha: float = 10.0,
        reg_dist: float = 0.1,
        reg_cl: float = 1.0,
        normalize_feature_cost: bool = True,
        transport_solver: str = "sinkhorn",
        sinkhorn_reg: float = 0.05,
        sinkhorn_num_iter_max: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alignment_scheduler = AdaptationWeightScheduler(
            base_weight=adaptation_weight,
            schedule=adaptation_schedule,
            max_steps=adaptation_max_steps,
            alpha=adaptation_schedule_alpha,
        )
        self.reg_dist = reg_dist
        self.reg_cl = reg_cl
        self.normalize_feature_cost = normalize_feature_cost
        self.transport_solver = transport_solver
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_num_iter_max = sinkhorn_num_iter_max

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source = self.forward(source_x)
        logits_target, features_target = self.forward(target_x)

        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_alignment = deepjdot_loss(
            source_y,
            logits_target,
            features_source,
            features_target,
            reg_dist=self.reg_dist,
            reg_cl=self.reg_cl,
            normalize_feature_cost=self.normalize_feature_cost,
            solver=self.transport_solver,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_num_iter_max=self.sinkhorn_num_iter_max,
        )
        if not torch.isfinite(loss_alignment):
            loss_alignment = torch.zeros_like(loss_alignment)
        current_weight = self.alignment_scheduler.step()
        loss_total = loss_cls + current_weight * loss_alignment
        if not torch.isfinite(loss_total):
            loss_total = loss_cls

        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_alignment.item()),
                "lambda_alignment": current_weight,
                "acc_source": accuracy_from_logits(logits_source, source_y),
            },
        )
