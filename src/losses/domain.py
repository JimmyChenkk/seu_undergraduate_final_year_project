"""Domain adaptation losses used by the benchmark reproduction methods."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


def coral_loss(features_source: torch.Tensor, features_target: torch.Tensor) -> torch.Tensor:
    """Correlation alignment loss."""

    mean_source = features_source.mean(dim=0, keepdim=True)
    mean_target = features_target.mean(dim=0, keepdim=True)
    centered_source = features_source - mean_source
    centered_target = features_target - mean_target
    covariance_source = centered_source.t().mm(centered_source) / max(len(features_source) - 1, 1)
    covariance_target = centered_target.t().mm(centered_target) / max(len(features_target) - 1, 1)
    mean_gap = (mean_source - mean_target).pow(2).mean()
    covariance_gap = (covariance_source - covariance_target).pow(2).mean()
    return mean_gap + covariance_gap


def _gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
) -> torch.Tensor:
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(dim=2)

    if fix_sigma is None:
        n_samples = total.shape[0]
        bandwidth = l2_distance.detach().sum() / max(n_samples * (n_samples - 1), 1)
    else:
        bandwidth = torch.tensor(fix_sigma, device=total.device, dtype=total.dtype)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernel_values = [
        torch.exp(-l2_distance / (bandwidth * (kernel_mul**index) + 1e-12))
        for index in range(kernel_num)
    ]
    return sum(kernel_values)


def multiple_kernel_mmd(features_source: torch.Tensor, features_target: torch.Tensor) -> torch.Tensor:
    """Multiple-kernel MMD with a conservative Gaussian kernel bank."""

    batch_size = min(features_source.shape[0], features_target.shape[0])
    if batch_size == 0:
        return torch.tensor(0.0, device=features_source.device)
    source = features_source[:batch_size]
    target = features_target[:batch_size]
    kernels = _gaussian_kernel(source, target)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()


class GradientReverseFunction(Function):
    """Classic GRL used by DANN."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


class GradientReverseLayer(nn.Module):
    """Small wrapper around the GRL autograd function."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReverseFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """Shallow discriminator for adversarial alignment."""

    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).squeeze(-1)


def domain_adversarial_loss(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    *,
    discriminator: DomainDiscriminator,
    grl: GradientReverseLayer,
) -> tuple[torch.Tensor, float]:
    """Compute adversarial domain loss and discriminator accuracy."""

    source_features = grl(features_source)
    target_features = grl(features_target)
    logits_source = discriminator(source_features)
    logits_target = discriminator(target_features)
    labels_source = torch.ones_like(logits_source)
    labels_target = torch.zeros_like(logits_target)

    loss_source = F.binary_cross_entropy_with_logits(logits_source, labels_source)
    loss_target = F.binary_cross_entropy_with_logits(logits_target, labels_target)
    loss = 0.5 * (loss_source + loss_target)

    with torch.no_grad():
        predictions_source = (torch.sigmoid(logits_source) >= 0.5).float()
        predictions_target = (torch.sigmoid(logits_target) >= 0.5).float()
        correct = (predictions_source == labels_source).float().sum() + (
            predictions_target == labels_target
        ).float().sum()
        total = labels_source.numel() + labels_target.numel()
        accuracy = float(correct / max(total, 1))
    return loss, accuracy
