"""Domain adaptation losses used by the benchmark reproduction methods."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


_EPSILON = 1e-6


def entropy(probabilities: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """Entropy helper used by confidence-aware DA methods."""

    values = -(probabilities * torch.log(probabilities.clamp_min(_EPSILON))).sum(dim=1)
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


class MinimumClassConfusionLoss(nn.Module):
    """Target-side class confusion regularizer from MCC."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("MCC temperature must be positive.")
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        probabilities = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = 1.0 + torch.exp(-entropy(probabilities).detach())
        entropy_weight = (batch_size * entropy_weight / entropy_weight.sum().clamp_min(_EPSILON)).unsqueeze(dim=1)
        class_confusion = (probabilities * entropy_weight).transpose(0, 1).mm(probabilities)
        class_confusion = class_confusion / class_confusion.sum(dim=1, keepdim=True).clamp_min(_EPSILON)
        return (class_confusion.sum() - torch.trace(class_confusion)) / float(max(num_classes, 1))


def _covariance_matrix(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.t().mm(centered) / max(features.shape[0] - 1, 1)


def coral_loss(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    *,
    align_mean: bool = True,
    normalize_covariance: bool = True,
) -> torch.Tensor:
    """Correlation alignment loss with optional mean matching."""

    if features_source.numel() == 0 or features_target.numel() == 0:
        return torch.zeros((), device=features_source.device, dtype=features_source.dtype)

    mean_source = features_source.mean(dim=0, keepdim=True)
    mean_target = features_target.mean(dim=0, keepdim=True)
    covariance_source = _covariance_matrix(features_source)
    covariance_target = _covariance_matrix(features_target)

    covariance_gap = (covariance_source - covariance_target).pow(2).sum()
    if normalize_covariance:
        feature_dim = max(features_source.shape[1], 1)
        covariance_gap = covariance_gap / float(4 * feature_dim * feature_dim)

    if not align_mean:
        return covariance_gap
    return covariance_gap + (mean_source - mean_target).pow(2).mean()


class GaussianKernel(nn.Module):
    """Gaussian kernel with running bandwidth statistics."""

    def __init__(self, sigma: float | None = None, *, track_running_stats: bool = True, alpha: float = 1.0) -> None:
        super().__init__()
        assert track_running_stats or sigma is not None
        self.track_running_stats = track_running_stats
        self.alpha = float(alpha)
        sigma_square = sigma * sigma if sigma is not None else 1.0
        self.register_buffer("sigma_square", torch.tensor(float(sigma_square)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distance_square = ((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(dim=2)
        if self.track_running_stats:
            sigma_square = self.alpha * distance_square.detach().mean().clamp_min(_EPSILON)
            self.sigma_square.copy_(sigma_square.to(self.sigma_square.device))
        sigma_square = self.sigma_square.to(device=x.device, dtype=x.dtype).clamp_min(_EPSILON)
        return torch.exp(-distance_square / (2.0 * sigma_square))


def _update_index_matrix(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    linear: bool = True,
) -> torch.Tensor:
    index_matrix = torch.zeros((2 * batch_size, 2 * batch_size), device=device, dtype=dtype)
    if linear:
        for i in range(batch_size):
            source_i = i
            source_j = (i + 1) % batch_size
            target_i = source_i + batch_size
            target_j = source_j + batch_size
            weight = 1.0 / float(batch_size)
            index_matrix[source_i, source_j] = weight
            index_matrix[target_i, target_j] = weight
            index_matrix[source_i, target_j] = -weight
            index_matrix[source_j, target_i] = -weight
        return index_matrix

    if batch_size > 1:
        intra_weight = 1.0 / float(batch_size * (batch_size - 1))
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                index_matrix[i, j] = intra_weight
                index_matrix[i + batch_size, j + batch_size] = intra_weight
    cross_weight = -1.0 / float(batch_size * batch_size)
    index_matrix[:batch_size, batch_size:] = cross_weight
    index_matrix[batch_size:, :batch_size] = cross_weight
    return index_matrix


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """MK-MMD using a configurable bank of Gaussian kernels."""

    def __init__(self, kernels: list[nn.Module], *, linear: bool = True) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(kernels)
        self.linear = linear

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = min(source.shape[0], target.shape[0])
        if batch_size <= 1:
            return torch.zeros((), device=source.device, dtype=source.dtype)

        source = source[:batch_size]
        target = target[:batch_size]
        features = torch.cat([source, target], dim=0)
        index_matrix = _update_index_matrix(
            batch_size,
            device=source.device,
            dtype=source.dtype,
            linear=self.linear,
        )
        kernel_matrix = sum(kernel(features) for kernel in self.kernels)
        loss = (kernel_matrix * index_matrix).sum()
        if self.linear:
            return loss
        return loss + 2.0 / float(batch_size - 1)


def multiple_kernel_mmd(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    *,
    kernel_scales: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0),
    linear: bool = True,
) -> torch.Tensor:
    """Convenience wrapper for MK-MMD with a standard Gaussian kernel bank."""

    kernels = [GaussianKernel(alpha=scale) for scale in kernel_scales]
    return MultipleKernelMaximumMeanDiscrepancy(kernels, linear=linear)(features_source, features_target)


def deepjdot_loss(
    source_labels: torch.Tensor,
    logits_target: torch.Tensor,
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    *,
    reg_dist: float = 0.1,
    reg_cl: float = 1.0,
    sample_weights: torch.Tensor | None = None,
    target_sample_weights: torch.Tensor | None = None,
    normalize_feature_cost: bool = False,
    solver: str = "emd",
    sinkhorn_reg: float = 0.05,
    sinkhorn_num_iter_max: int = 100,
) -> torch.Tensor:
    """Compute the minibatch DeepJDOT loss with a fixed OT coupling.

    This follows the reference DeepJDOT training recipe: compute the OT plan
    from the current minibatch using squared feature distance plus squared
    distance between source one-hot labels and target class probabilities, then
    optimize the network with that plan fixed. The optimized loss uses the same
    feature distance together with gamma-weighted target cross entropy.
    """

    try:
        import ot
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise ImportError("DeepJDOT requires the POT package (import ot).") from exc

    source_size = min(features_source.shape[0], source_labels.shape[0])
    target_size = min(features_target.shape[0], logits_target.shape[0])
    if source_size == 0 or target_size == 0:
        return torch.zeros((), device=features_source.device, dtype=features_source.dtype)

    source_labels = source_labels[:source_size]
    features_source = features_source[:source_size]
    logits_target = logits_target[:target_size]
    features_target = features_target[:target_size]

    features_source = torch.nan_to_num(features_source, nan=0.0, posinf=1e6, neginf=-1e6)
    features_target = torch.nan_to_num(features_target, nan=0.0, posinf=1e6, neginf=-1e6)
    logits_target = torch.nan_to_num(logits_target, nan=0.0, posinf=1e6, neginf=-1e6)
    num_classes = max(int(logits_target.shape[1]), 1)
    source_labels = source_labels.long()
    if source_labels.min() < 0 or source_labels.max() >= num_classes:
        raise ValueError(
            f"DeepJDOT source labels must be in [0, {num_classes - 1}], "
            f"got min={int(source_labels.min())}, max={int(source_labels.max())}."
        )

    feature_cost = torch.cdist(features_source, features_target, p=2).pow(2)
    if normalize_feature_cost:
        feature_cost = feature_cost / float(max(features_source.shape[1], 1))
    feature_cost = torch.nan_to_num(feature_cost, nan=1e6, posinf=1e6, neginf=1e6)

    target_log_probs = F.log_softmax(logits_target.float(), dim=1).to(dtype=logits_target.dtype)
    target_probabilities = target_log_probs.exp()
    source_one_hot = F.one_hot(source_labels, num_classes=num_classes).to(dtype=target_probabilities.dtype)
    class_cost_for_plan = torch.cdist(source_one_hot, target_probabilities, p=2).pow(2)
    class_cost_for_plan = torch.nan_to_num(class_cost_for_plan, nan=1e6, posinf=1e6, neginf=1e6)
    class_cost_for_loss = -target_log_probs[:, source_labels].transpose(0, 1)
    class_cost_for_loss = torch.nan_to_num(class_cost_for_loss, nan=1e6, posinf=1e6, neginf=1e6)
    transport_cost = reg_dist * feature_cost + reg_cl * class_cost_for_plan
    loss_cost = reg_dist * feature_cost + reg_cl * class_cost_for_loss

    if not torch.isfinite(transport_cost).all() or not torch.isfinite(loss_cost).all():
        raise RuntimeError("DeepJDOT encountered a non-finite transport cost.")

    cost_for_solver = transport_cost.detach().double().cpu()

    if sample_weights is None:
        sample_weights = torch.full(
            (source_size,),
            1.0 / source_size,
            device=features_source.device,
            dtype=features_source.dtype,
        )
    if target_sample_weights is None:
        target_sample_weights = torch.full(
            (target_size,),
            1.0 / target_size,
            device=features_target.device,
            dtype=features_target.dtype,
        )
    if sample_weights.numel() < source_size:
        raise ValueError(
            f"DeepJDOT source sample_weights length {sample_weights.numel()} is smaller than source batch {source_size}."
        )
    if target_sample_weights.numel() < target_size:
        raise ValueError(
            "DeepJDOT target_sample_weights length "
            f"{target_sample_weights.numel()} is smaller than target batch {target_size}."
        )
    sample_weights = sample_weights[:source_size].to(device=features_source.device, dtype=features_source.dtype)
    target_sample_weights = target_sample_weights[:target_size].to(
        device=features_target.device,
        dtype=features_target.dtype,
    )
    if not torch.isfinite(sample_weights).all() or not torch.isfinite(target_sample_weights).all():
        raise RuntimeError("DeepJDOT received non-finite sample weights.")
    sample_weights_sum = sample_weights.sum()
    target_sample_weights_sum = target_sample_weights.sum()
    if sample_weights_sum <= 0 or target_sample_weights_sum <= 0:
        raise RuntimeError("DeepJDOT sample weights must have positive mass.")
    sample_weights = sample_weights / sample_weights_sum
    target_sample_weights = target_sample_weights / target_sample_weights_sum

    solver_name = solver.strip().lower()
    sample_weights_np = sample_weights.detach().double().cpu().numpy()
    target_sample_weights_np = target_sample_weights.detach().double().cpu().numpy()
    cost_np = cost_for_solver.numpy()
    if solver_name in {"sinkhorn", "entropic", "regularized"}:
        gamma_np = ot.sinkhorn(
            sample_weights_np,
            target_sample_weights_np,
            cost_np,
            reg=max(float(sinkhorn_reg), 1e-6),
            numItermax=max(int(sinkhorn_num_iter_max), 10),
        )
    if solver_name in {"emd", "exact"}:
        gamma_np = ot.emd(sample_weights_np, target_sample_weights_np, cost_np)
    if solver_name not in {"sinkhorn", "entropic", "regularized", "emd", "exact"}:
        raise ValueError(f"Unsupported DeepJDOT solver: {solver}")

    gamma = torch.as_tensor(gamma_np, device=features_source.device, dtype=loss_cost.dtype)
    if not torch.isfinite(gamma).all():
        raise RuntimeError("DeepJDOT transport solver returned a non-finite coupling.")
    gamma_sum = gamma.sum()
    if gamma_sum <= 0:
        raise RuntimeError("DeepJDOT transport solver returned an empty coupling.")
    gamma = gamma / gamma_sum
    return (gamma.detach() * loss_cost).sum()


class GradientReverseFunction(Function):
    """Classic GRL used by DANN-like methods."""

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
        self.lambda_ = float(lambda_)
        self.last_coeff = self.lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_coeff = self.lambda_
        return GradientReverseFunction.apply(x, self.lambda_)


class WarmStartGradientReverseLayer(nn.Module):
    """Logistic warm-start GRL commonly used in adversarial DA training."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        lo: float = 0.0,
        hi: float = 1.0,
        max_iters: float = 1000.0,
        auto_step: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.lo = float(lo)
        self.hi = float(hi)
        self.max_iters = max(float(max_iters), 1.0)
        self.auto_step = auto_step
        self.iter_num = 0
        self.last_coeff = self.lo

    def _compute_coeff(self) -> float:
        return (
            2.0 * (self.hi - self.lo) / (1.0 + math.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo)
            + self.lo
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = self._compute_coeff()
        self.last_coeff = coeff
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(x, coeff)

    def step(self) -> None:
        self.iter_num += 1


class DomainDiscriminator(nn.Module):
    """Configurable MLP discriminator for adversarial alignment."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        hidden_layers = max(int(num_hidden_layers), 1)
        layers: list[nn.Module] = []
        current_dim = in_features
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).squeeze(-1)


def _binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    return float((predictions == labels).float().mean().item())


def domain_adversarial_loss(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    *,
    discriminator: DomainDiscriminator,
    grl: nn.Module,
    weights_source: torch.Tensor | None = None,
    weights_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Compute adversarial domain loss and discriminator accuracy."""

    features = grl(torch.cat([features_source, features_target], dim=0))
    logits_source, logits_target = discriminator(features).split([features_source.shape[0], features_target.shape[0]], dim=0)
    labels_source = torch.ones_like(logits_source)
    labels_target = torch.zeros_like(logits_target)

    loss_source = F.binary_cross_entropy_with_logits(logits_source, labels_source, weight=weights_source)
    loss_target = F.binary_cross_entropy_with_logits(logits_target, labels_target, weight=weights_target)
    accuracy = 0.5 * (
        _binary_accuracy_from_logits(logits_source, labels_source)
        + _binary_accuracy_from_logits(logits_target, labels_target)
    )
    return 0.5 * (loss_source + loss_target), accuracy


class MultiLinearMap(nn.Module):
    """Deterministic multilinear map used by CDAN."""

    def forward(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        conditioned = torch.bmm(probabilities.unsqueeze(2), features.unsqueeze(1))
        return conditioned.view(batch_size, -1)


class RandomizedMultiLinearMap(nn.Module):
    """Lower-dimensional randomized conditioning map used for stable CDAN training."""

    def __init__(self, feature_dim: int, num_classes: int, output_dim: int = 1024) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("rf", torch.randn(feature_dim, output_dim))
        self.register_buffer("rg", torch.randn(num_classes, output_dim))

    def forward(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        projected_features = features @ self.rf
        projected_probabilities = probabilities @ self.rg
        return (projected_features * projected_probabilities) / math.sqrt(float(self.output_dim))


class ConditionalDomainAdversarialLoss(nn.Module):
    """CDAN/CDAN+E loss with optional entropy conditioning."""

    def __init__(
        self,
        *,
        domain_discriminator: DomainDiscriminator,
        feature_dim: int,
        num_classes: int,
        entropy_conditioning: bool = False,
        randomized: bool = False,
        randomized_dim: int = 1024,
        grl: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.domain_discriminator = domain_discriminator
        self.entropy_conditioning = entropy_conditioning
        self.grl = grl if grl is not None else WarmStartGradientReverseLayer(auto_step=True)
        if randomized:
            self.map = RandomizedMultiLinearMap(feature_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()

    def _normalized_entropy_weight(self, probabilities: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + torch.exp(-entropy(probabilities.detach()))
        return weights / weights.detach().sum().clamp_min(_EPSILON) * weights.shape[0]

    def forward(
        self,
        logits_source: torch.Tensor,
        features_source: torch.Tensor,
        logits_target: torch.Tensor,
        features_target: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        probabilities_source = F.softmax(logits_source, dim=1).detach()
        probabilities_target = F.softmax(logits_target, dim=1).detach()

        conditioned_source = self.map(features_source, probabilities_source)
        conditioned_target = self.map(features_target, probabilities_target)
        conditioned = torch.cat([conditioned_source, conditioned_target], dim=0)
        logits_domain = self.domain_discriminator(self.grl(conditioned))
        labels_domain = torch.cat(
            [
                torch.ones(features_source.shape[0], device=logits_domain.device, dtype=logits_domain.dtype),
                torch.zeros(features_target.shape[0], device=logits_domain.device, dtype=logits_domain.dtype),
            ],
            dim=0,
        )

        weights = None
        if self.entropy_conditioning:
            probabilities = torch.cat([probabilities_source, probabilities_target], dim=0)
            weights = self._normalized_entropy_weight(probabilities).to(
                device=logits_domain.device,
                dtype=logits_domain.dtype,
            )

        loss_domain = F.binary_cross_entropy_with_logits(
            logits_domain,
            labels_domain,
            weight=weights,
        )
        accuracy_domain = _binary_accuracy_from_logits(logits_domain, labels_domain)
        return loss_domain, accuracy_domain
