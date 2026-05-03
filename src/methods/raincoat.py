"""RAINCOAT baseline adapted to the benchmark training interface."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import MethodStepOutput, accuracy_from_logits


_EPSILON = 1e-6


class _SpectralConv1d(nn.Module):
    """Low-frequency Fourier feature extractor from the RAINCOAT reference."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = max(int(modes), 1)
        scale = 1.0 / float(max(self.in_channels * self.out_channels, 1))
        self.weights = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_freq = torch.fft.rfft(torch.cos(x.float()), norm="ortho")
        usable_modes = min(self.modes, x_freq.shape[-1])
        out_freq = torch.zeros(
            x.shape[0],
            self.out_channels,
            x_freq.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        weights = self.weights[:, :, :usable_modes].to(device=x.device)
        out_freq[:, :, :usable_modes] = torch.einsum("bim,iom->bom", x_freq[:, :, :usable_modes], weights)
        low_freq = out_freq[:, :, :usable_modes]
        freq_features = torch.cat([low_freq.abs(), low_freq.angle()], dim=-1)
        return freq_features, out_freq


class _RaincoatCNN(nn.Module):
    """Time-domain CNN branch used by RAINCOAT."""

    def __init__(
        self,
        *,
        in_channels: int,
        mid_channels: int,
        final_out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
        features_len: int,
    ) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                final_out_channels,
                kernel_size=8,
                stride=1,
                bias=False,
                padding=4,
            ),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(features_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        return x.reshape(x.shape[0], -1)


class _RaincoatEncoder(nn.Module):
    """Time-frequency encoder with a normalized fused representation."""

    def __init__(
        self,
        *,
        in_channels: int,
        input_length: int,
        fourier_modes: int,
        mid_channels: int,
        final_out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
        features_len: int,
    ) -> None:
        super().__init__()
        max_modes = max(input_length // 2, 1)
        self.fourier_modes = min(max(int(fourier_modes), 1), max_modes)
        self.final_out_channels = int(final_out_channels)
        self.features_len = max(int(features_len), 1)
        self.freq = _SpectralConv1d(in_channels, in_channels, self.fourier_modes)
        self.freq_pool = nn.Conv1d(in_channels, 1, kernel_size=3, stride=1, bias=False, padding=1)
        self.freq_norm = nn.BatchNorm1d(self.fourier_modes * 2)
        self.time = _RaincoatCNN(
            in_channels=in_channels,
            mid_channels=mid_channels,
            final_out_channels=final_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            features_len=self.features_len,
        )
        self.out_dim = self.fourier_modes * 2 + self.final_out_channels * self.features_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freq_features, out_freq = self.freq(x)
        freq_features = self.freq_pool(freq_features).squeeze(1)
        freq_features = F.relu(self.freq_norm(freq_features), inplace=True)
        time_features = self.time(x)
        features = torch.cat([freq_features, time_features], dim=1)
        return F.normalize(features, dim=1), out_freq


class _RaincoatDecoder(nn.Module):
    """RAINCOAT reconstruction decoder for low/high frequency components."""

    def __init__(
        self,
        *,
        in_channels: int,
        input_length: int,
        final_out_channels: int,
        features_len: int,
        fourier_feature_dim: int,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.fourier_feature_dim = int(fourier_feature_dim)
        self.final_out_channels = int(final_out_channels)
        self.features_len = max(int(features_len), 1)
        self.low_norm = nn.BatchNorm1d(in_channels)
        self.high_norm = nn.BatchNorm1d(in_channels)
        self.high_decoder = nn.ConvTranspose1d(
            final_out_channels,
            input_length,
            kernel_size=in_channels,
            stride=1,
        )

    def forward(self, features: torch.Tensor, out_freq: torch.Tensor) -> torch.Tensor:
        low_reconstruction = self.low_norm(torch.fft.irfft(out_freq, n=self.input_length, norm="ortho"))
        time_features = features[:, self.fourier_feature_dim :]
        time_features = time_features.reshape(features.shape[0], self.final_out_channels, self.features_len)
        high_reconstruction = self.high_decoder(time_features).permute(0, 2, 1)
        high_reconstruction = F.relu(self.high_norm(high_reconstruction), inplace=True)
        return low_reconstruction + high_reconstruction


class _TemperatureClassifier(nn.Module):
    """Linear RAINCOAT classifier with reference temperature scaling."""

    def __init__(self, in_features: int, num_classes: int, temperature: float) -> None:
        super().__init__()
        self.logits = nn.Linear(in_features, num_classes, bias=False)
        self.temperature = max(float(temperature), _EPSILON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x) / self.temperature


class _SinkhornDistance(nn.Module):
    """Device-native differentiable Sinkhorn distance for minibatch features."""

    def __init__(self, *, eps: float = 0.05, max_iter: int = 50, threshold: float = 1e-3) -> None:
        super().__init__()
        self.eps = max(float(eps), _EPSILON)
        self.max_iter = max(int(max_iter), 1)
        self.threshold = max(float(threshold), 0.0)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = source.float()
        target = target.float()
        if source.shape[0] == 0 or target.shape[0] == 0:
            return torch.zeros((), device=source.device, dtype=source.dtype)

        cost = torch.cdist(source, target, p=2).pow(2)
        n_source, n_target = cost.shape
        mu = torch.full((n_source,), 1.0 / n_source, device=source.device, dtype=source.dtype)
        nu = torch.full((n_target,), 1.0 / n_target, device=source.device, dtype=source.dtype)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        log_mu = torch.log(mu.clamp_min(_EPSILON))
        log_nu = torch.log(nu.clamp_min(_EPSILON))

        for _ in range(self.max_iter):
            previous_u = u
            modified_cost = (-cost + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps
            u = self.eps * (log_mu - torch.logsumexp(modified_cost, dim=1)) + u
            modified_cost = (-cost + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps
            v = self.eps * (log_nu - torch.logsumexp(modified_cost.transpose(0, 1), dim=1)) + v
            if self.threshold > 0 and torch.mean(torch.abs(u - previous_u)).item() < self.threshold:
                break

        transport_plan = torch.exp((-cost + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps)
        return torch.sum(transport_plan * cost)


class RAINCOATMethod(nn.Module):
    """RAINCOAT with time-frequency reconstruction and Sinkhorn alignment."""

    method_name = "raincoat"
    supports_multi_source = True

    def __init__(
        self,
        *,
        num_classes: int,
        in_channels: int = 34,
        input_length: int = 600,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        backbone_name: str = "raincoat",
        backbone_kwargs: dict | None = None,
        fourier_modes: int = 64,
        mid_channels: int = 64,
        final_out_channels: int = 128,
        kernel_size: int = 5,
        stride: int = 1,
        features_len: int = 1,
        classifier_temperature: float = 0.1,
        sinkhorn_weight: float = 0.5,
        reconstruction_weight: float = 1e-4,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_max_iter: int = 50,
        sinkhorn_threshold: float = 1e-3,
        reconstruction_reduction: str = "mean",
    ) -> None:
        del classifier_hidden_dim, backbone_name
        super().__init__()
        config = dict(backbone_kwargs or {})
        fourier_modes = int(config.get("fourier_modes", fourier_modes))
        mid_channels = int(config.get("mid_channels", mid_channels))
        final_out_channels = int(config.get("final_out_channels", final_out_channels))
        kernel_size = int(config.get("kernel_size", kernel_size))
        stride = int(config.get("stride", stride))
        features_len = int(config.get("features_len", features_len))
        self.encoder = _RaincoatEncoder(
            in_channels=in_channels,
            input_length=input_length,
            fourier_modes=fourier_modes,
            mid_channels=mid_channels,
            final_out_channels=final_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            features_len=features_len,
        )
        self.decoder = _RaincoatDecoder(
            in_channels=in_channels,
            input_length=input_length,
            final_out_channels=final_out_channels,
            features_len=features_len,
            fourier_feature_dim=self.encoder.fourier_modes * 2,
        )
        self.classifier = _TemperatureClassifier(
            self.encoder.out_dim,
            num_classes,
            classifier_temperature,
        )
        self.sinkhorn = _SinkhornDistance(
            eps=sinkhorn_epsilon,
            max_iter=sinkhorn_max_iter,
            threshold=sinkhorn_threshold,
        )
        self.sinkhorn_weight = float(sinkhorn_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.reconstruction_reduction = str(reconstruction_reduction).strip().lower()
        if self.reconstruction_reduction not in {"mean", "sum"}:
            raise KeyError(f"Unsupported RAINCOAT reconstruction reduction: {reconstruction_reduction}")

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, out_freq = self.encoder(x)
        logits = self.classifier(features)
        return logits, features, out_freq

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, features, _ = self._encode(x)
        return logits, features

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.forward(x)
        return features

    def merge_source_batches(self, source_batches) -> tuple[torch.Tensor, torch.Tensor]:
        if len(source_batches) == 1:
            return source_batches[0]
        source_x = torch.cat([x_batch for x_batch, _ in source_batches], dim=0)
        source_y = torch.cat([y_batch for _, y_batch in source_batches], dim=0)
        return source_x, source_y

    def after_optimizer_step(self) -> dict[str, float]:
        return {}

    def _reconstruction_loss(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
        freq_source: torch.Tensor,
        freq_target: torch.Tensor,
    ) -> torch.Tensor:
        source_reconstruction = self.decoder(features_source, freq_source)
        target_reconstruction = self.decoder(features_target, freq_target)
        reduction = self.reconstruction_reduction
        source_loss = F.l1_loss(source_reconstruction, source_x.float(), reduction=reduction)
        target_loss = F.l1_loss(target_reconstruction, target_x.float(), reduction=reduction)
        if reduction == "sum":
            denominator = float(max(source_x.shape[0] + target_x.shape[0], 1))
            return (source_loss + target_loss) / denominator
        return source_loss + target_loss

    def compute_loss(self, source_batches, target_batch) -> MethodStepOutput:
        source_x, source_y = self.merge_source_batches(source_batches)
        target_x, _ = target_batch
        logits_source, features_source, freq_source = self._encode(source_x)
        _, features_target, freq_target = self._encode(target_x)

        loss_cls = F.cross_entropy(logits_source, source_y)
        loss_sinkhorn = self.sinkhorn(features_source, features_target).to(
            device=loss_cls.device,
            dtype=loss_cls.dtype,
        )
        loss_reconstruction = self._reconstruction_loss(
            source_x,
            target_x,
            features_source,
            features_target,
            freq_source,
            freq_target,
        ).to(device=loss_cls.device, dtype=loss_cls.dtype)
        loss_total = (
            loss_cls
            + self.sinkhorn_weight * loss_sinkhorn
            + self.reconstruction_weight * loss_reconstruction
        )
        if not torch.isfinite(loss_total).item():
            raise RuntimeError("RAINCOAT produced a non-finite total loss.")

        return MethodStepOutput(
            loss=loss_total,
            metrics={
                "loss_total": float(loss_total.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_alignment": float(loss_sinkhorn.item()),
                "loss_reconstruction": float(loss_reconstruction.item()),
                "lambda_alignment": self.sinkhorn_weight,
                "lambda_reconstruction": self.reconstruction_weight,
                "acc_source": accuracy_from_logits(logits_source, source_y),
            },
        )
