"""Lightweight target augmentations for consistency regularization."""

from __future__ import annotations

import torch


def augment_signal(
    x: torch.Tensor,
    *,
    noise_std: float = 0.01,
    time_mask_ratio: float = 0.06,
    channel_dropout_prob: float = 0.05,
) -> torch.Tensor:
    """Apply Gaussian noise, time masking, and channel dropout to ``(B,C,T)``."""

    augmented = x
    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * float(noise_std)

    if time_mask_ratio > 0 and augmented.shape[-1] > 1:
        mask_len = max(1, int(round(augmented.shape[-1] * float(time_mask_ratio))))
        max_start = max(augmented.shape[-1] - mask_len, 0)
        starts = torch.randint(0, max_start + 1, (augmented.shape[0],), device=augmented.device)
        time_masked = augmented.clone()
        for row, start in enumerate(starts.tolist()):
            time_masked[row, :, start : start + mask_len] = 0.0
        augmented = time_masked

    if channel_dropout_prob > 0:
        keep = torch.rand(
            augmented.shape[0],
            augmented.shape[1],
            1,
            device=augmented.device,
            dtype=augmented.dtype,
        ) >= float(channel_dropout_prob)
        augmented = augmented * keep

    return augmented
