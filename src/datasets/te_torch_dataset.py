"""Torch-ready TEP dataset utilities for benchmark reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .te_da_dataset import (
    DEFAULT_FOLD_NAME,
    DomainAdaptationSetting,
    TEDADatasetConfig,
    TEDADatasetInterface,
    canonicalize_domain_id,
    slice_domain_split,
)


@dataclass
class DomainSplitTensors:
    """Materialized tensors for one domain and one held-out fold."""

    domain_id: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    eval_x: torch.Tensor
    eval_y: torch.Tensor
    train_indices: np.ndarray
    eval_indices: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(torch.unique(torch.cat([self.train_y, self.eval_y])).numel())

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.train_x.shape[1:])


@dataclass
class PreparedBenchmarkData:
    """Grouped loaders and metadata for one experiment setting."""

    setting: DomainAdaptationSetting
    source_splits: list[DomainSplitTensors]
    target_split: DomainSplitTensors
    source_train_loaders: list[DataLoader]
    source_train_eval_loaders: list[DataLoader]
    source_eval_loaders: list[DataLoader]
    target_train_loader: DataLoader
    target_eval_loader: DataLoader


def _make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
    )


def load_domain_split(
    interface: TEDADatasetInterface,
    domain_id: str,
    *,
    fold_name: str | None = None,
) -> DomainSplitTensors:
    """Load one domain into train/eval tensors using the selected fold."""

    canonical = canonicalize_domain_id(domain_id)
    payload = interface.load_raw_domain_payload(canonical)
    train_indices, eval_indices = interface.build_train_eval_indices(
        canonical,
        fold_name=fold_name,
    )
    train_x, train_y = slice_domain_split(
        payload,
        train_indices,
        normalization=interface.config.normalization,
        channels_first=interface.config.channels_first,
    )
    eval_x, eval_y = slice_domain_split(
        payload,
        eval_indices,
        normalization=interface.config.normalization,
        channels_first=interface.config.channels_first,
    )
    return DomainSplitTensors(
        domain_id=canonical,
        train_x=torch.from_numpy(train_x).float(),
        train_y=torch.from_numpy(train_y).long(),
        eval_x=torch.from_numpy(eval_x).float(),
        eval_y=torch.from_numpy(eval_y).long(),
        train_indices=train_indices,
        eval_indices=eval_indices,
    )


def prepare_benchmark_data(
    *,
    config: TEDADatasetConfig,
    setting: DomainAdaptationSetting,
    batch_size: int,
    num_workers: int = 0,
    fold_name: str | None = None,
) -> PreparedBenchmarkData:
    """Prepare loaders for single-source or multi-source benchmark experiments."""

    interface = TEDADatasetInterface(config)
    selected_fold = fold_name or config.preferred_fold or DEFAULT_FOLD_NAME

    source_splits = [
        load_domain_split(interface, reference.domain.name, fold_name=selected_fold)
        for reference in setting.source_domains
    ]
    target_split = load_domain_split(
        interface,
        setting.target_domain.domain.name,
        fold_name=selected_fold,
    )

    source_train_loaders = [
        _make_loader(
            split.train_x,
            split.train_y,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for split in source_splits
    ]
    source_train_eval_loaders = [
        _make_loader(
            split.train_x,
            split.train_y,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        for split in source_splits
    ]
    source_eval_loaders = [
        _make_loader(
            split.eval_x,
            split.eval_y,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        for split in source_splits
    ]
    target_train_loader = _make_loader(
        target_split.train_x,
        target_split.train_y,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    target_eval_loader = _make_loader(
        target_split.eval_x,
        target_split.eval_y,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return PreparedBenchmarkData(
        setting=setting,
        source_splits=source_splits,
        target_split=target_split,
        source_train_loaders=source_train_loaders,
        source_train_eval_loaders=source_train_eval_loaders,
        source_eval_loaders=source_eval_loaders,
        target_train_loader=target_train_loader,
        target_eval_loader=target_eval_loader,
    )
