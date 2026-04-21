"""Torch-ready TEP dataset utilities for benchmark reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .te_da_dataset import (
    DEFAULT_FOLD_NAME,
    DomainAdaptationSetting,
    TEDADatasetConfig,
    TEDADatasetInterface,
    canonicalize_domain_id,
    compute_normalization_statistics,
    normalize_signals,
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
    cache_key: str | None = None
    cache_hit: bool = False


def _make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    dataset = TensorDataset(x, y)
    drop_last = bool(shuffle and len(dataset) >= batch_size)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)


def _cache_root(config: TEDADatasetConfig) -> Path:
    return Path(config.cache_dir) / "benchmark_prepared"


def _cache_key(
    *,
    config: TEDADatasetConfig,
    setting: DomainAdaptationSetting,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    source_fold_name: str,
    target_fold_name: str,
) -> str:
    payload = {
        "setting": setting.setting_name,
        "source_domains": [reference.domain.name for reference in setting.source_domains],
        "target_domain": setting.target_domain.domain.name,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "source_fold_name": str(source_fold_name),
        "target_fold_name": str(target_fold_name),
        "normalization": config.normalization,
        "normalization_scope": config.normalization_scope,
        "channels_first": bool(config.channels_first),
        "preferred_fold": config.preferred_fold,
        "raw_dir": str(config.raw_dir),
        "manifest_path": str(config.manifest_path),
        "raw_file_pattern": config.raw_file_pattern,
    }
    digest = sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _cache_path(config: TEDADatasetConfig, cache_key: str) -> Path:
    return _cache_root(config) / f"{cache_key}.npz"


def _build_domain_split_from_arrays(
    *,
    domain_id: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> DomainSplitTensors:
    return DomainSplitTensors(
        domain_id=domain_id,
        train_x=torch.from_numpy(train_x).float(),
        train_y=torch.from_numpy(train_y).long(),
        eval_x=torch.from_numpy(eval_x).float(),
        eval_y=torch.from_numpy(eval_y).long(),
        train_indices=train_indices,
        eval_indices=eval_indices,
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

    normalization_scope = str(interface.config.normalization_scope).strip().lower()
    normalization_stats = None
    if normalization_scope in {"domain", "full_domain", "global"}:
        full_signals = np.asarray(payload["Signals"])
        normalized_full = normalize_signals(
            full_signals,
            normalization=interface.config.normalization,
            channels_first=interface.config.channels_first,
        )
        train_x = normalized_full[train_indices]
        eval_x = normalized_full[eval_indices]
    else:
        stats_source_indices = train_indices if normalization_scope in {"train", "train_split"} else None
        if stats_source_indices is not None:
            stats_source = np.asarray(payload["Signals"])[stats_source_indices]
            normalization_stats = compute_normalization_statistics(
                stats_source,
                normalization=interface.config.normalization,
            )
        train_x, train_y = slice_domain_split(
            payload,
            train_indices,
            normalization=interface.config.normalization,
            normalization_stats=normalization_stats,
            channels_first=interface.config.channels_first,
        )
        eval_x, eval_y = slice_domain_split(
            payload,
            eval_indices,
            normalization=interface.config.normalization,
            normalization_stats=normalization_stats,
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

    train_y = np.asarray(payload["Labels"], dtype=np.int64)[train_indices]
    eval_y = np.asarray(payload["Labels"], dtype=np.int64)[eval_indices]
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
    pin_memory: bool = False,
    persistent_workers: bool = False,
    fold_name: str | None = None,
) -> PreparedBenchmarkData:
    """Prepare loaders for single-source or multi-source benchmark experiments."""

    interface = TEDADatasetInterface(config)
    random_fold_enabled = bool(getattr(config, "random_fold_enabled", False))
    if random_fold_enabled:
        source_fold_name = getattr(config, "source_fold", None) or fold_name or DEFAULT_FOLD_NAME
        target_fold_name = getattr(config, "target_fold", None) or fold_name or DEFAULT_FOLD_NAME
    else:
        source_fold_name = getattr(config, "source_fold", None) or fold_name or config.preferred_fold or DEFAULT_FOLD_NAME
        target_fold_name = getattr(config, "target_fold", None) or fold_name or config.preferred_fold or DEFAULT_FOLD_NAME
    cache_key = _cache_key(
        config=config,
        setting=setting,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        source_fold_name=source_fold_name,
        target_fold_name=target_fold_name,
    )
    cache_path = _cache_path(config, cache_key)

    source_splits: list[DomainSplitTensors]
    target_split: DomainSplitTensors
    cache_hit = False
    if cache_path.exists():
        try:
            payload = np.load(cache_path, allow_pickle=False)
            split_names = [reference.domain.name for reference in setting.source_domains]
            source_splits = []
            for index, domain_name in enumerate(split_names):
                source_splits.append(
                    _build_domain_split_from_arrays(
                        domain_id=domain_name,
                        train_x=payload[f"source_{index}_train_x"],
                        train_y=payload[f"source_{index}_train_y"],
                        eval_x=payload[f"source_{index}_eval_x"],
                        eval_y=payload[f"source_{index}_eval_y"],
                        train_indices=payload[f"source_{index}_train_indices"],
                        eval_indices=payload[f"source_{index}_eval_indices"],
                    )
                )
            target_split = _build_domain_split_from_arrays(
                domain_id=setting.target_domain.domain.name,
                train_x=payload["target_train_x"],
                train_y=payload["target_train_y"],
                eval_x=payload["target_eval_x"],
                eval_y=payload["target_eval_y"],
                train_indices=payload["target_train_indices"],
                eval_indices=payload["target_eval_indices"],
            )
            cache_hit = True
        except Exception:
            cache_path.unlink(missing_ok=True)
            cache_hit = False
    if not cache_hit:
        source_splits = [
            load_domain_split(interface, reference.domain.name, fold_name=source_fold_name)
            for reference in setting.source_domains
        ]
        target_split = load_domain_split(
            interface,
            setting.target_domain.domain.name,
            fold_name=target_fold_name,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload: dict[str, np.ndarray] = {}
        for index, split in enumerate(source_splits):
            cache_payload[f"source_{index}_train_x"] = split.train_x.numpy()
            cache_payload[f"source_{index}_train_y"] = split.train_y.numpy()
            cache_payload[f"source_{index}_eval_x"] = split.eval_x.numpy()
            cache_payload[f"source_{index}_eval_y"] = split.eval_y.numpy()
            cache_payload[f"source_{index}_train_indices"] = split.train_indices
            cache_payload[f"source_{index}_eval_indices"] = split.eval_indices
        cache_payload["target_train_x"] = target_split.train_x.numpy()
        cache_payload["target_train_y"] = target_split.train_y.numpy()
        cache_payload["target_eval_x"] = target_split.eval_x.numpy()
        cache_payload["target_eval_y"] = target_split.eval_y.numpy()
        cache_payload["target_train_indices"] = target_split.train_indices
        cache_payload["target_eval_indices"] = target_split.eval_indices
        np.savez_compressed(cache_path, **cache_payload)

    source_train_loaders = [
        _make_loader(
            split.train_x,
            split.train_y,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
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
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
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
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        for split in source_splits
    ]
    target_train_loader = _make_loader(
        target_split.train_x,
        target_split.train_y,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    target_eval_loader = _make_loader(
        target_split.eval_x,
        target_split.eval_y,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
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
        cache_key=cache_key,
        cache_hit=cache_hit,
    )
