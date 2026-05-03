"""Data loading and split protocol for TEP domain adaptation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import canonical_domain, fold_to_name


@dataclass
class DomainSplit:
    """One domain materialized with one held-out evaluation fold."""

    domain_id: str
    fold_name: str
    train_x: np.ndarray
    train_y: np.ndarray
    eval_x: np.ndarray
    eval_y: np.ndarray
    train_indices: np.ndarray
    eval_indices: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ExperimentData:
    """Source and target splits for a single experiment."""

    source_splits: list[DomainSplit]
    target_split: DomainSplit
    num_classes: int = 29

    @property
    def source_domains(self) -> list[str]:
        return [split.domain_id for split in self.source_splits]

    @property
    def target_domain(self) -> str:
        return self.target_split.domain_id


class TEPDomainLoader:
    """Load Kaggle TEP raw pickles without target eval label leakage."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        *,
        normalization_scope: str = "domain",
        eps: float = 1e-6,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.normalization_scope = str(normalization_scope).strip().lower()
        self.eps = float(eps)

    def resolve_pickle(self, domain: int | str) -> Path:
        domain_id = canonical_domain(domain)
        number = int(domain_id.replace("mode", ""))
        path = self.raw_dir / f"TEPDataset_Mode{number}.pickle"
        if not path.exists():
            candidates = sorted(self.raw_dir.glob(f"*Mode{number}*.pickle"))
            if candidates:
                path = candidates[0]
        if not path.exists():
            raise FileNotFoundError(f"Cannot find raw pickle for {domain_id} under {self.raw_dir}.")
        return path

    def load_raw(self, domain: int | str) -> dict[str, Any]:
        path = self.resolve_pickle(domain)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        for key in ("Signals", "Labels", "Folds"):
            if key not in payload:
                raise KeyError(f"{path} is missing required key {key!r}.")
        return payload

    def split_indices(self, payload: dict[str, Any], fold: int | str) -> tuple[np.ndarray, np.ndarray, str]:
        fold_name = fold_to_name(fold)
        folds = payload["Folds"]
        if fold_name not in folds:
            raise KeyError(f"{fold_name} not found. Available folds: {sorted(folds)}")
        eval_indices = np.asarray(folds[fold_name], dtype=np.int64)
        eval_set = set(int(item) for item in eval_indices.tolist())
        sample_count = int(np.asarray(payload["Signals"]).shape[0])
        train_indices = np.asarray(
            [index for index in range(sample_count) if index not in eval_set],
            dtype=np.int64,
        )
        return train_indices, eval_indices, fold_name

    def compute_stats(self, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signals = np.asarray(signals, dtype=np.float32)
        mean = signals.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std = signals.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std = np.where(std < self.eps, 1.0, std).astype(np.float32)
        return mean, std

    def normalize(self, signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        values = np.asarray(signals, dtype=np.float32)
        normalized = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return np.transpose(normalized, (0, 2, 1)).astype(np.float32, copy=False)

    def load_domain_split(self, domain: int | str, fold: int | str) -> DomainSplit:
        domain_id = canonical_domain(domain)
        payload = self.load_raw(domain_id)
        signals = np.asarray(payload["Signals"])
        labels = np.asarray(payload["Labels"], dtype=np.int64)
        train_indices, eval_indices, fold_name = self.split_indices(payload, fold)

        if self.normalization_scope in {"train", "train_only", "train_split"}:
            mean, std = self.compute_stats(signals[train_indices])
        elif self.normalization_scope in {"domain", "full_domain", "global"}:
            mean, std = self.compute_stats(signals)
        else:
            raise ValueError(f"Unsupported normalization_scope={self.normalization_scope!r}.")

        return DomainSplit(
            domain_id=domain_id,
            fold_name=fold_name,
            train_x=self.normalize(signals[train_indices], mean, std),
            train_y=labels[train_indices],
            eval_x=self.normalize(signals[eval_indices], mean, std),
            eval_y=labels[eval_indices],
            train_indices=train_indices,
            eval_indices=eval_indices,
            mean=mean,
            std=std,
        )

    def load_experiment(self, sources: list[int | str], target: int | str, fold: int | str) -> ExperimentData:
        source_splits = [self.load_domain_split(source, fold) for source in sources]
        target_split = self.load_domain_split(target, fold)
        return ExperimentData(source_splits=source_splits, target_split=target_split)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    hide_labels: bool = False,
    drop_last: bool | None = None,
) -> DataLoader:
    labels = np.full_like(y, -1) if hide_labels else y
    tensors = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(labels).long())
    if drop_last is None:
        drop_last = bool(shuffle and len(tensors) >= batch_size)
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
