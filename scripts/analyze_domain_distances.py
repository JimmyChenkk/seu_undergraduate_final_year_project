#!/usr/bin/env python
"""Compute lightweight mode-level distance proxies for TEP domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.te_da_dataset import TEDADatasetConfig, TEDADatasetInterface
from src.trainers.train_benchmark import load_yaml


def _domain_summary(interface: TEDADatasetInterface, domain_id: str, *, max_samples: int, seed: int) -> dict:
    payload = interface.load_raw_domain_payload(domain_id)
    signals = np.asarray(payload["Signals"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    if max_samples > 0 and signals.shape[0] > max_samples:
        indices = np.sort(rng.choice(signals.shape[0], size=max_samples, replace=False))
        signals = signals[indices]
    sample_channel_means = signals.mean(axis=1)
    mean = sample_channel_means.mean(axis=0)
    covariance = np.cov(sample_channel_means, rowvar=False)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "domain_id": domain_id,
        "sample_count": int(signals.shape[0]),
        "mean": mean,
        "covariance": covariance,
    }


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def _bures_proxy(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    sqrt_a = _sqrtm_psd(cov_a)
    middle = sqrt_a @ cov_b @ sqrt_a
    return float(np.trace(cov_a + cov_b - 2.0 * _sqrtm_psd(middle)))


def analyze_distances(data_config_path: Path, *, domains: list[str] | None, max_samples: int, seed: int) -> dict:
    data_payload = load_yaml(data_config_path)
    config = TEDADatasetConfig.from_dict(data_payload)
    interface = TEDADatasetInterface(config)
    domain_ids = domains or interface.list_manifest_domains()
    summaries = [_domain_summary(interface, domain_id, max_samples=max_samples, seed=seed) for domain_id in domain_ids]
    pairs = []
    for left_index, left in enumerate(summaries):
        for right in summaries[left_index + 1 :]:
            mean_distance = float(np.linalg.norm(left["mean"] - right["mean"]))
            covariance_distance = float(np.linalg.norm(left["covariance"] - right["covariance"], ord="fro"))
            bures_distance = _bures_proxy(left["covariance"], right["covariance"])
            pairs.append(
                {
                    "left": left["domain_id"],
                    "right": right["domain_id"],
                    "mean_l2": mean_distance,
                    "cov_fro": covariance_distance,
                    "bures_proxy": bures_distance,
                    "combined_proxy": float(mean_distance + bures_distance),
                }
            )
    return {
        "data_config": str(data_config_path),
        "domains": domain_ids,
        "max_samples": max_samples,
        "seed": seed,
        "pairwise": sorted(pairs, key=lambda item: item["combined_proxy"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/te_da.yaml"))
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_distances(
        args.data_config,
        domains=args.domains,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
