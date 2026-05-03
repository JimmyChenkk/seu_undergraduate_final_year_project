from __future__ import annotations

import inspect
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from src.datasets.te_da_dataset import (
    DomainAdaptationSetting,
    DomainDataReference,
    DomainSpec,
    TEDADatasetConfig,
)
from src.datasets.te_torch_dataset import _cache_key, _make_loader
from src.datasets.te_torch_dataset import prepare_benchmark_data


class TETorchDatasetTests(unittest.TestCase):
    def test_shuffled_loader_keeps_small_training_split(self) -> None:
        x = torch.randn(3, 2, 4)
        y = torch.tensor([0, 1, 2], dtype=torch.long)

        loader = _make_loader(
            x,
            y,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        self.assertEqual(len(loader), 1)
        x_batch, y_batch = next(iter(loader))
        self.assertEqual(tuple(x_batch.shape), (3, 2, 4))
        self.assertEqual(int(y_batch.numel()), 3)

    def test_prepared_cache_disabled_by_default(self) -> None:
        config = TEDADatasetConfig.from_dict({"loading": {}})

        self.assertFalse(config.prepared_cache_enabled)

    def test_prepared_cache_key_is_loader_independent(self) -> None:
        signature = inspect.signature(_cache_key)
        self.assertNotIn("batch_size", signature.parameters)
        self.assertNotIn("num_workers", signature.parameters)

        config = TEDADatasetConfig()
        setting = DomainAdaptationSetting(
            setting_name="single_source",
            source_domains=[
                DomainDataReference(
                    domain=DomainSpec(name="mode1"),
                    storage="raw",
                )
            ],
            target_domain=DomainDataReference(
                domain=DomainSpec(name="mode2"),
                storage="raw",
            ),
        )

        key = _cache_key(
            config=config,
            setting=setting,
            source_fold_name="Fold 1",
            target_fold_name="Fold 2",
        )

        self.assertEqual(len(key), 40)

    def test_unlabeled_target_train_loader_masks_labels(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            benchmark_dir = root / "benchmark"
            raw_dir.mkdir()
            benchmark_dir.mkdir()
            payload = {
                "Signals": np.random.randn(6, 4, 2).astype(np.float32),
                "Labels": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
                "Folds": {"Fold 1": np.asarray([0, 1], dtype=np.int64)},
            }
            for mode in ("Mode1", "Mode2"):
                with (raw_dir / f"TEPDataset_{mode}.pickle").open("wb") as handle:
                    pickle.dump(payload, handle)
            manifest = {
                "dataset_name": "mini",
                "domain_count": 2,
                "domains": [
                    {
                        "domain_id": "mode1",
                        "source_file": "TEPDataset_Mode1.pickle",
                        "signals_shape": [6, 4, 2],
                        "fold_names": ["Fold 1"],
                    },
                    {
                        "domain_id": "mode2",
                        "source_file": "TEPDataset_Mode2.pickle",
                        "signals_shape": [6, 4, 2],
                        "fold_names": ["Fold 1"],
                    },
                ],
            }
            manifest_path = benchmark_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            config = TEDADatasetConfig(
                raw_dir=raw_dir,
                benchmark_dir=benchmark_dir,
                cache_dir=root / "cache",
                manifest_path=manifest_path,
                normalization_scope="train",
            )
            setting = DomainAdaptationSetting(
                setting_name="single_source",
                source_domains=[DomainDataReference(DomainSpec(name="mode1"), storage="raw")],
                target_domain=DomainDataReference(DomainSpec(name="mode2"), storage="raw"),
                target_label_mode="unlabeled",
            )

            prepared = prepare_benchmark_data(config=config, setting=setting, batch_size=2)
            _, target_y = next(iter(prepared.target_train_loader))

            self.assertTrue(torch.equal(target_y, torch.full_like(target_y, -1)))


if __name__ == "__main__":
    unittest.main()
