from __future__ import annotations

import inspect
import unittest

import torch

from src.datasets.te_da_dataset import (
    DomainAdaptationSetting,
    DomainDataReference,
    DomainSpec,
    TEDADatasetConfig,
)
from src.datasets.te_torch_dataset import _cache_key, _make_loader


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


if __name__ == "__main__":
    unittest.main()
