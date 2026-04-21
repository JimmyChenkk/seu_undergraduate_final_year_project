from __future__ import annotations

import unittest

import torch

from src.datasets.te_torch_dataset import _make_loader


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


if __name__ == "__main__":
    unittest.main()
