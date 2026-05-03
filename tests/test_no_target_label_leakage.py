from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from src.methods import build_method
from tests.test_ccs_rpl_tc_cdan import _ccs_config
from tests.test_rpl_tc_cdan import _rpl_config
from tests.test_tc_cdan import _tc_config


class NoTargetLabelLeakageTests(unittest.TestCase):
    def _loss_with_target_labels(self, config: dict, target_y: torch.Tensor) -> float:
        torch.manual_seed(10)
        method = build_method(deepcopy(config), num_classes=29, in_channels=34, input_length=32, num_sources=1)
        method.train()
        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0])
        target_x = torch.randn(4, 34, 32)
        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
        return float(output.loss.item())

    def test_unsupervised_progressive_methods_ignore_target_batch_labels(self) -> None:
        configs = [
            _tc_config(),
            _rpl_config(no_reliable=False),
            _ccs_config(no_reliable=False),
        ]
        labels_a = torch.tensor([0, 1, 2, 3])
        labels_b = torch.tensor([28, 27, 26, 25])
        for config in configs:
            with self.subTest(method=config["method_name"]):
                loss_a = self._loss_with_target_labels(config, labels_a)
                loss_b = self._loss_with_target_labels(config, labels_b)
                self.assertAlmostEqual(loss_a, loss_b, places=6)


if __name__ == "__main__":
    unittest.main()
