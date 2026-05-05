from __future__ import annotations

from copy import deepcopy
import unittest

import torch

from src.methods import build_method
from tests.test_ccs_rpl_tc_cdan import _ccs_config
from tests.test_rpl_tc_cdan import _rpl_config
from tests.test_tc_cdan import _tc_config


def _ca_ccsr_wjdot_config() -> dict:
    return {
        "method_name": "ca_ccsr_wjdot",
        "backbone": {
            "name": "fcn",
            "classifier_hidden_dim": 16,
            "dropout": 0.0,
            "embedding_dim": 32,
        },
        "loss": {
            "transport_solver": "sinkhorn",
            "sinkhorn_reg": 0.1,
            "sinkhorn_num_iter_max": 10,
            "alignment_start_step": 0,
            "alignment_ramp_steps": 1,
            "reliability_start_step": 0,
            "reliability_ramp_steps": 1,
            "lambda_teacher": 0.1,
            "teacher_ramp_steps": 1,
            "domain_hidden_dim": 16,
            "target_label_assist_weight": 0.0,
        },
    }


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
            _ca_ccsr_wjdot_config(),
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
