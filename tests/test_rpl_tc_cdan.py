from __future__ import annotations

import unittest

import torch

from src.methods import build_method
from tests.test_tc_cdan import _tc_config


def _rpl_config(*, no_reliable: bool = False) -> dict:
    config = _tc_config()
    config["method_name"] = "rpl_tc_cdan"
    config["loss"].update(
        {
            "pseudo_weight": 0.2,
            "pseudo_start_step": 0,
            "pseudo_warmup_steps": 0,
            "pseudo_confidence_threshold": 1.0 if no_reliable else 0.0,
            "pseudo_entropy_threshold": 0.0 if no_reliable else 1.0,
            "pseudo_max_per_class": 2,
            "pseudo_use_reliability_weighting": True,
            "reliability_weights": {
                "confidence": 1.0,
                "inverse_entropy": 1.0,
                "agreement": 1.0,
            },
        }
    )
    return config


class RPLTCCDANTests(unittest.TestCase):
    def test_pseudo_loss_handles_empty_reliable_batch(self) -> None:
        torch.manual_seed(1)
        method = build_method(_rpl_config(no_reliable=True), num_classes=29, in_channels=34, input_length=32, num_sources=1)
        method.train()
        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0])
        target_x = torch.randn(4, 34, 32)
        target_y = torch.full((4,), -1)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertEqual(output.metrics["pseudo_accept_count"], 0.0)
        self.assertEqual(output.metrics["pseudo_loss"], 0.0)
        self.assertIn("pseudo_class_histogram_00", output.metrics)
        self.assertIn("target_prediction_class_histogram_00", output.metrics)

    def test_pseudo_loss_can_select_reliable_samples(self) -> None:
        torch.manual_seed(2)
        method = build_method(_rpl_config(no_reliable=False), num_classes=29, in_channels=34, input_length=32, num_sources=1)
        method.train()
        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0])
        target_x = torch.randn(4, 34, 32)
        target_y = torch.full((4,), -1)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertGreaterEqual(output.metrics["pseudo_accept_count"], 1.0)
        self.assertGreaterEqual(output.metrics["pseudo_accept_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
