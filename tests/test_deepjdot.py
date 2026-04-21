from __future__ import annotations

import importlib.util
import unittest

import torch

from src.methods import build_method


@unittest.skipUnless(importlib.util.find_spec("ot") is not None, "POT is required for DeepJDOT")
class DeepJDOTMethodTests(unittest.TestCase):
    def test_deepjdot_method_backward_uses_reference_defaults(self) -> None:
        torch.manual_seed(11)
        method = build_method(
            {
                "method_name": "deepjdot",
                "backbone": {
                    "name": "fcn",
                    "classifier_hidden_dim": 32,
                    "dropout": 0.0,
                },
                "loss": {
                    "adaptation_weight": 1.0,
                    "adaptation_schedule": "constant",
                    "reg_dist": 0.1,
                    "reg_cl": 1.0,
                    "normalize_feature_cost": False,
                    "transport_solver": "emd",
                },
            },
            num_classes=29,
            in_channels=34,
            input_length=32,
            num_sources=1,
        )
        method.train()
        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        target_x = torch.randn(4, 34, 32)
        target_y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        step_output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
        step_output.loss.backward()

        gradients = [parameter.grad for parameter in method.encoder.parameters() if parameter.grad is not None]
        self.assertEqual(step_output.metrics["lambda_alignment"], 1.0)
        self.assertGreater(step_output.metrics["loss_alignment"], 0.0)
        self.assertTrue(gradients)
        self.assertGreater(float(sum(gradient.abs().sum() for gradient in gradients)), 0.0)


if __name__ == "__main__":
    unittest.main()
