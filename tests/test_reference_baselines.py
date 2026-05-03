from __future__ import annotations

from copy import deepcopy
import importlib.util
import unittest

import torch

from src.methods import build_method


def _base_config(method_name: str) -> dict:
    return {
        "method_name": method_name,
        "backbone": {
            "name": "raincoat" if method_name == "raincoat" else "fcn",
            "classifier_hidden_dim": 16,
            "dropout": 0.0,
            "fourier_modes": 8,
            "mid_channels": 8,
            "final_out_channels": 8,
            "features_len": 1,
        },
        "loss": {
            "adaptation_weight": 0.5,
            "adaptation_schedule": "constant",
            "randomized": True,
            "randomized_dim": 32,
            "domain_hidden_dim": 16,
            "domain_num_hidden_layers": 1,
            "kernel_mul": 2.0,
            "kernel_num": 3,
            "sinkhorn_weight": 0.5,
            "reconstruction_weight": 1e-4,
            "sinkhorn_epsilon": 0.1,
            "sinkhorn_max_iter": 5,
        },
    }


class ReferenceBaselineMethodTests(unittest.TestCase):
    def _build(self, method_name: str):
        return build_method(
            _base_config(method_name),
            num_classes=5,
            in_channels=4,
            input_length=32,
            num_sources=1,
        )

    def test_new_reference_baselines_backward(self) -> None:
        methods = ["codats", "cdan_ts", "dsan", "raincoat"]
        if importlib.util.find_spec("ot") is not None:
            methods.append("deepjdot")

        for method_name in methods:
            with self.subTest(method=method_name):
                torch.manual_seed(12)
                method = self._build(method_name)
                method.train()
                source_x = torch.randn(4, 4, 32)
                source_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                target_x = torch.randn(4, 4, 32)
                target_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

                step_output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
                step_output.loss.backward()

                gradients = [parameter.grad for parameter in method.parameters() if parameter.grad is not None]
                self.assertTrue(torch.isfinite(step_output.loss).item())
                self.assertTrue(gradients)
                self.assertGreater(float(sum(gradient.abs().sum() for gradient in gradients)), 0.0)

    def test_unsupervised_reference_baselines_ignore_target_labels(self) -> None:
        for method_name in ["codats", "cdan_ts", "dsan", "raincoat"]:
            with self.subTest(method=method_name):
                labels_a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                labels_b = torch.tensor([4, 3, 2, 1], dtype=torch.long)
                loss_values = []
                for labels in (labels_a, labels_b):
                    torch.manual_seed(14)
                    method = self._build(method_name)
                    method.train()
                    source_x = torch.randn(4, 4, 32)
                    source_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                    target_x = torch.randn(4, 4, 32)
                    output = method.compute_loss([(source_x, source_y)], (target_x, labels))
                    loss_values.append(float(output.loss.item()))

                self.assertAlmostEqual(loss_values[0], loss_values[1], places=6)


if __name__ == "__main__":
    unittest.main()
