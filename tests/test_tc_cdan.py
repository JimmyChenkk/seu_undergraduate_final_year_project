from __future__ import annotations

import unittest

import torch

from src.methods import build_method
from src.methods.tc_cdan import TCCDANMethod


def _tc_config() -> dict:
    return {
        "method_name": "tc_cdan",
        "backbone": {"name": "fcn", "classifier_hidden_dim": 32, "dropout": 0.0},
        "loss": {
            "adaptation_weight": 0.2,
            "adaptation_schedule": "warm_start",
            "adaptation_max_steps": 8,
            "grl_lambda": 1.0,
            "grl_warm_start": True,
            "grl_max_iters": 8,
            "randomized": True,
            "randomized_dim": 32,
            "entropy_conditioning": True,
            "domain_hidden_dim": 64,
            "domain_num_hidden_layers": 1,
            "teacher_ema_decay": 0.9,
            "teacher_temperature": 1.0,
            "consistency_weight": 0.1,
            "consistency_start_step": 0,
            "consistency_warmup_steps": 1,
            "augment": {
                "weak_jitter_std": 0.0,
                "weak_scaling_std": 0.0,
                "strong_jitter_std": 0.0,
                "strong_scaling_std": 0.0,
                "strong_time_mask_ratio": 0.0,
                "strong_channel_dropout_prob": 0.0,
            },
        },
    }


class TCCDANTests(unittest.TestCase):
    def test_forward_and_single_batch_loss_are_finite(self) -> None:
        torch.manual_seed(0)
        method = build_method(_tc_config(), num_classes=29, in_channels=34, input_length=32, num_sources=1)
        self.assertIsInstance(method, TCCDANMethod)
        method.train()

        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0])
        target_x = torch.randn(4, 34, 32)
        target_y = torch.full((4,), -1)

        logits, features = method(target_x)
        self.assertEqual(tuple(logits.shape), (4, 29))
        self.assertEqual(tuple(features.shape), (4, 128))

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
        self.assertTrue(torch.isfinite(output.loss).item())
        output.loss.backward()
        hook_metrics = method.after_optimizer_step()
        self.assertIn("consistency_loss", output.metrics)
        self.assertIn("domain_accuracy", output.metrics)
        self.assertIn("teacher_ema_decay", hook_metrics)


if __name__ == "__main__":
    unittest.main()
