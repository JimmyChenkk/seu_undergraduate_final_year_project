from __future__ import annotations

import unittest

import torch

from src.methods import build_method
from tests.test_rpl_tc_cdan import _rpl_config


def _ccs_config(*, no_reliable: bool = False) -> dict:
    config = _rpl_config(no_reliable=no_reliable)
    config["method_name"] = "ccs_rpl_tc_cdan"
    config["loss"].update(
        {
            "prototype_weight": 0.1,
            "prototype_start_step": 0,
            "prototype_warmup_steps": 0,
            "prototype_momentum": 0.9,
            "prototype_min_target_per_class": 1,
            "target_prototype_blend": 0.25,
            "class_separation_weight": 0.1,
            "class_separation_margin": 0.2,
        }
    )
    return config


class CCSRPLTCCDANTests(unittest.TestCase):
    def test_reference_prototypes_keep_float32_under_half_features(self) -> None:
        torch.manual_seed(2)
        method = build_method(_ccs_config(no_reliable=False), num_classes=29, in_channels=34, input_length=32, num_sources=1)
        features = torch.randn(4, 128).half()
        labels = torch.tensor([0, 1, 2, 0])

        reference, active = method._reference_prototypes(features, labels)

        self.assertEqual(reference.dtype, torch.float32)
        self.assertTrue(active[:3].all().item())

    def test_prototype_terms_handle_no_reliable_targets(self) -> None:
        torch.manual_seed(3)
        method = build_method(_ccs_config(no_reliable=True), num_classes=29, in_channels=34, input_length=32, num_sources=1)
        method.train()
        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0])
        target_x = torch.randn(4, 34, 32)
        target_y = torch.full((4,), -1)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
        output.loss.backward()
        hook_metrics = method.after_optimizer_step()

        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertEqual(output.metrics["prototype_loss"], 0.0)
        self.assertEqual(output.metrics["prototype_update_count"], 0.0)
        self.assertGreater(hook_metrics["prototype_source_update_count"], 0.0)
        self.assertEqual(hook_metrics["prototype_target_update_count"], 0.0)

    def test_prototype_terms_are_finite_with_reliable_targets(self) -> None:
        torch.manual_seed(4)
        method = build_method(_ccs_config(no_reliable=False), num_classes=29, in_channels=34, input_length=32, num_sources=1)
        method.train()
        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0])
        target_x = torch.randn(4, 34, 32)
        target_y = torch.full((4,), -1)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertIn("class_separation_loss", output.metrics)
        self.assertIn("class_inter_intra_ratio", output.metrics)
        self.assertIn("reliable_target_per_class_00", output.metrics)


if __name__ == "__main__":
    unittest.main()
