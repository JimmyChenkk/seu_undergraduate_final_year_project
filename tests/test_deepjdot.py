from __future__ import annotations

import importlib.util
import unittest

import torch

from src.methods import build_method
from src.methods.deepjdot import _inverse_sqrt_class_weights


def _deepjdot_family_config(method_name: str) -> dict:
    return {
        "method_name": method_name,
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
            "transport_solver": "sinkhorn",
            "sinkhorn_reg": 0.1,
            "sinkhorn_num_iter_max": 30,
            "prototype_weight": 0.05,
            "prototype_start_step": 0,
            "prototype_warmup_steps": 1,
            "pseudo_weight": 0.02,
            "pseudo_start_step": 0,
            "pseudo_warmup_steps": 1,
            "tau_start": 0.0,
            "tau_end": 0.0,
        },
    }


def _tpu_family_config(method_name: str) -> dict:
    return {
        "method_name": method_name,
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
            "normalize_feature_cost": True,
            "transport_solver": "sinkhorn_unbalanced",
            "sinkhorn_reg": 0.1,
            "sinkhorn_num_iter_max": 30,
            "unbalanced_transport": True,
            "uot_tau_s": 1.0,
            "uot_tau_t": 1.0,
            "source_warmup_steps": 0,
            "alignment_start_step": 0,
            "supcon_weight": 0.03,
            "supcon_warmup_only": False,
            "prototype_cost_weight": 0.01,
            "prototype_start_step": 0,
            "prototype_warmup_steps": 1,
            "temporal_cost_weight": 0.01,
            "temporal_start_step": 0,
            "temporal_warmup_steps": 1,
            "pseudo_weight": 0.02,
            "pseudo_start_step": 0,
            "pseudo_warmup_steps": 1,
            "tau_start": 0.0,
            "tau_end": 0.0,
            "consistency_weight": 0.01,
            "consistency_start_step": 0,
            "consistency_warmup_steps": 1,
            "q_ot_entropy_threshold": 1.0,
            "js_threshold": 1.0,
        },
    }


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

    def test_deepjdot_innovation_family_backward(self) -> None:
        for method_name in ["tp_deepjdot", "cbtp_deepjdot"]:
            with self.subTest(method=method_name):
                torch.manual_seed(21)
                method = build_method(
                    _deepjdot_family_config(method_name),
                    num_classes=5,
                    in_channels=4,
                    input_length=32,
                    num_sources=1,
                )
                method.train()
                source_x = torch.randn(4, 4, 32)
                source_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                target_x = torch.randn(4, 4, 32)
                target_y = torch.full((4,), -1, dtype=torch.long)

                step_output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
                step_output.loss.backward()

                gradients = [parameter.grad for parameter in method.parameters() if parameter.grad is not None]
                self.assertTrue(torch.isfinite(step_output.loss).item())
                self.assertGreater(step_output.metrics["loss_alignment"], 0.0)
                self.assertGreater(step_output.metrics["lambda_prototype"], 0.0)
                self.assertIn("loss_prototype", step_output.metrics)
                if method_name == "cbtp_deepjdot":
                    self.assertGreater(step_output.metrics["lambda_pseudo"], 0.0)
                    self.assertIn("loss_pseudo", step_output.metrics)
                self.assertTrue(gradients)
                self.assertGreater(float(sum(gradient.abs().sum() for gradient in gradients)), 0.0)

    def test_unbalanced_tpu_family_backward_and_records_diagnostics(self) -> None:
        for method_name in ["u_deepjdot", "tpu_deepjdot", "cbtpu_deepjdot"]:
            with self.subTest(method=method_name):
                torch.manual_seed(31)
                config = (
                    _tpu_family_config(method_name)
                    if method_name != "u_deepjdot"
                    else _deepjdot_family_config(method_name)
                )
                if method_name == "u_deepjdot":
                    config["loss"].update(
                        {
                            "transport_solver": "sinkhorn_unbalanced",
                            "unbalanced_transport": True,
                            "uot_tau_s": 1.0,
                            "uot_tau_t": 1.0,
                        }
                    )
                method = build_method(
                    config,
                    num_classes=5,
                    in_channels=4,
                    input_length=32,
                    num_sources=1,
                )
                method.train()
                source_x = torch.randn(6, 4, 32)
                source_y = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
                target_x = torch.randn(5, 4, 32)
                target_y = torch.full((5,), -1, dtype=torch.long)

                step_output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
                step_output.loss.backward()

                gradients = [parameter.grad for parameter in method.parameters() if parameter.grad is not None]
                self.assertTrue(torch.isfinite(step_output.loss).item())
                self.assertGreater(step_output.metrics["loss_alignment"], 0.0)
                self.assertIn("uot_transported_mass", step_output.metrics)
                self.assertIn("uot_row_mass_deviation", step_output.metrics)
                self.assertIn("q_ot_entropy_mean", step_output.metrics)
                if method_name in {"tpu_deepjdot", "cbtpu_deepjdot"}:
                    self.assertIn("source_supcon_loss", step_output.metrics)
                    self.assertIn("prototype_relative_cost_mean", step_output.metrics)
                if method_name == "cbtpu_deepjdot":
                    self.assertIn("q_ot_cls_proto_agreement_rate", step_output.metrics)
                    self.assertIn("accepted_target_ratio", step_output.metrics)
                    self.assertIn("accepted_pseudo_label_class_00", step_output.metrics)
                self.assertTrue(gradients)
                self.assertGreater(float(sum(gradient.abs().sum() for gradient in gradients)), 0.0)

    def test_deepjdot_innovation_family_ignores_target_labels(self) -> None:
        for method_name in ["tp_deepjdot", "cbtp_deepjdot", "u_deepjdot", "tpu_deepjdot", "cbtpu_deepjdot"]:
            losses = []
            for target_y in (
                torch.tensor([0, 1, 2, 3], dtype=torch.long),
                torch.tensor([4, 3, 2, 1], dtype=torch.long),
            ):
                torch.manual_seed(22)
                config = (
                    _tpu_family_config(method_name)
                    if method_name in {"tpu_deepjdot", "cbtpu_deepjdot"}
                    else _deepjdot_family_config(method_name)
                )
                if method_name == "u_deepjdot":
                    config["loss"].update(
                        {
                            "transport_solver": "sinkhorn_unbalanced",
                            "unbalanced_transport": True,
                            "uot_tau_s": 1.0,
                            "uot_tau_t": 1.0,
                        }
                    )
                method = build_method(
                    config,
                    num_classes=5,
                    in_channels=4,
                    input_length=32,
                    num_sources=1,
                )
                method.train()
                source_x = torch.randn(4, 4, 32)
                source_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                target_x = torch.randn(4, 4, 32)
                output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
                losses.append(float(output.loss.item()))

            with self.subTest(method=method_name):
                self.assertAlmostEqual(losses[0], losses[1], places=6)

    def test_cbtp_class_weights_preserve_amp_dtype(self) -> None:
        labels = torch.tensor([0, 0, 2, 4], dtype=torch.long)
        mask = torch.tensor([True, True, False, True])

        weights = _inverse_sqrt_class_weights(
            labels,
            mask=mask,
            num_classes=5,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )

        self.assertEqual(weights.dtype, torch.float16)
        self.assertTrue(torch.isfinite(weights).all().item())


if __name__ == "__main__":
    unittest.main()
