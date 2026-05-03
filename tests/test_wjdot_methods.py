from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.methods import build_method
from src.trainers.train_benchmark import _export_reliability_tables
from src.tep_ot.ot_losses import OTLossConfig, jdot_transport_loss


def _config(method_name: str) -> dict:
    return {
        "method_name": method_name,
        "backbone": {
            "name": "fcn_temporal" if "tp" in method_name else "fcn",
            "classifier_hidden_dim": 16,
            "dropout": 0.0,
            "embedding_dim": 32,
        },
        "loss": {
            "transport_solver": "sinkhorn",
            "sinkhorn_reg": 0.1,
            "sinkhorn_num_iter_max": 30,
            "tau_steps": 10,
        },
    }


class WJDOTMethodTests(unittest.TestCase):
    def test_wjdot_family_backward(self) -> None:
        for method_name in [
            "jdot",
            "tp_jdot",
            "cbtp_jdot",
            "wjdot",
            "tp_wjdot",
            "cbtp_wjdot",
            "ms_cbtp_wjdot",
        ]:
            with self.subTest(method=method_name):
                torch.manual_seed(7)
                method = build_method(
                    _config(method_name),
                    num_classes=5,
                    in_channels=4,
                    input_length=32,
                    num_sources=2 if method_name == "ms_cbtp_wjdot" else 1,
                )
                method.train()
                source_x = torch.randn(4, 4, 32)
                source_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                target_x = torch.randn(4, 4, 32)
                target_y = torch.full((4,), -1, dtype=torch.long)
                source_batches = [(source_x, source_y)]
                if method_name == "ms_cbtp_wjdot":
                    source_batches.append(
                        (
                            torch.randn(4, 4, 32),
                            torch.tensor([1, 2, 3, 4], dtype=torch.long),
                        )
                    )

                step_output = method.compute_loss(source_batches, (target_x, target_y))
                step_output.loss.backward()
                gradients = [parameter.grad for parameter in method.parameters() if parameter.grad is not None]

                self.assertTrue(torch.isfinite(step_output.loss).item())
                self.assertGreater(step_output.metrics["loss_alignment"], 0.0)
                self.assertTrue(gradients)
                self.assertGreater(float(sum(gradient.abs().sum() for gradient in gradients)), 0.0)

    def test_jdot_family_keeps_source_ce_unweighted(self) -> None:
        for method_name in ["jdot", "tp_jdot", "cbtp_jdot"]:
            with self.subTest(method=method_name):
                method = build_method(
                    _config(method_name),
                    num_classes=5,
                    in_channels=4,
                    input_length=32,
                    num_sources=1,
                )
                self.assertFalse(method.use_source_class_balance)

    def test_wjdot_family_keeps_source_class_balancing(self) -> None:
        for method_name in ["wjdot", "tp_wjdot", "cbtp_wjdot", "ms_cbtp_wjdot"]:
            with self.subTest(method=method_name):
                method = build_method(
                    _config(method_name),
                    num_classes=5,
                    in_channels=4,
                    input_length=32,
                    num_sources=2 if method_name == "ms_cbtp_wjdot" else 1,
                )
                self.assertTrue(method.use_source_class_balance)

    def test_cbtp_jdot_uses_confidence_curriculum_without_source_balance(self) -> None:
        config = _config("cbtp_jdot")
        config["loss"].update(
            {
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
                "prototype_start_step": 0,
                "prototype_warmup_steps": 1,
                "pseudo_start_step": 0,
                "tau_start": 0.0,
                "tau_end": 0.0,
            }
        )
        torch.manual_seed(11)
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
        target_y = torch.full((4,), -1, dtype=torch.long)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertFalse(method.use_source_class_balance)
        self.assertIn("pseudo_acceptance", output.metrics)
        self.assertGreaterEqual(output.metrics["pseudo_acceptance"], 0.0)
        self.assertGreaterEqual(output.metrics["loss_pseudo"], 0.0)

    def test_target_labels_are_ignored(self) -> None:
        losses = []
        for target_y in (
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.tensor([4, 3, 2, 1], dtype=torch.long),
        ):
            torch.manual_seed(13)
            method = build_method(
                _config("tp_wjdot"),
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

        self.assertAlmostEqual(losses[0], losses[1], places=6)

    def test_target_label_assist_uses_visible_target_labels(self) -> None:
        losses = []
        config = _config("tp_wjdot")
        config["loss"].update(
            {
                "target_label_assist_weight": 1.0,
                "target_label_assist_start_step": 0,
                "target_label_assist_warmup_steps": 1,
            }
        )
        for target_y in (
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.tensor([4, 3, 2, 1], dtype=torch.long),
        ):
            torch.manual_seed(13)
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

            self.assertGreater(output.metrics["lambda_target_label_assist"], 0.0)
            self.assertEqual(output.metrics["target_label_assist_count"], 4.0)

        self.assertNotAlmostEqual(losses[0], losses[1], places=4)

    def test_prototype_gate_masks_prototype_cost(self) -> None:
        source_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        source_labels = torch.tensor([0, 1], dtype=torch.long)
        target_logits = torch.tensor([[6.0, -6.0], [-6.0, 6.0]])
        source_prototypes = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        config = OTLossConfig(
            feature_weight=0.0,
            label_weight=0.0,
            prototype_weight=1.0,
            normalize_costs=True,
            solver="sinkhorn",
            sinkhorn_reg=0.1,
            sinkhorn_num_iter=30,
        )

        _, metrics_open = jdot_transport_loss(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_logits=target_logits,
            num_classes=2,
            config=config,
            source_prototypes=source_prototypes,
        )
        _, metrics_closed = jdot_transport_loss(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_logits=target_logits,
            num_classes=2,
            config=config,
            source_prototypes=source_prototypes,
            prototype_gate=torch.zeros(2),
        )

        self.assertGreater(metrics_open["ot_prototype_cost"], 0.0)
        self.assertEqual(metrics_closed["ot_prototype_cost"], 0.0)

    def test_tp_prototype_weight_can_be_delayed(self) -> None:
        config = _config("tp_wjdot")
        config["loss"].update(
            {
                "prototype_weight": 0.5,
                "prototype_start_step": 5,
                "prototype_warmup_steps": 5,
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
            }
        )
        torch.manual_seed(17)
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
        target_y = torch.full((4,), -1, dtype=torch.long)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertEqual(output.metrics["lambda_prototype"], 0.0)
        self.assertEqual(output.metrics["ot_prototype_cost"], 0.0)

    def test_cbtp_prototype_gate_follows_confidence_curriculum(self) -> None:
        config = _config("cbtp_wjdot")
        config["loss"].update(
            {
                "prototype_weight": 0.5,
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
                "prototype_start_step": 0,
                "prototype_warmup_steps": 1,
                "pseudo_start_step": 0,
                "tau_start": 1.0,
                "tau_end": 1.0,
            }
        )
        torch.manual_seed(19)
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
        target_y = torch.full((4,), -1, dtype=torch.long)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertEqual(output.metrics["prototype_acceptance"], 0.0)
        self.assertEqual(output.metrics["ot_prototype_cost"], 0.0)

    def test_tp_ignores_classifier_confidence_for_prototype_gate(self) -> None:
        config = _config("tp_wjdot")
        config["loss"].update(
            {
                "prototype_mode": "tp_residual_safe",
                "prototype_weight": 0.5,
                "prototype_in_coupling": False,
                "prototype_confidence_threshold": 1.0,
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
                "prototype_start_step": 0,
                "prototype_warmup_steps": 1,
            }
        )
        torch.manual_seed(23)
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
        target_y = torch.full((4,), -1, dtype=torch.long)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertEqual(output.metrics["prototype_acceptance"], 1.0)
        self.assertGreater(output.metrics["ot_prototype_cost"], 0.0)

    def test_barycentric_prototype_loss_uses_ot_class_mass(self) -> None:
        source_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        target_features = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
        source_labels = torch.tensor([0, 1], dtype=torch.long)
        target_logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
        source_prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        config = OTLossConfig(
            feature_weight=0.1,
            label_weight=1.0,
            prototype_weight=0.5,
            prototype_in_coupling=False,
            prototype_mode="tp_barycentric",
            ot_class_entropy_gate=True,
            solver="sinkhorn",
            sinkhorn_reg=0.1,
            sinkhorn_num_iter=30,
        )

        loss, metrics = jdot_transport_loss(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_logits=target_logits,
            num_classes=2,
            config=config,
            source_prototypes=source_prototypes,
        )
        loss.backward()

        self.assertGreater(metrics["ot_barycentric_active_classes"], 0.0)
        self.assertIn("ot_class_entropy_gate_mean", metrics)
        self.assertIsNotNone(target_features.grad)
        self.assertGreater(float(target_features.grad.abs().sum()), 0.0)

    def test_ms_cbtp_single_source_logs_degeneracy(self) -> None:
        config = _config("ms_cbtp_wjdot")
        config["loss"].update(
            {
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
                "prototype_start_step": 0,
                "prototype_warmup_steps": 1,
            }
        )
        torch.manual_seed(29)
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
        target_y = torch.full((4,), -1, dtype=torch.long)

        output = method.compute_loss([(source_x, source_y)], (target_x, target_y))

        self.assertEqual(output.metrics["ms_cbtp_single_source_degenerate_to_cbtp"], 1.0)

    def test_ms_cbtp_class_alpha_only_uses_uniform_source_loss_weights(self) -> None:
        config = _config("ms_cbtp_wjdot")
        config["loss"].update(
            {
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
                "prototype_start_step": 0,
                "prototype_warmup_steps": 1,
                "class_alpha_only": True,
                "class_alpha_top_m_sources": 1,
            }
        )
        torch.manual_seed(31)
        method = build_method(
            config,
            num_classes=5,
            in_channels=4,
            input_length=32,
            num_sources=2,
        )
        method.train()
        source_batches = [
            (torch.randn(4, 4, 32), torch.tensor([0, 1, 2, 3], dtype=torch.long)),
            (torch.randn(4, 4, 32), torch.tensor([1, 2, 3, 4], dtype=torch.long)),
        ]
        target_x = torch.randn(4, 4, 32)
        target_y = torch.full((4,), -1, dtype=torch.long)

        output = method.compute_loss(source_batches, (target_x, target_y))
        snapshot = method.reliability_snapshot()

        self.assertEqual(output.metrics["ms_weighting_mode_id"], 1.0)
        self.assertAlmostEqual(output.metrics["loss_source_weight_0"], 0.5, places=6)
        self.assertAlmostEqual(output.metrics["loss_source_weight_1"], 0.5, places=6)
        self.assertIn("class_source_weights", snapshot)
        self.assertIn("transport_mass_matrix", snapshot)
        self.assertEqual(tuple(snapshot["class_source_weights"].shape), (2, 5))

    def test_ms_cbtp_unbalanced_records_transport_mass_matrix(self) -> None:
        config = _config("ms_cbtp_wjdot")
        config["loss"].update(
            {
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
                "prototype_start_step": 0,
                "prototype_warmup_steps": 1,
                "ms_weighting_mode": "class_alpha_unbalanced",
                "class_alpha_top_m_sources": 1,
            }
        )
        torch.manual_seed(37)
        method = build_method(
            config,
            num_classes=5,
            in_channels=4,
            input_length=32,
            num_sources=2,
        )
        method.train()
        source_batches = [
            (torch.randn(4, 4, 32), torch.tensor([0, 1, 2, 3], dtype=torch.long)),
            (torch.randn(4, 4, 32), torch.tensor([1, 2, 3, 4], dtype=torch.long)),
        ]
        target_x = torch.randn(4, 4, 32)
        target_y = torch.full((4,), -1, dtype=torch.long)

        output = method.compute_loss(source_batches, (target_x, target_y))
        snapshot = method.reliability_snapshot()

        self.assertEqual(output.metrics["ms_weighting_mode_id"], 2.0)
        self.assertIn("transport_mass_total", output.metrics)
        self.assertIn("transport_mass_matrix", snapshot)
        self.assertEqual(tuple(snapshot["transport_mass_matrix"].shape), (2, 5))

    def test_ms_cbtp_target_label_calibration_does_not_enable_target_ce(self) -> None:
        config = _config("ms_cbtp_wjdot")
        config["loss"].update(
            {
                "alignment_start_step": 0,
                "alignment_ramp_steps": 1,
                "prototype_start_step": 0,
                "prototype_warmup_steps": 1,
                "class_alpha_only": True,
                "target_label_calibration": True,
                "target_label_assist_weight": 0.0,
            }
        )
        torch.manual_seed(41)
        method = build_method(
            config,
            num_classes=5,
            in_channels=4,
            input_length=32,
            num_sources=2,
        )
        method.train()
        source_batches = [
            (torch.randn(4, 4, 32), torch.tensor([0, 1, 2, 3], dtype=torch.long)),
            (torch.randn(4, 4, 32), torch.tensor([1, 2, 3, 4], dtype=torch.long)),
        ]
        target_x = torch.randn(4, 4, 32)
        target_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        output = method.compute_loss(source_batches, (target_x, target_y))

        self.assertEqual(output.metrics["target_label_calibration_used"], 1.0)
        self.assertEqual(output.metrics["lambda_target_label_assist"], 0.0)
        self.assertEqual(output.metrics["loss_target_label_assist"], 0.0)

    def test_ms_cbtp_reliability_export_writes_diagnostic_matrices(self) -> None:
        class FakeModel:
            def reliability_snapshot(self):
                matrix = torch.tensor([[0.7, 0.3], [0.3, 0.7]], dtype=torch.float32)
                return {
                    "source_weights": matrix.mean(dim=1),
                    "class_source_weights": matrix,
                    "transport_mass_matrix": matrix * 0.5,
                    "per_class_ot_cost_matrix": matrix + 1.0,
                    "per_class_proto_distance_matrix": matrix + 2.0,
                }

        with TemporaryDirectory() as temp_dir:
            paths = _export_reliability_tables(
                model=FakeModel(),
                history=[{"epoch": 1}],
                tables_dir=Path(temp_dir),
                source_domain_ids=["mode1", "mode2"],
            )

            for key in [
                "class_alpha_matrix_path",
                "transport_mass_matrix_path",
                "per_class_ot_cost_matrix_path",
                "per_class_proto_distance_matrix_path",
            ]:
                self.assertIsNotNone(paths[key])
                self.assertTrue(Path(paths[key]).exists())


if __name__ == "__main__":
    unittest.main()
