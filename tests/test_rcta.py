from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.te_da_dataset import DomainAdaptationSetting, DomainDataReference, DomainSpec
from src.datasets.te_torch_dataset import DomainSplitTensors, PreparedBenchmarkData
from src.methods import build_method
from src.methods.rcta import RCTAMethod, _ReliabilityGate
from src.trainers.train_benchmark import run_deep_experiment, save_json


def _method_config(*, base_align: str = "cdan", gate_score_floor: float = 0.0) -> dict:
    return {
        "method_name": "rcta",
        "backbone": {
            "name": "fcn",
            "classifier_hidden_dim": 32,
            "dropout": 0.0,
        },
        "optimization": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "loss": {
            "base_align": base_align,
            "use_mcc": True,
            "mcc_weight": 0.05,
            "mcc_temperature": 2.0,
            "teacher_ema_decay": 0.9,
            "teacher_temperature": 1.2,
            "reliability_weights": {
                "cal_conf": 1.0,
                "inv_entropy": 1.0,
                "consistency": 1.0,
                "proto_sim": 1.0,
            },
            "gate_score_floor": gate_score_floor,
            "gate_accept_ratio_start": 1.0,
            "gate_accept_ratio_end": 1.0,
            "gate_curriculum_steps": 1,
            "pseudo_label_weight": 0.2,
            "prototype_weight": 0.1,
            "prototype_separation_weight": 0.1,
            "consistency_weight": 0.1,
            "alignment_start_step": 2,
            "alignment_use_reliable_only": True,
            "semi_reliable_consistency_weight": 0.25,
            "unreliable_entropy_weight": 0.08,
            "prototype_momentum": 0.9,
            "prototype_separation_margin": 0.2,
            "augment": {
                "weak_jitter_std": 0.0,
                "weak_scaling_std": 0.0,
                "strong_jitter_std": 0.0,
                "strong_scaling_std": 0.0,
                "strong_time_mask_ratio": 0.0,
                "strong_channel_dropout_prob": 0.0,
            },
            "cdan": {
                "adaptation_weight": 0.2,
                "adaptation_schedule": "warm_start",
                "adaptation_max_steps": 8,
                "adaptation_schedule_alpha": 10.0,
                "grl_lambda": 1.0,
                "grl_warm_start": True,
                "grl_max_iters": 8,
                "randomized": True,
                "randomized_dim": 32,
                "entropy_conditioning": True,
                "domain_hidden_dim": 64,
                "domain_num_hidden_layers": 1,
            },
            "dann": {
                "adaptation_weight": 0.5,
                "adaptation_schedule": "warm_start",
                "adaptation_max_steps": 8,
                "adaptation_schedule_alpha": 10.0,
                "grl_lambda": 1.0,
                "grl_warm_start": True,
                "grl_max_iters": 8,
                "domain_hidden_dim": 64,
                "domain_num_hidden_layers": 1,
            },
            "deepjdot": {
                "adaptation_weight": 1.0,
                "adaptation_schedule": "constant",
                "adaptation_max_steps": 8,
                "adaptation_schedule_alpha": 10.0,
                "reg_dist": 0.1,
                "reg_cl": 1.0,
                "normalize_feature_cost": False,
                "transport_solver": "emd",
                "sinkhorn_reg": 0.05,
                "sinkhorn_num_iter_max": 100,
            },
        },
    }


def _make_domain_split(domain_id: str, *, offset: float = 0.0) -> DomainSplitTensors:
    generator = torch.Generator().manual_seed(abs(hash((domain_id, offset))) % 2**31)
    train_x = torch.randn(8, 34, 32, generator=generator) + offset
    eval_x = torch.randn(4, 34, 32, generator=generator) + offset
    train_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    eval_y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    return DomainSplitTensors(
        domain_id=domain_id,
        train_x=train_x,
        train_y=train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        train_indices=np.arange(len(train_x)),
        eval_indices=np.arange(len(eval_x)),
    )


def _make_loader(x: torch.Tensor, y: torch.Tensor, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _make_prepared_data() -> PreparedBenchmarkData:
    source_split = _make_domain_split("mode1", offset=0.0)
    target_split = _make_domain_split("mode4", offset=0.5)
    setting = DomainAdaptationSetting(
        setting_name="single_source",
        source_domains=[DomainDataReference(DomainSpec(name="mode1"), storage="raw")],
        target_domain=DomainDataReference(DomainSpec(name="mode4"), storage="raw"),
    )
    batch_size = 4
    return PreparedBenchmarkData(
        setting=setting,
        source_splits=[source_split],
        target_split=target_split,
        source_train_loaders=[_make_loader(source_split.train_x, source_split.train_y, batch_size=batch_size, shuffle=True)],
        source_train_eval_loaders=[
            _make_loader(source_split.train_x, source_split.train_y, batch_size=batch_size, shuffle=False)
        ],
        source_eval_loaders=[_make_loader(source_split.eval_x, source_split.eval_y, batch_size=batch_size, shuffle=False)],
        target_train_loader=_make_loader(target_split.train_x, target_split.train_y, batch_size=batch_size, shuffle=True),
        target_eval_loader=_make_loader(target_split.eval_x, target_split.eval_y, batch_size=batch_size, shuffle=False),
    )


class RCTATests(unittest.TestCase):
    def test_build_method_instantiates_rcta_for_both_aligners(self) -> None:
        for base_align in ("cdan", "dann", "deepjdot"):
            method = build_method(
                _method_config(base_align=base_align),
                num_classes=29,
                in_channels=34,
                input_length=32,
                num_sources=1,
            )
            self.assertIsInstance(method, RCTAMethod)
            self.assertEqual(method.base_align, base_align)
            self.assertAlmostEqual(method.semi_reliable_consistency_weight, 0.25)
            self.assertAlmostEqual(method.unreliable_entropy_weight, 0.08)

    def test_reliability_gate_applies_score_floor_and_classwise_top_ratio(self) -> None:
        gate = _ReliabilityGate(
            score_floor=0.5,
            accept_ratio_start=0.5,
            accept_ratio_end=0.5,
            curriculum_steps=10,
        )
        scores = torch.tensor([0.9, 0.7, 0.6, 0.4, 0.8], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

        selected, ratio = gate.select(scores, labels, step_num=0)

        self.assertEqual(ratio, 0.5)
        self.assertTrue(torch.equal(selected, torch.tensor([True, False, False, False, True])))

    def test_after_optimizer_step_updates_teacher_and_prototypes(self) -> None:
        torch.manual_seed(0)
        method = build_method(
            _method_config(base_align="cdan", gate_score_floor=0.0),
            num_classes=29,
            in_channels=34,
            input_length=32,
            num_sources=1,
        )
        optimizer = torch.optim.SGD(method.parameters(), lr=0.05)
        method.train()

        source_x = torch.randn(4, 34, 32)
        source_y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        target_x = torch.randn(4, 34, 32)
        target_y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        teacher_before = next(method.teacher_classifier.parameters()).detach().clone()
        step_output = method.compute_loss([(source_x, source_y)], (target_x, target_y))
        step_output.loss.backward()
        optimizer.step()
        hook_metrics = method.after_optimizer_step()
        teacher_after = next(method.teacher_classifier.parameters()).detach().clone()

        self.assertFalse(torch.allclose(teacher_before, teacher_after))
        self.assertTrue(all(not parameter.requires_grad for parameter in method.teacher_encoder.parameters()))
        self.assertGreater(int(method.source_prototype_counts.sum().item()), 0)
        self.assertGreater(int(method.target_prototype_counts.sum().item()), 0)
        self.assertIn("source_prototype_active_classes", hook_metrics)
        self.assertIn("target_prototype_active_classes", hook_metrics)

    def test_alignment_is_delayed_until_configured_step(self) -> None:
        torch.manual_seed(4)
        method = build_method(
            _method_config(base_align="cdan", gate_score_floor=0.0),
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
        self.assertEqual(step_output.metrics["lambda_alignment"], 0.0)

        method.step_num.fill_(2)
        step_output_after = method.compute_loss([(source_x, source_y)], (target_x, target_y))
        self.assertGreater(step_output_after.metrics["lambda_alignment"], 0.0)

    def test_rcta_smoke_backward_for_cdan_and_deepjdot(self) -> None:
        torch.manual_seed(1)
        cases = ["cdan", "dann"]
        if importlib.util.find_spec("ot") is not None:
            cases.append("deepjdot")

        for base_align in cases:
            method = build_method(
                _method_config(base_align=base_align, gate_score_floor=0.0),
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
            self.assertTrue(gradients, msg=f"Expected encoder gradients for base_align={base_align}")

    def test_run_deep_experiment_smoke_emits_rcta_metrics(self) -> None:
        torch.manual_seed(2)
        prepared_data = _make_prepared_data()
        method_config = _method_config(base_align="cdan", gate_score_floor=0.0)
        experiment_config = {
            "device": "cpu",
            "runtime": {
                "dry_run": True,
                "save_checkpoint": False,
                "save_analysis": False,
                "show_progress": False,
                "model_selection": "hybrid_source_eval_inverse_entropy",
                "selection_weights": {"source_eval": 0.7, "target_entropy": 0.3},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_paths = {
                "analysis_path": temp_root / "artifacts" / "analysis.npz",
                "checkpoints_dir": temp_root / "checkpoints",
            }
            result = run_deep_experiment(
                method_config=method_config,
                experiment_config=experiment_config,
                prepared_data=prepared_data,
                run_paths=run_paths,
                scenario_id="mode1_to_mode4",
            )

            self.assertTrue(result["history"])
            summary = result["history"][0]
            self.assertIn("gate_accept_ratio", summary)
            self.assertIn("gate_mean_score", summary)
            self.assertIn("pseudo_label_kept", summary)
            self.assertIn("teacher_ema_decay", summary)
            self.assertIn("source_prototype_active_classes", summary)
            self.assertIn("target_prototype_active_classes", summary)
            self.assertIn("target_eval_balanced_acc", result)
            self.assertIn("target_confusion_matrix", result)

            metrics_path = temp_root / "tables" / "result.json"
            save_json(
                metrics_path,
                {
                    "method_name": "rcta",
                    "scenario_id": "mode1_to_mode4",
                    "result": result,
                },
            )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["method_name"], "rcta")
            self.assertIn("history", payload["result"])


if __name__ == "__main__":
    unittest.main()
