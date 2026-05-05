from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.ccsr_wjdot_fusion import export_ccsr_wjdot_fusion_artifacts
from src.methods import build_method


def _loader(x_values, y_values) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.as_tensor(x_values, dtype=torch.float32),
            torch.as_tensor(y_values, dtype=torch.long),
        ),
        batch_size=4,
        shuffle=False,
    )


class TinyFeatureModel(nn.Module):
    def forward(self, x: torch.Tensor):
        features = x.float()
        logits = 4.0 * features[:, :3]
        return logits, features

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits

    def source_expert_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        expert_one = torch.softmax(logits, dim=1)
        expert_two = torch.softmax(logits + torch.tensor([0.0, 0.2, -0.1], device=x.device), dim=1)
        return torch.stack([expert_one, expert_two], dim=0)


class CCSRWDOTFusionTests(unittest.TestCase):
    def test_method_registry_builds_ccsr_wjdot_fusion(self) -> None:
        method = build_method(
            {
                "method_name": "ccsr_wjdot_fusion",
                "backbone": {
                    "name": "fcn",
                    "classifier_hidden_dim": 8,
                    "dropout": 0.0,
                    "embedding_dim": 8,
                },
                "loss": {
                    "transport_solver": "sinkhorn",
                    "sinkhorn_num_iter_max": 5,
                },
            },
            num_classes=3,
            in_channels=2,
            input_length=16,
            num_sources=2,
        )

        self.assertEqual(method.method_name, "ccsr_wjdot_fusion")

    def test_fusion_exports_alpha_and_final_metrics(self) -> None:
        source1_train = _loader(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.1],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.1, 0.0, 0.9, 0.0],
            ],
            [0, 0, 1, 1, 2, 2],
        )
        source2_train = _loader(
            [
                [0.8, 0.2, 0.0, 0.1],
                [0.7, 0.2, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.1],
                [0.0, 0.7, 0.2, 0.0],
                [0.2, 0.0, 0.8, 0.1],
                [0.1, 0.0, 0.7, 0.0],
            ],
            [0, 0, 1, 1, 2, 2],
        )
        target_train = _loader(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            [-1, -1, -1],
        )
        target_eval = _loader(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            [0, 1, 2],
        )
        prepared_data = SimpleNamespace(
            source_splits=[SimpleNamespace(domain_id="mode1"), SimpleNamespace(domain_id="mode2")],
            target_split=SimpleNamespace(domain_id="mode5"),
            source_train_eval_loaders=[source1_train, source2_train],
            source_eval_loaders=[source1_train, source2_train],
            target_train_loader=target_train,
            target_eval_loader=target_eval,
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = export_ccsr_wjdot_fusion_artifacts(
                model=TinyFeatureModel(),
                prepared_data=prepared_data,
                device=torch.device("cpu"),
                analysis_path=root / "artifacts" / "ccsr_analysis.npz",
                tables_dir=root / "tables",
                figures_dir=root / "figures",
                scenario_id="mode1-mode2_to_mode5",
                method_name="ccsr_wjdot_fusion",
                ccsr_config={"num_classes": 3, "tau_proto": 0.0},
                max_batches=None,
                non_blocking=False,
                amp_enabled=False,
            )

            self.assertEqual(len(summary["target_confusion_matrix"]), 3)
            self.assertTrue(Path(summary["analysis_path"]).exists())
            self.assertTrue(Path(summary["reliability_components_path"]).exists())
            self.assertTrue(Path(summary["class_source_alpha_path"]).exists())
            self.assertIn("ccsr_wjdot_fusion", summary)
            self.assertGreaterEqual(summary["target_eval_acc"], 0.0)
            self.assertLessEqual(summary["target_eval_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
