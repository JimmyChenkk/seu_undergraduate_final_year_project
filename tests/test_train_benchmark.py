from __future__ import annotations

import unittest

from src.evaluation.review import build_run_review
from src.trainers.train_benchmark import _resolve_metric_score, apply_method_overrides


class TrainBenchmarkTests(unittest.TestCase):
    def test_apply_method_overrides_merges_runtime_and_method_sections(self) -> None:
        experiment_payload = {
            "runtime": {"show_progress": True},
            "tracking": {"batch_root_name": "base"},
            "method_overrides": {
                "*": {
                    "runtime": {"show_progress": False},
                    "optimization": {"epochs": 60},
                },
                "dann": {
                    "runtime": {"pin_memory": True},
                    "optimization": {"learning_rate": 2e-4},
                    "loss": {"adaptation_weight": 0.8},
                },
            },
        }
        method_payload = {
            "method_name": "dann",
            "optimization": {"epochs": 70, "batch_size": 32, "learning_rate": 1e-4},
            "loss": {"adaptation_weight": 0.5},
        }

        merged_experiment, merged_method = apply_method_overrides(experiment_payload, method_payload)

        self.assertFalse(merged_experiment["runtime"]["show_progress"])
        self.assertTrue(merged_experiment["runtime"]["pin_memory"])
        self.assertEqual(merged_method["optimization"]["epochs"], 60)
        self.assertEqual(merged_method["optimization"]["batch_size"], 32)
        self.assertEqual(merged_method["optimization"]["learning_rate"], 2e-4)
        self.assertEqual(merged_method["loss"]["adaptation_weight"], 0.8)

    def test_hybrid_selection_can_prefer_lower_entropy_epoch(self) -> None:
        weights = {"source_eval": 0.7, "target_entropy": 0.3}
        early_epoch = {
            "acc_source_eval": 0.80,
            "target_train_mean_entropy": 0.50,
        }
        later_epoch = {
            "acc_source_eval": 0.79,
            "target_train_mean_entropy": 0.10,
        }

        early_score = _resolve_metric_score(
            early_epoch,
            "hybrid_source_eval_inverse_entropy",
            weights=weights,
        )
        later_score = _resolve_metric_score(
            later_epoch,
            "hybrid_source_eval_inverse_entropy",
            weights=weights,
        )

        self.assertIsNotNone(early_score)
        self.assertIsNotNone(later_score)
        self.assertGreater(later_score, early_score)

    def test_review_payload_keeps_selection_metadata(self) -> None:
        result_payload = {
            "method_name": "dann",
            "scenario_id": "mode1_to_mode4",
            "setting": "single_source",
            "run_root": "runs/example",
            "result": {
                "selected_epoch": 15,
                "model_selection": "hybrid_source_eval_inverse_entropy",
                "model_selection_weights": {"source_eval": 0.7, "target_entropy": 0.3},
                "selected_model_selection_score": 0.42,
                "selected_target_train_mean_confidence": 0.9,
                "selected_target_train_mean_entropy": 0.2,
                "early_stopped": True,
                "early_stopping_metric": "hybrid_source_eval_inverse_entropy",
                "early_stopping_weights": {"source_eval": 0.7, "target_entropy": 0.3},
                "early_stopping_best_score": 0.42,
                "selected_source_train_acc": 0.8,
                "selected_source_eval_acc": 0.75,
                "selected_target_eval_acc": 0.6,
            },
        }

        review = build_run_review(
            result_payload,
            figure_paths={
                "tsne_domain": "runs/example/figures/tsne_domain.png",
                "tsne_class": "runs/example/figures/tsne_class.png",
                "confusion_matrix": "runs/example/figures/confusion_matrix.png",
            },
        )

        self.assertEqual(review["selection"]["selected_epoch"], 15)
        self.assertEqual(
            review["selection"]["model_selection"],
            "hybrid_source_eval_inverse_entropy",
        )
        self.assertEqual(review["selection"]["selected_target_train_mean_entropy"], 0.2)


if __name__ == "__main__":
    unittest.main()
