from __future__ import annotations

import unittest
from pathlib import Path

from src.automation.run_small_scale_round import _load_yaml, build_run_plan


ROOT = Path(__file__).resolve().parents[1]


class AutomationPlanTests(unittest.TestCase):
    def test_quick_debug_plan_expands_to_12_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/quick_debug.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(len(plan["methods"]), 6)
        self.assertEqual(len(plan["scene_settings"]), 2)
        self.assertEqual(len(plan["runs"]), 12)

    def test_benchmark_72_plan_expands_to_72_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_72.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(len(plan["methods"]), 6)
        self.assertEqual(len(plan["scene_settings"]), 12)
        self.assertEqual(len(plan["runs"]), 72)

        single_source = [item for item in plan["scene_settings"] if item["setting"] == "single_source"]
        multi_source = [item for item in plan["scene_settings"] if item["setting"] == "multi_source"]
        self.assertEqual(len(single_source), 6)
        self.assertEqual(len(multi_source), 6)

    def test_benchmark_56_plan_expands_to_56_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_56_8scenes_7methods_rcta_best.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(len(plan["methods"]), 7)
        self.assertEqual(len(plan["scene_settings"]), 8)
        self.assertEqual(len(plan["runs"]), 56)

        single_source = [item for item in plan["scene_settings"] if item["setting"] == "single_source"]
        multi_source = [item for item in plan["scene_settings"] if item["setting"] == "multi_source"]
        self.assertEqual(len(single_source), 4)
        self.assertEqual(len(multi_source), 4)
        self.assertEqual(plan["scene_settings"][4]["source_domains"], ["mode2", "mode3", "mode5"])
        self.assertEqual(plan["scene_settings"][5]["source_domains"], ["mode2", "mode3", "mode6"])
        self.assertEqual(plan["scene_settings"][6]["source_domains"], ["mode2", "mode5", "mode6"])
        self.assertEqual(plan["scene_settings"][7]["source_domains"], ["mode3", "mode5", "mode6"])

    def test_benchmark_88_random_fold_plan_reuses_one_fold_pair_per_scene(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_88_11scenes_8methods_randomfold.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(len(plan["methods"]), 8)
        self.assertEqual(len(plan["scene_settings"]), 11)
        self.assertEqual(len(plan["runs"]), 88)

        fold_pairs_by_scene: dict[str, set[tuple[str, str]]] = {}
        for run in plan["runs"]:
            fold_pairs_by_scene.setdefault(str(run["label"]), set()).add(
                (str(run["source_fold"]), str(run["target_fold"]))
            )

        self.assertEqual(set(fold_pairs_by_scene.keys()), {str(item["label"]) for item in plan["scene_settings"]})
        self.assertTrue(all(len(fold_pairs) == 1 for fold_pairs in fold_pairs_by_scene.values()))

    def test_random_fold_plan_is_reproducible_for_same_seed(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_88_11scenes_8methods_goal_tune.yaml")
        first_plan = build_run_plan(payload)
        second_plan = build_run_plan(payload)

        first_pairs = [(run["label"], run["source_fold"], run["target_fold"]) for run in first_plan["runs"]]
        second_pairs = [(run["label"], run["source_fold"], run["target_fold"]) for run in second_plan["runs"]]

        self.assertEqual(first_pairs, second_pairs)

    def test_random_fold_plan_accepts_auto_seed(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_88_11scenes_8methods_goal_tune.yaml")
        payload["seed"] = None
        plan = build_run_plan(payload)

        self.assertEqual(plan["seed_mode"], "auto")
        self.assertIsInstance(plan["seed"], int)
        self.assertEqual(len(plan["runs"]), 88)


if __name__ == "__main__":
    unittest.main()
