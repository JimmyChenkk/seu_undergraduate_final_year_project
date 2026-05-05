from __future__ import annotations

import unittest
from pathlib import Path

from src.automation.run_small_scale_round import (
    _load_yaml,
    _result_matches_run,
    build_run_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class AutomationPlanTests(unittest.TestCase):
    def test_single_source_mainline_plan_expands_to_48_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/tep_ot_single_source_8methods_stage1_fold0.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(
            plan["methods"],
            [
                "source_only",
                "dsan",
                "cdan_ts",
                "codats",
                "deepjdot",
                "tpu_deepjdot",
                "cbtpu_deepjdot",
                "target_only",
            ],
        )
        self.assertEqual(len(plan["scene_settings"]), 6)
        self.assertEqual(len(plan["runs"]), 48)
        self.assertTrue(all(item["setting"] == "single_source" for item in plan["scene_settings"]))

    def test_multisource_mainline_plan_expands_to_45_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/tep_ot_multisource_ca_ccsr_wjdot_stage1_probe_fold0.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(
            plan["methods"],
            [
                "source_only",
                "codats",
                "wjdot",
                "ca_ccsr_wjdot_prior20",
                "target_ref",
            ],
        )
        self.assertEqual(len(plan["scene_settings"]), 9)
        self.assertEqual(len(plan["runs"]), 45)
        self.assertTrue(all(item["setting"] == "multi_source" for item in plan["scene_settings"]))

    def test_single_source_mainline_keeps_fold0_policy(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/tep_ot_single_source_8methods_stage1_fold0.yaml")
        plan = build_run_plan(payload)

        self.assertFalse(plan["fold_policy"]["random_fold_enabled"])
        self.assertEqual(
            [run["source_fold"] for run in plan["runs"][:: len(plan["methods"])]],
            ["Fold 1"] * 6,
        )
        self.assertEqual(
            [run["target_fold"] for run in plan["runs"][:: len(plan["methods"])]],
            ["Fold 1"] * 6,
        )

    def test_single_source_tpu_diagnostic_plan_expands_to_18_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/tep_ot_single_source_tpu_stage1_fold0.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(
            plan["methods"],
            [
                "deepjdot",
                "tpu_deepjdot",
                "cbtpu_deepjdot",
            ],
        )
        self.assertEqual(len(plan["scene_settings"]), 6)
        self.assertEqual(len(plan["runs"]), 18)

    def _random_fold_payload(self) -> dict:
        return {
            "seed": 42,
            "automation": {
                "methods": ["source_only", "cdan"],
                "single_source_scenes": ["mode1->mode2", "mode2->mode1", "mode1->mode5"],
            },
            "protocol_override": {
                "preferred_fold": "Fold 1",
                "source_folds": [1, 2, 3, 4, 5],
                "target_folds": [1, 2, 3, 4, 5],
                "fold_sampling": {
                    "enabled": True,
                    "strategy": "random_per_scene",
                    "random_per_scene": True,
                    "random_per_run": False,
                },
            },
        }

    def test_random_fold_plan_reuses_one_fold_pair_per_scene(self) -> None:
        plan = build_run_plan(self._random_fold_payload())

        self.assertEqual(len(plan["methods"]), 2)
        self.assertEqual(len(plan["scene_settings"]), 3)
        self.assertEqual(len(plan["runs"]), 6)

        fold_pairs_by_scene: dict[str, set[tuple[str, str]]] = {}
        for run in plan["runs"]:
            fold_pairs_by_scene.setdefault(str(run["label"]), set()).add(
                (str(run["source_fold"]), str(run["target_fold"]))
            )

        self.assertEqual(set(fold_pairs_by_scene.keys()), {str(item["label"]) for item in plan["scene_settings"]})
        self.assertTrue(all(len(fold_pairs) == 1 for fold_pairs in fold_pairs_by_scene.values()))

    def test_random_fold_plan_is_reproducible_for_same_seed(self) -> None:
        payload = self._random_fold_payload()
        first_plan = build_run_plan(payload)
        second_plan = build_run_plan(payload)

        first_pairs = [(run["label"], run["source_fold"], run["target_fold"]) for run in first_plan["runs"]]
        second_pairs = [(run["label"], run["source_fold"], run["target_fold"]) for run in second_plan["runs"]]

        self.assertEqual(first_pairs, second_pairs)

    def test_random_fold_plan_accepts_auto_seed(self) -> None:
        payload = self._random_fold_payload()
        payload["seed"] = None
        plan = build_run_plan(payload)

        self.assertEqual(plan["seed_mode"], "auto")
        self.assertIsInstance(plan["seed"], int)
        self.assertEqual(len(plan["runs"]), 6)

    def test_cli_scene_accepts_hyphen_separator_to_avoid_shell_redirection(self) -> None:
        payload = {
            "seed": 42,
            "automation": {"methods": ["source_only"]},
            "protocol_override": {
                "preferred_fold": "Fold 1",
                "random_fold_enabled": False,
            },
        }

        plan = build_run_plan(payload, cli_scenes=["mode1-mode5"])

        self.assertEqual(len(plan["scene_settings"]), 1)
        self.assertEqual(plan["runs"][0]["source_domains"], ["mode1"])
        self.assertEqual(plan["runs"][0]["target_domain"], "mode5")

    def test_cli_scene_accepts_multisource_plus_separator(self) -> None:
        payload = {
            "seed": 42,
            "automation": {"methods": ["wjdot"]},
            "protocol_override": {
                "preferred_fold": "Fold 1",
                "random_fold_enabled": False,
            },
        }

        plan = build_run_plan(payload, cli_scenes=["mode1+mode2->mode5"])

        self.assertEqual(len(plan["scene_settings"]), 1)
        self.assertEqual(plan["scene_settings"][0]["setting"], "multi_source")
        self.assertEqual(plan["runs"][0]["source_domains"], ["mode1", "mode2"])
        self.assertEqual(plan["runs"][0]["target_domain"], "mode5")

    def test_result_matcher_finds_same_scene_base_wjdot_run(self) -> None:
        payload = {
            "seed": 42,
            "automation": {"methods": ["wjdot"]},
            "protocol_override": {
                "preferred_fold": "Fold 1",
                "random_fold_enabled": False,
            },
        }
        run = build_run_plan(payload, cli_scenes=["mode1+mode2->mode5"])["runs"][0]
        result_payload = {
            "method_name": "wjdot",
            "method_base_name": "wjdot",
            "scenario_id": "mode1-mode2_to_mode5",
            "source_domains": ["mode1", "mode2"],
            "target_domain": "mode5",
            "source_fold": "Fold 1",
            "target_fold": "Fold 1",
        }

        self.assertTrue(_result_matches_run(result_payload, run, "wjdot"))
        self.assertFalse(_result_matches_run(result_payload, run, "ca_ccsr_wjdot"))

    def test_method_overrides_resolve_for_method_and_scene_maps(self) -> None:
        payload = {
            "seed": 42,
            "automation": {
                "methods": ["deepjdot", "cbtpu_deepjdot"],
                "single_source_scenes": ["mode1->mode2"],
            },
            "method_overrides": {
                "deepjdot": {
                    "loss": {
                        "adaptation_weight": 0.55,
                    },
                },
                "all": {
                    "cbtpu_deepjdot": {
                        "loss": {
                            "pseudo_weight": 0.03,
                        },
                    },
                },
                "m1_m2": {
                    "cbtpu_deepjdot": {
                        "loss": {
                            "pseudo_start_step": 2200,
                        },
                    },
                },
            },
            "protocol_override": {
                "random_fold_enabled": False,
                "preferred_fold": "Fold 1",
            },
        }

        plan = build_run_plan(payload)
        runs_by_method = {run["method_name"]: run for run in plan["runs"]}

        self.assertEqual(
            runs_by_method["deepjdot"]["method_overrides"]["loss"]["adaptation_weight"],
            0.55,
        )
        self.assertEqual(
            runs_by_method["cbtpu_deepjdot"]["method_overrides"]["loss"]["pseudo_weight"],
            0.03,
        )
        self.assertEqual(
            runs_by_method["cbtpu_deepjdot"]["method_overrides"]["loss"]["pseudo_start_step"],
            2200,
        )


if __name__ == "__main__":
    unittest.main()
