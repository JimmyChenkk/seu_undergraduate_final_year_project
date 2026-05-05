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
    def test_rescue_smoke_plan_expands_to_6_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/rescue_smoke.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(len(plan["methods"]), 6)
        self.assertEqual(len(plan["scene_settings"]), 1)
        self.assertEqual(len(plan["runs"]), 6)

    def test_main_90_benchmark_plan_expands_to_90_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_90_mode125_9scenes_10methods_fixedfold.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(
            plan["methods"],
            [
                "source_only",
                "target_only",
                "coral",
                "dan",
                "dann",
                "cdan",
                "deepjdot",
                "tc_cdan",
                "rpl_tc_cdan",
                "ccs_rpl_tc_cdan",
            ],
        )
        self.assertEqual(len(plan["scene_settings"]), 9)
        self.assertEqual(len(plan["runs"]), 90)

        single_source = [item for item in plan["scene_settings"] if item["setting"] == "single_source"]
        multi_source = [item for item in plan["scene_settings"] if item["setting"] == "multi_source"]
        self.assertEqual(len(single_source), 6)
        self.assertEqual(len(multi_source), 3)

    def test_mode125_baseline_plan_expands_to_63_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_63_mode125_9scenes_7baselines_fixedfold.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(
            plan["methods"],
            [
                "source_only",
                "codats",
                "cdan_ts",
                "dsan",
                "deepjdot",
                "raincoat",
                "target_only",
            ],
        )
        self.assertEqual(len(plan["scene_settings"]), 9)
        self.assertEqual(len(plan["runs"]), 63)

    def test_single_source_deepjdot_plan_expands_to_48_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/tep_ot_single_source_deepjdot_125_fixedfold.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(
            plan["methods"],
            [
                "source_only",
                "deepjdot",
                "tp_deepjdot",
                "cbtp_deepjdot",
                "cdan_ts",
                "codats",
                "dsan",
                "target_only",
            ],
        )
        self.assertEqual(len(plan["scene_settings"]), 6)
        self.assertEqual(len(plan["runs"]), 48)

    def test_rescue_9scenes_plan_expands_to_90_runs(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/rescue_9scenes_10methods.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(len(plan["methods"]), 10)
        self.assertEqual(len(plan["scene_settings"]), 9)
        self.assertEqual(len(plan["runs"]), 90)

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

    def test_mode125_fixed_fold_plan_uses_configured_scene_folds(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/benchmark_90_mode125_9scenes_10methods_fixedfold.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(len(plan["methods"]), 10)
        self.assertEqual(len(plan["scene_settings"]), 9)
        self.assertEqual(len(plan["runs"]), 90)
        self.assertFalse(plan["fold_policy"]["random_fold_enabled"])

        expected = {
            "mode1_to_mode2": ("Fold 2", {"mode1": "Fold 2"}, "Fold 4"),
            "mode2_to_mode1": ("Fold 5", {"mode2": "Fold 5"}, "Fold 3"),
            "mode1_to_mode5": ("Fold 4", {"mode1": "Fold 4"}, "Fold 3"),
            "mode5_to_mode1": ("Fold 4", {"mode5": "Fold 4"}, "Fold 4"),
            "mode2_to_mode5": ("Fold 1", {"mode2": "Fold 1"}, "Fold 3"),
            "mode5_to_mode2": ("Fold 3", {"mode5": "Fold 3"}, "Fold 4"),
            "mode1-mode2_to_mode5": (
                "Fold 4+Fold 1",
                {"mode1": "Fold 4", "mode2": "Fold 1"},
                "Fold 3",
            ),
            "mode1-mode5_to_mode2": (
                "Fold 2+Fold 3",
                {"mode1": "Fold 2", "mode5": "Fold 3"},
                "Fold 4",
            ),
            "mode2-mode5_to_mode1": (
                "Fold 4",
                {"mode2": "Fold 4", "mode5": "Fold 4"},
                "Fold 4",
            ),
        }

        first_run_by_scene = {str(run["label"]): run for run in plan["runs"][:: len(plan["methods"])]}
        self.assertEqual(set(first_run_by_scene), set(expected))
        for label, (source_fold, source_folds_by_domain, target_fold) in expected.items():
            run = first_run_by_scene[label]
            self.assertEqual(run["source_fold"], source_fold)
            self.assertEqual(run["source_folds_by_domain"], source_folds_by_domain)
            self.assertEqual(run["target_fold"], target_fold)

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

    def test_ccsr_wjdot_stage_plan_keeps_wjdot_before_posthoc_fusion(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/tep_ot_multisource_ccsr_wjdot_stage1_fold0.yaml")
        plan = build_run_plan(
            payload,
            cli_methods=["wjdot", "ccsr_wjdot_fusion"],
            cli_scenes=["mode1+mode2->mode5"],
        )

        self.assertEqual([run["method_name"] for run in plan["runs"]], ["wjdot", "ccsr_wjdot_fusion"])

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
        self.assertFalse(_result_matches_run(result_payload, run, "ccsr_wjdot_fusion"))

    def test_rcta_mode125_ablation_uses_three_cumulative_stages(self) -> None:
        payload = _load_yaml(ROOT / "configs/experiment/rcta_mode125_ablation_fixedfold.yaml")
        plan = build_run_plan(payload)

        self.assertEqual(plan["methods"], ["rcta_a_teacher_temporal", "rcta_ab_gate", "rcta_abc_proto_multi"])
        self.assertEqual(len(plan["scene_settings"]), 9)
        self.assertEqual(len(plan["runs"]), 27)

    def test_method_overrides_resolve_for_method_and_scene_maps(self) -> None:
        payload = {
            "seed": 42,
            "automation": {
                "methods": ["deepjdot", "rpl_tc_cdan"],
                "single_source_scenes": ["mode1->mode2"],
            },
            "method_overrides": {
                "deepjdot": {
                    "loss": {
                        "adaptation_weight": 0.55,
                    },
                },
                "all": {
                    "rpl_tc_cdan": {
                        "loss": {
                            "pseudo_weight": 0.03,
                        },
                    },
                },
                "m1_m2": {
                    "rpl_tc_cdan": {
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
            runs_by_method["rpl_tc_cdan"]["method_overrides"]["loss"]["pseudo_weight"],
            0.03,
        )
        self.assertEqual(
            runs_by_method["rpl_tc_cdan"]["method_overrides"]["loss"]["pseudo_start_step"],
            2200,
        )


if __name__ == "__main__":
    unittest.main()
