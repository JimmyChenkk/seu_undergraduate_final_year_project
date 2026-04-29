"""Run config-driven quick-debug or benchmark batch plans."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Iterable

from src.utils.fold_policy import canonicalize_fold_choice
from src.utils.random_seed import resolve_seed
from src.utils.run_layout import build_timestamp


ALL_MODES = [
    "mode1",
    "mode2",
    "mode3",
    "mode4",
    "mode5",
    "mode6",
]


class AutomationDependencyError(RuntimeError):
    """Raised when the batch automation entrypoint is missing YAML support."""


def _import_yaml():
    if importlib.util.find_spec("yaml") is None:
        raise AutomationDependencyError(
            "Missing automation dependency: yaml. Install it manually in tep_env, for example with "
            + "`pip install -r requirements-benchmark.txt`."
        )

    import yaml

    return yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    yaml = _import_yaml()
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _validate_domain(domain_name: str) -> str:
    normalized = str(domain_name).strip()
    if normalized not in ALL_MODES:
        raise ValueError(f"Unsupported domain in automation plan: {domain_name}")
    return normalized


def _parse_scene_tokens(scene_tokens: Iterable[str]) -> list[tuple[str, str]]:
    parsed = []
    for token in scene_tokens:
        normalized = str(token).replace(":", "->")
        if "->" not in normalized:
            raise ValueError(f"Invalid scene token: {token}")
        source_domain, target_domain = [item.strip() for item in normalized.split("->", maxsplit=1)]
        source_domain = _validate_domain(source_domain)
        target_domain = _validate_domain(target_domain)
        if source_domain == target_domain:
            raise ValueError(f"Automation scene must use different source and target domains: {token}")
        parsed.append((source_domain, target_domain))
    return parsed


def _build_single_source_settings(scene_tokens: Iterable[str]) -> list[dict[str, object]]:
    return [
        {
            "setting": "single_source",
            "source_domains": [source_domain],
            "target_domain": target_domain,
            "label": f"{source_domain}_to_{target_domain}",
        }
        for source_domain, target_domain in _parse_scene_tokens(scene_tokens)
    ]


def _build_all_directed_scenes() -> list[tuple[str, str]]:
    return [
        (source_domain, target_domain)
        for source_domain in ALL_MODES
        for target_domain in ALL_MODES
        if source_domain != target_domain
    ]


def _build_multisource_settings(scene_tokens: Iterable[str]) -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []
    for token in scene_tokens:
        normalized = str(token).replace(":", "->")
        if "->" not in normalized:
            raise ValueError(f"Invalid multi-source scene token: {token}")
        source_token, target_domain = [item.strip() for item in normalized.split("->", maxsplit=1)]
        source_domains = [item.strip() for item in source_token.replace("+", "-").split("-") if item.strip()]
        for domain_name in source_domains:
            _validate_domain(domain_name)
        target_domain = _validate_domain(target_domain)
        if len(source_domains) < 2:
            raise ValueError(f"Multi-source scene must include at least two source domains: {token}")
        if target_domain in source_domains:
            raise ValueError(f"Multi-source scene target must differ from sources: {token}")
        settings.append(
            {
                "setting": "multi_source",
                "source_domains": source_domains,
                "target_domain": target_domain,
                "label": f"{'-'.join(source_domains)}_to_{target_domain}",
            }
        )
    return settings


def _mode_compact(domain_name: str) -> str:
    return str(domain_name).replace("mode", "m")


def _mode_number(domain_name: str) -> str:
    return str(domain_name).replace("mode", "")


def _scene_override_keys(scene: dict[str, object]) -> list[str]:
    source_domains = [str(item) for item in scene["source_domains"]]
    target_domain = str(scene["target_domain"])
    label = str(scene["label"])
    compact_full = "_".join([*[_mode_compact(item) for item in source_domains], _mode_compact(target_domain)])
    compact_cluster = "_".join(
        [_mode_compact(source_domains[0]), *[_mode_number(item) for item in source_domains[1:]], _mode_compact(target_domain)]
    )
    return [
        label,
        label.replace("-", "_"),
        f"{'+'.join(source_domains)}->{target_domain}",
        f"{'-'.join(source_domains)}->{target_domain}",
        f"{'_'.join(source_domains)}_to_{target_domain}",
        compact_full,
        compact_cluster,
    ]


def _scene_fold_override(
    scene: dict[str, object],
    raw_overrides: Any,
) -> dict[str, Any]:
    keys = set(_scene_override_keys(scene))
    if isinstance(raw_overrides, dict):
        for key in keys:
            value = raw_overrides.get(key)
            if isinstance(value, dict):
                return value
        return {}
    if isinstance(raw_overrides, list):
        for item in raw_overrides:
            if not isinstance(item, dict):
                continue
            scene_key = item.get("scene") or item.get("label")
            if scene_key is not None and str(scene_key) in keys:
                return item
    return {}


def _source_fold_mapping(
    scene: dict[str, object],
    override: dict[str, Any],
    default_source_fold: str,
) -> dict[str, str]:
    source_domains = [str(item) for item in scene["source_domains"]]
    raw_by_domain = override.get("source_folds_by_domain")
    raw_source_folds = override.get("source_folds")
    raw_source_fold = override.get("source_fold")

    if isinstance(raw_by_domain, dict):
        return {
            domain_name: canonicalize_fold_choice(raw_by_domain.get(domain_name, default_source_fold))
            for domain_name in source_domains
        }
    if isinstance(raw_source_folds, dict):
        return {
            domain_name: canonicalize_fold_choice(raw_source_folds.get(domain_name, default_source_fold))
            for domain_name in source_domains
        }
    if isinstance(raw_source_folds, list):
        return {
            domain_name: canonicalize_fold_choice(raw_source_folds[index])
            if index < len(raw_source_folds)
            else canonicalize_fold_choice(default_source_fold)
            for index, domain_name in enumerate(source_domains)
        }
    if isinstance(raw_source_fold, list):
        return {
            domain_name: canonicalize_fold_choice(raw_source_fold[index])
            if index < len(raw_source_fold)
            else canonicalize_fold_choice(default_source_fold)
            for index, domain_name in enumerate(source_domains)
        }

    source_fold = canonicalize_fold_choice(raw_source_fold if raw_source_fold is not None else default_source_fold)
    return {domain_name: source_fold for domain_name in source_domains}


def _source_fold_display(source_domains: list[str], source_folds_by_domain: dict[str, str]) -> str:
    ordered_folds = [source_folds_by_domain[domain_name] for domain_name in source_domains]
    if len(set(ordered_folds)) == 1:
        return ordered_folds[0]
    return "+".join(ordered_folds)


def _source_pool_from_targets(experiment_payload: dict[str, Any], target_tokens: Iterable[str]) -> list[str]:
    automation = _automation_section(experiment_payload)
    configured_pool = automation.get("multisource_source_pool")
    if isinstance(configured_pool, list) and configured_pool:
        return [_validate_domain(str(item)) for item in configured_pool]

    target_domains = [_validate_domain(str(item)) for item in target_tokens]
    if set(target_domains) == set(ALL_MODES):
        return list(ALL_MODES)

    pool: list[str] = []
    for source_domain, target_domain in _parse_scene_tokens(automation.get("single_source_scenes", [])):
        for domain_name in (source_domain, target_domain):
            if domain_name not in pool:
                pool.append(domain_name)
    for domain_name in target_domains:
        if domain_name not in pool:
            pool.append(domain_name)
    return [domain_name for domain_name in ALL_MODES if domain_name in pool]


def _build_multisource_target_settings(
    target_tokens: Iterable[str],
    *,
    source_pool: list[str],
) -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []
    for token in target_tokens:
        target_domain = _validate_domain(str(token))
        source_domains = [domain_name for domain_name in source_pool if domain_name != target_domain]
        if len(source_domains) < 2:
            raise ValueError(
                "Multi-source targets need at least two source domains after leaving out "
                f"{target_domain!r}; source_pool={source_pool}"
            )
        settings.append(
            {
                "setting": "multi_source",
                "source_domains": source_domains,
                "target_domain": target_domain,
                "label": f"{'-'.join(source_domains)}_to_{target_domain}",
            }
        )
    return settings


def _automation_section(experiment_payload: dict[str, Any]) -> dict[str, Any]:
    automation = experiment_payload.get("automation", {})
    if not isinstance(automation, dict):
        return {}
    return automation


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _scene_override_payload(
    experiment_payload: dict[str, Any],
    scene: dict[str, object],
    method_name: str,
) -> dict[str, Any]:
    overrides = experiment_payload.get("method_overrides", {})
    if not isinstance(overrides, dict):
        return {}

    scene_keys = _scene_override_keys(scene)
    method_keys = {
        "*",
        "all",
        method_name,
        str(method_name).lower(),
    }
    merged: dict[str, Any] = {}

    for scene_key in ["*", "all", *scene_keys]:
        scene_payload = overrides.get(scene_key)
        if not isinstance(scene_payload, dict):
            continue
        for method_key in ["*", "all", *method_keys]:
            method_payload = scene_payload.get(method_key)
            if not isinstance(method_payload, dict):
                continue
            merged = _deep_merge_dict(merged, method_payload)

    return merged


def _discover_method_names() -> list[str]:
    return sorted(path.stem for path in Path("configs/method").glob("*.yaml"))


def resolve_method_names(
    experiment_payload: dict[str, Any],
    cli_methods: list[str] | None,
) -> list[str]:
    if cli_methods is not None:
        methods = [str(item).strip() for item in cli_methods if str(item).strip()]
    else:
        configured_methods = _automation_section(experiment_payload).get("methods", [])
        if isinstance(configured_methods, list):
            methods = [str(item).strip() for item in configured_methods if str(item).strip()]
        else:
            methods = []

    if not methods:
        methods = _discover_method_names()
    if not methods:
        raise ValueError("No methods resolved for automation.")
    return methods


def resolve_scene_settings(
    experiment_payload: dict[str, Any],
    cli_scenes: list[str] | None,
    *,
    all_scenes: bool,
    include_multisource_targets: bool | None,
) -> list[dict[str, object]]:
    automation = _automation_section(experiment_payload)
    explicit_multisource_scenes = (
        isinstance(automation.get("multisource_scenes"), list)
        and bool(automation.get("multisource_scenes"))
    )
    include_configured_multisource_targets = (
        bool(automation.get("include_multisource_targets", not explicit_multisource_scenes))
        if include_multisource_targets is None
        else bool(include_multisource_targets)
    )

    scene_settings = []

    if all_scenes:
        return _build_single_source_settings([f"{source}->{target}" for source, target in _build_all_directed_scenes()])

    if cli_scenes is not None:
        return _build_single_source_settings(cli_scenes)

    single_source_scenes = automation.get("single_source_scenes", [])
    multisource_scenes = automation.get("multisource_scenes", [])
    multisource_targets = automation.get("multisource_targets", [])
    if isinstance(single_source_scenes, list) and single_source_scenes:
        scene_settings.extend(_build_single_source_settings(single_source_scenes))
    if isinstance(multisource_scenes, list) and multisource_scenes:
        scene_settings.extend(_build_multisource_settings(multisource_scenes))
    if (
        include_configured_multisource_targets
        and isinstance(multisource_targets, list)
        and multisource_targets
    ):
        scene_settings.extend(
            _build_multisource_target_settings(
                multisource_targets,
                source_pool=_source_pool_from_targets(experiment_payload, multisource_targets),
            )
        )

    if not scene_settings:
        protocol = experiment_payload.get("protocol_override", {})
        source_domains = protocol.get("source_domains", [])
        target_domain = protocol.get("target_domain")
        if len(source_domains) == 1 and target_domain:
            scene_settings.extend(_build_single_source_settings([f"{source_domains[0]}->{target_domain}"]))

    if not scene_settings:
        raise ValueError(
            "No automation scenes resolved. Configure automation.single_source_scenes / "
            "automation.multisource_scenes / automation.multisource_targets, or pass --scenes/--all-scenes."
        )
    return scene_settings


def build_run_plan(
    experiment_payload: dict[str, Any],
    *,
    cli_methods: list[str] | None = None,
    cli_scenes: list[str] | None = None,
    all_scenes: bool = False,
    include_multisource_targets: bool | None = None,
) -> dict[str, Any]:
    methods = resolve_method_names(experiment_payload, cli_methods)
    scene_settings = resolve_scene_settings(
        experiment_payload,
        cli_scenes,
        all_scenes=all_scenes,
        include_multisource_targets=include_multisource_targets,
    )
    protocol_override = experiment_payload.get("protocol_override", {})
    fold_sampling = protocol_override.get("fold_sampling", {})
    if not isinstance(fold_sampling, dict):
        fold_sampling = {}
    random_fold_enabled = bool(fold_sampling.get("enabled", protocol_override.get("random_fold_enabled", False)))
    source_folds = [str(item) for item in protocol_override.get("source_folds", [1, 2, 3, 4, 5])]
    target_folds = [str(item) for item in protocol_override.get("target_folds", [1, 2, 3, 4, 5])]
    preferred_fold = str(protocol_override.get("preferred_fold", "Fold 1"))
    scene_fold_overrides = protocol_override.get("scene_fold_overrides", {})
    rng_seed, seed_mode = resolve_seed(experiment_payload.get("seed"))
    rng = random.Random(rng_seed)
    runs = []
    for scene in scene_settings:
        sampled_source_fold = canonicalize_fold_choice(rng.choice(source_folds) if random_fold_enabled else preferred_fold)
        sampled_target_fold = canonicalize_fold_choice(rng.choice(target_folds) if random_fold_enabled else preferred_fold)
        fold_override = _scene_fold_override(scene, scene_fold_overrides)
        source_domains = [str(item) for item in scene["source_domains"]]
        source_folds_by_domain = _source_fold_mapping(scene, fold_override, sampled_source_fold)
        source_fold = _source_fold_display(source_domains, source_folds_by_domain)
        target_fold = canonicalize_fold_choice(fold_override.get("target_fold", sampled_target_fold))
        for method_name in methods:
            runs.append(
                {
                    "method_name": method_name,
                    "setting": str(scene["setting"]),
                    "source_domains": source_domains,
                    "target_domain": str(scene["target_domain"]),
                    "label": str(scene["label"]),
                    "source_fold": source_fold,
                    "source_folds_by_domain": dict(source_folds_by_domain),
                    "target_fold": target_fold,
                    "method_overrides": _scene_override_payload(experiment_payload, scene, method_name),
                }
            )
    return {
        "methods": methods,
        "scene_settings": scene_settings,
        "runs": runs,
        "fold_policy": {
            "random_fold_enabled": random_fold_enabled,
            "source_folds": source_folds,
            "target_folds": target_folds,
            "preferred_fold": preferred_fold,
            "seed": rng_seed,
            "seed_mode": seed_mode,
        },
        "seed": rng_seed,
        "seed_mode": seed_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured DA batch plan.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/te_da.yaml"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiment/quick_debug.yaml"),
    )
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Run all 30 directed single-source single-target mode pairs.",
    )
    parser.add_argument(
        "--include-multisource-targets",
        action="store_true",
        default=None,
        help="Force-enable configured automation.multisource_targets in addition to single-source scenes.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only print the resolved run plan without launching training.",
    )
    parser.add_argument("--batch-root-name", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed for this batch launch without editing the YAML config.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _refresh_batch_outputs(batch_root: Path) -> None:
    from src.evaluation.evaluate import export_comparison_summary
    from src.evaluation.report_figures import export_summary_figures

    summary_dir = export_comparison_summary(batch_root)
    if summary_dir is None:
        return
    export_summary_figures(batch_root, summary_dir.parent / "figures")


def main() -> None:
    args = parse_args()
    base_experiment = _load_yaml(args.experiment_config)
    if args.seed is not None:
        base_experiment["seed"] = int(args.seed)
    plan = build_run_plan(
        base_experiment,
        cli_methods=args.methods,
        cli_scenes=args.scenes,
        all_scenes=args.all_scenes,
        include_multisource_targets=args.include_multisource_targets,
    )
    methods = plan["methods"]
    scene_settings = plan["scene_settings"]
    run_plan = plan["runs"]
    experiment_name = str(base_experiment.get("experiment_name", "batch_run"))
    fold_policy = plan.get("fold_policy", {})
    base_experiment["seed"] = int(plan["seed"])
    base_experiment["seed_mode"] = str(plan["seed_mode"])
    print(
        f"Planned {len(run_plan)} runs from {len(scene_settings)} settings x {len(methods)} methods "
        f"using {args.experiment_config}."
    )
    if args.plan_only:
        print(
            "Fold policy: "
            f"random={fold_policy.get('random_fold_enabled', False)}, "
            f"seed={fold_policy.get('seed')}, "
            f"seed_mode={fold_policy.get('seed_mode', 'fixed')}, "
            f"source_choices={fold_policy.get('source_folds', [])}, "
            f"target_choices={fold_policy.get('target_folds', [])}, "
            f"preferred={fold_policy.get('preferred_fold', 'Fold 1')}"
        )
        for run in run_plan:
            scene_label = run['label']
            fold_text = f"src{run['source_fold']}__tgt{run['target_fold']}"
            print(
                f"{run['method_name']}: {','.join(run['source_domains'])} -> "
                f"{run['target_domain']} ({scene_label}; {fold_text})"
            )
        return

    batch_root_name = args.batch_root_name or f"{build_timestamp()}_{experiment_name}"
    should_refresh_batch_outputs = bool(
        base_experiment.get("runtime", {}).get("refresh_batch_outputs", True)
    )

    with tempfile.TemporaryDirectory(prefix="tep_batch_plan_") as temp_dir:
        temp_root = Path(temp_dir)
        for run in run_plan:
            method_name = str(run["method_name"])
            method_config_path = Path("configs/method") / f"{method_name}.yaml"
            if not method_config_path.exists():
                raise FileNotFoundError(f"Method config not found: {method_config_path}")

            experiment_payload = deepcopy(base_experiment)
            experiment_payload["experiment_name"] = f"{experiment_name}_{run['label']}_src{run['source_fold']}__tgt{run['target_fold']}"
            experiment_payload.setdefault("tracking", {})
            experiment_payload["tracking"]["batch_root_name"] = batch_root_name
            experiment_payload.setdefault("runtime", {})
            if args.dry_run:
                experiment_payload["runtime"]["dry_run"] = True
            experiment_payload["runtime"].setdefault("save_checkpoint", True)
            experiment_payload["runtime"].setdefault("save_analysis", True)
            experiment_payload["runtime"]["refresh_batch_outputs"] = False
            experiment_payload.setdefault("protocol_override", {})
            protocol_update = {
                "setting": str(run["setting"]),
                "source_domains": [str(item) for item in run["source_domains"]],
                "target_domain": str(run["target_domain"]),
                "preferred_fold": "Fold 1",
                "source_fold": str(run["source_fold"]),
                "source_folds_by_domain": dict(run.get("source_folds_by_domain", {})),
                "target_fold": str(run["target_fold"]),
            }
            experiment_payload["protocol_override"].update(protocol_update)
            if run.get("method_overrides"):
                scene_key = str(run["label"])
                method_key = str(run["method_name"])
                experiment_payload["method_overrides"] = {
                    scene_key: {
                        method_key: deepcopy(run["method_overrides"]),
                    }
                }

            temp_experiment_path = temp_root / f"{method_name}_{run['label']}.yaml"
            _save_yaml(temp_experiment_path, experiment_payload)
            command = [
                "bash",
                "scripts/train.sh",
                str(args.data_config),
                str(method_config_path),
                str(temp_experiment_path),
                "--batch-root-name",
                batch_root_name,
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    "Batch round stopped at "
                    f"{method_name} on {run['label']} "
                    f"(exit code {completed.returncode})."
                )

    batch_root = Path(str(base_experiment.get("output_dir", "runs"))) / batch_root_name
    if should_refresh_batch_outputs:
        _refresh_batch_outputs(batch_root)
    print(f"Batch results written under {batch_root}")
    print(f"Comparison summary expected at {batch_root / 'comparison_summary'}/")


if __name__ == "__main__":
    try:
        main()
    except AutomationDependencyError as exc:
        raise SystemExit(str(exc))
