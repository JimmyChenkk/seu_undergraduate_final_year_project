"""Run config-driven quick-debug or benchmark batch plans."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

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


def _normalize_multisource_targets(targets: Any) -> list[str]:
    if targets in (None, False):
        return []
    if targets is True:
        return list(ALL_MODES)

    raw_items = [targets] if isinstance(targets, str) else list(targets)
    normalized_targets: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        token = str(item).strip()
        if not token:
            continue
        if token.lower() in {"all", "*"}:
            for domain_name in ALL_MODES:
                if domain_name not in seen:
                    seen.add(domain_name)
                    normalized_targets.append(domain_name)
            continue
        domain_name = _validate_domain(token)
        if domain_name not in seen:
            seen.add(domain_name)
            normalized_targets.append(domain_name)
    return normalized_targets


def _build_multisource_settings(scene_tokens: Iterable[str]) -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []
    for source_token, target_domain in _parse_scene_tokens(scene_tokens):
        source_domains = [item.strip() for item in source_token.replace("+", "-").split("-") if item.strip()]
        for domain_name in source_domains:
            _validate_domain(domain_name)
        if len(source_domains) < 2:
            raise ValueError(f"Multi-source scene must include at least two source domains: {source_token}->{target_domain}")
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
    include_multisource_targets: bool,
) -> list[dict[str, object]]:
    automation = _automation_section(experiment_payload)

    scene_settings = []

    if all_scenes:
        return _build_single_source_settings([f"{source}->{target}" for source, target in _build_all_directed_scenes()])

    if cli_scenes is not None:
        return _build_single_source_settings(cli_scenes)

    for key, value in automation.items():
        if key == "single_source_scenes" and isinstance(value, list) and value:
            scene_settings.extend(_build_single_source_settings(value))
        elif key in {"multisource_scenes", "multisource_targets"}:
            if key == "multisource_targets" and not include_multisource_targets:
                continue
            if isinstance(value, list) and value:
                scene_settings.extend(_build_multisource_settings(value))

    if not scene_settings:
        protocol = experiment_payload.get("protocol_override", {})
        source_domains = protocol.get("source_domains", [])
        target_domain = protocol.get("target_domain")
        if len(source_domains) == 1 and target_domain:
            scene_settings.extend(_build_single_source_settings([f"{source_domains[0]}->{target_domain}"]))

    if not scene_settings:
        raise ValueError(
            "No automation scenes resolved. Configure automation.single_source_scenes or "
            "automation.multisource_scenes, or pass --scenes/--all-scenes."
        )
    return scene_settings


def build_run_plan(
    experiment_payload: dict[str, Any],
    *,
    cli_methods: list[str] | None = None,
    cli_scenes: list[str] | None = None,
    all_scenes: bool = False,
    include_multisource_targets: bool = False,
) -> dict[str, Any]:
    methods = resolve_method_names(experiment_payload, cli_methods)
    scene_settings = resolve_scene_settings(
        experiment_payload,
        cli_scenes,
        all_scenes=all_scenes,
        include_multisource_targets=include_multisource_targets,
    )
    runs = [
        {
            "method_name": method_name,
            "setting": str(scene["setting"]),
            "source_domains": [str(item) for item in scene["source_domains"]],
            "target_domain": str(scene["target_domain"]),
            "label": str(scene["label"]),
        }
        for scene in scene_settings
        for method_name in methods
    ]
    return {
        "methods": methods,
        "scene_settings": scene_settings,
        "runs": runs,
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
        help="Also run the 6 five-source-to-one-target benchmark settings.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only print the resolved run plan without launching training.",
    )
    parser.add_argument("--batch-root-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_experiment = _load_yaml(args.experiment_config)
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
    print(
        f"Planned {len(run_plan)} runs from {len(scene_settings)} settings x {len(methods)} methods "
        f"using {args.experiment_config}."
    )
    if args.plan_only:
        for run in run_plan:
            print(
                f"{run['method_name']}: {','.join(run['source_domains'])} -> "
                f"{run['target_domain']} ({run['setting']})"
            )
        return

    batch_root_name = args.batch_root_name or f"{build_timestamp()}_{experiment_name}"

    with tempfile.TemporaryDirectory(prefix="tep_batch_plan_") as temp_dir:
        temp_root = Path(temp_dir)
        for run in run_plan:
            method_name = str(run["method_name"])
            method_config_path = Path("configs/method") / f"{method_name}.yaml"
            if not method_config_path.exists():
                raise FileNotFoundError(f"Method config not found: {method_config_path}")

            experiment_payload = deepcopy(base_experiment)
            experiment_payload["experiment_name"] = f"{experiment_name}_{run['label']}"
            experiment_payload.setdefault("tracking", {})
            experiment_payload["tracking"]["batch_root_name"] = batch_root_name
            experiment_payload.setdefault("runtime", {})
            if args.dry_run:
                experiment_payload["runtime"]["dry_run"] = True
            experiment_payload["runtime"].setdefault("save_checkpoint", True)
            experiment_payload["runtime"].setdefault("save_analysis", True)
            experiment_payload.setdefault("protocol_override", {})
            experiment_payload["protocol_override"].update(
                {
                    "setting": str(run["setting"]),
                    "source_domains": [str(item) for item in run["source_domains"]],
                    "target_domain": str(run["target_domain"]),
                    "preferred_fold": "Fold 1",
                }
            )

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

    print(f"Batch results written under runs/{batch_root_name}")
    print(f"Comparison summary expected at runs/{batch_root_name}/comparison_summary/")


if __name__ == "__main__":
    try:
        main()
    except AutomationDependencyError as exc:
        raise SystemExit(str(exc))
