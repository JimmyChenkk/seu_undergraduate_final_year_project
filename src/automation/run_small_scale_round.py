"""Run the default small-scale single-source DA sweep."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable

from src.utils.run_layout import build_timestamp


DEFAULT_METHODS = [
    "source_only",
    "coral",
    "dan",
    "dann",
    "cdan",
    "deepjdot",
]

ALL_MODES = [
    "mode1",
    "mode2",
    "mode3",
    "mode4",
    "mode5",
    "mode6",
]

DEFAULT_SCENES = [
    ("mode1", "mode4"),
    ("mode4", "mode1"),
    ("mode2", "mode5"),
    ("mode5", "mode2"),
    ("mode3", "mode6"),
    ("mode6", "mode3"),
]


class AutomationDependencyError(RuntimeError):
    """Raised when the small-scale automation entrypoint is missing YAML support."""


def _import_yaml():
    if importlib.util.find_spec("yaml") is None:
        raise AutomationDependencyError(
            "Missing automation dependency: yaml. Install it manually in tep_env, for example with "
            + "`pip install -r requirements-benchmark.txt`."
        )

    import yaml

    return yaml


def _load_yaml(path: Path) -> dict:
    yaml = _import_yaml()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def _save_yaml(path: Path, payload: dict) -> None:
    yaml = _import_yaml()
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _parse_scene_tokens(scene_tokens: Iterable[str]) -> list[tuple[str, str]]:
    parsed = []
    for token in scene_tokens:
        normalized = token.replace(":", "->")
        if "->" not in normalized:
            raise ValueError(f"Invalid scene token: {token}")
        source_domain, target_domain = [item.strip() for item in normalized.split("->", maxsplit=1)]
        parsed.append((source_domain, target_domain))
    return parsed


def _build_all_directed_scenes() -> list[tuple[str, str]]:
    return [
        (source_domain, target_domain)
        for source_domain in ALL_MODES
        for target_domain in ALL_MODES
        if source_domain != target_domain
    ]


def _build_all_multisource_target_settings() -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []
    for target_domain in ALL_MODES:
        source_domains = [domain for domain in ALL_MODES if domain != target_domain]
        settings.append(
            {
                "setting": "multi_source",
                "source_domains": source_domains,
                "target_domain": target_domain,
                "label": f"{'-'.join(source_domains)}_to_{target_domain}",
            }
        )
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the default small-scale DA sweep.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/te_da.yaml"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiment/autonomous_small_scale.yaml"),
    )
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=[f"{source}->{target}" for source, target in DEFAULT_SCENES],
    )
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
    parser.add_argument("--batch-root-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_experiment = _load_yaml(args.experiment_config)
    scene_settings: list[dict[str, object]]
    if args.all_scenes:
        scene_settings = [
            {
                "setting": "single_source",
                "source_domains": [source_domain],
                "target_domain": target_domain,
                "label": f"{source_domain}_to_{target_domain}",
            }
            for source_domain, target_domain in _build_all_directed_scenes()
        ]
    else:
        scene_settings = [
            {
                "setting": "single_source",
                "source_domains": [source_domain],
                "target_domain": target_domain,
                "label": f"{source_domain}_to_{target_domain}",
            }
            for source_domain, target_domain in _parse_scene_tokens(args.scenes)
        ]
    if args.include_multisource_targets:
        scene_settings.extend(_build_all_multisource_target_settings())
    batch_root_name = args.batch_root_name or f"{build_timestamp()}_full_run"

    with tempfile.TemporaryDirectory(prefix="tep_small_scale_") as temp_dir:
        temp_root = Path(temp_dir)
        for scene in scene_settings:
            setting_name = str(scene["setting"])
            source_domains = [str(item) for item in scene["source_domains"]]
            target_domain = str(scene["target_domain"])
            scene_label = str(scene["label"])
            for method_name in args.methods:
                method_config_path = Path("configs/method") / f"{method_name}.yaml"
                if not method_config_path.exists():
                    raise FileNotFoundError(f"Method config not found: {method_config_path}")

                experiment_payload = deepcopy(base_experiment)
                experiment_payload["experiment_name"] = (
                    f"{base_experiment.get('experiment_name', 'autonomous_small_scale')}_{scene_label}"
                )
                experiment_payload.setdefault("tracking", {})
                experiment_payload["tracking"]["batch_root_name"] = batch_root_name
                experiment_payload.setdefault("runtime", {})
                if args.dry_run:
                    experiment_payload["runtime"]["dry_run"] = True
                experiment_payload["runtime"].setdefault("save_analysis", True)
                experiment_payload.setdefault("protocol_override", {})
                experiment_payload["protocol_override"].update(
                    {
                        "setting": setting_name,
                        "source_domains": source_domains,
                        "target_domain": target_domain,
                        "preferred_fold": "Fold 1",
                    }
                )

                temp_experiment_path = temp_root / f"{method_name}_{scene_label}.yaml"
                _save_yaml(temp_experiment_path, experiment_payload)
                # Reuse the stable shell wrapper so automation and one-off runs share
                # the same environment setup (conda activation, MPL cache, etc.).
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
                        "Small-scale round stopped at "
                        f"{method_name} on {scene_label} "
                        f"(exit code {completed.returncode})."
                    )

    print(f"Batch results written under runs/{batch_root_name}")
    print(f"Comparison summary expected at runs/{batch_root_name}/comparison_summary/")


if __name__ == "__main__":
    try:
        main()
    except AutomationDependencyError as exc:
        raise SystemExit(str(exc))
