"""Run the default small-scale single-source DA sweep."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable

import yaml

from src.utils.run_layout import build_timestamp


DEFAULT_METHODS = [
    "source_only",
    "coral",
    "dan",
    "dann",
    "cdan",
    "jdot",
]

DEFAULT_SCENES = [
    ("mode1", "mode4"),
    ("mode4", "mode1"),
    ("mode2", "mode5"),
    ("mode5", "mode2"),
    ("mode3", "mode6"),
    ("mode6", "mode3"),
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def _save_yaml(path: Path, payload: dict) -> None:
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
    parser.add_argument("--batch-root-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_experiment = _load_yaml(args.experiment_config)
    scenes = _parse_scene_tokens(args.scenes)
    batch_root_name = args.batch_root_name or f"{build_timestamp()}_full_run"

    with tempfile.TemporaryDirectory(prefix="tep_small_scale_") as temp_dir:
        temp_root = Path(temp_dir)
        for source_domain, target_domain in scenes:
            for method_name in args.methods:
                method_config_path = Path("configs/method") / f"{method_name}.yaml"
                if not method_config_path.exists():
                    raise FileNotFoundError(f"Method config not found: {method_config_path}")

                experiment_payload = deepcopy(base_experiment)
                experiment_payload["experiment_name"] = (
                    f"{base_experiment.get('experiment_name', 'autonomous_small_scale')}_{source_domain}_to_{target_domain}"
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
                        "setting": "single_source",
                        "source_domains": [source_domain],
                        "target_domain": target_domain,
                        "preferred_fold": "Fold 1",
                    }
                )

                temp_experiment_path = temp_root / f"{method_name}_{source_domain}_to_{target_domain}.yaml"
                _save_yaml(temp_experiment_path, experiment_payload)
                command = [
                    sys.executable,
                    "-m",
                    "src.trainers.train_benchmark",
                    "--data-config",
                    str(args.data_config),
                    "--method-config",
                    str(method_config_path),
                    "--experiment-config",
                    str(temp_experiment_path),
                    "--batch-root-name",
                    batch_root_name,
                ]
                subprocess.run(command, check=True)

    print(f"Batch results written under runs/{batch_root_name}")
    print(f"Comparison summary expected at runs/{batch_root_name}/comparison_summary/")


if __name__ == "__main__":
    main()
