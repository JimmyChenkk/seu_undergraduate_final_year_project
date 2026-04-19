#!/usr/bin/env python3
"""Temporary isolated source-only runner for 6-scene TE experiments.

This script stays entirely under ``data/eval`` and does not modify the project's
main training/data code. It launches the existing training entry point through
``scripts/train.sh`` while reading the raw datasets directly from ``data/raw``.

Strategy
========
- Keep the project architecture untouched.
- Never copy or duplicate raw ``.pickle`` files into ``runs/`` or any task output.
- Create only temporary experiment configuration files.
- Reuse the existing trainer/data loader semantics as-is.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALL_MODES = ["mode1", "mode2", "mode3", "mode4", "mode5", "mode6"]
ALL_FOLDS = [f"Fold {index}" for index in range(1, 6)]
DEFAULT_SCENES = [("mode1", "mode4"), ("mode4", "mode1"), ("mode2", "mode5"), ("mode5", "mode2"), ("mode3", "mode6"), ("mode6", "mode3")]
SOURCE_ONLY_METHOD = "source_only"
DEFAULT_DATA_CONFIG = Path("configs/data/te_da.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment/quick_debug.yaml")
DEFAULT_METHOD_CONFIG = Path("configs/method/source_only.yaml")


class RunnerDependencyError(RuntimeError):
    """Raised when optional YAML support is missing."""


def _import_yaml():
    if importlib.util.find_spec("yaml") is None:
        raise RunnerDependencyError(
            "Missing dependency: yaml. Install PyYAML before running this script."
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_mode(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in ALL_MODES:
        raise ValueError(f"Unsupported mode: {name}")
    return normalized


def _canonical_fold(name: str) -> str:
    normalized = str(name).strip()
    if normalized not in ALL_FOLDS:
        raise ValueError(f"Unsupported fold: {name}")
    return normalized


def build_task_plan() -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for source_domain, target_domain in DEFAULT_SCENES:
        for source_fold in ALL_FOLDS:
            for target_fold in ALL_FOLDS:
                tasks.append(
                    {
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "source_fold": source_fold,
                        "target_fold": target_fold,
                        "label": (
                            f"{source_domain}_to_{target_domain}__"
                            f"src-{source_fold.replace(' ', '')}__tgt-{target_fold.replace(' ', '')}"
                        ),
                    }
                )
    return tasks


def _resolve_task_root(output_root: Path, timestamp: str) -> Path:
    return output_root / f"{timestamp}_sourceonly_6scene"


def _task_run_dir(task_root: Path, task: dict[str, str]) -> Path:
    return task_root / f"{task['source_domain']}_to_{task['target_domain']}" / (
        f"src_{task['source_fold'].replace(' ', '')}__tgt_{task['target_fold'].replace(' ', '')}"
    )



def _build_temp_experiment_payload(base_experiment: dict[str, Any], *, batch_root_name: str, task: dict[str, str], dry_run: bool) -> dict[str, Any]:
    payload = deepcopy(base_experiment)
    payload["experiment_name"] = f"{base_experiment.get('experiment_name', 'sourceonly_6scene')}_{task['label']}"
    payload.setdefault("tracking", {})
    payload["tracking"]["batch_root_name"] = batch_root_name
    payload.setdefault("runtime", {})
    payload["runtime"]["dry_run"] = bool(dry_run)
    payload["runtime"]["save_checkpoint"] = True
    payload["runtime"]["save_analysis"] = True
    payload["runtime"]["final_epoch_evaluation"] = True
    payload["runtime"]["final_selection_evaluation"] = True
    payload.setdefault("automation", {})
    payload["automation"]["methods"] = [SOURCE_ONLY_METHOD]
    payload.setdefault("protocol_override", {})
    payload["protocol_override"].update(
        {
            "setting": "single_source",
            "source_domains": [task["source_domain"]],
            "target_domain": task["target_domain"],
            "preferred_fold": task["source_fold"],
            "source_fold": task["source_fold"],
            "target_fold": task["target_fold"],
        }
    )
    return payload


def _build_result_summary(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "tables" / "result.json"
    review_path = run_dir / "tables" / "review.json"
    payload: dict[str, Any] = {"run_dir": str(run_dir), "exists": run_dir.exists()}
    if result_path.exists():
        result_payload = _load_json(result_path)
        payload.update(
            {
                "method_name": result_payload.get("method_name"),
                "setting": result_payload.get("setting"),
                "source_domains": result_payload.get("source_domains", []),
                "target_domain": result_payload.get("target_domain"),
                "scenario_id": result_payload.get("scenario_id"),
                "fold_name": result_payload.get("fold_name"),
                "source_train_acc": result_payload.get("source_train_acc"),
                "source_eval_acc": result_payload.get("source_eval_acc"),
                "target_eval_acc": result_payload.get("target_eval_acc"),
                "target_eval_balanced_acc": result_payload.get("target_eval_balanced_acc"),
                "figure_paths": result_payload.get("figure_paths", {}),
                "metrics_path": result_payload.get("metrics_path"),
                "review_path": result_payload.get("review_path"),
            }
        )
    if review_path.exists():
        payload["review"] = _load_json(review_path)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporary isolated runner for 6-scene source-only TE tasks.")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--method-config", type=Path, default=DEFAULT_METHOD_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("runs/sourceonly_6scene"))
    parser.add_argument("--batch-root-name", type=str, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_data = _load_yaml(args.data_config)
    base_experiment = _load_yaml(args.experiment_config)
    method_config_path = args.method_config
    if not method_config_path.exists():
        raise FileNotFoundError(f"Method config not found: {method_config_path}")

    tasks = build_task_plan()
    if args.limit is not None:
        tasks = tasks[: max(args.limit, 0)]

    if args.plan_only:
        print(f"Planned {len(tasks)} tasks.")
        for task in tasks:
            print(
                f"{task['source_domain']} -> {task['target_domain']} | "
                f"{task['source_fold']} x {task['target_fold']} | {task['label']}"
            )
        return 0

    from src.utils.run_layout import build_timestamp

    timestamp = build_timestamp()
    batch_root_name = args.batch_root_name or f"{timestamp}_sourceonly_6scene"
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    task_root = output_root / batch_root_name
    task_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    summary.append(
        {
            "batch_root_name": batch_root_name,
            "task_root": str(task_root),
            "task_count": len(tasks),
            "dry_run": bool(args.dry_run),
        }
    )

    with tempfile.TemporaryDirectory(prefix="sourceonly_6scene_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, task in enumerate(tasks, start=1):
            _canonical_mode(task["source_domain"])
            _canonical_mode(task["target_domain"])
            _canonical_fold(task["source_fold"])
            _canonical_fold(task["target_fold"])

            run_dir = _task_run_dir(task_root, task)
            if args.skip_existing and (run_dir / "tables" / "result.json").exists():
                summary.append({"task": task, "skipped": True, **_build_result_summary(run_dir)})
                continue

            run_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_bundle_root = temp_root / f"{index:04d}_{task['label']}"
            temp_bundle_root.mkdir(parents=True, exist_ok=True)

            temp_data_payload = deepcopy(base_data)
            temp_experiment_payload = _build_temp_experiment_payload(
                base_experiment,
                batch_root_name=batch_root_name,
                task=task,
                dry_run=args.dry_run,
            )

            temp_data_config = temp_bundle_root / "data.yaml"
            temp_experiment_config = temp_bundle_root / "experiment.yaml"
            _save_yaml(temp_data_config, temp_data_payload)
            _save_yaml(temp_experiment_config, temp_experiment_payload)

            command = [
                "bash",
                "scripts/train.sh",
                str(temp_data_config),
                str(method_config_path),
                str(temp_experiment_config),
                "--batch-root-name",
                batch_root_name,
            ]
            completed = subprocess.run(command, check=False)
            summary.append(
                {
                    "task": task,
                    "returncode": completed.returncode,
                    **_build_result_summary(run_dir),
                }
            )
            if completed.returncode != 0:
                raise SystemExit(
                    "Batch stopped at "
                    f"{task['source_domain']} -> {task['target_domain']} "
                    f"({task['source_fold']} x {task['target_fold']}) with exit code {completed.returncode}."
                )

    summary_path = task_root / "sourceonly_6scene_summary.json"
    _save_json(summary_path, {"row_count": len(summary), "rows": summary})
    print(f"Wrote summary to {summary_path}")
    print(f"Batch results written under {task_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
