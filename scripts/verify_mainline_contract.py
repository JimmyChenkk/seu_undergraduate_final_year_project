#!/usr/bin/env python3
"""Verify ranking contracts for frozen benchmark result batches."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_IGNORED_METHODS = {"target_only", "target_ref", "target_supervised_reference"}


@dataclass(frozen=True)
class ResultRow:
    scene_label: str
    method_name: str
    source_count: int
    accuracy: float
    path: Path


@dataclass(frozen=True)
class Violation:
    scene_label: str
    leader: str
    challenger: str
    leader_accuracy: float
    challenger_accuracy: float
    margin: float


def _iter_result_rows(root: Path) -> Iterable[ResultRow]:
    for path in sorted(root.glob("*/tables/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = payload.get("result", {})
        method_name = str(payload.get("method_name") or result.get("method_name") or "").strip()
        scene_label = str(payload.get("scene_label") or payload.get("scenario_id") or "").strip()
        accuracy = result.get("selected_target_eval_acc", result.get("target_eval_acc"))
        if not method_name or not scene_label or accuracy is None:
            continue
        rows_key = (scene_label, method_name)
        del rows_key
        yield ResultRow(
            scene_label=scene_label,
            method_name=method_name,
            source_count=len(payload.get("source_domains", [])),
            accuracy=float(accuracy),
            path=path,
        )


def _latest_rows_by_scene_method(root: Path) -> dict[tuple[str, str], ResultRow]:
    rows: dict[tuple[str, str], ResultRow] = {}
    for row in _iter_result_rows(root):
        key = (row.scene_label, row.method_name)
        current = rows.get(key)
        if current is None or row.path.stat().st_mtime >= current.path.stat().st_mtime:
            rows[key] = row
    return rows


def _scene_methods(rows: dict[tuple[str, str], ResultRow]) -> dict[str, dict[str, ResultRow]]:
    scenes: dict[str, dict[str, ResultRow]] = {}
    for (scene_label, method_name), row in rows.items():
        scenes.setdefault(scene_label, {})[method_name] = row
    return scenes


def _filter_source_count(
    rows: dict[tuple[str, str], ResultRow],
    source_count: int | None,
) -> dict[tuple[str, str], ResultRow]:
    if source_count is None:
        return rows
    return {
        key: row
        for key, row in rows.items()
        if row.source_count == int(source_count)
    }


def _check_leader_contract(
    scenes: dict[str, dict[str, ResultRow]],
    *,
    leader: str,
    ignored_methods: set[str],
    margin: float,
) -> list[Violation]:
    violations: list[Violation] = []
    for scene_label, methods in sorted(scenes.items()):
        leader_row = methods.get(leader)
        if leader_row is None:
            continue
        for method_name, challenger_row in sorted(methods.items()):
            if method_name == leader or method_name in ignored_methods:
                continue
            gap = leader_row.accuracy - challenger_row.accuracy
            if gap <= margin:
                violations.append(
                    Violation(
                        scene_label=scene_label,
                        leader=leader,
                        challenger=method_name,
                        leader_accuracy=leader_row.accuracy,
                        challenger_accuracy=challenger_row.accuracy,
                        margin=gap,
                    )
                )
    return violations


def _check_pair_contract(
    scenes: dict[str, dict[str, ResultRow]],
    *,
    higher: str,
    lower: str,
    margin: float,
) -> list[Violation]:
    violations: list[Violation] = []
    for scene_label, methods in sorted(scenes.items()):
        higher_row = methods.get(higher)
        lower_row = methods.get(lower)
        if higher_row is None or lower_row is None:
            continue
        gap = higher_row.accuracy - lower_row.accuracy
        if gap <= margin:
            violations.append(
                Violation(
                    scene_label=scene_label,
                    leader=higher,
                    challenger=lower,
                    leader_accuracy=higher_row.accuracy,
                    challenger_accuracy=lower_row.accuracy,
                    margin=gap,
                )
            )
    return violations


def _format_table(scenes: dict[str, dict[str, ResultRow]], methods: list[str]) -> str:
    lines = []
    header = ["scene", *methods]
    lines.append("\t".join(header))
    for scene_label, scene_rows in sorted(scenes.items()):
        values = [scene_label]
        for method_name in methods:
            row = scene_rows.get(method_name)
            values.append("NA" if row is None else f"{row.accuracy:.6f}")
        lines.append("\t".join(values))
    return "\n".join(lines)


def _print_violations(title: str, violations: list[Violation]) -> None:
    if not violations:
        print(f"{title}: PASS")
        return
    print(f"{title}: FAIL ({len(violations)} violation(s))")
    for item in violations:
        print(
            "  "
            f"{item.scene_label}: {item.leader}={item.leader_accuracy:.6f}, "
            f"{item.challenger}={item.challenger_accuracy:.6f}, "
            f"margin={item.margin:.6f}"
        )


def _check_scene_count(title: str, scenes: dict[str, dict[str, ResultRow]], expected: int | None) -> bool:
    if expected is None:
        if scenes:
            return True
        print(f"{title}: FAIL (no scenes found)")
        return False
    actual = len(scenes)
    if actual == int(expected):
        return True
    print(f"{title}: FAIL (expected {int(expected)} scene(s), found {actual})")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify mainline single-source and multi-source ranking contracts."
    )
    parser.add_argument("--single-root", type=Path, help="Batch root for single-source results.")
    parser.add_argument("--multi-root", type=Path, help="Batch root for multi-source results.")
    parser.add_argument(
        "--single-source-count",
        type=int,
        help="Only check single-root rows with this many source domains.",
    )
    parser.add_argument(
        "--multi-source-count",
        type=int,
        help="Only check multi-root rows with this many source domains, e.g. 2 for multi-source 15 or 5 for 5-source 30.",
    )
    parser.add_argument("--single-scene-count", type=int, help="Require this many single-source scenes.")
    parser.add_argument("--multi-scene-count", type=int, help="Require this many multi-source scenes.")
    parser.add_argument("--margin", type=float, default=0.0, help="Required strict margin above competitors.")
    parser.add_argument(
        "--ignore-method",
        action="append",
        default=[],
        help="Method to ignore as a competitor. Can be provided multiple times.",
    )
    parser.add_argument("--print-tables", action="store_true", help="Print per-scene accuracy tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignored_methods = DEFAULT_IGNORED_METHODS | {str(item) for item in args.ignore_method}
    all_violations: list[Violation] = []
    contract_failed = False

    if args.single_root:
        single_rows = _filter_source_count(
            _latest_rows_by_scene_method(args.single_root),
            args.single_source_count,
        )
        single_scenes = _scene_methods(single_rows)
        contract_failed = contract_failed or not _check_scene_count(
            "single-source scene count",
            single_scenes,
            args.single_scene_count,
        )
        single_methods = [
            "source_only",
            "dsan",
            "cdan_ts",
            "codats",
            "deepjdot",
            "tpu_dpjdot",
            "cbtpu_dpjdot",
            "target_only",
        ]
        if args.print_tables:
            print("\n[single-source]")
            print(_format_table(single_scenes, single_methods))
        violations = _check_leader_contract(
            single_scenes,
            leader="cbtpu_dpjdot",
            ignored_methods=ignored_methods,
            margin=float(args.margin),
        )
        violations.extend(
            _check_pair_contract(
                single_scenes,
                higher="tpu_dpjdot",
                lower="deepjdot",
                margin=float(args.margin),
            )
        )
        _print_violations("single-source contract", violations)
        all_violations.extend(violations)

    if args.multi_root:
        multi_rows = _filter_source_count(
            _latest_rows_by_scene_method(args.multi_root),
            args.multi_source_count,
        )
        multi_scenes = _scene_methods(multi_rows)
        contract_failed = contract_failed or not _check_scene_count(
            "multi-source scene count",
            multi_scenes,
            args.multi_scene_count,
        )
        multi_methods = ["source_only", "codats", "wjdot", "ca_ccsr_wjdot", "target_ref"]
        if args.print_tables:
            print("\n[multi-source]")
            print(_format_table(multi_scenes, multi_methods))
        violations = _check_leader_contract(
            multi_scenes,
            leader="ca_ccsr_wjdot",
            ignored_methods=ignored_methods,
            margin=float(args.margin),
        )
        _print_violations("multi-source contract", violations)
        all_violations.extend(violations)

    raise SystemExit(1 if contract_failed or all_violations else 0)


if __name__ == "__main__":
    main()
