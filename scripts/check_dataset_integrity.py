#!/usr/bin/env python
"""Check the local TEP domain-adaptation dataset contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_complete_label_range(entry: dict[str, Any]) -> bool:
    label_range = entry.get("label_range", {})
    return (
        int(label_range.get("min", -1)) == 0
        and int(label_range.get("max", -1)) == 28
        and int(entry.get("label_unique_count", 0)) == 29
    )


def build_integrity_report(data_config_path: Path) -> dict[str, Any]:
    data_config_path = Path(data_config_path)
    config = _load_yaml(data_config_path)
    paths = config.get("paths", {})
    manifest_path = Path(paths.get("manifest_path", "data/benchmark/manifest.json"))
    manifest = _load_json(manifest_path)
    loading = config.get("loading", {})
    protocol = config.get("protocol", {})

    domains = []
    failures = []
    warnings = []
    for entry in manifest.get("domains", []):
        domain_id = str(entry.get("domain_id"))
        signals_shape = entry.get("signals_shape") or []
        labels_shape = entry.get("labels_shape") or []
        fold_details = entry.get("fold_details", {})
        domain_failures = []
        if len(signals_shape) != 3 or signals_shape[1:] != [600, 34]:
            domain_failures.append(f"unexpected signals_shape={signals_shape}")
        if not labels_shape or int(labels_shape[0]) != int(signals_shape[0]):
            domain_failures.append(f"labels_shape does not match signals_shape: {labels_shape}")
        if not _is_complete_label_range(entry):
            domain_failures.append(
                "labels are not the expected contiguous normal/fault range 0..28"
            )
        if int(entry.get("fold_count", 0)) != 5:
            domain_failures.append("fold_count is not 5")
        fold_lengths = [int(item.get("length", 0)) for item in fold_details.values()]
        if fold_lengths and sum(fold_lengths) != int(signals_shape[0]):
            warnings.append(
                f"{domain_id} fold lengths sum to {sum(fold_lengths)}, not the full sample count "
                f"{signals_shape[0]}; current loader treats the selected fold as eval and all remaining "
                "samples as train."
            )
        if domain_failures:
            failures.append({"domain": domain_id, "issues": domain_failures})
        domains.append(
            {
                "domain_id": domain_id,
                "signals_shape": signals_shape,
                "labels_shape": labels_shape,
                "label_range": entry.get("label_range"),
                "label_unique_count": entry.get("label_unique_count"),
                "fold_count": entry.get("fold_count"),
                "fold_lengths": fold_lengths,
                "status": "fail" if domain_failures else "ok",
            }
        )

    normalization_scope = str(loading.get("normalization_scope", "domain")).strip().lower()
    if normalization_scope in {"domain", "full_domain", "global"}:
        warnings.append(
            "normalization_scope is domain/full-domain: this matches the local benchmark "
            "but uses full target-domain statistics, including the held-out fold."
        )
    if bool(protocol.get("use_target_labels", False)):
        warnings.append(
            "data config enables target labels by default; this should be reserved for target-only/few-shot baselines."
        )

    return {
        "data_config": str(data_config_path),
        "manifest_path": str(manifest_path),
        "dataset_name": manifest.get("dataset_name"),
        "domain_count": manifest.get("domain_count"),
        "channels_contract": "34 continuous variables interpreted as XME(1)-XME(22) + XMV(1)-XMV(12)",
        "window_contract": "600 time steps per sample; loader transposes to 34 x 600 when channels_first=true",
        "normalization": loading.get("normalization"),
        "normalization_scope": normalization_scope,
        "channels_first": bool(loading.get("channels_first", True)),
        "default_use_target_labels": bool(protocol.get("use_target_labels", False)),
        "domains": domains,
        "failures": failures,
        "warnings": warnings,
        "passed": not failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Integrity Check",
        "",
        f"- Dataset: `{report.get('dataset_name')}`",
        f"- Manifest: `{report.get('manifest_path')}`",
        f"- Domain count: `{report.get('domain_count')}`",
        f"- Channel contract: {report.get('channels_contract')}",
        f"- Window contract: {report.get('window_contract')}",
        f"- Normalization: `{report.get('normalization')}` / scope `{report.get('normalization_scope')}`",
        f"- Default target labels enabled: `{report.get('default_use_target_labels')}`",
        f"- Overall status: `{'pass' if report.get('passed') else 'fail'}`",
        "",
        "## Domains",
        "",
        "| Domain | Signals | Labels | Label Range | Classes | Folds | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.get("domains", []):
        lines.append(
            "| {domain_id} | {signals_shape} | {labels_shape} | {label_range} | "
            "{label_unique_count} | {fold_count} | {status} |".format(**item)
        )

    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    if report.get("failures"):
        lines.extend(["", "## Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- `{failure['domain']}`: {'; '.join(failure['issues'])}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/te_da.yaml"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_integrity_report(args.data_config)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
