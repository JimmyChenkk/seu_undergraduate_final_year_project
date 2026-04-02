#!/usr/bin/env python3
"""Build a metadata-first TE benchmark manifest from raw inspection output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.te_da_dataset import (  # noqa: E402
    TEDADatasetConfig,
    build_manifest_from_inspection,
    inspect_raw_directory,
    load_json_file,
    render_inspection_markdown,
    write_json_file,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Please run this script inside tep_env.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TE benchmark manifest from raw inspection results.")
    parser.add_argument("--config", default="configs/data/te_da.yaml", help="Dataset configuration YAML.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_config = TEDADatasetConfig.from_dict(payload)

    if dataset_config.inspection_json_path.exists():
        inspection = load_json_file(dataset_config.inspection_json_path)
    else:
        inspection = inspect_raw_directory(dataset_config.raw_dir, pattern=dataset_config.raw_file_pattern)
        markdown = render_inspection_markdown(inspection)
        dataset_config.inspection_report_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_config.inspection_report_path.write_text(markdown, encoding="utf-8")
        write_json_file(dataset_config.inspection_json_path, inspection)

    manifest = build_manifest_from_inspection(
        inspection,
        dataset_name=str(payload.get("dataset_name", "te_domain_adaptation")),
    )
    write_json_file(dataset_config.manifest_path, manifest)

    print(f"Manifest written to: {dataset_config.manifest_path}")
    print(f"Domain count: {manifest['domain_count']}")
    print(f"Domains: {[entry['domain_id'] for entry in manifest['domains']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
