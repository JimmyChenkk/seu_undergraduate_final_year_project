"""Dataset metadata interfaces for the TE domain adaptation project."""

from .te_da_dataset import (
    DomainAdaptationSetting,
    DomainDataReference,
    DomainSpec,
    build_manifest_from_inspection,
    canonicalize_domain_id,
    inspect_raw_directory,
    normalize_signals,
    render_inspection_markdown,
    slice_domain_split,
    TEDADatasetConfig,
    TEDADatasetInterface,
    write_json_file,
)

__all__ = [
    "DomainAdaptationSetting",
    "DomainDataReference",
    "DomainSpec",
    "TEDADatasetConfig",
    "TEDADatasetInterface",
    "build_manifest_from_inspection",
    "canonicalize_domain_id",
    "inspect_raw_directory",
    "normalize_signals",
    "render_inspection_markdown",
    "slice_domain_split",
    "write_json_file",
]
