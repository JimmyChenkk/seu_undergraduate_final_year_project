"""Minimal TE dataset metadata utilities for stage-two benchmark work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
from typing import Any, Iterable

import numpy as np


REQUIRED_TOP_LEVEL_KEYS = ("Signals", "Labels", "Folds")
DEFAULT_FOLD_NAME = "Fold 1"
DOMAIN_PATTERN = re.compile(r"mode[_\s-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DomainSpec:
    """Lightweight domain description used by later experiment configs."""

    name: str
    split: str = "train"
    labeled: bool = True
    max_labeled_samples: int | None = None


@dataclass
class DomainDataReference:
    """Manifest-backed domain reference."""

    domain: DomainSpec
    storage: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainAdaptationSetting:
    """Single-source or multi-source setting description."""

    setting_name: str
    source_domains: list[DomainDataReference]
    target_domain: DomainDataReference
    target_label_mode: str = "unlabeled"
    notes: list[str] = field(default_factory=list)


@dataclass
class TEDADatasetConfig:
    """Paths and metadata defaults for the TE DA workspace."""

    raw_dir: Path = Path("data/raw")
    benchmark_dir: Path = Path("data/benchmark")
    cache_dir: Path = Path("data/cache")
    manifest_path: Path = Path("data/benchmark/manifest.json")
    inspection_json_path: Path = Path("data/cache/te_raw_data_inspection.json")
    inspection_report_path: Path = Path("data/benchmark/te_raw_data_inspection.md")
    raw_file_pattern: str = "*.pickle"
    preferred_fold: str = DEFAULT_FOLD_NAME
    normalization: str = "standardization"
    normalization_scope: str = "domain"
    channels_first: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TEDADatasetConfig":
        paths = data.get("paths", {})
        loading = data.get("loading", {})
        protocol = data.get("protocol", {})
        return cls(
            raw_dir=Path(paths.get("raw_dir", "data/raw")),
            benchmark_dir=Path(paths.get("benchmark_dir", "data/benchmark")),
            cache_dir=Path(paths.get("cache_dir", "data/cache")),
            manifest_path=Path(paths.get("manifest_path", "data/benchmark/manifest.json")),
            inspection_json_path=Path(paths.get("inspection_json_path", "data/cache/te_raw_data_inspection.json")),
            inspection_report_path=Path(
                paths.get("inspection_report_path", "data/benchmark/te_raw_data_inspection.md")
            ),
            raw_file_pattern=str(loading.get("raw_file_pattern", "*.pickle")),
            preferred_fold=str(protocol.get("preferred_fold", DEFAULT_FOLD_NAME)),
            normalization=str(loading.get("normalization", "standardization")),
            normalization_scope=str(loading.get("normalization_scope", "domain")),
            channels_first=bool(loading.get("channels_first", True)),
        )


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON payload from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def canonicalize_domain_id(name: str) -> str:
    """Convert file names or user input to canonical mode ids such as ``mode1``."""

    match = DOMAIN_PATTERN.search(Path(name).stem)
    if match is not None:
        return f"mode{int(match.group(1))}"
    return name.strip().lower().replace(" ", "_")


def _relative_source_path(path: Path, raw_dir: Path) -> str:
    try:
        return str(path.relative_to(raw_dir.parent))
    except ValueError:
        return str(path)


def iter_valid_raw_pickles(raw_dir: Path, pattern: str = "*.pickle") -> list[Path]:
    """Return raw pickle files while ignoring Windows metadata and hidden files."""

    if not raw_dir.exists():
        return []

    files: list[Path] = []
    for path in sorted(raw_dir.glob(pattern)):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.endswith(".Zone.Identifier"):
            continue
        files.append(path)
    return files


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _length_of(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def summarize_array(value: Any) -> dict[str, Any]:
    """Summarize a numpy-like top-level object without altering it."""

    return {
        "python_type": type(value).__name__,
        "shape": _shape_of(value),
        "length": _length_of(value),
        "dtype": _dtype_of(value),
    }


def summarize_labels(labels: np.ndarray) -> dict[str, Any]:
    """Capture conservative label statistics for manifest and notes."""

    unique_values = np.unique(labels)
    summary = summarize_array(labels)
    summary["label_stats"] = {
        "size": int(labels.size),
        "min": int(labels.min()),
        "max": int(labels.max()),
        "unique_count": int(unique_values.size),
        "unique_values_preview": [int(value) for value in unique_values[:20]],
    }
    if unique_values.size > 20:
        summary["label_stats"]["unique_values_truncated"] = True
    return summary


def summarize_folds(folds: Any) -> dict[str, Any]:
    """Summarize fold metadata without assuming a more complex nested schema."""

    summary = summarize_array(folds)
    if not isinstance(folds, dict):
        summary["fold_names"] = []
        summary["fold_count"] = 0
        return summary

    entries: dict[str, Any] = {}
    for fold_name, indices in folds.items():
        fold_summary = summarize_array(indices)
        if isinstance(indices, np.ndarray) and indices.size > 0:
            fold_summary["index_range"] = {
                "size": int(indices.size),
                "min": int(indices.min()),
                "max": int(indices.max()),
            }
        entries[str(fold_name)] = fold_summary

    summary["fold_names"] = list(entries.keys())
    summary["fold_count"] = len(entries)
    summary["entries"] = entries
    return summary


def inspect_raw_pickle(path: Path, raw_dir: Path) -> dict[str, Any]:
    """Inspect one raw pickle file and return metadata only."""

    result: dict[str, Any] = {
        "file_name": path.name,
        "domain_id": canonicalize_domain_id(path.name),
        "source_path": _relative_source_path(path, raw_dir),
        "size_bytes": path.stat().st_size,
    }

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["python_type"] = type(payload).__name__

    if not isinstance(payload, dict):
        result["top_level_keys"] = []
        result["root_summary"] = summarize_array(payload)
        return result

    top_level_keys = [str(key) for key in payload.keys()]
    result["top_level_keys"] = top_level_keys
    result["required_keys_present"] = {key: key in payload for key in REQUIRED_TOP_LEVEL_KEYS}

    key_summaries: dict[str, Any] = {}
    for key, value in payload.items():
        key_name = str(key)
        if key_name == "Labels" and isinstance(value, np.ndarray):
            key_summaries[key_name] = summarize_labels(value)
        elif key_name == "Folds":
            key_summaries[key_name] = summarize_folds(value)
        else:
            key_summaries[key_name] = summarize_array(value)
    result["key_summaries"] = key_summaries
    return result


def inspect_raw_directory(raw_dir: Path, pattern: str = "*.pickle") -> dict[str, Any]:
    """Inspect every valid raw pickle file under ``raw_dir``."""

    files = iter_valid_raw_pickles(raw_dir, pattern=pattern)
    inspection = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "file_count": len(files),
        "files": [inspect_raw_pickle(path, raw_dir) for path in files],
    }
    inspection["domain_ids"] = [item.get("domain_id") for item in inspection["files"]]
    return inspection


def render_inspection_markdown(inspection: dict[str, Any]) -> str:
    """Render a human-readable schema inspection report."""

    lines: list[str] = [
        "# TE Raw Data Inspection",
        "",
        "This note is auto-generated by `scripts/inspect_raw_data.py`.",
        "",
        "## File Summary",
        "",
        "| File | Domain | Type | Top-level keys | Error |",
        "| --- | --- | --- | --- | --- |",
    ]

    files = inspection.get("files", [])
    signals_shapes: set[tuple[int, ...]] = set()
    signals_dtypes: set[str] = set()
    labels_dtypes: set[str] = set()
    all_ready = True

    for item in files:
        keys = ", ".join(item.get("top_level_keys", [])) or "N/A"
        error = item.get("error", "None")
        lines.append(
            f"| {item['file_name']} | {item.get('domain_id', 'N/A')} | "
            f"{item.get('python_type', 'N/A')} | {keys} | {error} |"
        )

    for item in files:
        lines.extend(
            [
                "",
                f"## {item['file_name']}",
                "",
                f"- Domain ID: `{item.get('domain_id', 'N/A')}`",
                f"- Source path: `{item.get('source_path', 'N/A')}`",
                f"- File size (bytes): `{item.get('size_bytes', 'N/A')}`",
                f"- Python type: `{item.get('python_type', 'N/A')}`",
            ]
        )

        if item.get("error"):
            all_ready = False
            lines.append(f"- Error: `{item['error']}`")
            continue

        lines.append(f"- Top-level keys: `{item.get('top_level_keys', [])}`")
        lines.append(f"- Required keys present: `{item.get('required_keys_present', {})}`")
        lines.append("")
        lines.append("| Key | Type | Shape | Length | Dtype | Extra |")
        lines.append("| --- | --- | --- | --- | --- | --- |")

        key_summaries = item.get("key_summaries", {})
        for key in item.get("top_level_keys", []):
            summary = key_summaries.get(key, {})
            extra_parts: list[str] = []
            if key == "Signals" and summary.get("shape") is not None:
                signals_shapes.add(tuple(summary["shape"]))
            if key == "Signals" and summary.get("dtype") is not None:
                signals_dtypes.add(str(summary["dtype"]))
            if key == "Labels" and summary.get("dtype") is not None:
                labels_dtypes.add(str(summary["dtype"]))
            if key == "Labels" and summary.get("label_stats") is not None:
                extra_parts.append(f"label_stats={summary['label_stats']}")
            if key == "Folds":
                extra_parts.append(f"fold_names={summary.get('fold_names', [])}")
                extra_parts.append(f"fold_count={summary.get('fold_count', 0)}")
            lines.append(
                f"| {key} | {summary.get('python_type')} | {summary.get('shape')} | "
                f"{summary.get('length')} | {summary.get('dtype')} | "
                f"{'; '.join(extra_parts) if extra_parts else 'None'} |"
            )
            if key == "Folds" and summary.get("entries"):
                lines.append("")
                lines.append("| Fold | Shape | Length | Dtype | Index range |")
                lines.append("| --- | --- | --- | --- | --- |")
                for fold_name, fold_summary in summary["entries"].items():
                    lines.append(
                        f"| {fold_name} | {fold_summary.get('shape')} | {fold_summary.get('length')} | "
                        f"{fold_summary.get('dtype')} | {fold_summary.get('index_range', 'None')} |"
                    )

    lines.extend(
        [
            "",
            "## Short Conclusion",
            "",
            f"- Direct foundation for single-source / multi-source DA: `{'yes' if all_ready else 'not yet'}`",
            "- Reason: the raw pickle files already expose per-domain `Signals`, `Labels`, and `Folds`, "
            "which is enough to build a metadata-first benchmark manifest without copying arrays.",
            f"- Signals shape variants observed: `{sorted(signals_shapes) if signals_shapes else 'N/A'}`",
            f"- Signals dtype variants observed: `{sorted(signals_dtypes) if signals_dtypes else 'N/A'}`",
            f"- Labels dtype variants observed: `{sorted(labels_dtypes) if labels_dtypes else 'N/A'}`",
            "- Most important next step from raw to benchmark: lock a manifest-backed domain/fold mapping so later "
            "loaders stop guessing filenames and split structure.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest_from_inspection(
    inspection: dict[str, Any],
    *,
    dataset_name: str = "te_domain_adaptation",
) -> dict[str, Any]:
    """Build a metadata-first manifest from inspection results."""

    domains: list[dict[str, Any]] = []
    for item in inspection.get("files", []):
        if item.get("error"):
            raise ValueError(f"Inspection failed for {item['file_name']}: {item['error']}")

        top_level_keys = item.get("top_level_keys", [])
        missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in top_level_keys]
        if missing:
            raise ValueError(f"{item['file_name']} is missing required keys: {missing}")

        key_summaries = item.get("key_summaries", {})
        labels = key_summaries["Labels"]
        folds = key_summaries["Folds"]
        if not folds.get("fold_names"):
            raise ValueError(f"{item['file_name']} has missing or invalid fold structure.")

        label_stats = labels.get("label_stats", {})
        domains.append(
            {
                "domain_id": item["domain_id"],
                "source_file": item["file_name"],
                "source_path": item["source_path"],
                "signals_shape": key_summaries["Signals"].get("shape"),
                "signals_dtype": key_summaries["Signals"].get("dtype"),
                "labels_shape": labels.get("shape"),
                "labels_dtype": labels.get("dtype"),
                "label_range": {
                    "min": label_stats.get("min"),
                    "max": label_stats.get("max"),
                },
                "label_unique_count": label_stats.get("unique_count"),
                "label_unique_values_preview": label_stats.get("unique_values_preview"),
                "fold_names": folds.get("fold_names", []),
                "fold_count": folds.get("fold_count", 0),
                "fold_details": folds.get("entries", {}),
            }
        )

    domains.sort(key=lambda item: int(item["domain_id"].replace("mode", "")))
    return {
        "dataset_name": dataset_name,
        "schema_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_kind": "immutable_pickle_snapshot",
        "domain_count": len(domains),
        "required_top_level_keys": list(REQUIRED_TOP_LEVEL_KEYS),
        "domains": domains,
    }


class TEDADatasetInterface:
    """Manifest-backed domain resolver for later single-source / multi-source work."""

    def __init__(self, config: TEDADatasetConfig) -> None:
        self.config = config
        self._manifest_cache: dict[str, Any] | None = None

    def load_manifest(self) -> dict[str, Any]:
        """Load and cache the benchmark manifest."""

        if self._manifest_cache is None:
            self._manifest_cache = load_json_file(self.config.manifest_path)
        return self._manifest_cache

    def list_manifest_domains(self) -> list[str]:
        """Return canonical domain ids from the manifest."""

        manifest = self.load_manifest()
        return [str(entry["domain_id"]) for entry in manifest.get("domains", [])]

    def get_domain_entry(self, domain_id: str) -> dict[str, Any]:
        """Return one manifest entry by canonical domain id."""

        canonical = canonicalize_domain_id(domain_id)
        manifest = self.load_manifest()
        for entry in manifest.get("domains", []):
            if str(entry["domain_id"]) == canonical:
                return entry
        raise KeyError(f"Domain not found in manifest: {domain_id}")

    def resolve_domain_reference(self, domain: DomainSpec) -> DomainDataReference:
        """Resolve one domain into a manifest-backed reference."""

        entry = self.get_domain_entry(domain.name)
        metadata = {
            "fold_names": entry.get("fold_names", []),
            "fold_count": entry.get("fold_count", 0),
            "preferred_fold": self.config.preferred_fold,
            "signals_shape": entry.get("signals_shape"),
            "signals_dtype": entry.get("signals_dtype"),
            "labels_shape": entry.get("labels_shape"),
            "labels_dtype": entry.get("labels_dtype"),
            "label_range": entry.get("label_range"),
        }
        return DomainDataReference(
            domain=DomainSpec(
                name=str(entry["domain_id"]),
                split=domain.split,
                labeled=domain.labeled,
                max_labeled_samples=domain.max_labeled_samples,
            ),
            storage="benchmark_manifest",
            path=Path(entry["source_path"]),
            metadata=metadata,
        )

    def build_single_source_setting(
        self,
        source_domain: str,
        target_domain: str,
        *,
        split: str = "train",
        target_label_mode: str = "unlabeled",
        few_shot_target_samples: int | None = None,
    ) -> DomainAdaptationSetting:
        """Construct a manifest-backed single-source setting description."""

        source_spec = DomainSpec(name=source_domain, split=split, labeled=True)
        target_spec = DomainSpec(
            name=target_domain,
            split=split,
            labeled=target_label_mode != "unlabeled",
            max_labeled_samples=few_shot_target_samples,
        )
        return DomainAdaptationSetting(
            setting_name="single_source",
            source_domains=[self.resolve_domain_reference(source_spec)],
            target_domain=self.resolve_domain_reference(target_spec),
            target_label_mode=target_label_mode,
            notes=[f"Preferred fold: {self.config.preferred_fold}"],
        )

    def build_multi_source_setting(
        self,
        source_domains: Iterable[str],
        target_domain: str,
        *,
        split: str = "train",
        target_label_mode: str = "unlabeled",
        few_shot_target_samples: int | None = None,
    ) -> DomainAdaptationSetting:
        """Construct a manifest-backed multi-source setting description."""

        source_refs = [
            self.resolve_domain_reference(DomainSpec(name=domain_name, split=split, labeled=True))
            for domain_name in source_domains
        ]
        target_spec = DomainSpec(
            name=target_domain,
            split=split,
            labeled=target_label_mode != "unlabeled",
            max_labeled_samples=few_shot_target_samples,
        )
        return DomainAdaptationSetting(
            setting_name="multi_source",
            source_domains=source_refs,
            target_domain=self.resolve_domain_reference(target_spec),
            target_label_mode=target_label_mode,
            notes=[f"Preferred fold: {self.config.preferred_fold}"],
        )

    def describe_setting(self, setting: DomainAdaptationSetting) -> dict[str, Any]:
        """Return a serializable summary for dry-run style checks."""

        return {
            "setting_name": setting.setting_name,
            "target_label_mode": setting.target_label_mode,
            "available_domains": self.list_manifest_domains(),
            "source_domains": [self._describe_reference(reference) for reference in setting.source_domains],
            "target_domain": self._describe_reference(setting.target_domain),
            "notes": setting.notes,
        }

    def _describe_reference(self, reference: DomainDataReference) -> dict[str, Any]:
        return {
            "domain": reference.domain.name,
            "split": reference.domain.split,
            "labeled": reference.domain.labeled,
            "max_labeled_samples": reference.domain.max_labeled_samples,
            "storage": reference.storage,
            "path": str(reference.path) if reference.path is not None else None,
            "metadata": reference.metadata,
        }

    def resolve_raw_pickle_path(self, domain_id: str) -> Path:
        """Resolve the raw pickle path for a canonical domain id."""

        entry = self.get_domain_entry(domain_id)
        raw_name = entry.get("source_file")
        if not raw_name:
            raise KeyError(f"Manifest entry for {domain_id} has no source_file.")
        return self.config.raw_dir / str(raw_name)

    def load_raw_domain_payload(self, domain_id: str) -> dict[str, Any]:
        """Load one raw pickle payload for later benchmark reproduction work."""

        raw_path = self.resolve_raw_pickle_path(domain_id)
        with raw_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"{raw_path} does not contain a dictionary payload.")
        missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in payload]
        if missing:
            raise KeyError(f"{raw_path} is missing required keys: {missing}")
        return payload

    def load_fold_indices(self, domain_id: str, fold_name: str | None = None) -> np.ndarray:
        """Load the authoritative fold indices for one domain."""

        payload = self.load_raw_domain_payload(domain_id)
        fold_key = fold_name or self.config.preferred_fold
        folds = payload["Folds"]
        if fold_key not in folds:
            available = sorted(str(key) for key in folds.keys())
            raise KeyError(f"Fold {fold_key!r} not found for {domain_id}. Available: {available}")
        return np.asarray(folds[fold_key], dtype=np.int64)

    def build_train_eval_indices(
        self,
        domain_id: str,
        *,
        fold_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build train/eval indices using one held-out fold."""

        entry = self.get_domain_entry(domain_id)
        sample_count = int(entry["signals_shape"][0])
        eval_indices = self.load_fold_indices(domain_id, fold_name=fold_name)
        eval_set = set(int(index) for index in eval_indices.tolist())
        train_indices = np.asarray(
            [index for index in range(sample_count) if index not in eval_set],
            dtype=np.int64,
        )
        return train_indices, eval_indices


def normalize_signals(
    signals: np.ndarray,
    *,
    normalization: str = "standardization",
    normalization_stats: tuple[np.ndarray, np.ndarray] | None = None,
    channels_first: bool = True,
) -> np.ndarray:
    """Normalize one domain with the same convention as the public benchmark repo."""

    if signals.ndim != 3:
        raise ValueError(f"Expected 3D signals, got shape {signals.shape}")

    features = np.asarray(signals, dtype=np.float32).copy()
    if normalization_stats is None:
        bias, scale = compute_normalization_statistics(
            features,
            normalization=normalization,
        )
    else:
        bias, scale = normalization_stats
        bias = np.asarray(bias, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)

    for feature_index in range(features.shape[-1]):
        if np.isclose(scale[feature_index], 0.0):
            denominator = bias[feature_index] if not np.isclose(bias[feature_index], 0.0) else 1.0
            features[..., feature_index] = features[..., feature_index] / denominator
        else:
            features[..., feature_index] = (
                features[..., feature_index] - bias[feature_index]
            ) / scale[feature_index]

    if channels_first:
        return np.transpose(features, (0, 2, 1))
    return features


def compute_normalization_statistics(
    signals: np.ndarray,
    *,
    normalization: str = "standardization",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel normalization statistics for one domain."""

    if signals.ndim != 3:
        raise ValueError(f"Expected 3D signals, got shape {signals.shape}")

    features = np.asarray(signals, dtype=np.float32)
    _, _, n_features = features.shape
    min_vals = features.min(axis=(0, 1))
    max_vals = features.max(axis=(0, 1))
    mean_vals = features.mean(axis=(0, 1))
    std_vals = features.std(axis=(0, 1))

    if normalization == "standardization":
        return mean_vals, std_vals
    if normalization == "scaling":
        return min_vals, max_vals - min_vals
    return (
        np.zeros([n_features], dtype=np.float32),
        np.ones([n_features], dtype=np.float32),
    )


def slice_domain_split(
    payload: dict[str, Any],
    indices: np.ndarray,
    *,
    normalization: str = "standardization",
    normalization_stats: tuple[np.ndarray, np.ndarray] | None = None,
    channels_first: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice and normalize one payload split conservatively."""

    signals = np.asarray(payload["Signals"])[indices]
    labels = np.asarray(payload["Labels"], dtype=np.int64)[indices]
    normalized = normalize_signals(
        signals,
        normalization=normalization,
        normalization_stats=normalization_stats,
        channels_first=channels_first,
    )
    return normalized, labels
