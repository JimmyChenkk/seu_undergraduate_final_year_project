"""Generate report-style figures for TEP benchmark experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _load_result_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        rows.append(
            {
                "path": path,
                "method": str(payload.get("method_name")),
                "setting": str(payload.get("setting")),
                "scenario_id": str(payload.get("scenario_id", path.stem)),
                "source_domains": payload.get("source_domains", []),
                "target_domain": str(payload.get("target_domain")),
                "target_eval_acc": float(
                    result.get("final_target_eval_acc", result.get("target_eval_acc", 0.0))
                ),
                "target_eval_balanced_acc": result.get("target_eval_balanced_acc"),
                "run_root": payload.get("run_root"),
                "analysis_path": result.get("analysis_path"),
            }
        )
    return rows


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(path: Path) -> None:
    _ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def export_mean_bar_chart(rows: list[dict], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["setting"], row["method"])
        grouped.setdefault(key, []).append(row["target_eval_acc"])

    settings = sorted(set(item[0] for item in grouped))
    methods = sorted(set(item[1] for item in grouped))
    x = np.arange(len(methods))
    width = 0.35 if len(settings) > 1 else 0.6

    plt.figure(figsize=(10, 5))
    for index, setting in enumerate(settings):
        values = [np.mean(grouped.get((setting, method), [np.nan])) for method in methods]
        offset = (index - (len(settings) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=setting.replace("_", "-"))

    plt.xticks(x, methods, rotation=20)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title("Method Mean Accuracy by Setting")
    if len(settings) > 1:
        plt.legend()
    _save_figure(output_path)


def export_setting_heatmap(rows: list[dict], setting_name: str, output_path: Path) -> None:
    subset = [row for row in rows if row["setting"] == setting_name]
    if not subset:
        return

    scenarios = sorted(set(row["scenario_id"] for row in subset))
    methods = sorted(set(row["method"] for row in subset))
    matrix = np.full((len(scenarios), len(methods)), np.nan, dtype=float)
    for row in subset:
        scenario_index = scenarios.index(row["scenario_id"])
        method_index = methods.index(row["method"])
        matrix[scenario_index, method_index] = row["target_eval_acc"]

    plt.figure(figsize=(max(7, len(methods) * 1.2), max(4, len(scenarios) * 0.5)))
    image = plt.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    plt.colorbar(image, label="Accuracy")
    plt.xticks(np.arange(len(methods)), methods, rotation=30)
    plt.yticks(np.arange(len(scenarios)), scenarios)
    plt.title(f"{setting_name.replace('_', '-').title()} Task Heatmap")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if not np.isnan(value):
                plt.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8)

    _save_figure(output_path)


def _fit_tsne(features: np.ndarray) -> np.ndarray:
    if len(features) < 5:
        return np.zeros((len(features), 2), dtype=np.float32)
    perplexity = min(30, max(5, len(features) // 10))
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=42)
    return tsne.fit_transform(features)


def export_tsne_figures(artifact_path: Path, output_dir: Path) -> None:
    payload = np.load(artifact_path, allow_pickle=True)
    source_embeddings = payload["source_embeddings"]
    source_labels = payload["source_labels"]
    source_domains = payload["source_domains"].astype(str)
    target_embeddings = payload["target_embeddings"]
    target_labels = payload["target_labels"]
    target_domains = payload["target_domains"].astype(str)

    features = np.concatenate([source_embeddings, target_embeddings], axis=0)
    label_values = np.concatenate([source_labels, target_labels], axis=0)
    domain_values = np.concatenate([source_domains, target_domains], axis=0)
    domain_flags = np.array(["source"] * len(source_embeddings) + ["target"] * len(target_embeddings))
    embedding_2d = _fit_tsne(features)

    plt.figure(figsize=(6, 5))
    for domain_name in ["source", "target"]:
        mask = domain_flags == domain_name
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=14,
            alpha=0.65,
            label=domain_name,
        )
    plt.legend()
    plt.title("Domain Fusion t-SNE")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    _save_figure(output_dir / "tsne_domain.png")

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=label_values,
        s=12,
        alpha=0.75,
        cmap="tab20",
    )
    plt.colorbar(scatter, label="Class label")
    plt.title("Class Aggregation t-SNE")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    _save_figure(output_dir / "tsne_class.png")


def export_confusion_matrix_figure(artifact_path: Path, output_path: Path) -> None:
    payload = np.load(artifact_path, allow_pickle=True)
    labels = payload["target_labels"]
    predictions = payload["target_predictions"]

    matrix = np.zeros((29, 29), dtype=int)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1

    plt.figure(figsize=(8, 7))
    image = plt.imshow(matrix, aspect="auto", cmap="Blues")
    plt.colorbar(image, label="Count")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Target Confusion Matrix")
    _save_figure(output_path)


def export_domain_comparison_figure(
    left_artifact: Path,
    right_artifact: Path,
    output_path: Path,
) -> None:
    """Create a side-by-side domain-fusion t-SNE comparison like the proposal figures."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, artifact_path in zip(axes, [left_artifact, right_artifact]):
        payload = np.load(artifact_path, allow_pickle=True)
        source_embeddings = payload["source_embeddings"]
        target_embeddings = payload["target_embeddings"]
        features = np.concatenate([source_embeddings, target_embeddings], axis=0)
        domain_flags = np.array(["source"] * len(source_embeddings) + ["target"] * len(target_embeddings))
        embedding_2d = _fit_tsne(features)
        for domain_name in ["source", "target"]:
            mask = domain_flags == domain_name
            axis.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                s=14,
                alpha=0.65,
                label=domain_name,
            )
        method_name = str(payload["method_name"][0]) if "method_name" in payload else artifact_path.parent.name
        scenario_id = str(payload["scenario_id"][0]) if "scenario_id" in payload else artifact_path.parent.name
        axis.set_title(f"{method_name.upper()} ({scenario_id})")
        axis.set_xlabel("t-SNE-1")
        axis.set_ylabel("t-SNE-2")
    axes[0].legend(loc="best")
    plt.suptitle("Domain Fusion Comparison")
    _save_figure(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export report-style TEP benchmark figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("runs/tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    parser.add_argument("--artifact", action="append", default=[], help="Optional artifact .npz path for t-SNE/confusion.")
    parser.add_argument(
        "--compare-domain-artifacts",
        nargs=2,
        action="append",
        default=[],
        metavar=("LEFT", "RIGHT"),
        help="Optional pair of artifact .npz files to export a side-by-side domain-fusion comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_result_rows(args.results_dir)
    if not rows:
        print("No result JSON files found.")
        return

    _ensure_dir(args.output_dir)
    export_mean_bar_chart(rows, args.output_dir / "method_mean_accuracy.png")
    export_setting_heatmap(rows, "single_source", args.output_dir / "single_source_heatmap.png")
    export_setting_heatmap(rows, "multi_source", args.output_dir / "multi_source_heatmap.png")

    for artifact_item in args.artifact:
        artifact_path = Path(artifact_item)
        if not artifact_path.exists():
            continue
        payload = np.load(artifact_path, allow_pickle=True)
        scenario_value = str(payload["scenario_id"][0]) if "scenario_id" in payload else artifact_path.parent.name
        method_value = str(payload["method_name"][0]) if "method_name" in payload else artifact_path.parent.parent.parent.name
        figure_dir = args.output_dir / f"{method_value}_{scenario_value}"
        _ensure_dir(figure_dir)
        export_tsne_figures(artifact_path, figure_dir)
        export_confusion_matrix_figure(artifact_path, figure_dir / "confusion_matrix.png")

    for pair_index, (left_item, right_item) in enumerate(args.compare_domain_artifacts, start=1):
        left_path = Path(left_item)
        right_path = Path(right_item)
        if not left_path.exists() or not right_path.exists():
            continue
        export_domain_comparison_figure(
            left_path,
            right_path,
            args.output_dir / f"domain_comparison_{pair_index}.png",
        )

    print(f"Figures exported to {args.output_dir}")


if __name__ == "__main__":
    main()
