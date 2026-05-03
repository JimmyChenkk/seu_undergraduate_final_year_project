"""Generate core figures from TEP DA result artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_ORDER = [
    "deepjdot",
    "tpu_deepjdot",
    "cbtpu_deepjdot",
]
FIGURE_HIDDEN_METHODS = {
    "u_deepjdot",
}
METHOD_DISPLAY_NAMES = {
    "deepjdot": "deepjdot",
    "tpu_deepjdot": "tpu_dpjdot",
    "cbtpu_deepjdot": "cbtpu_dpjdot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TEP DA result figures.")
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tep_ot_plots"))
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1", "balanced_accuracy"])
    parser.add_argument("--confusion-matrix", type=Path, default=None)
    parser.add_argument("--source-weights", type=Path, default=None)
    parser.add_argument("--class-source-weights", type=Path, default=None)
    parser.add_argument("--analysis", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_existing(rows: list[dict[str, str]], field: str) -> Path | None:
    for row in rows:
        value = row.get(field, "")
        if value and Path(value).exists():
            return Path(value)
    return None


def sort_methods(methods) -> list[str]:
    rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(
        (str(method) for method in methods if str(method) not in FIGURE_HIDDEN_METHODS),
        key=lambda method: (rank.get(method, len(rank)), method),
    )


def display_method(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method), str(method))


def plot_method_bar(rows: list[dict[str, str]], output_dir: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric):
            grouped[row["method"]].append(float(row[metric]))
    methods = sort_methods(grouped)
    means = [float(np.mean(grouped[method])) for method in methods]
    stds = [float(np.std(grouped[method], ddof=1)) if len(grouped[method]) > 1 else 0.0 for method in methods]
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 0.8), 4.5))
    labels = [display_method(method) for method in methods]
    ax.bar(labels, means, yerr=stds, color="#4477aa", capsize=3)
    ax.set_ylabel(metric)
    ax.set_title("Method comparison")
    ax.set_ylim(0, max(1.0, max(means, default=0.0) * 1.15))
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "method_comparison_bar.png", dpi=180)
    plt.close(fig)


def plot_task_heatmap(rows: list[dict[str, str]], output_dir: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    tasks = sorted({row["task"] for row in rows})
    methods = sort_methods({row["method"] for row in rows})
    for row in rows:
        values[(row["task"], row["method"])].append(float(row[metric]))
    matrix = np.full((len(tasks), len(methods)), np.nan, dtype=float)
    for i, task in enumerate(tasks):
        for j, method in enumerate(methods):
            item = values.get((task, method), [])
            if item:
                matrix[i, j] = float(np.mean(item))
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 0.75), max(5, len(tasks) * 0.45)))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([display_method(method) for method in methods], rotation=35, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title(f"Task-method {metric}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "task_method_heatmap.png", dpi=180)
    plt.close(fig)


def plot_confusion(confusion_path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    confusion = np.load(confusion_path) if confusion_path.suffix == ".npy" else np.loadtxt(confusion_path, delimiter=",")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    ax.set_xticks(range(0, confusion.shape[1], 4))
    ax.set_yticks(range(0, confusion.shape[0], 4))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_source_weights(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = load_rows(path)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if "epoch" in rows[0]:
        epochs = [int(float(row["epoch"])) for row in rows]
        for key in rows[0].keys():
            if key == "epoch":
                continue
            ax.plot(epochs, [float(row.get(key, 0.0)) for row in rows], marker="o", label=key)
        ax.set_xlabel("Epoch")
    else:
        labels = [row["source_domain"] for row in rows]
        values = [float(row["weight"]) for row in rows]
        ax.bar(labels, values, color="#66aa55")
    ax.set_ylabel("source weight")
    ax.set_title("Source weight curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "source_weight_curve.png", dpi=180)
    plt.close(fig)


def plot_class_source(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = load_rows(path)
    if not rows:
        return
    sources = [key for key in rows[0].keys() if key != "class_id"]
    matrix = np.asarray([[float(row[source]) for source in sources] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, len(sources) * 1.4), 8))
    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources)
    ax.set_yticks(range(0, matrix.shape[0], 2))
    ax.set_yticklabels([str(i) for i in range(0, matrix.shape[0], 2)])
    ax.set_xlabel("Source")
    ax.set_ylabel("Class")
    ax.set_title("Class-source weights")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "class_source_weight_heatmap.png", dpi=180)
    plt.close(fig)


def plot_embedding(analysis_path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    payload = np.load(analysis_path, allow_pickle=False)
    source_embeddings = payload["source_embeddings"]
    target_embeddings = payload["target_embeddings"]
    if source_embeddings.size == 0 or target_embeddings.size == 0:
        return
    embeddings = np.concatenate([source_embeddings, target_embeddings], axis=0)
    domains = np.concatenate(
        [
            payload["source_domains"].astype(str),
            np.asarray(["target"] * target_embeddings.shape[0]),
        ]
    )
    max_points = min(1500, embeddings.shape[0])
    rng = np.random.default_rng(0)
    indices = rng.choice(embeddings.shape[0], size=max_points, replace=False)
    embedding_sample = embeddings[indices]
    domains_sample = domains[indices]
    perplexity = min(30, max(5, (max_points - 1) // 3))
    projected = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=0).fit_transform(
        embedding_sample
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    for domain in sorted(set(domains_sample.tolist())):
        mask = domains_sample == domain
        ax.scatter(projected[mask, 0], projected[mask, 1], s=10, alpha=0.7, label=domain)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Embedding t-SNE")
    ax.legend(loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_tsne.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_csv)
    plot_method_bar(rows, args.output_dir, args.metric)
    plot_task_heatmap(rows, args.output_dir, args.metric)

    confusion_path = args.confusion_matrix or first_existing(rows, "confusion_matrix_path")
    source_weights = args.source_weights or first_existing(rows, "source_weight_path")
    class_source = args.class_source_weights or first_existing(rows, "class_source_weight_path")
    analysis = args.analysis or first_existing(rows, "analysis_path")
    if confusion_path:
        plot_confusion(confusion_path, args.output_dir)
    if source_weights:
        plot_source_weights(source_weights, args.output_dir)
    if class_source:
        plot_class_source(class_source, args.output_dir)
    if analysis:
        plot_embedding(analysis, args.output_dir)
    print(f"figures: {args.output_dir}")


if __name__ == "__main__":
    main()
