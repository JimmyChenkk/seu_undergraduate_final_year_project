"""JDOT wrapper around skada for a shallow OT-style baseline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np


def _import_skada_jdot():
    """Import JDOTClassifier from an installed package or the local reference repo."""

    try:
        from skada import JDOTClassifier  # type: ignore
        return JDOTClassifier
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        local_repo = repo_root / "external" / "skada"
        if local_repo.exists():
            sys.path.insert(0, str(local_repo))
            from skada import JDOTClassifier  # type: ignore
            return JDOTClassifier
        raise


def _flatten_tensor_batch(tensor) -> np.ndarray:
    return tensor.numpy().reshape(tensor.shape[0], -1)


def _save_jdot_analysis_artifacts(
    *,
    model,
    prepared_data,
    analysis_path: Path,
    scenario_id: str,
    method_name: str,
) -> None:
    source_split = prepared_data.source_splits[0]
    target_split = prepared_data.target_split

    x_source_eval = _flatten_tensor_batch(source_split.eval_x)
    x_target_eval = _flatten_tensor_batch(target_split.eval_x)

    source_probabilities = model.predict_proba(x_source_eval)
    target_probabilities = model.predict_proba(x_target_eval)
    np.savez_compressed(
        analysis_path,
        scenario_id=np.array([scenario_id], dtype=object),
        method_name=np.array([method_name], dtype=object),
        source_embeddings=x_source_eval.astype(np.float32),
        source_labels=source_split.eval_y.numpy(),
        source_predictions=source_probabilities.argmax(axis=1),
        source_domains=np.array([source_split.domain_id] * len(x_source_eval), dtype=object),
        target_embeddings=x_target_eval.astype(np.float32),
        target_labels=target_split.eval_y.numpy(),
        target_predictions=target_probabilities.argmax(axis=1),
        target_domains=np.array([target_split.domain_id] * len(x_target_eval), dtype=object),
        target_logits=target_probabilities.astype(np.float32),
    )


def run_jdot_experiment(
    prepared_data,
    method_config: dict[str, Any],
    *,
    analysis_path: Path | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Run a shallow JDOT baseline on flattened TEP trajectories."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    source_split = prepared_data.source_splits[0]
    target_split = prepared_data.target_split

    x_source = _flatten_tensor_batch(source_split.train_x)
    y_source = source_split.train_y.numpy()
    x_source_eval = _flatten_tensor_batch(source_split.eval_x)
    y_source_eval = source_split.eval_y.numpy()
    x_target_train = _flatten_tensor_batch(target_split.train_x)
    x_target_eval = _flatten_tensor_batch(target_split.eval_x)
    y_target_eval = target_split.eval_y.numpy()

    x_joint = np.concatenate([x_source, x_target_train], axis=0)
    y_joint = np.concatenate([y_source, np.full(shape=x_target_train.shape[0], fill_value=-1, dtype=y_source.dtype)])
    sample_domain = np.concatenate(
        [
            np.full(shape=x_source.shape[0], fill_value=1, dtype=np.int64),
            np.full(shape=x_target_train.shape[0], fill_value=-2, dtype=np.int64),
        ]
    )

    jdot_kwargs = method_config.get("jdot", {})
    estimator = LogisticRegression(
        max_iter=int(jdot_kwargs.get("logreg_max_iter", 200)),
        multi_class="auto",
    )
    JDOTClassifier = _import_skada_jdot()
    model = JDOTClassifier(
        base_estimator=estimator,
        alpha=float(jdot_kwargs.get("alpha", 0.5)),
        n_iter_max=int(jdot_kwargs.get("n_iter_max", 10)),
        verbose=bool(jdot_kwargs.get("verbose", False)),
    )
    model.fit(x_joint, y_joint, sample_domain=sample_domain)

    source_predictions = model.predict(x_source)
    source_eval_predictions = model.predict(x_source_eval)
    target_predictions = model.predict(x_target_eval)
    source_train_acc = float(accuracy_score(y_source, source_predictions))
    source_eval_acc = float(accuracy_score(y_source_eval, source_eval_predictions))
    target_eval_acc = float(accuracy_score(y_target_eval, target_predictions))
    target_eval_balanced_acc = float(balanced_accuracy_score(y_target_eval, target_predictions))
    target_confusion = confusion_matrix(
        y_target_eval,
        target_predictions,
        labels=np.arange(29),
    )

    if analysis_path is not None and scenario_id is not None:
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        _save_jdot_analysis_artifacts(
            model=model,
            prepared_data=prepared_data,
            analysis_path=analysis_path,
            scenario_id=scenario_id,
            method_name="jdot",
        )

    history = [
        {
            "epoch": 1,
            "acc_source_train": source_train_acc,
            "acc_source_eval": source_eval_acc,
            "target_eval_acc": target_eval_acc,
        }
    ]
    return {
        "method_name": "jdot",
        "history": history,
        "source_train_acc": source_train_acc,
        "source_eval_acc": source_eval_acc,
        "final_source_train_acc": source_train_acc,
        "best_source_train_acc": source_train_acc,
        "final_source_eval_acc": source_eval_acc,
        "best_source_eval_acc": source_eval_acc,
        "target_eval_acc": target_eval_acc,
        "final_target_eval_acc": target_eval_acc,
        "best_target_eval_acc": target_eval_acc,
        "target_eval_balanced_acc": target_eval_balanced_acc,
        "target_confusion_matrix": target_confusion.tolist(),
        "analysis_path": str(analysis_path) if analysis_path is not None else None,
        "source_train_size": int(x_source.shape[0]),
        "source_eval_size": int(x_source_eval.shape[0]),
        "target_train_size": int(x_target_train.shape[0]),
        "target_eval_size": int(x_target_eval.shape[0]),
    }
