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


def run_jdot_experiment(prepared_data, method_config: dict[str, Any]) -> dict[str, Any]:
    """Run a shallow JDOT baseline on flattened TEP trajectories."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    source_split = prepared_data.source_splits[0]
    target_split = prepared_data.target_split

    x_source = source_split.train_x.numpy().reshape(source_split.train_x.shape[0], -1)
    y_source = source_split.train_y.numpy()
    x_target_train = target_split.train_x.numpy().reshape(target_split.train_x.shape[0], -1)
    x_target_eval = target_split.eval_x.numpy().reshape(target_split.eval_x.shape[0], -1)
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
    target_predictions = model.predict(x_target_eval)
    return {
        "method_name": "jdot",
        "source_train_acc": float(accuracy_score(y_source, source_predictions)),
        "target_eval_acc": float(accuracy_score(y_target_eval, target_predictions)),
        "source_train_size": int(x_source.shape[0]),
        "target_train_size": int(x_target_train.shape[0]),
        "target_eval_size": int(x_target_eval.shape[0]),
    }
