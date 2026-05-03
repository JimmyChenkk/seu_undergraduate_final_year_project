"""Batch runner for the fixed TEP DA task grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from run_experiment import run_from_args
from src.tep_ot.utils import timestamp


DEFAULT_TASKS = [
    "1->2",
    "2->1",
    "1->5",
    "5->1",
    "2->5",
    "5->2",
    "1+2->5",
    "2+5->1",
    "1+5->2",
]

DEFAULT_METHODS = [
    "source_only",
    "target_only",
    "mmd",
    "coral",
    "dann",
    "jdot",
    "wjdot",
    "tp_wjdot",
    "cbtp_wjdot",
    "ms_cbtp_wjdot",
]


def parse_task(task: str) -> tuple[list[str], str]:
    left, right = task.split("->", maxsplit=1)
    return left.split("+"), right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed 9-task TEP DA grid.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tep_ot_batch"))
    parser.add_argument("--normalization-scope", choices=["domain", "train"], default="domain")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--ot-solver", choices=["sinkhorn", "emd"], default="sinkhorn")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--prototype-weight", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_root = args.output_dir / timestamp()
    aggregate_csv = batch_root / "all_results.csv"
    for seed in args.seeds:
        for fold in args.folds:
            for task in args.tasks:
                sources, target = parse_task(task)
                for method in args.methods:
                    run_args = SimpleNamespace(
                        method=method,
                        sources=sources,
                        target=target,
                        fold=fold,
                        seed=seed,
                        raw_dir=args.raw_dir,
                        output_dir=batch_root,
                        aggregate_csv=aggregate_csv,
                        normalization_scope=args.normalization_scope,
                        epochs=args.epochs,
                        pretrain_epochs=args.pretrain_epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        device=args.device,
                        num_workers=args.num_workers,
                        max_train_batches=args.max_train_batches,
                        eval_max_batches=args.eval_max_batches,
                        embedding_dim=128,
                        dropout=0.1,
                        adaptation_weight=1.0,
                        ot_solver=args.ot_solver,
                        sinkhorn_reg=args.sinkhorn_reg,
                        prototype_weight=args.prototype_weight,
                    )
                    print(f"[batch] method={method} task={task} fold={fold} seed={seed}", flush=True)
                    run_from_args(run_args)
    print(f"[batch] results: {aggregate_csv}")


if __name__ == "__main__":
    main()
