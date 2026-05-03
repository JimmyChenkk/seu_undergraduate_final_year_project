"""Command-line entry for one TEP domain-adaptation experiment.

Example:
    python run_experiment.py --method source_only --sources 1 --target 2 --fold 0 --seed 0
    python run_experiment.py --method ms_cbtp_wjdot --sources 1 5 --target 2 --fold 0 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tep_ot.data import TEPDomainLoader
from src.tep_ot.train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TEP DA experiment.")
    parser.add_argument(
        "--method",
        required=True,
        help=(
            "source_only, target_only, mmd, coral, dann, jdot, tp_jdot, "
            "cbtp_jdot, wjdot, tp_wjdot, cbtp_wjdot, ms_cbtp_wjdot"
        ),
    )
    parser.add_argument("--sources", nargs="+", required=True, help="Source domain ids, e.g. 1 5.")
    parser.add_argument("--target", required=True, help="Target domain id, e.g. 2.")
    parser.add_argument("--fold", default=0, help="Zero-based eval fold; 0 means Fold 1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tep_ot"))
    parser.add_argument("--aggregate-csv", type=Path, default=None)
    parser.add_argument("--normalization-scope", choices=["domain", "train"], default="domain")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test limiter.")
    parser.add_argument("--eval-max-batches", type=int, default=None, help="Smoke-test limiter.")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adaptation-weight", type=float, default=1.0)
    parser.add_argument("--ot-solver", choices=["sinkhorn", "emd"], default="sinkhorn")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--prototype-weight", type=float, default=None)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> dict:
    fold = int(args.fold)
    loader = TEPDomainLoader(args.raw_dir, normalization_scope=args.normalization_scope)
    data = loader.load_experiment(args.sources, args.target, fold)
    return run_experiment(
        data=data,
        method_name=args.method,
        fold=fold,
        seed=args.seed,
        output_dir=args.output_dir,
        aggregate_csv=args.aggregate_csv,
        device_name=args.device,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        eval_max_batches=args.eval_max_batches,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        adaptation_weight=args.adaptation_weight,
        ot_solver=args.ot_solver,
        sinkhorn_reg=args.sinkhorn_reg,
        prototype_weight=args.prototype_weight,
    )


def main() -> None:
    result = run_from_args(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
