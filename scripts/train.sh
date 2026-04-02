#!/usr/bin/env bash
set -euo pipefail

DATA_CONFIG="${1:-configs/data/te_da.yaml}"
METHOD_CONFIG="${2:-configs/method/source_only.yaml}"
EXPERIMENT_CONFIG="${3:-configs/experiment/benchmark_single_source.yaml}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m src.trainers.train_benchmark \
  --data-config "${DATA_CONFIG}" \
  --method-config "${METHOD_CONFIG}" \
  --experiment-config "${EXPERIMENT_CONFIG}"
