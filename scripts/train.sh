#!/usr/bin/env bash
set -euo pipefail

DATA_CONFIG="configs/data/te_da.yaml"
METHOD_CONFIG="configs/method/source_only.yaml"
EXPERIMENT_CONFIG="configs/experiment/benchmark_single_source.yaml"

if [[ $# -ge 1 && "$1" != --* ]]; then
  DATA_CONFIG="$1"
  shift
fi
if [[ $# -ge 1 && "$1" != --* ]]; then
  METHOD_CONFIG="$1"
  shift
fi
if [[ $# -ge 1 && "$1" != --* ]]; then
  EXPERIMENT_CONFIG="$1"
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m src.trainers.train_benchmark \
  --data-config "${DATA_CONFIG}" \
  --method-config "${METHOD_CONFIG}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  "$@"
