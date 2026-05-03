#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"

DATA_CONFIG="configs/data/te_da.yaml"
METHOD_CONFIG="configs/method/source_only.yaml"
EXPERIMENT_CONFIG="configs/experiment/quick_debug.yaml"

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

cd "${ROOT_DIR}"
resolve_python_runner "train.sh"

"${PYTHON_RUNNER[@]}" -m src.trainers.train_benchmark \
  --data-config "${DATA_CONFIG}" \
  --method-config "${METHOD_CONFIG}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  "$@"
