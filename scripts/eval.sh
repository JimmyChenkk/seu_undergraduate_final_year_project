#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"

RESULTS_DIR="runs"
if [[ $# -ge 1 && "$1" != --* ]]; then
  RESULTS_DIR="$1"
  shift
fi

cd "${ROOT_DIR}"
resolve_python_runner "eval.sh"

"${PYTHON_RUNNER[@]}" -m src.evaluation.evaluate --results-dir "${RESULTS_DIR}" "$@"
