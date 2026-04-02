#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-runs/tables}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m src.evaluation.evaluate --results-dir "${RESULTS_DIR}"
