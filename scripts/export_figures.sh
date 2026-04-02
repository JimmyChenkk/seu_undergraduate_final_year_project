#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-runs/tables}"
OUTPUT_DIR="${2:-runs/figures}"
shift 2 || true

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

ARGS=()
POSITIONAL=("$@")
INDEX=0
while [[ ${INDEX} -lt ${#POSITIONAL[@]} ]]; do
  ITEM="${POSITIONAL[${INDEX}]}"
  if [[ "${ITEM}" == "--compare" ]]; then
    LEFT_INDEX=$((INDEX + 1))
    RIGHT_INDEX=$((INDEX + 2))
    if [[ ${RIGHT_INDEX} -ge ${#POSITIONAL[@]} ]]; then
      echo "export_figures.sh: --compare requires two artifact paths" >&2
      exit 1
    fi
    ARGS+=(
      "--compare-domain-artifacts"
      "${POSITIONAL[${LEFT_INDEX}]}"
      "${POSITIONAL[${RIGHT_INDEX}]}"
    )
    INDEX=$((INDEX + 3))
    continue
  fi

  ARGS+=("--artifact" "${ITEM}")
  INDEX=$((INDEX + 1))
done

"${PYTHON_BIN}" -m src.evaluation.report_figures \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "${ARGS[@]}"
