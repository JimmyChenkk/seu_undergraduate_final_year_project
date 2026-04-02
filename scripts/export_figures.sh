#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="runs"
if [[ $# -ge 1 && "$1" != --* ]]; then
  RESULTS_DIR="$1"
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

ARGS=()
if [[ $# -ge 2 && "$1" == "--output-dir" ]]; then
  ARGS+=("--output-dir" "$2")
  shift 2
fi

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
  "${ARGS[@]}"
