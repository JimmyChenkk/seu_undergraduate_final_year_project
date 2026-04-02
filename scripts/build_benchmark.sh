#!/usr/bin/env bash
set -euo pipefail

# Build a metadata-first TE benchmark manifest.
# This script does not copy raw arrays. It only creates small metadata artifacts
# such as the inspection report, inspection JSON, and benchmark manifest.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-configs/data/te_da.yaml}"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[build_benchmark.sh] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "tep_env" ]]; then
  PYTHON_BIN="python"
elif command -v conda >/dev/null 2>&1; then
  exec conda run -n tep_env python scripts/build_benchmark.py --config "${CONFIG_PATH}"
else
  echo "[build_benchmark.sh] tep_env is not active and conda is unavailable." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/build_benchmark.py --config "${CONFIG_PATH}"
