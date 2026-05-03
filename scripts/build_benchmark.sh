#!/usr/bin/env bash
set -euo pipefail

# Build a metadata-first TE benchmark manifest.
# This script does not copy raw arrays. It only creates small metadata artifacts
# such as the inspection report, inspection JSON, and benchmark manifest.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"
CONFIG_PATH="${1:-configs/data/te_da.yaml}"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[build_benchmark.sh] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

resolve_python_runner "build_benchmark.sh"

"${PYTHON_RUNNER[@]}" scripts/build_benchmark.py --config "${CONFIG_PATH}"
