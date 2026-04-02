#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"

cd "${ROOT_DIR}"
resolve_python_runner "run_small_scale_round.sh"

"${PYTHON_RUNNER[@]}" -m src.automation.run_small_scale_round "$@"
