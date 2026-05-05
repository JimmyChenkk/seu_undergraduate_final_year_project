#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"

cd "${ROOT_DIR}"
resolve_python_runner "export_ccsr_wjdot_posthoc.sh"

"${PYTHON_RUNNER[@]}" -m src.evaluation.ccsr_wjdot_posthoc "$@"
