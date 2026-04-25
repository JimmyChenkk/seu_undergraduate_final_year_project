#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT_DIR}/scripts/run_small_scale_round.sh" \
  --experiment-config "${ROOT_DIR}/configs/experiment/rcta_mode125_ablation_fixedfold.yaml" \
  "$@"
