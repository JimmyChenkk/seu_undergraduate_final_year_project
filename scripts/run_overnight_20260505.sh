#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-overnight_20260505_$(date +%H%M%S)}"
ROUNDS="${ROUNDS:-3}"
PLAN_ONLY="${PLAN_ONLY:-0}"

DEFAULT_SEEDS=(42 43 44)
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<<"${SEEDS}"
else
  SEED_LIST=("${DEFAULT_SEEDS[@]}")
fi

SINGLE_CONFIG="configs/experiment/tep_ot_single_source_8methods_stage1_fold0_overnight_20260505.yaml"
MULTI_CONFIG="configs/experiment/tep_ot_multisource_ca_ccsr_wjdot_stage1_probe_fold0.yaml"
MULTI_5SOURCE_CONFIG="configs/experiment/tep_ot_multisource_5source_prior20_overnight_20260505.yaml"

TWO_SOURCE_SCENES=(
  "mode1+mode2->mode5"
  "mode2+mode5->mode1"
  "mode1+mode5->mode2"
)

run_batch() {
  local label="$1"
  local seed="$2"
  shift 2

  local batch_root="${label}_${RUN_TAG}_seed${seed}"
  local command=(
    bash scripts/run_small_scale_round.sh
    "$@"
    --seed "${seed}"
    --batch-root-name "${batch_root}"
  )

  if [[ "${PLAN_ONLY}" == "1" ]]; then
    command+=(--plan-only)
  fi

  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "${command[*]}"
  "${command[@]}"
}

for ((round_index = 1; round_index <= ROUNDS; round_index++)); do
  seed_index=$((round_index - 1))
  if (( seed_index < ${#SEED_LIST[@]} )); then
    seed="${SEED_LIST[seed_index]}"
  else
    seed="$((42 + seed_index))"
  fi

  round_tag="r${round_index}"
  run_batch "single_source_48_${round_tag}" "${seed}" \
    --experiment-config "${SINGLE_CONFIG}"

  run_batch "multisource_15_${round_tag}" "${seed}" \
    --experiment-config "${MULTI_CONFIG}" \
    --scenes "${TWO_SOURCE_SCENES[@]}"

  run_batch "multisource_30_${round_tag}" "${seed}" \
    --experiment-config "${MULTI_5SOURCE_CONFIG}"
done
