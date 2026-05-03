#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: GPU_GUARD_LOG_DIR=runs/<batch> scripts/run_with_gpu_guard.sh -- <command> [args...]" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu_guard] nvidia-smi not found; refusing to start an unmonitored GPU run." >&2
  exit 127
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "[gpu_guard] timeout not found; refusing to start without bounded GPU probes." >&2
  exit 127
fi

LOG_DIR="${GPU_GUARD_LOG_DIR:-runs/gpu_guard_$(date +%Y%m%d_%H%M%S)}"
INTERVAL_SECONDS="${GPU_GUARD_INTERVAL_SECONDS:-2}"
PROBE_TIMEOUT_SECONDS="${GPU_GUARD_PROBE_TIMEOUT_SECONDS:-5}"
TEMP_WARN_C="${GPU_GUARD_TEMP_WARN_C:-80}"
TEMP_STOP_C="${GPU_GUARD_TEMP_STOP_C:-84}"
MEM_STOP_MB="${GPU_GUARD_MEM_STOP_MB:-23500}"
POWER_STOP_W="${GPU_GUARD_POWER_STOP_W:-}"
QUERY_FAIL_STOP_COUNT="${GPU_GUARD_QUERY_FAIL_STOP_COUNT:-2}"
KILL_GRACE_SECONDS="${GPU_GUARD_KILL_GRACE_SECONDS:-20}"

mkdir -p "${LOG_DIR}"
GPU_CSV="${LOG_DIR}/gpu_monitor.csv"
GUARD_LOG="${LOG_DIR}/gpu_guard.log"
COMMAND_LOG="${LOG_DIR}/command.log"
STOP_REASON="${LOG_DIR}/stop_reason.txt"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

is_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

ge_number() {
  awk -v lhs="$1" -v rhs="$2" 'BEGIN { exit !((lhs + 0) >= (rhs + 0)) }'
}

log_guard() {
  local message="$1"
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "${message}" | tee -a "${GUARD_LOG}" >&2
}

stop_child_group() {
  local reason="$1"
  if [[ -f "${STOP_REASON}" ]]; then
    return 0
  fi
  printf '%s\n' "${reason}" >"${STOP_REASON}"
  log_guard "STOP: ${reason}"
  if kill -0 "${CHILD_PID}" >/dev/null 2>&1; then
    kill -TERM -- "-${CHILD_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 "${KILL_GRACE_SECONDS}"); do
      if ! kill -0 "${CHILD_PID}" >/dev/null 2>&1; then
        return 0
      fi
      sleep 1
    done
    log_guard "Child process group still alive after ${KILL_GRACE_SECONDS}s; sending KILL."
    kill -KILL -- "-${CHILD_PID}" >/dev/null 2>&1 || true
  fi
}

cleanup_on_signal() {
  stop_child_group "gpu_guard received a signal"
}

trap cleanup_on_signal INT TERM

{
  printf 'host_time,nvidia_timestamp,gpu_index,temp_c,util_pct,mem_used_mb,mem_total_mb,power_w,pstate,clock_graphics_mhz,clock_mem_mhz,throttle_active\n'
} >"${GPU_CSV}"

log_guard "Log directory: ${LOG_DIR}"
log_guard "Thresholds: temp_warn=${TEMP_WARN_C}C temp_stop=${TEMP_STOP_C}C mem_stop=${MEM_STOP_MB}MB power_stop=${POWER_STOP_W:-disabled}W interval=${INTERVAL_SECONDS}s"
log_guard "Command: $*"

if ! command -v setsid >/dev/null 2>&1; then
  log_guard "setsid not found; cannot isolate the child process group."
  exit 127
fi

setsid "$@" > >(tee -a "${COMMAND_LOG}") 2> >(tee -a "${COMMAND_LOG}" >&2) &
CHILD_PID=$!
QUERY_FAIL_COUNT=0

while kill -0 "${CHILD_PID}" >/dev/null 2>&1; do
  HOST_TIME="$(date --iso-8601=seconds)"
  QUERY_OUTPUT="$(
    timeout "${PROBE_TIMEOUT_SECONDS}s" nvidia-smi \
      --query-gpu=timestamp,index,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,pstate,clocks.gr,clocks.mem,clocks_throttle_reasons.active \
      --format=csv,noheader,nounits 2>&1
  )" || {
    QUERY_FAIL_COUNT=$((QUERY_FAIL_COUNT + 1))
    log_guard "nvidia-smi query failed (${QUERY_FAIL_COUNT}/${QUERY_FAIL_STOP_COUNT}): ${QUERY_OUTPUT}"
    if (( QUERY_FAIL_COUNT >= QUERY_FAIL_STOP_COUNT )); then
      stop_child_group "nvidia-smi failed ${QUERY_FAIL_COUNT} consecutive times"
      break
    fi
    sleep "${INTERVAL_SECONDS}"
    continue
  }

  QUERY_FAIL_COUNT=0
  while IFS=',' read -r NVIDIA_TIME GPU_INDEX TEMP_C UTIL_PCT MEM_USED_MB MEM_TOTAL_MB POWER_W PSTATE CLOCK_GR CLOCK_MEM THROTTLE_ACTIVE; do
    NVIDIA_TIME="$(trim "${NVIDIA_TIME}")"
    GPU_INDEX="$(trim "${GPU_INDEX}")"
    TEMP_C="$(trim "${TEMP_C}")"
    UTIL_PCT="$(trim "${UTIL_PCT}")"
    MEM_USED_MB="$(trim "${MEM_USED_MB}")"
    MEM_TOTAL_MB="$(trim "${MEM_TOTAL_MB}")"
    POWER_W="$(trim "${POWER_W}")"
    PSTATE="$(trim "${PSTATE}")"
    CLOCK_GR="$(trim "${CLOCK_GR}")"
    CLOCK_MEM="$(trim "${CLOCK_MEM}")"
    THROTTLE_ACTIVE="$(trim "${THROTTLE_ACTIVE}")"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${HOST_TIME}" "${NVIDIA_TIME}" "${GPU_INDEX}" "${TEMP_C}" "${UTIL_PCT}" \
      "${MEM_USED_MB}" "${MEM_TOTAL_MB}" "${POWER_W}" "${PSTATE}" \
      "${CLOCK_GR}" "${CLOCK_MEM}" "${THROTTLE_ACTIVE}" >>"${GPU_CSV}"

    if is_number "${TEMP_C}" && ge_number "${TEMP_C}" "${TEMP_STOP_C}"; then
      stop_child_group "GPU ${GPU_INDEX} temperature reached ${TEMP_C}C >= ${TEMP_STOP_C}C"
      break
    fi
    if is_number "${TEMP_C}" && ge_number "${TEMP_C}" "${TEMP_WARN_C}"; then
      log_guard "WARN: GPU ${GPU_INDEX} temperature ${TEMP_C}C >= ${TEMP_WARN_C}C"
    fi
    if is_number "${MEM_USED_MB}" && ge_number "${MEM_USED_MB}" "${MEM_STOP_MB}"; then
      stop_child_group "GPU ${GPU_INDEX} memory reached ${MEM_USED_MB}MB >= ${MEM_STOP_MB}MB"
      break
    fi
    if [[ -n "${POWER_STOP_W}" ]] && is_number "${POWER_W}" && ge_number "${POWER_W}" "${POWER_STOP_W}"; then
      stop_child_group "GPU ${GPU_INDEX} power reached ${POWER_W}W >= ${POWER_STOP_W}W"
      break
    fi
  done <<<"${QUERY_OUTPUT}"

  if [[ -f "${STOP_REASON}" ]]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done

set +e
wait "${CHILD_PID}"
CHILD_STATUS=$?
set -e

if [[ -f "${STOP_REASON}" ]]; then
  log_guard "Child stopped by gpu_guard; original wait status=${CHILD_STATUS}."
  exit 86
fi

log_guard "Child finished with status ${CHILD_STATUS}."
exit "${CHILD_STATUS}"
