#!/usr/bin/env bash

resolve_python_runner() {
  local script_name="${1:-script}"

  if [[ "${CONDA_DEFAULT_ENV:-}" == "tep_env" ]]; then
    local python_bin="${PYTHON_BIN:-python}"
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
      python_bin="python3"
    fi
    PYTHON_RUNNER=("${python_bin}")
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    PYTHON_RUNNER=(conda run -n tep_env python)
    return 0
  fi

  echo "[${script_name}] tep_env is not active and conda is unavailable." >&2
  return 1
}
