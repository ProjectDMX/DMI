#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_runner() {
  local baseline="$1"
  case "${baseline}" in
    hf_ideal) echo "${SCRIPT_DIR}/scripts/run_step_breakdown_hf_ideal.py" ;;
    hf_api) echo "${SCRIPT_DIR}/scripts/run_step_breakdown_hf_api.py" ;;
    torch_hooks) echo "${SCRIPT_DIR}/scripts/run_step_breakdown_torch_hooks.py" ;;
    proj_dmi) echo "${SCRIPT_DIR}/scripts/run_step_breakdown_proj_dmi.py" ;;
    proj_dmi_manual) echo "${SCRIPT_DIR}/scripts/run_step_breakdown_proj_dmi_manual.py" ;;
    proj_dmi_legacy) echo "${SCRIPT_DIR}/scripts/run_step_breakdown_proj_dmi_legacy.py" ;;
    *)
      echo "Unknown step-breakdown baseline: ${baseline}" >&2
      return 1
      ;;
  esac
}

run_step_breakdown_baseline() {
  local baseline="$1"
  shift
  "${PYTHON_BIN}" "$(resolve_runner "${baseline}")" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  baseline="${1:?usage: run_step_breakdown_baseline.sh <baseline> [args...]}"
  shift
  run_step_breakdown_baseline "${baseline}" "$@"
fi
