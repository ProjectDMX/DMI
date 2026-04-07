#!/usr/bin/env bash
set -euo pipefail

OFFLINE_E2E_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_E2E_PROJECT_ROOT="$(cd -- "${OFFLINE_E2E_SCRIPT_DIR}/../../.." && pwd)"

offline_e2e_setup_local_env() {
  cd "${OFFLINE_E2E_PROJECT_ROOT}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export PYTHONPATH="${OFFLINE_E2E_PROJECT_ROOT}/transformers/src:${OFFLINE_E2E_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export TL_ENABLE_NVTX="${TL_ENABLE_NVTX:-1}"
  export PYTHON_BIN="${PYTHON_BIN:-python}"
}
