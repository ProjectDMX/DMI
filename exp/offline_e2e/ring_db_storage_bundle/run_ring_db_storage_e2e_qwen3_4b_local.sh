#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/nengneng/miniconda3/envs/proj-dmx/bin/python}"
GPU="${GPU:-1}"
MODEL="${MODEL:-qwen3-4b}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREFILL_TOKENS="${PREFILL_TOKENS:-128}"
DECODE_STEPS="${DECODE_STEPS:-500}"
WARMUP="${WARMUP:-1}"
NUM_BATCHES="${NUM_BATCHES:-10}"
STORAGE_LABEL="${STORAGE_LABEL:-disk}"
PROJ_DMI_MODE="${PROJ_DMI_MODE:-ring_db}"
RESULTS_ROOT="${RESULTS_ROOT:-exp/offline_e2e/results/ring_db_storage_e2e}"
HOOK_SELECTION="${HOOK_SELECTION:-hidden-states,final_ln,logits}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:---local-files-only}"

export CUDA_VISIBLE_DEVICES="${GPU}"

"${PYTHON_BIN}" exp/offline_e2e/run_ring_db_storage_e2e.py \
  --model "${MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --prefill-tokens "${PREFILL_TOKENS}" \
  --decode-steps "${DECODE_STEPS}" \
  --warmup "${WARMUP}" \
  --num-batches "${NUM_BATCHES}" \
  --storage-label "${STORAGE_LABEL}" \
  --proj-dmi-mode "${PROJ_DMI_MODE}" \
  --results-dir "${RESULTS_ROOT}" \
  --hook-selection "${HOOK_SELECTION}" \
  ${LOCAL_FILES_ONLY}
