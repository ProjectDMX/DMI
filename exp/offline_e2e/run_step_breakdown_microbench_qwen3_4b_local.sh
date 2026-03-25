#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/nengneng/miniconda3/envs/proj-dmx/bin/python}"
GPU="${GPU:-1}"
MODEL="${MODEL:-qwen3-4b}"
RESULTS_DIR="${RESULTS_DIR:-exp/offline_e2e/results/step_breakdown_qwen3_4b_local}"
PREFILL_TOKENS="${PREFILL_TOKENS:-128}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
HOOK_SELECTION="${HOOK_SELECTION:-hidden-states,final_ln,logits}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:---local-files-only}"
RING_PAYLOAD_MB="${RING_PAYLOAD_MB:-1024}"
RING_PINNED_MB="${RING_PINNED_MB:-1024}"
RING_TASK_ENTRIES="${RING_TASK_ENTRIES:-65536}"

export CUDA_VISIBLE_DEVICES="${GPU}"

run_one() {
  local baseline="$1"
  local batch_size="$2"
  echo "============================================================"
  echo "baseline=${baseline} bs=${batch_size}"
  echo "============================================================"
  "${PYTHON_BIN}" exp/offline_e2e/run_step_breakdown_microbench.py \
    --baseline "${baseline}" \
    --model "${MODEL}" \
    --batch-size "${batch_size}" \
    --prefill-tokens "${PREFILL_TOKENS}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --results-dir "${RESULTS_DIR}" \
    ${LOCAL_FILES_ONLY} \
    --hook-selection "${HOOK_SELECTION}" \
    --proj-dmi-mode ring_db \
    --ring-payload-mb "${RING_PAYLOAD_MB}" \
    --ring-pinned-mb "${RING_PINNED_MB}" \
    --ring-task-entries "${RING_TASK_ENTRIES}" \
    --drain-flush-task-ratio 0.15 \
    --drain-flush-payload-ratio 0.15 \
    --db-host localhost \
    --db-port 9000 \
    --db-user default \
    --db-database default \
    --db-table offload
}

for bs in 1 8 32 64; do
  run_one hf_ideal "${bs}"
  run_one hf_api "${bs}"
  run_one torch_hooks "${bs}"
  run_one proj_dmi "${bs}"
done
