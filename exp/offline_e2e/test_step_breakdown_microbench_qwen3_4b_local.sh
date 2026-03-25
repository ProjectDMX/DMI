#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/nengneng/miniconda3/envs/proj-dmx/bin/python}"
GPU="${GPU:-1}"
MODEL="${MODEL:-qwen3-4b}"
RESULTS_DIR="${RESULTS_DIR:-exp/offline_e2e/results/step_breakdown_qwen3_4b_local_smoke}"
HOOK_SELECTION="${HOOK_SELECTION:-hidden-states,final_ln,logits}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:---local-files-only}"

export CUDA_VISIBLE_DEVICES="${GPU}"

run_one() {
  local baseline="$1"
  echo "============================================================"
  echo "smoke baseline=${baseline}"
  echo "============================================================"
  "${PYTHON_BIN}" exp/offline_e2e/run_step_breakdown_microbench.py \
    --baseline "${baseline}" \
    --model "${MODEL}" \
    --batch-size 1 \
    --prefill-tokens 128 \
    --warmup 0 \
    --iters 1 \
    --results-dir "${RESULTS_DIR}" \
    ${LOCAL_FILES_ONLY} \
    --hook-selection "${HOOK_SELECTION}" \
  --proj-dmi-mode ring_null \
    --ring-payload-mb 10240 \
    --ring-pinned-mb 10240 \
    --drain-flush-task-ratio 0.0 \
    --drain-flush-payload-ratio 0.0 \
    --drain-flush-timeout-us 1000 \
    --ch-parallelism 1 \
    --ch-queue-max-items 128 \
    --ch-queue-max-size-mb 128 \
    --db-host localhost \
    --db-port 9000 \
    --db-user default \
    --db-database default \
    --db-table offload
}

for baseline in hf_ideal hf_api torch_hooks proj_dmi proj_dmi_legacy; do
  run_one "${baseline}"
done
