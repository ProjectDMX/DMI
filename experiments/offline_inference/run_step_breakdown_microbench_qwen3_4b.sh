#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_local_env.sh"
offline_e2e_setup_local_env

RESULTS_ROOT="${RESULTS_ROOT:-experiments/offline_inference/results/step_breakdown_qwen3_4b_local}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_ROOT}/qwen3-4b_${RUN_TAG}"
mkdir -p "${RESULTS_DIR}"

for BATCH_SIZE in 1 8 32 64; do
  bash experiments/offline_inference/run_step_breakdown_baseline.sh hf_ideal --model qwen3-4b --local-files-only --batch-size "${BATCH_SIZE}" --prefill-tokens 128 --warmup 2 --iters 5 --results-dir "${RESULTS_DIR}" --baseline-label hf_ideal || true
  bash experiments/offline_inference/run_step_breakdown_baseline.sh hf_api --model qwen3-4b --local-files-only --batch-size "${BATCH_SIZE}" --prefill-tokens 128 --warmup 2 --iters 5 --results-dir "${RESULTS_DIR}" --baseline-label hf_api || true
  bash experiments/offline_inference/run_step_breakdown_baseline.sh torch_hooks --model qwen3-4b --local-files-only --batch-size "${BATCH_SIZE}" --prefill-tokens 128 --warmup 2 --iters 5 --results-dir "${RESULTS_DIR}" --baseline-label torch_hooks --hook-selection "hidden-states,final_ln,logits" || true
  bash experiments/offline_inference/run_step_breakdown_baseline.sh proj_dmi_manual --model qwen3-4b --local-files-only --batch-size "${BATCH_SIZE}" --prefill-tokens 128 --warmup 2 --iters 5 --results-dir "${RESULTS_DIR}" --baseline-label dmi_manual --proj-dmi-mode ring_db --hook-selection "hidden-states,final_ln,logits" --ring-payload-mb 8192 --ring-pinned-mb 8192 --ring-task-entries 65536 || true
  bash experiments/offline_inference/run_step_breakdown_baseline.sh proj_dmi_manual --model qwen3-4b --local-files-only --batch-size "${BATCH_SIZE}" --prefill-tokens 128 --warmup 2 --iters 5 --results-dir "${RESULTS_DIR}" --baseline-label dmi_manual_eager --disable-compile --proj-dmi-mode ring_db --hook-selection "hidden-states,final_ln,logits" --ring-payload-mb 8192 --ring-pinned-mb 8192 --ring-task-entries 65536 || true
done

