#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/_local_env.sh"
offline_e2e_setup_local_env

RESULTS_DIR="${RESULTS_DIR:-experiments/offline_inference/results/prefill_backpressure_fixed_$(date '+%Y%m%d_%H%M%S')}"
COMMON=(
  --model qwen3-4b
  --local-files-only
  --batch-size 64
  --prefill-tokens 64
  --num-microbatches 32
  --results-dir "${RESULTS_DIR}"
)

"${PYTHON_BIN}" experiments/offline_inference/scripts/run_prefill_backpressure.py "${COMMON[@]}" --baseline hf_native --baseline-label hf_native
"${PYTHON_BIN}" experiments/offline_inference/scripts/run_prefill_backpressure.py "${COMMON[@]}" --baseline torch_hooks --baseline-label torch_hooks --hook-selection "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits"
"${PYTHON_BIN}" experiments/offline_inference/scripts/run_prefill_backpressure.py "${COMMON[@]}" --baseline proj_dmi --baseline-label dmi_light --proj-dmi-mode ring_db --hook-selection "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits" --ring-payload-mb 61440 --ring-pinned-mb 61440 --ring-task-entries 65536 --drain-flush-payload-ratio 0.15 --drain-flush-task-ratio 0.15
"${PYTHON_BIN}" experiments/offline_inference/scripts/run_prefill_backpressure.py "${COMMON[@]}" --baseline proj_dmi --baseline-label dmi_heavy --proj-dmi-mode ring_db --hook-selection "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits" --ring-payload-mb 16384 --ring-pinned-mb 16384 --ring-task-entries 65536 --drain-flush-payload-ratio 0.15 --drain-flush-task-ratio 0.15
