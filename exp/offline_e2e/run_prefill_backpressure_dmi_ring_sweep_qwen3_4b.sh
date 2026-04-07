#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_local_env.sh"
offline_e2e_setup_local_env

RESULTS_DIR="${RESULTS_DIR:-exp/offline_e2e/results/prefill_backpressure_dmi_ring_sweep_$(date '+%Y%m%d_%H%M%S')}"
COMMON=(
  --model qwen3-4b
  --local-files-only
  --batch-size 64
  --prefill-tokens 64
  --num-microbatches 50
  --results-dir "${RESULTS_DIR}"
)

for ring_mb in 10240 20480 30720 40960 51200; do
  "${PYTHON_BIN}" exp/offline_e2e/scripts/run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline proj_dmi --baseline-label "dmi_light_ring${ring_mb}" --proj-dmi-mode ring_db \
    --hook-selection "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits" \
    --ring-payload-mb "${ring_mb}" --ring-pinned-mb "${ring_mb}" --ring-task-entries 65536 \
    --drain-flush-payload-ratio 0 --drain-flush-task-ratio 0 --drain-flush-timeout-us 200 || true
  "${PYTHON_BIN}" exp/offline_e2e/scripts/run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline proj_dmi --baseline-label "dmi_heavy_ring${ring_mb}" --proj-dmi-mode ring_db \
    --hook-selection "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits" \
    --ring-payload-mb "${ring_mb}" --ring-pinned-mb "${ring_mb}" --ring-task-entries 65536 \
    --drain-flush-payload-ratio 0 --drain-flush-task-ratio 0 --drain-flush-timeout-us 200 || true
done

for label_and_hooks in \
  "dmi_hs_logits_ring61440:hidden-states,final_ln,logits" \
  "dmi_73_ring61440:q,k,logits" \
  "dmi_145_ring61440:q,k,attn_scores,logits"; do
  label="${label_and_hooks%%:*}"
  hooks="${label_and_hooks#*:}"
  "${PYTHON_BIN}" exp/offline_e2e/scripts/run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline proj_dmi --baseline-label "${label}" --proj-dmi-mode ring_db \
    --hook-selection "${hooks}" \
    --ring-payload-mb 61440 --ring-pinned-mb 61440 --ring-task-entries 65536 \
    --drain-flush-payload-ratio 0 --drain-flush-task-ratio 0 --drain-flush-timeout-us 200 || true
done

