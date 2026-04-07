#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_local_env.sh"
offline_e2e_setup_local_env

DATASET="${DATASET:-sharegpt}"
RESULTS_DIR="${RESULTS_DIR:-experiments/offline_inference/results/hook_count_qwen3_4b_${DATASET}_$(date '+%Y%m%d_%H%M%S')}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LOCAL_FLAG=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FLAG+=(--local-files-only)
fi

case "${DATASET}" in
  sharegpt) SAMPLE_FILE="benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"; MAX_INPUT=384; MAX_OUTPUT=64 ;;
  wildchat) SAMPLE_FILE="benchmark/data/offline_e2e/wildchat_500_sample1.jsonl"; MAX_INPUT=384; MAX_OUTPUT=64 ;;
  *) echo "ERROR: unsupported DATASET=${DATASET}" >&2; exit 1 ;;
esac

HOOK_SELECTIONS=(
  "logits"
  "q,k,logits"
  "q,k,pattern,attn_scores,logits"
  "q,k,v,z,pattern,attn_scores,resid_mid,logits"
  "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits"
)

for hook_sel in "${HOOK_SELECTIONS[@]}"; do
  echo "=== hook_selection=${hook_sel} ==="
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_upper_bound.py \
    --model qwen3-4b --batch-size 64 --sample-file "${SAMPLE_FILE}" \
    --max-input-tokens "${MAX_INPUT}" --max-new-tokens "${MAX_OUTPUT}" \
    --limit 512 --pad-buckets "128,256,384,512" --capture-mode hs_logits \
    --results-dir "${RESULTS_DIR}" "${LOCAL_FLAG[@]}"
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_torch_hooks.py \
    --model qwen3-4b --batch-size 64 --sample-file "${SAMPLE_FILE}" \
    --max-input-tokens "${MAX_INPUT}" --max-new-tokens "${MAX_OUTPUT}" \
    --limit 512 --pad-buckets "128,256,384,512" --capture-mode hs_logits \
    --results-dir "${RESULTS_DIR}" --disable-compile --hook-selection "${hook_sel}" "${LOCAL_FLAG[@]}"
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_proj_dmi.py \
    --model qwen3-4b --batch-size 64 --sample-file "${SAMPLE_FILE}" \
    --max-input-tokens "${MAX_INPUT}" --max-new-tokens "${MAX_OUTPUT}" \
    --limit 512 --pad-buckets "128,256,384,512" --capture-mode hs_logits \
    --results-dir "${RESULTS_DIR}" --hook-selection "${hook_sel}" \
    --ring-payload-mb 40960 --ring-pinned-mb 40960 --ring-task-entries 131072 \
    --drain-flush-payload-ratio 0.15 --drain-flush-task-ratio 0.15 "${LOCAL_FLAG[@]}" || true
done
