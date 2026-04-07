#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_local_env.sh"
offline_e2e_setup_local_env

MODEL="${MODEL:-qwen3-4b}"
BS="${BS:-32}"
SAMPLE_FILE="${SAMPLE_FILE:-benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl}"
MAX_INPUT="${MAX_INPUT:-200}"
MAX_OUTPUT="${MAX_OUTPUT:-750}"
LIMIT="${LIMIT:-50}"
RING_MB="${RING_MB:-56320}"
FLUSH="${FLUSH:-0.15}"
PAD_BUCKETS="${PAD_BUCKETS:-64,128,256,512}"
CAPTURE_MODE="${CAPTURE_MODE:-hs}"
RESULTS_DIR="${RESULTS_DIR:-experiments/offline_inference/results}"
LOCAL_ONLY_FLAG=()
if [[ "${LOCAL_FILES_ONLY:-1}" == "1" ]]; then
  LOCAL_ONLY_FLAG+=(--local-files-only)
fi

COMMON=(
  --model "${MODEL}"
  --batch-size "${BS}"
  --sample-file "${SAMPLE_FILE}"
  --max-input-tokens "${MAX_INPUT}"
  --max-new-tokens "${MAX_OUTPUT}"
  --limit "${LIMIT}"
  --pad-buckets "${PAD_BUCKETS}"
  --capture-mode "${CAPTURE_MODE}"
  --results-dir "${RESULTS_DIR}"
  "${LOCAL_ONLY_FLAG[@]}"
)

declare -a RESULTS=()

cleanup_gpu() {
  echo "--- GPU cleanup ---"
  "${PYTHON_BIN}" -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
  sleep 5
}

run_baseline() {
  local label="$1"
  shift
  echo ""
  echo "=== ${label} ==="
  local output
  if ! output=$("$@" 2>&1); then
    echo "${output}"
    echo "ERROR: baseline failed: ${label}" >&2
    return 1
  fi
  echo "${output}"
  local target_toks compute_toks
  target_toks=$(echo "${output}" | grep -oP 'target_tok/s=\K[0-9.]+' | tail -1 || true)
  compute_toks=$(echo "${output}" | grep -oP 'compute_tok/s=\K[0-9.]+' | tail -1 || true)
  RESULTS+=("${label}|${target_toks}|${compute_toks}")
  cleanup_gpu
}

run_baseline "hf_upper_bound (compile)" \
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_upper_bound.py "${COMMON[@]}"
run_baseline "hf_upper_bound (eager)" \
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_upper_bound.py "${COMMON[@]}" --disable-compile
run_baseline "hf_monitor (generate, eager)" \
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_monitor.py "${COMMON[@]}"
run_baseline "hf_monitor_manual (compile)" \
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_hf_monitor_manual.py "${COMMON[@]}"
run_baseline "proj_dmi (compile) ${RING_MB}MB" \
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_proj_dmi.py "${COMMON[@]}" \
    --ring-payload-mb "${RING_MB}" --ring-pinned-mb "${RING_MB}" \
    --ring-task-entries 131072 \
    --drain-flush-payload-ratio "${FLUSH}" --drain-flush-task-ratio "${FLUSH}"
run_baseline "proj_dmi (eager) ${RING_MB}MB" \
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_proj_dmi.py "${COMMON[@]}" --disable-compile \
    --ring-payload-mb "${RING_MB}" --ring-pinned-mb "${RING_MB}" \
    --ring-task-entries 131072 \
    --drain-flush-payload-ratio "${FLUSH}" --drain-flush-task-ratio "${FLUSH}"
run_baseline "nnsight (eager)" \
  "${PYTHON_BIN}" experiments/offline_inference/scripts/run_nnsight.py "${COMMON[@]}"

echo ""
echo "Completed run_baselines.sh"

