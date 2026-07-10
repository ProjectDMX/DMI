#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXPERIMENT_ID="${EXPERIMENT_ID:-pcie_scheduler_overhead_$(date +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${SCRIPT_DIR}/results/${EXPERIMENT_ID}}"
TRACE_DIR="${TRACE_DIR:-${EXPERIMENT_ROOT}/trace_data}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
PORT="${PORT:-8000}"
CONDA_ENV="${CONDA_ENV:-agent-dmi}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TARGET_COUNT="${TARGET_COUNT:-64}"
PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-128}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
WARM_QPS="${WARM_QPS:-16}"
REPLAY_QPS="${REPLAY_QPS:-16}"

mkdir -p "${EXPERIMENT_ROOT}/runs"

run_trial() {
  local repetition="$1"
  local mode="$2"
  local enabled=0
  if [[ "${mode}" == "on" ]]; then
    enabled=1
  fi

  local run_id="${EXPERIMENT_ID}_${mode}_rep${repetition}"
  local run_root="${EXPERIMENT_ROOT}/runs/${mode}_rep${repetition}"
  if [[ -s "${run_root}/ablation_summary.json" ]]; then
    echo "[scheduler-overhead] skip completed ${mode} rep${repetition}"
    return
  fi

  echo "[scheduler-overhead] start ${mode} rep${repetition}"
  RUN_ID_BASE="${run_id}" \
  RUN_ROOT="${run_root}" \
  TRACE_DIR="${TRACE_DIR}" \
  CONDITIONS=no_move \
  CONDA_ENV="${CONDA_ENV}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  TENSOR_PARALLEL_SIZE=1 \
  PORT="${PORT}" \
  MODEL="${MODEL}" \
  TOKENIZER="${MODEL}" \
  VLLM_MAX_MODEL_LEN=1024 \
  VLLM_GPU_MEMORY_UTILIZATION=0.45 \
  VLLM_MAX_NUM_SEQS=16 \
  KV_OFFLOAD_ENABLED=false \
  LMCACHE_CPU_SIZE_GB=1 \
  LMCACHE_LOCAL_DISK_ENABLED=false \
  TRACE_MODE=reentry \
  TARGET_COUNT="${TARGET_COUNT}" \
  EVICT_COUNT=16 \
  MIN_PROMPT_TOKENS="${PROMPT_MIN_TOKENS}" \
  MAX_PROMPT_TOKENS="${PROMPT_MAX_TOKENS}" \
  WARM_QPS="${WARM_QPS}" \
  EVICT_QPS="${WARM_QPS}" \
  REPLAY_QPS="${REPLAY_QPS}" \
  REPLAY_MAX_TOKENS="${OUTPUT_TOKENS}" \
  REPLAY_MAX_WORKERS=64 \
  REPLAY_MODE=open-loop \
  INTER_PHASE_SLEEP_S=2 \
  DMI_ENABLED=true \
  DMI_HOOK_SELECTION=resid_pre \
  DMI_RING_PAYLOAD_MB=256 \
  DMI_RING_PINNED_MB=256 \
  DMI_NULL_MODE=false \
  DMI_DRAIN_FLUSH_TIMEOUT_US=100000 \
  DMX_PCIE_GOVERNOR_ENABLED="${enabled}" \
  DMX_PCIE_GOVERNOR_DEBUG=0 \
  DMX_PCIE_GOVERNOR_MAX_CHUNK_MB=32 \
  DMX_PCIE_GOVERNOR_MAX_DEFER_US=1000000 \
  bash "${SCRIPT_DIR}/run_dmi_lmcache_reentry_ablation.sh"
}

# Alternate order to reduce thermal and time-of-run bias.
run_trial 1 off
run_trial 1 on
run_trial 2 on
run_trial 2 off
run_trial 3 off
run_trial 3 on

python "${SCRIPT_DIR}/scripts/summarize_scheduler_overhead.py" \
  --root "${EXPERIMENT_ROOT}" \
  --out "${EXPERIMENT_ROOT}/scheduler_overhead_summary.json"

echo "[scheduler-overhead] completed ${EXPERIMENT_ROOT}"
