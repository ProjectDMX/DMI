#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-pcie_scheduler_effectiveness_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/results/${RUN_ID}}"
TRACE_DIR="${TRACE_DIR:-${RUN_ROOT}/trace_data}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-formal}"
case "${BENCHMARK_PROFILE}" in
  formal)
    TRACE_CONFIG="${TRACE_CONFIG:-${SCRIPT_DIR}/configs/kv_store_isolated_bursts.json}"
    REPETITIONS="${REPETITIONS:-3}"
    WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
    MIN_MEASURED_REQUESTS="${MIN_MEASURED_REQUESTS:-32}"
    DEFAULT_DMI_HOOK_SELECTION="resid_pre,ln1,attn_out,resid_mid,ln2,mlp_in,mlp_out"
    DEFAULT_DMI_RING_PAYLOAD_MB=4096
    DEFAULT_DMI_RING_PINNED_MB=2048
    DEFAULT_GOVERNOR_HARD_WATERMARK_RATIO=0.95
    ;;
  smoke)
    TRACE_CONFIG="${TRACE_CONFIG:-${SCRIPT_DIR}/configs/kv_store_smoke.json}"
    REPETITIONS="${REPETITIONS:-1}"
    WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
    MIN_MEASURED_REQUESTS="${MIN_MEASURED_REQUESTS:-1}"
    DEFAULT_DMI_HOOK_SELECTION="resid_pre"
    DEFAULT_DMI_RING_PAYLOAD_MB=1024
    DEFAULT_DMI_RING_PINNED_MB=1024
    DEFAULT_GOVERNOR_HARD_WATERMARK_RATIO=0.80
    ;;
  *)
    echo "unknown BENCHMARK_PROFILE=${BENCHMARK_PROFILE}; expected formal or smoke" >&2
    exit 2
    ;;
esac
TRIAL_CONDITIONS="${TRIAL_CONDITIONS:-dmi_off,gov_off,gov_on}"
REQUIRE_GOVERNOR_ENGAGED="${REQUIRE_GOVERNOR_ENGAGED:-1}"
GPU_INDEX="${GPU_INDEX:-2}"
PORT="${PORT:-8000}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
CONDA_ENV="${CONDA_ENV:-agent-dmi}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"

DMI_RING_PAYLOAD_MB="${DMI_RING_PAYLOAD_MB:-${DEFAULT_DMI_RING_PAYLOAD_MB}}"
DMI_RING_PINNED_MB="${DMI_RING_PINNED_MB:-${DEFAULT_DMI_RING_PINNED_MB}}"
DMI_HOOK_SELECTION="${DMI_HOOK_SELECTION:-${DEFAULT_DMI_HOOK_SELECTION}}"
DMI_DRAIN_FLUSH_TIMEOUT_US="${DMI_DRAIN_FLUSH_TIMEOUT_US:-1000}"
GOVERNOR_MAX_CHUNK_MB="${GOVERNOR_MAX_CHUNK_MB:-32}"
GOVERNOR_MAX_DEFER_US="${GOVERNOR_MAX_DEFER_US:-1000000}"
GOVERNOR_HARD_WATERMARK_RATIO="${GOVERNOR_HARD_WATERMARK_RATIO:-${DEFAULT_GOVERNOR_HARD_WATERMARK_RATIO}}"
LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-false}"
LMCACHE_CONNECTOR_MODE="${LMCACHE_CONNECTOR_MODE:-inprocess}"
LMCACHE_MP_HOST="${LMCACHE_MP_HOST:-tcp://localhost}"
LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-5555}"
LMCACHE_MP_HTTP_PORT="${LMCACHE_MP_HTTP_PORT:-8081}"
LMCACHE_MP_L1_SIZE_GB="${LMCACHE_MP_L1_SIZE_GB:-8}"
LMCACHE_MP_L1_INIT_SIZE_GB="${LMCACHE_MP_L1_INIT_SIZE_GB:-1}"
HINT_AUDIT="${HINT_AUDIT:-1}"

mkdir -p "${RUN_ROOT}/runs" "${TRACE_DIR}"

if [[ ! -s "${TRACE_DIR}/trace.jsonl" ]]; then
  python "${SCRIPT_DIR}/scripts/generate_kv_pressure_trace.py" \
    --config "${TRACE_CONFIG}" \
    --out-dir "${TRACE_DIR}"
fi

echo "[scheduler-effectiveness] run_root=${RUN_ROOT}"
echo "[scheduler-effectiveness] profile=${BENCHMARK_PROFILE} gpu=${GPU_INDEX} repetitions=${REPETITIONS} warmup=${WARMUP_REQUESTS}"

run_trial() {
  local condition="$1"
  local repetition="$2"
  local dmi_enabled=true
  local governor_enabled=0
  local governor_debug=0

  case "${condition}" in
    dmi_off)
      dmi_enabled=false
      ;;
    gov_off)
      ;;
    gov_on)
      governor_enabled=1
      governor_debug=1
      ;;
    *)
      echo "unknown condition: ${condition}" >&2
      return 2
      ;;
  esac

  local trial_name="${condition}_rep${repetition}"
  local trial_root="${RUN_ROOT}/runs/${trial_name}"
  echo "[scheduler-effectiveness] starting ${trial_name}"

  RUN_ID_BASE="${RUN_ID}_${trial_name}" \
  RUN_ROOT="${trial_root}" \
  TRACE_DIR="${TRACE_DIR}" \
  CONDITIONS=cold_store \
  CONDA_ENV="${CONDA_ENV}" \
  CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
  TENSOR_PARALLEL_SIZE=1 \
  PORT="${PORT}" \
  MODEL="${MODEL}" \
  TOKENIZER="${MODEL}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  VLLM_ENFORCE_EAGER=true \
  KV_OFFLOAD_ENABLED=true \
  VLLM_KV_OFFLOAD_GB=32 \
  LMCACHE_CPU_SIZE_GB=48 \
  LMCACHE_LOCAL_DISK_ENABLED=false \
  LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE}" \
  LMCACHE_CONNECTOR_MODE="${LMCACHE_CONNECTOR_MODE}" \
  LMCACHE_MP_HOST="${LMCACHE_MP_HOST}" \
  LMCACHE_MP_PORT="${LMCACHE_MP_PORT}" \
  LMCACHE_MP_HTTP_PORT="${LMCACHE_MP_HTTP_PORT}" \
  LMCACHE_MP_L1_SIZE_GB="${LMCACHE_MP_L1_SIZE_GB}" \
  LMCACHE_MP_L1_INIT_SIZE_GB="${LMCACHE_MP_L1_INIT_SIZE_GB}" \
  REPLAY_MODE=open-loop \
  REPLAY_ENDPOINT=completions \
  REPLAY_MAX_WORKERS=32 \
  REPLAY_TIMEOUT_S=1200 \
  REPLAY_REQUIRE_NONEMPTY_PHASES=true \
  INTER_PHASE_SLEEP_S=0 \
  DMI_ENABLED="${dmi_enabled}" \
  DMI_HOOK_SELECTION="${DMI_HOOK_SELECTION}" \
  DMI_RING_PAYLOAD_MB="${DMI_RING_PAYLOAD_MB}" \
  DMI_RING_PINNED_MB="${DMI_RING_PINNED_MB}" \
  DMI_NULL_MODE=false \
  DMI_DRAIN_FLUSH_TIMEOUT_US="${DMI_DRAIN_FLUSH_TIMEOUT_US}" \
  DMX_PCIE_GOVERNOR_ENABLED="${governor_enabled}" \
  DMX_PCIE_GOVERNOR_DEBUG="${governor_debug}" \
  DMX_PCIE_GOVERNOR_MAX_CHUNK_MB="${GOVERNOR_MAX_CHUNK_MB}" \
  DMX_PCIE_GOVERNOR_MAX_DEFER_US="${GOVERNOR_MAX_DEFER_US}" \
  DMX_PCIE_GOVERNOR_HARD_WATERMARK_RATIO="${GOVERNOR_HARD_WATERMARK_RATIO}" \
  DMX_PCIE_HINT_AUDIT="${HINT_AUDIT}" \
  bash "${SCRIPT_DIR}/run_dmi_lmcache_reentry_ablation.sh"
}

IFS=',' read -ra base_order <<< "${TRIAL_CONDITIONS}"
for repetition in $(seq 1 "${REPETITIONS}"); do
  order=()
  offset=$(( (repetition - 1) % ${#base_order[@]} ))
  for ((i=0; i<${#base_order[@]}; i++)); do
    order+=("${base_order[(i + offset) % ${#base_order[@]}]}")
  done
  for condition in "${order[@]}"; do
    run_trial "${condition}" "${repetition}"
  done
done

SUMMARY_ARGS=(
  --run-root "${RUN_ROOT}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --min-measured-requests "${MIN_MEASURED_REQUESTS}"
)
if [[ "${REQUIRE_GOVERNOR_ENGAGED,,}" =~ ^(1|true|yes|on)$ ]]; then
  SUMMARY_ARGS+=(--require-governor-engaged)
fi
if [[ "${BENCHMARK_PROFILE}" == "formal" ]]; then
  SUMMARY_ARGS+=(--min-repetitions 3)
fi
python "${SCRIPT_DIR}/scripts/summarize_scheduler_effectiveness.py" "${SUMMARY_ARGS[@]}"

echo "[scheduler-effectiveness] completed ${RUN_ROOT}"
