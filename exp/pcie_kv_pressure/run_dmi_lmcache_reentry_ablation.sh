#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HIDDENCACHE_DIR="${HIDDENCACHE_DIR:-$(cd "${REPO_DIR}/.." && pwd)/hiddencache}"
GENERATOR_DIR="${HIDDENCACHE_DIR}/workload_generator"

RUN_ID_BASE="${RUN_ID_BASE:-dmi_lmcache_reentry_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/results/${RUN_ID_BASE}}"
TRACE_DIR="${TRACE_DIR:-${RUN_ROOT}/trace_data}"
CONDITIONS="${CONDITIONS:-no_move,move}"

CONDA_ENV="${CONDA_ENV:-agent-dmi}"
CONDA_SH="${CONDA_SH:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-1}"
KV_OFFLOAD_ENABLED="${KV_OFFLOAD_ENABLED:-true}"
VLLM_KV_OFFLOAD_GB="${VLLM_KV_OFFLOAD_GB:-64}"
LMCACHE_CPU_SIZE_GB="${LMCACHE_CPU_SIZE_GB:-64}"
LMCACHE_LOCAL_DISK_ENABLED="${LMCACHE_LOCAL_DISK_ENABLED:-false}"
LMCACHE_DISK_SIZE_GB="${LMCACHE_DISK_SIZE_GB:-100}"
LMCACHE_DISK_BASE="${LMCACHE_DISK_BASE:-/tmp/lmcache_kv}"
LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-false}"
LMCACHE_ENABLE_ASYNC_LOADING="${LMCACHE_ENABLE_ASYNC_LOADING:-false}"
LMCACHE_CONNECTOR_MODE="${LMCACHE_CONNECTOR_MODE:-inprocess}"
LMCACHE_MP_HOST="${LMCACHE_MP_HOST:-tcp://localhost}"
LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-5555}"
LMCACHE_MP_HTTP_PORT="${LMCACHE_MP_HTTP_PORT:-8081}"
LMCACHE_MP_L1_SIZE_GB="${LMCACHE_MP_L1_SIZE_GB:-8}"
LMCACHE_MP_L1_INIT_SIZE_GB="${LMCACHE_MP_L1_INIT_SIZE_GB:-1}"
LMCACHE_MP_SETTLE_S="${LMCACHE_MP_SETTLE_S:-1}"

case "${LMCACHE_CONNECTOR_MODE}" in
  inprocess|mp) ;;
  *)
    echo "[ablation] unknown LMCACHE_CONNECTOR_MODE=${LMCACHE_CONNECTOR_MODE}" >&2
    exit 2
    ;;
esac

TRACE_MODE="${TRACE_MODE:-reentry}"
TARGET_COUNT="${TARGET_COUNT:-16}"
EVICT_COUNT="${EVICT_COUNT:-64}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-512}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
BUCKET_SPEC="${BUCKET_SPEC:-512:1024:8,1024:2048:6,2048:4096:2}"
WARM_QPS="${WARM_QPS:-2.0}"
EVICT_QPS="${EVICT_QPS:-4.0}"
REPLAY_QPS="${REPLAY_QPS:-1.0}"
REPLAY_MAX_TOKENS="${REPLAY_MAX_TOKENS:-64}"
REPLAY_MAX_WORKERS="${REPLAY_MAX_WORKERS:-64}"
REPLAY_TIMEOUT_S="${REPLAY_TIMEOUT_S:-1200}"
REPLAY_MODE="${REPLAY_MODE:-open-loop}"
REPLAY_ENDPOINT="${REPLAY_ENDPOINT:-chat}"
REPLAY_REQUIRE_NONEMPTY_PHASES="${REPLAY_REQUIRE_NONEMPTY_PHASES:-false}"
INTER_PHASE_SLEEP_S="${INTER_PHASE_SLEEP_S:-5}"

DMI_ENABLED="${DMI_ENABLED:-true}"
DMI_HOOK_SELECTION="${DMI_HOOK_SELECTION:-resid_pre}"
DMI_RING_PAYLOAD_MB="${DMI_RING_PAYLOAD_MB:-2048}"
DMI_RING_PINNED_MB="${DMI_RING_PINNED_MB:-2048}"
# Important: DMX null mode is producer-null, not drain-null. When true, the
# producer kernels return before writing the ring, so there is no D2H traffic.
# Keep it false here and omit dmx_db_host to exercise ring D2H without
# ClickHouse storage.
DMI_NULL_MODE="${DMI_NULL_MODE:-false}"
DMI_DRAIN_FLUSH_TIMEOUT_US="${DMI_DRAIN_FLUSH_TIMEOUT_US:-100000}"

mkdir -p "${RUN_ROOT}"

if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -n "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    echo "[ablation] unable to find conda.sh; set CONDA_SH" >&2
    exit 1
  fi
  set +u
  conda activate "${CONDA_ENV}"
  set -u
fi

export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[ablation] run_root=${RUN_ROOT}"
echo "[ablation] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[ablation] model=${MODEL}"

if [[ ! -s "${TRACE_DIR}/trace.jsonl" ]]; then
  echo "[ablation] building trace in ${TRACE_DIR}"
  BUILD_ARGS=(
    --hiddencache-dir "${HIDDENCACHE_DIR}"
    --mode "${TRACE_MODE}"
    --tokenizer "${TOKENIZER}"
    --out-dir "${TRACE_DIR}"
    --warm-qps "${WARM_QPS}"
    --evict-qps "${EVICT_QPS}"
    --replay-qps "${REPLAY_QPS}"
    --replay-max-tokens "${REPLAY_MAX_TOKENS}"
  )
  if [[ "${TRACE_MODE}" == "length-bucket" ]]; then
    BUILD_ARGS+=(--bucket-spec "${BUCKET_SPEC}" --evict-count "${EVICT_COUNT}")
  else
    BUILD_ARGS+=(
      --target-count "${TARGET_COUNT}"
      --evict-count "${EVICT_COUNT}"
      --min-prompt-tokens "${MIN_PROMPT_TOKENS}"
      --max-prompt-tokens "${MAX_PROMPT_TOKENS}"
    )
  fi
  python "${SCRIPT_DIR}/scripts/build_real_reentry_trace.py" "${BUILD_ARGS[@]}"
fi

PIDS=()
VLLM_PID=""
WATCHER_PID=""
LMCACHE_SERVER_PID=""

stop_monitors() {
  set +e
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done
}

wait_monitors() {
  set +e
  for pid in "${PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  PIDS=()
}

cleanup_server() {
  set +e
  if [[ -n "${WATCHER_PID}" ]]; then
    kill -TERM "${WATCHER_PID}" >/dev/null 2>&1 || true
    wait "${WATCHER_PID}" >/dev/null 2>&1 || true
    WATCHER_PID=""
  fi
  stop_monitors
  wait_monitors
  if [[ -n "${VLLM_PID}" ]]; then
    kill -INT "${VLLM_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 60); do
      kill -0 "${VLLM_PID}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -TERM "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
    VLLM_PID=""
  fi
  if [[ -n "${LMCACHE_SERVER_PID}" ]]; then
    kill -INT "${LMCACHE_SERVER_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      kill -0 "${LMCACHE_SERVER_PID}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -TERM "${LMCACHE_SERVER_PID}" >/dev/null 2>&1 || true
    wait "${LMCACHE_SERVER_PID}" >/dev/null 2>&1 || true
    LMCACHE_SERVER_PID=""
  fi
}
trap cleanup_server EXIT

start_lmcache_mp_server() {
  local run_dir="$1"
  local log_path="${run_dir}/logs/lmcache_server.log"

  (
    cd "${REPO_DIR}"
    exec lmcache server \
      --host localhost \
      --port "${LMCACHE_MP_PORT}" \
      --http-host 127.0.0.1 \
      --http-port "${LMCACHE_MP_HTTP_PORT}" \
      --transfer-mode gpu \
      --l1-size-gb "${LMCACHE_MP_L1_SIZE_GB}" \
      --l1-init-size-gb "${LMCACHE_MP_L1_INIT_SIZE_GB}" \
      --eviction-policy LRU \
      --disable-observability
  ) >"${log_path}" 2>&1 &
  LMCACHE_SERVER_PID=$!
  echo "${LMCACHE_SERVER_PID}" > "${run_dir}/lmcache_server.pid"

  for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${LMCACHE_MP_HTTP_PORT}/healthcheck" \
      >/dev/null 2>&1; then
      echo "[ablation] LMCacheMP server ready after ${i} checks"
      return
    fi
    if ! kill -0 "${LMCACHE_SERVER_PID}" >/dev/null 2>&1; then
      echo "[ablation] LMCacheMP server exited before ready" >&2
      tail -200 "${log_path}" >&2 || true
      return 1
    fi
    sleep 1
  done

  echo "[ablation] LMCacheMP server health timeout" >&2
  tail -200 "${log_path}" >&2 || true
  return 1
}

phase_order_for_condition() {
  case "$1" in
    no_move) echo "warm,replay" ;;
    move) echo "warm,evict,replay" ;;
    *) echo "$1" ;;
  esac
}

start_monitors() {
  local run_dir="$1"
  local log_dir="${run_dir}/logs"
  python "${SCRIPT_DIR}/scripts/monitor_prometheus.py" \
    --run-dir "${run_dir}" --source vllm_metrics --out "${run_dir}/vllm_metrics.jsonl" \
    --interval 1 --url "http://127.0.0.1:${PORT}/metrics" \
    --keep-prefix vllm: --keep-prefix vllm_ --keep-prefix lmcache: --keep-prefix lmcache_ \
    >"${log_dir}/monitor_vllm_metrics.log" 2>&1 &
  PIDS+=("$!")

  python "${SCRIPT_DIR}/scripts/monitor_prometheus.py" \
    --run-dir "${run_dir}" --source lmcache_metrics --out "${run_dir}/lmcache_metrics.jsonl" \
    --interval 1 --url "http://127.0.0.1:6999/metrics" --url "http://127.0.0.1:7000/metrics" \
    --url "http://127.0.0.1:7001/metrics" \
    --keep-prefix lmcache: --keep-prefix lmcache_ \
    >"${log_dir}/monitor_lmcache_metrics.log" 2>&1 &
  PIDS+=("$!")
}

run_condition() {
  local condition="$1"
  local phase_order
  phase_order="$(phase_order_for_condition "${condition}")"
  local run_dir="${RUN_ROOT}/${condition}"
  local log_dir="${run_dir}/logs"
  local lmcache_disk_path=""
  local prom_dir="/tmp/lmcache_prometheus_${RUN_ID_BASE}_${condition}"

  mkdir -p "${log_dir}"
  cp "${TRACE_DIR}/trace.jsonl" "${run_dir}/trace.jsonl"
  cp "${TRACE_DIR}/prefix_bank.jsonl" "${run_dir}/prefix_bank.jsonl"
  cp "${TRACE_DIR}/trace_summary.json" "${run_dir}/trace_summary.json"

  if [[ "${LMCACHE_LOCAL_DISK_ENABLED}" == "true" ]]; then
    for gpu_idx in $(seq 0 $((TENSOR_PARALLEL_SIZE - 1))); do
      local path="${LMCACHE_DISK_BASE}/${RUN_ID_BASE}_${condition}/gpu${gpu_idx}"
      mkdir -p "${path}"
      if [[ -z "${lmcache_disk_path}" ]]; then
        lmcache_disk_path="${path}"
      else
        lmcache_disk_path="${lmcache_disk_path},${path}"
      fi
    done
  fi

  export PROMETHEUS_MULTIPROC_DIR="${prom_dir}"
  export LMCACHE_INTERNAL_API_SERVER_ENABLED=true
  export LMCACHE_INTERNAL_API_SERVER_PORT_START=6999
  export LMCACHE_INTERNAL_API_SERVER_HOST=127.0.0.1
  export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_CPU_SIZE_GB}"
  export LMCACHE_LOCAL_CPU=true
  if [[ "${LMCACHE_LOCAL_DISK_ENABLED}" == "true" ]]; then
    export LMCACHE_LOCAL_DISK="${lmcache_disk_path}"
    export LMCACHE_LOCAL_DISK_PATH_SHARDING=by_gpu
    export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_DISK_SIZE_GB}"
  else
    unset LMCACHE_LOCAL_DISK
    unset LMCACHE_LOCAL_DISK_PATH_SHARDING
    export LMCACHE_MAX_LOCAL_DISK_SIZE=0
  fi
  export LMCACHE_CONFIG_FILE="${run_dir}/lmcache_config.yaml"
  export LMCACHE_KV_WAIT_TRACE="${run_dir}/kv_wait_trace.jsonl"

  rm -rf "${prom_dir}"
  mkdir -p "${prom_dir}"

  {
    cat <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: ${LMCACHE_CPU_SIZE_GB}
EOF
    if [[ "${LMCACHE_LOCAL_DISK_ENABLED}" == "true" ]]; then
      cat <<EOF
local_disk: ${lmcache_disk_path}
local_disk_path_sharding: by_gpu
max_local_disk_size: ${LMCACHE_DISK_SIZE_GB}
EOF
    else
      cat <<EOF
# local_disk:
max_local_disk_size: 0
EOF
    fi
    cat <<EOF
remote_url: null
remote_serde: naive
internal_api_server_enabled: true
internal_api_server_port_start: 6999
internal_api_server_host: 127.0.0.1
use_layerwise: ${LMCACHE_USE_LAYERWISE}
enable_async_loading: ${LMCACHE_ENABLE_ASYNC_LOADING}
EOF
  } > "${LMCACHE_CONFIG_FILE}"

  if [[ "${LMCACHE_CONNECTOR_MODE}" == "mp" ]]; then
    # The MP connector is configured through KVTransferConfig and the server
    # CLI. Do not let an in-process LMCache config leak into either side.
    unset LMCACHE_CONFIG_FILE
    unset LMCACHE_INTERNAL_API_SERVER_ENABLED
    unset LMCACHE_INTERNAL_API_SERVER_PORT_START
    unset LMCACHE_INTERNAL_API_SERVER_HOST
    start_lmcache_mp_server "${run_dir}"
  fi

  local config_path="${TRACE_DIR}/dmi_real_reentry_manifest.json"
  if [[ ! -s "${config_path}" ]]; then
    config_path="${TRACE_DIR}/trace_summary.json"
  fi

  python "${SCRIPT_DIR}/scripts/init_run.py" \
    --run-dir "${run_dir}" --run-id "${RUN_ID_BASE}_${condition}" \
    --mode "dmi-${DMI_ENABLED}-lmcache-reentry-${condition}" \
    --config "${config_path}" \
    --notes "vLLM+LMCache RealReentry ablation condition=${condition} phases=${phase_order} dmi_enabled=${DMI_ENABLED} connector=${LMCACHE_CONNECTOR_MODE}" \
    >"${log_dir}/init_run.log" 2>&1

  python - <<PY >"${run_dir}/experiment_config.json"
import json
print(json.dumps({
  "condition": "${condition}",
  "phase_order": "${phase_order}",
  "model": "${MODEL}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "tensor_parallel_size": int("${TENSOR_PARALLEL_SIZE}"),
  "vllm_max_model_len": int("${VLLM_MAX_MODEL_LEN}"),
  "vllm_gpu_memory_utilization": float("${VLLM_GPU_MEMORY_UTILIZATION}"),
  "vllm_max_num_seqs": int("${VLLM_MAX_NUM_SEQS}"),
  "vllm_max_num_batched_tokens": int("${VLLM_MAX_NUM_BATCHED_TOKENS}") if "${VLLM_MAX_NUM_BATCHED_TOKENS}" else None,
  "vllm_enforce_eager": "${VLLM_ENFORCE_EAGER}".lower() in {"1", "true", "yes", "on"},
  "kv_offload_enabled": "${KV_OFFLOAD_ENABLED}".lower() == "true",
  "vllm_kv_offload_gb": int("${VLLM_KV_OFFLOAD_GB}"),
  "lmcache_cpu_size_gb": int("${LMCACHE_CPU_SIZE_GB}"),
  "lmcache_local_disk_enabled": "${LMCACHE_LOCAL_DISK_ENABLED}".lower() == "true",
  "lmcache_disk_path": "${lmcache_disk_path}",
  "lmcache_use_layerwise": "${LMCACHE_USE_LAYERWISE}".lower() in {"1", "true", "yes", "on"},
  "lmcache_connector_mode": "${LMCACHE_CONNECTOR_MODE}",
  "lmcache_mp_host": "${LMCACHE_MP_HOST}",
  "lmcache_mp_port": int("${LMCACHE_MP_PORT}"),
  "lmcache_mp_http_port": int("${LMCACHE_MP_HTTP_PORT}"),
  "lmcache_mp_l1_size_gb": int("${LMCACHE_MP_L1_SIZE_GB}"),
  "dmi_enabled": "${DMI_ENABLED}".lower() == "true",
  "dmi_hook_selection": "${DMI_HOOK_SELECTION}",
  "dmi_ring_payload_mb": int("${DMI_RING_PAYLOAD_MB}"),
  "dmi_ring_pinned_mb": int("${DMI_RING_PINNED_MB}"),
  "dmi_null_mode": "${DMI_NULL_MODE}",
  "pcie_governor_enabled": "${DMX_PCIE_GOVERNOR_ENABLED:-0}".lower() in {"1", "true", "yes", "on"},
  "pcie_governor_max_chunk_mb": int("${DMX_PCIE_GOVERNOR_MAX_CHUNK_MB:-32}"),
  "pcie_governor_max_defer_us": int("${DMX_PCIE_GOVERNOR_MAX_DEFER_US:-1000000}"),
  "pcie_governor_hard_watermark_ratio": float("${DMX_PCIE_GOVERNOR_HARD_WATERMARK_RATIO:-0.80}"),
}, indent=2, sort_keys=True))
PY

  local additional_config
  additional_config="$(python - <<PY
import json
print(json.dumps({
  "dmx_model_id": "${RUN_ID_BASE}_${condition}",
  "dmx_hook_selection": "${DMI_HOOK_SELECTION}",
  "dmx_ring_payload_mb": int("${DMI_RING_PAYLOAD_MB}"),
  "dmx_ring_pinned_mb": int("${DMI_RING_PINNED_MB}"),
  "dmx_null_mode": "${DMI_NULL_MODE}".lower() == "true",
  "dmx_drain_flush_timeout_us": int("${DMI_DRAIN_FLUSH_TIMEOUT_US}"),
}))
PY
)"

	  local vllm_args=(
	    serve "${MODEL}"
	    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
	    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
	    --port "${PORT}"
	  )
  if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    vllm_args+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
  fi
  if [[ "${VLLM_ENFORCE_EAGER,,}" =~ ^(1|true|yes|on)$ ]]; then
    vllm_args+=(--enforce-eager)
  fi
  if [[ "${KV_OFFLOAD_ENABLED}" == "true" ]]; then
    if [[ "${LMCACHE_CONNECTOR_MODE}" == "mp" ]]; then
      local kv_transfer_config
      kv_transfer_config="$(python - <<PY
import json
print(json.dumps({
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "${LMCACHE_MP_HOST}",
    "lmcache.mp.port": int("${LMCACHE_MP_PORT}"),
  },
}))
PY
)"
      vllm_args+=(
        --kv-transfer-config "${kv_transfer_config}"
        --disable-hybrid-kv-cache-manager
      )
    else
      vllm_args+=(
        --kv-offloading-size "${VLLM_KV_OFFLOAD_GB}"
        --kv-offloading-backend lmcache
        --disable-hybrid-kv-cache-manager
      )
    fi
  fi
  if [[ "${DMI_ENABLED}" == "true" ]]; then
    vllm_args+=(
      --worker-cls integration.vllm_adapter.DMXGPUWorker
      --additional-config "${additional_config}"
    )
  fi
  if [[ "${VLLM_DISABLE_LOG_STATS}" == "1" ]]; then
    vllm_args+=(--disable-log-stats)
  fi

  echo "[ablation] starting condition=${condition} phases=${phase_order}"
  (
    cd "${REPO_DIR}"
    PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" exec vllm "${vllm_args[@]}"
  ) >"${log_dir}/vllm.log" 2>&1 &
  VLLM_PID=$!
  echo "${VLLM_PID}" > "${run_dir}/vllm.pid"

  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "[ablation] condition=${condition} vLLM ready after ${i} checks"
      break
    fi
    if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
      echo "[ablation] condition=${condition} vLLM exited before ready" >&2
      tail -200 "${log_dir}/vllm.log" >&2 || true
      return 1
    fi
    if grep -qE "torch\.OutOfMemoryError|Engine core initialization failed" "${log_dir}/vllm.log"; then
      echo "[ablation] condition=${condition} vLLM initialization failed" >&2
      tail -200 "${log_dir}/vllm.log" >&2 || true
      return 1
    fi
    if [[ "${i}" == "180" ]]; then
      echo "[ablation] condition=${condition} vLLM health timeout" >&2
      tail -200 "${log_dir}/vllm.log" >&2 || true
      return 1
    fi
    sleep 5
  done

  start_monitors "${run_dir}"
  (
    while [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; do
      sleep 2
    done
    stop_monitors
  ) &
  WATCHER_PID=$!

  local replay_extra_args=()
  if [[ "${REPLAY_REQUIRE_NONEMPTY_PHASES,,}" =~ ^(1|true|yes|on)$ ]]; then
    replay_extra_args+=(--require-nonempty-phases)
  fi

	  python "${SCRIPT_DIR}/scripts/replay_phases_with_markers.py" \
    --hiddencache-dir "${HIDDENCACHE_DIR}" \
    --trace "${run_dir}/trace.jsonl" \
    --prefix-bank "${run_dir}/prefix_bank.jsonl" \
    --results "${run_dir}/client_results.jsonl" \
    --markers "${run_dir}/phase_markers.jsonl" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${MODEL}" \
    --mode "${REPLAY_MODE}" \
    --endpoint "${REPLAY_ENDPOINT}" \
    --max-workers "${REPLAY_MAX_WORKERS}" \
    --timeout-s "${REPLAY_TIMEOUT_S}" \
	    --inter-phase-sleep-s "${INTER_PHASE_SLEEP_S}" \
	    --phase-order "${phase_order}" \
	    --ignore-eos \
	    "${replay_extra_args[@]}" \
	    >"${log_dir}/replay_phases.log" 2>&1

  if [[ "${LMCACHE_CONNECTOR_MODE}" == "mp" ]]; then
    sleep "${LMCACHE_MP_SETTLE_S}"
  fi

  stop_monitors
  wait_monitors
  python "${GENERATOR_DIR}/monitoring/summarize_kv_wait_trace.py" \
    --run-dir "${run_dir}" \
    >"${log_dir}/summarize_kv_wait_trace.log" 2>&1 || true
  cleanup_server
}

IFS=',' read -ra CONDITION_LIST <<< "${CONDITIONS}"
for condition in "${CONDITION_LIST[@]}"; do
  run_condition "${condition}"
done

python "${SCRIPT_DIR}/scripts/summarize_ablation.py" --run-root "${RUN_ROOT}"
echo "[ablation] completed ${RUN_ROOT}"
