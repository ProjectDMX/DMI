#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-pcie_kv_pressure_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/results/${RUN_ID}}"
LOG_DIR="${RUN_DIR}/logs"
CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/synthetic_smoke.json}"

DRY_RUN="${DRY_RUN:-}"
GPUS="${GPUS:-}"
DIRECTION="${DIRECTION:-}"
BLOCK_MB="${BLOCK_MB:-}"
INFLIGHT="${INFLIGHT:-}"
DURATION_S="${DURATION_S:-}"
TARGET_MB_S="${TARGET_MB_S:-}"
SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-}"

METRICS_INTERVAL_S="${METRICS_INTERVAL_S:-1.0}"
VLLM_METRICS_URL="${VLLM_METRICS_URL:-}"
LMCACHE_METRICS_URLS="${LMCACHE_METRICS_URLS:-}"
DMI_METRICS_URLS="${DMI_METRICS_URLS:-}"

PRE_HOOK_CMD="${PRE_HOOK_CMD:-}"
SERVE_CMD="${SERVE_CMD:-}"
SERVE_HEALTH_URL="${SERVE_HEALTH_URL:-}"
SERVE_HEALTH_TIMEOUT_S="${SERVE_HEALTH_TIMEOUT_S:-300}"
EXTRA_CORUN_CMD="${EXTRA_CORUN_CMD:-}"
POST_HOOK_CMD="${POST_HOOK_CMD:-}"

mkdir -p "${LOG_DIR}"
export RUN_ID RUN_DIR LOG_DIR SCRIPT_DIR REPO_DIR

PIDS=()
SERVE_PID=""
EXTRA_PID=""

cleanup() {
  set +e
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -TERM "${pid}" >/dev/null 2>&1
    fi
  done
  if [[ -n "${EXTRA_PID}" ]] && kill -0 "${EXTRA_PID}" >/dev/null 2>&1; then
    kill -TERM "${EXTRA_PID}" >/dev/null 2>&1
  fi
  if [[ -n "${SERVE_PID}" ]] && kill -0 "${SERVE_PID}" >/dev/null 2>&1; then
    kill -INT "${SERVE_PID}" >/dev/null 2>&1
    for _ in $(seq 1 30); do
      kill -0 "${SERVE_PID}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -0 "${SERVE_PID}" >/dev/null 2>&1 && kill -TERM "${SERVE_PID}" >/dev/null 2>&1
  fi
  for pid in "${PIDS[@]:-}" "${EXTRA_PID:-}" "${SERVE_PID:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

echo "[pcie-pressure] run_id=${RUN_ID}"
echo "[pcie-pressure] run_dir=${RUN_DIR}"
echo "[pcie-pressure] config=${CONFIG}"

python "${SCRIPT_DIR}/scripts/init_run.py" \
  --run-dir "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --config "${CONFIG}" \
  --notes "KV on/offload PCIe pressure scaffold run" \
  >"${LOG_DIR}/init_run.log" 2>&1

if [[ -n "${PRE_HOOK_CMD}" ]]; then
  echo "[pcie-pressure] running PRE_HOOK_CMD"
  bash -lc "${PRE_HOOK_CMD}" >"${LOG_DIR}/pre_hook.log" 2>&1
fi

if [[ -n "${SERVE_CMD}" ]]; then
  echo "[pcie-pressure] starting SERVE_CMD"
  bash -lc "${SERVE_CMD}" >"${LOG_DIR}/serve.log" 2>&1 &
  SERVE_PID=$!
  echo "${SERVE_PID}" >"${RUN_DIR}/serve.pid"
fi

if [[ -n "${SERVE_HEALTH_URL}" ]]; then
  echo "[pcie-pressure] waiting for ${SERVE_HEALTH_URL}"
  deadline=$((SECONDS + SERVE_HEALTH_TIMEOUT_S))
  until curl -fsS "${SERVE_HEALTH_URL}" >/dev/null 2>&1; do
    if [[ -n "${SERVE_PID}" ]] && ! kill -0 "${SERVE_PID}" >/dev/null 2>&1; then
      echo "[pcie-pressure] SERVE_CMD exited before health check passed" >&2
      tail -200 "${LOG_DIR}/serve.log" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[pcie-pressure] health check timed out: ${SERVE_HEALTH_URL}" >&2
      tail -200 "${LOG_DIR}/serve.log" >&2 || true
      exit 1
    fi
    sleep 2
  done
fi

if [[ -n "${EXTRA_CORUN_CMD}" ]]; then
  echo "[pcie-pressure] starting EXTRA_CORUN_CMD"
  bash -lc "${EXTRA_CORUN_CMD}" >"${LOG_DIR}/extra_corun.log" 2>&1 &
  EXTRA_PID=$!
  echo "${EXTRA_PID}" >"${RUN_DIR}/extra_corun.pid"
fi

start_prom_monitor() {
  local source="$1"
  local urls_csv="$2"
  local out="$3"
  shift 3
  [[ -z "${urls_csv}" ]] && return 0
  local args=()
  IFS=',' read -ra urls <<< "${urls_csv}"
  for url in "${urls[@]}"; do
    [[ -n "${url}" ]] && args+=(--url "${url}")
  done
  [[ "${#args[@]}" -eq 0 ]] && return 0
  python "${SCRIPT_DIR}/scripts/monitor_prometheus.py" \
    --run-dir "${RUN_DIR}" \
    --source "${source}" \
    --out "${RUN_DIR}/${out}" \
    --interval "${METRICS_INTERVAL_S}" \
    "${args[@]}" \
    "$@" \
    >"${LOG_DIR}/monitor_${source}.log" 2>&1 &
  PIDS+=("$!")
}

start_prom_monitor "vllm_metrics" "${VLLM_METRICS_URL}" "vllm_metrics.jsonl" --keep-prefix vllm: --keep-prefix vllm_ --keep-prefix lmcache: --keep-prefix lmcache_
start_prom_monitor "lmcache_metrics" "${LMCACHE_METRICS_URLS}" "lmcache_metrics.jsonl" --keep-prefix lmcache: --keep-prefix lmcache_
start_prom_monitor "dmi_metrics" "${DMI_METRICS_URLS}" "dmi_metrics.jsonl" --keep-prefix dmi --keep-prefix dmx --keep-prefix DMI --keep-prefix DMX

HOG_ARGS=(--run-dir "${RUN_DIR}" --config "${CONFIG}")
[[ -n "${DRY_RUN}" ]] && HOG_ARGS+=(--dry-run "${DRY_RUN}")
[[ -n "${GPUS}" ]] && HOG_ARGS+=(--gpus "${GPUS}")
[[ -n "${DIRECTION}" ]] && HOG_ARGS+=(--direction "${DIRECTION}")
[[ -n "${BLOCK_MB}" ]] && HOG_ARGS+=(--block-mb "${BLOCK_MB}")
[[ -n "${INFLIGHT}" ]] && HOG_ARGS+=(--inflight "${INFLIGHT}")
[[ -n "${DURATION_S}" ]] && HOG_ARGS+=(--duration-s "${DURATION_S}")
[[ -n "${TARGET_MB_S}" ]] && HOG_ARGS+=(--target-mb-s "${TARGET_MB_S}")
[[ -n "${SAMPLE_INTERVAL_S}" ]] && HOG_ARGS+=(--sample-interval-s "${SAMPLE_INTERVAL_S}")

echo "[pcie-pressure] starting synthetic pressure"
python "${SCRIPT_DIR}/scripts/synthetic_pcie_hog.py" "${HOG_ARGS[@]}" \
  >"${LOG_DIR}/synthetic_pcie_hog.log" 2>&1

echo "[pcie-pressure] stopping monitors"
for pid in "${PIDS[@]:-}"; do
  kill -TERM "${pid}" >/dev/null 2>&1 || true
done
for pid in "${PIDS[@]:-}"; do
  wait "${pid}" >/dev/null 2>&1 || true
done
PIDS=()

if [[ -n "${POST_HOOK_CMD}" ]]; then
  echo "[pcie-pressure] running POST_HOOK_CMD"
  bash -lc "${POST_HOOK_CMD}" >"${LOG_DIR}/post_hook.log" 2>&1
fi

python "${SCRIPT_DIR}/scripts/summarize_run.py" \
  --run-dir "${RUN_DIR}" \
  >"${LOG_DIR}/summarize_run.log" 2>&1

echo "[pcie-pressure] completed ${RUN_DIR}"
