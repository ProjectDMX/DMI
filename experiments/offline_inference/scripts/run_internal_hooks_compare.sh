#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="qwen3-1.7b"
DATASETS="sharegpt"
SAMPLE_IDS="1"
BASELINES="hf_upper_bound,torch_hooks,nnsight,proj_dmi"
BATCH_SIZES="16"
REPEATS=1
MAX_NEW_TOKENS=0
LIMIT=0
RESULTS_DIR="${SCRIPT_DIR}/results"
LOCAL_ONLY=0
NO_SORT=0
PAD_TO_MULTIPLE_OF=0
PAD_BUCKETS=""
MAX_INPUT_TOKENS=0
SAMPLE_SIZE=500
PROJ_DMI_MODE="ring_null"
PROJ_DMI_COMPILE=1
PROJ_DMI_RETRY=0
RING_STEP_MB=5120
RING_MIN_MB=5120
HOOK_SELECTION="q,k,v,z,mlp_in,mlp_out,resid_mid"
EXTRA_ARGS=()
PROJ_DMI_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --sample-ids) SAMPLE_IDS="$2"; shift 2 ;;
    --baselines) BASELINES="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --local-files-only) LOCAL_ONLY=1; shift ;;
    --no-sort-by-length) NO_SORT=1; shift ;;
    --pad-to-multiple-of) PAD_TO_MULTIPLE_OF="$2"; shift 2 ;;
    --pad-buckets) PAD_BUCKETS="$2"; shift 2 ;;
    --max-input-tokens) MAX_INPUT_TOKENS="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --proj-dmi-mode) PROJ_DMI_MODE="$2"; shift 2 ;;
    --proj-dmi-compile) PROJ_DMI_COMPILE=1; shift ;;
    --proj-dmi-retry) PROJ_DMI_RETRY=1; shift ;;
    --ring-step-mb) RING_STEP_MB="$2"; shift 2 ;;
    --ring-min-mb) RING_MIN_MB="$2"; shift 2 ;;
    --hook-selection) HOOK_SELECTION="$2"; shift 2 ;;
    --ring-task-entries|--ring-payload-mb|--ring-pinned-mb|--drain-poll-timeout-us|--drain-flush-task-ratio|--drain-flush-payload-ratio|--drain-flush-entry-threshold|--drain-flush-byte-threshold|--drain-flush-timeout-us|--ch-parallelism|--ch-queue-max-items|--ch-queue-max-size-mb|--db-host|--db-port|--db-user|--db-password|--db-database|--db-table)
      PROJ_DMI_ARGS+=("$1" "$2"); shift 2 ;;
    --clone-slices)
      PROJ_DMI_ARGS+=("$1"); shift ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

cleanup_gpu() {
  echo "--- GPU cleanup ---"
  python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
  sleep 5
}

is_oom_log() {
  local log_file="$1"
  grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM|oom_kill|Killed' "${log_file}"
}

estimate_proj_dmi_ring_mb() {
  local model="$1"
  local bs="$2"
  local max_input="$3"
  local max_output="$4"
  local total_mb=81920
  local base_reserve
  case "${model}" in
    qwen3-4b) base_reserve=11264 ;;
    llama3.1-8b) base_reserve=19456 ;;
    qwen3-14b) base_reserve=24576 ;;
    *) base_reserve=24576 ;;
  esac
  local batch_penalty
  case "${bs}" in
    16) batch_penalty=2048 ;;
    32) batch_penalty=4096 ;;
    64) batch_penalty=8192 ;;
    *) batch_penalty=6144 ;;
  esac
  local seq_penalty=2048
  local total_seq=$((max_input + max_output))
  if [[ "${total_seq}" -ge 1200 ]]; then
    seq_penalty=5120
  elif [[ "${total_seq}" -ge 900 ]]; then
    seq_penalty=3072
  fi
  local slack_mb=4096
  local ring_mb=$(( total_mb - base_reserve - batch_penalty - seq_penalty - slack_mb ))
  if [[ "${ring_mb}" -gt 56320 ]]; then
    ring_mb=56320
  fi
  if [[ "${ring_mb}" -lt "${RING_MIN_MB}" ]]; then
    ring_mb="${RING_MIN_MB}"
  fi
  ring_mb=$(( (ring_mb / RING_STEP_MB) * RING_STEP_MB ))
  if [[ "${ring_mb}" -lt "${RING_MIN_MB}" ]]; then
    ring_mb="${RING_MIN_MB}"
  fi
  echo "${ring_mb}"
}

run_proj_dmi_with_retries() {
  local model="$1"
  local dataset="$2"
  local batch_size="$3"
  local repeat_index="$4"
  local sample_file="$5"
  local max_input="$6"
  local max_output="$7"
  shift 7

  local ring_mb
  ring_mb=$(estimate_proj_dmi_ring_mb "${model}" "${batch_size}" "${max_input}" "${max_output}")
  local log_dir="${RUN_RESULTS_DIR}/logs"
  mkdir -p "${log_dir}"

  while [[ "${ring_mb}" -ge "${RING_MIN_MB}" ]]; do
    local log_file="${log_dir}/${model}__${dataset}__bs${batch_size}__rep${repeat_index}__ring${ring_mb}.log"
    echo "=== baseline=proj_dmi dataset=${dataset} sample=${sample_id} bs=${batch_size} repeat=${repeat_index} hooks=${HOOK_SELECTION} ring=${ring_mb}MB ==="
    local proj_args=()
    if [[ "${PROJ_DMI_COMPILE}" != "1" ]]; then
      proj_args+=(--disable-compile)
    fi
    if python experiments/offline_inference/scripts/run_proj_dmi.py \
      --sample-file "${sample_file}" \
      --batch-size "${batch_size}" \
      --repeat-index "${repeat_index}" \
      --proj-dmi-mode "${PROJ_DMI_MODE}" \
      --hook-selection "${HOOK_SELECTION}" \
      --ring-payload-mb "${ring_mb}" \
      --ring-pinned-mb "${ring_mb}" \
      "${proj_args[@]}" \
      "${shared_args[@]}" \
      "${PROJ_DMI_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1; then
      cat "${log_file}"
      cleanup_gpu
      return 0
    fi
    cat "${log_file}"
    if is_oom_log "${log_file}"; then
      ring_mb=$((ring_mb - RING_STEP_MB))
      cleanup_gpu
      continue
    fi
    cleanup_gpu
    return 1
  done
  return 1
}

mkdir -p "${RESULTS_DIR}"

HOOK_TAG="${HOOK_SELECTION//,/__}"
HOOK_TAG="${HOOK_TAG// /}"
RUN_RESULTS_DIR="${RESULTS_DIR}/internal_hooks_${HOOK_TAG}"
mkdir -p "${RUN_RESULTS_DIR}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
IFS=',' read -r -a SAMPLE_LIST <<< "${SAMPLE_IDS}"
IFS=',' read -r -a BASELINE_LIST <<< "${BASELINES}"
IFS=',' read -r -a BATCH_LIST <<< "${BATCH_SIZES}"

shared_args=(
  --model "${MODEL}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --results-dir "${RUN_RESULTS_DIR}"
)

if [[ "${LIMIT}" != "0" ]]; then
  shared_args+=(--limit "${LIMIT}")
fi
if [[ "${LOCAL_ONLY}" == "1" ]]; then
  shared_args+=(--local-files-only)
fi
if [[ "${NO_SORT}" == "1" ]]; then
  shared_args+=(--no-sort-by-length)
fi
if [[ "${PAD_TO_MULTIPLE_OF}" != "0" ]]; then
  shared_args+=(--pad-to-multiple-of "${PAD_TO_MULTIPLE_OF}")
fi
if [[ -n "${PAD_BUCKETS}" ]]; then
  shared_args+=(--pad-buckets "${PAD_BUCKETS}")
fi
if [[ "${MAX_INPUT_TOKENS}" != "0" ]]; then
  shared_args+=(--max-input-tokens "${MAX_INPUT_TOKENS}")
fi

for dataset in "${DATASET_LIST[@]}"; do
  for sample_id in "${SAMPLE_LIST[@]}"; do
    sample_file="benchmark/data/offline_e2e/${dataset}_${SAMPLE_SIZE}_sample${sample_id}.jsonl"
    if [[ ! -f "${sample_file}" ]]; then
      echo "Missing sample file: ${sample_file}" >&2
      exit 1
    fi
    for batch_size in "${BATCH_LIST[@]}"; do
      for repeat_index in $(seq 1 "${REPEATS}"); do
        for baseline in "${BASELINE_LIST[@]}"; do
          echo "=== baseline=${baseline} dataset=${dataset} sample=${sample_id} bs=${batch_size} repeat=${repeat_index} hooks=${HOOK_SELECTION} ==="
          case "${baseline}" in
            hf_upper_bound)
              python experiments/offline_inference/scripts/run_hf_upper_bound.py \
                --sample-file "${sample_file}" \
                --batch-size "${batch_size}" \
                --repeat-index "${repeat_index}" \
                "${shared_args[@]}" \
                "${EXTRA_ARGS[@]}"
              ;;
            torch_hooks)
              python experiments/offline_inference/scripts/run_torch_hooks.py \
                --sample-file "${sample_file}" \
                --batch-size "${batch_size}" \
                --repeat-index "${repeat_index}" \
                --hook-selection "${HOOK_SELECTION}" \
                --disable-compile \
                "${shared_args[@]}" \
                "${EXTRA_ARGS[@]}"
              ;;
            nnsight)
              python experiments/offline_inference/scripts/run_nnsight.py \
                --sample-file "${sample_file}" \
                --batch-size "${batch_size}" \
                --repeat-index "${repeat_index}" \
                --hook-selection "${HOOK_SELECTION}" \
                --disable-compile \
                "${shared_args[@]}" \
                "${EXTRA_ARGS[@]}"
              ;;
            proj_dmi)
              if [[ "${PROJ_DMI_RETRY}" == "1" ]]; then
                run_proj_dmi_with_retries "${MODEL}" "${dataset}" "${batch_size}" "${repeat_index}" "${sample_file}" "${MAX_INPUT_TOKENS:-0}" "${MAX_NEW_TOKENS}" || true
              else
                proj_args=()
                if [[ "${PROJ_DMI_COMPILE}" != "1" ]]; then
                  proj_args+=(--disable-compile)
                fi
                python experiments/offline_inference/scripts/run_proj_dmi.py \
                  --sample-file "${sample_file}" \
                  --batch-size "${batch_size}" \
                  --repeat-index "${repeat_index}" \
                  --proj-dmi-mode "${PROJ_DMI_MODE}" \
                  --hook-selection "${HOOK_SELECTION}" \
                  "${proj_args[@]}" \
                  "${shared_args[@]}" \
                  "${PROJ_DMI_ARGS[@]}" \
                  "${EXTRA_ARGS[@]}"
              fi
              ;;
            *)
              echo "Unknown baseline: ${baseline}" >&2
              exit 1
              ;;
          esac
          cleanup_gpu
        done
      done
    done
  done
done
