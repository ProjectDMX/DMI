#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="qwen3-1.7b"
DATASETS="sharegpt,wildchat"
SAMPLE_IDS="1"
BASELINES="hf_upper_bound,hf_monitor,proj_dmi"
BATCH_SIZES="16"
REPEATS=1
MAX_NEW_TOKENS=0
LIMIT=0
RESULTS_DIR="${SCRIPT_DIR}/results"
LOCAL_ONLY=0
NO_SORT=0
DISABLE_COMPILE=0
PAD_TO_MULTIPLE_OF=0
PAD_BUCKETS=""
MAX_INPUT_TOKENS=0
PROJ_DMI_MODE="ring_null"
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
    --disable-compile) DISABLE_COMPILE=1; shift ;;
    --pad-to-multiple-of) PAD_TO_MULTIPLE_OF="$2"; shift 2 ;;
    --pad-buckets) PAD_BUCKETS="$2"; shift 2 ;;
    --max-input-tokens) MAX_INPUT_TOKENS="$2"; shift 2 ;;
    --proj-dmi-mode) PROJ_DMI_MODE="$2"; shift 2 ;;
    --ring-task-entries|--ring-payload-mb|--ring-pinned-mb|--drain-poll-timeout-us|--drain-flush-task-ratio|--drain-flush-payload-ratio|--drain-flush-entry-threshold|--drain-flush-byte-threshold|--drain-flush-timeout-us|--ch-parallelism|--ch-queue-max-items|--ch-queue-max-size-mb|--db-host|--db-port|--db-user|--db-password|--db-database|--db-table)
      PROJ_DMI_ARGS+=("$1" "$2"); shift 2 ;;
    --clone-slices)
      PROJ_DMI_ARGS+=("$1"); shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "${RESULTS_DIR}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
IFS=',' read -r -a SAMPLE_LIST <<< "${SAMPLE_IDS}"
IFS=',' read -r -a BASELINE_LIST <<< "${BASELINES}"
IFS=',' read -r -a BATCH_LIST <<< "${BATCH_SIZES}"

shared_args=(
  --model "${MODEL}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --results-dir "${RESULTS_DIR}"
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
if [[ "${DISABLE_COMPILE}" == "1" ]]; then
  shared_args+=(--disable-compile)
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
    sample_file="benchmark/data/offline_e2e/${dataset}_1000_sample${sample_id}.jsonl"
    if [[ ! -f "${sample_file}" ]]; then
      echo "Missing sample file: ${sample_file}" >&2
      exit 1
    fi
    for batch_size in "${BATCH_LIST[@]}"; do
      for repeat_index in $(seq 1 "${REPEATS}"); do
        for baseline in "${BASELINE_LIST[@]}"; do
          echo "=== baseline=${baseline} dataset=${dataset} sample=${sample_id} bs=${batch_size} repeat=${repeat_index} ==="
          case "${baseline}" in
            hf_upper_bound)
              python exp/offline_e2e/run_hf_upper_bound.py \
                --sample-file "${sample_file}" \
                --batch-size "${batch_size}" \
                --repeat-index "${repeat_index}" \
                "${shared_args[@]}" \
                "${EXTRA_ARGS[@]}"
              ;;
            hf_monitor)
              python exp/offline_e2e/run_hf_monitor.py \
                --sample-file "${sample_file}" \
                --batch-size "${batch_size}" \
                --repeat-index "${repeat_index}" \
                "${shared_args[@]}" \
                "${EXTRA_ARGS[@]}"
              ;;
            proj_dmi)
              python exp/offline_e2e/run_proj_dmi.py \
                --sample-file "${sample_file}" \
                --batch-size "${batch_size}" \
                --repeat-index "${repeat_index}" \
                --proj-dmi-mode "${PROJ_DMI_MODE}" \
                "${shared_args[@]}" \
                "${PROJ_DMI_ARGS[@]}" \
                "${EXTRA_ARGS[@]}"
              ;;
            *)
              echo "Unknown baseline: ${baseline}" >&2
              exit 1
              ;;
          esac
        done
      done
    done
  done
done
