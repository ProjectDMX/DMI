#!/bin/bash

set -uo pipefail

SCRATCH=${SCRATCH:-/scratch/zt1/project/zaoxing-prj/user/ynn1999}
ENV_DIR=${ENV_DIR:-${SCRATCH}/proj-dmx}
PROJECT=${PROJECT:-${SCRATCH}/DMI/DMI}

if [ ! -d "${ENV_DIR}" ]; then
    echo "Extracting conda environment ..."
    tar xzf "${SCRATCH}/proj-dmx.tar.gz" -C "${SCRATCH}"
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "ERROR: ${ENV_DIR}/bin/python not found!"
    exit 1
fi

export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_HOME="${ENV_DIR}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${PROJECT}/transformers/src:${PROJECT}"
export TL_ENABLE_NVTX=1

cd "${PROJECT}/exp/offline_e2e"

MODEL=${MODEL:-qwen3-14b}
DATASET=${DATASET:-wildchat}
CAPTURE_MODE=${CAPTURE_MODE:-hs_logits}
LIMIT_BASE=${LIMIT_BASE:-500}
PAD_BUCKETS=${PAD_BUCKETS:-64,128,256,512}
MAX_BS_CAP=${MAX_BS_CAP:-500}
SEARCH_HIGH_NORMAL=${SEARCH_HIGH_NORMAL:-128}
SEARCH_HIGH_HF_MONITOR=${SEARCH_HIGH_HF_MONITOR:-32}
MEASURE_AFTER_SEARCH=${MEASURE_AFTER_SEARCH:-1}
SMOKE=${SMOKE:-0}
RING_TASK_ENTRIES=${RING_TASK_ENTRIES:-131072}
RING_FLUSH=${RING_FLUSH:-0.15}

if [ "${SMOKE}" = "1" ]; then
    MAX_BS_CAP=${MAX_BS_CAP_SMOKE:-64}
    SEARCH_HIGH_NORMAL=${SEARCH_HIGH_NORMAL_SMOKE:-64}
    SEARCH_HIGH_HF_MONITOR=${SEARCH_HIGH_HF_MONITOR_SMOKE:-16}
    MEASURE_AFTER_SEARCH=${MEASURE_AFTER_SEARCH_SMOKE:-0}
fi

case "${DATASET}" in
    wildchat)
        SAMPLE_FILE="../../benchmark/data/offline_e2e/wildchat_500_sample1.jsonl"
        MAX_INPUT=250
        MAX_OUTPUT=1000
        ;;
    sharegpt)
        SAMPLE_FILE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
        MAX_INPUT=200
        MAX_OUTPUT=750
        ;;
    *)
        echo "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

BASELINES=(
    "hf_upper_bound_eager"
    "hf_monitor_generate_eager"
    "hf_monitor_manual_compile"
    "torch_hooks_eager"
    "nnsight_eager"
    "proj_dmi_compile_ring5g"
    "proj_dmi_compile_ring10g"
)

RUN_TAG="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-manual}"
TAG_SUFFIX=""
if [ "${SMOKE}" = "1" ]; then
    TAG_SUFFIX="_smoke"
fi
RESULTS_DIR="${PROJECT}/exp/offline_e2e/results/max_batch_${MODEL}_${DATASET}_${CAPTURE_MODE}${TAG_SUFFIX}_${RUN_TAG}"
LOG_DIR="${RESULTS_DIR}/logs"
ATTEMPT_DIR="${RESULTS_DIR}/attempts"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
ATTEMPTS_CSV="${RESULTS_DIR}/attempts.csv"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${ATTEMPT_DIR}"

echo "baseline,ring_mb,max_batch_size,last_ok_bs,first_fail_bs,measure_limit,measure_status,target_tok_s,compute_tok_s,prompts_s,total_seconds,search_total_seconds,measure_log_file,measure_json" > "${SUMMARY_CSV}"
echo "baseline,ring_mb,phase,batch_size,limit,status,target_tok_s,compute_tok_s,prompts_s,total_seconds,log_file,json_file" > "${ATTEMPTS_CSV}"

LOCAL_NVME_HF="/tmp/${USER}_hf_cache_${SLURM_JOB_ID:-manual}"
LOCAL_HF="${LOCAL_NVME_HF}"
CLEANUP_DIR="${LOCAL_NVME_HF}"
mkdir -p "${LOCAL_NVME_HF}/hub"
SRC_HUB="${SCRATCH}/hf_cache/hub"

trap 'if [ -n "${CLEANUP_DIR}" ] && [ -d "${CLEANUP_DIR}" ]; then echo "Cleaning up ${CLEANUP_DIR}..."; rm -rf "${CLEANUP_DIR}"; fi' EXIT

copy_model() {
    local model_dir="$1"
    local src="${SRC_HUB}/${model_dir}"
    local dst="${LOCAL_NVME_HF}/hub/${model_dir}"

    if [ ! -d "${src}" ]; then
        echo "ERROR: source model not found: ${src}"
        echo "Falling back to scratch HF cache"
        LOCAL_HF="${SCRATCH}/hf_cache"
        return 0
    fi
    if [ ! -d "${dst}" ]; then
        echo "Copying ${model_dir} to local NVMe ..."
        cp -a "${src}" "${dst}"
    fi
    LOCAL_HF="${LOCAL_NVME_HF}"
}

prepare_model_cache() {
    case "${MODEL}" in
        qwen3-14b)
            copy_model "models--Qwen--Qwen3-14B"
            ;;
        *)
            echo "Unknown model '${MODEL}', falling back to scratch HF cache"
            LOCAL_HF="${SCRATCH}/hf_cache"
            ;;
    esac
    export HF_HOME="${LOCAL_HF}"
    echo "HF_HOME=${HF_HOME}"
}

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

is_oom_log() {
    local log_file="$1"
    grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM' "${log_file}"
}

next_probe_up() {
    local bs="$1"
    if [ "${bs}" -lt 128 ]; then
        echo 128
    elif [ "${bs}" -lt 256 ] && [ "${MAX_BS_CAP}" -gt 128 ]; then
        if [ "${MAX_BS_CAP}" -lt 256 ]; then
            echo "${MAX_BS_CAP}"
        else
            echo 256
        fi
    elif [ "${bs}" -lt "${MAX_BS_CAP}" ]; then
        echo "${MAX_BS_CAP}"
    else
        echo 0
    fi
}

get_fallbacks() {
    local baseline="$1"
    if [ "${baseline}" = "hf_monitor_generate_eager" ]; then
        echo "16 8 4 2 1"
    else
        echo "64 32 16 8 4 2 1"
    fi
}

get_start_probe() {
    local baseline="$1"
    if [ "${baseline}" = "hf_monitor_generate_eager" ]; then
        echo "${SEARCH_HIGH_HF_MONITOR}"
    else
        echo "${SEARCH_HIGH_NORMAL}"
    fi
}

build_command() {
    local baseline="$1"
    local bs="$2"
    local limit="$3"
    local run_dir="$4"

    COMMON=(
        --model "${MODEL}"
        --batch-size "${bs}"
        --sample-file "${SAMPLE_FILE}"
        --local-files-only
        --max-input-tokens "${MAX_INPUT}"
        --max-new-tokens "${MAX_OUTPUT}"
        --limit "${limit}"
        --pad-buckets "${PAD_BUCKETS}"
        --results-dir "${run_dir}"
        --capture-mode "${CAPTURE_MODE}"
    )

    RING_MB=""
    case "${baseline}" in
        hf_upper_bound_eager)
            CMD=(python run_hf_upper_bound.py "${COMMON[@]}" --disable-compile)
            ;;
        hf_monitor_generate_eager)
            CMD=(python run_hf_monitor.py "${COMMON[@]}")
            ;;
        hf_monitor_manual_compile)
            CMD=(python run_hf_monitor_manual.py "${COMMON[@]}")
            ;;
        torch_hooks_eager)
            CMD=(python run_torch_hooks.py "${COMMON[@]}" --disable-compile)
            ;;
        nnsight_eager)
            CMD=(python run_nnsight.py "${COMMON[@]}")
            ;;
        proj_dmi_compile_ring5g)
            RING_MB=5120
            CMD=(python run_proj_dmi.py "${COMMON[@]}" --ring-payload-mb "${RING_MB}" --ring-pinned-mb "${RING_MB}" --ring-task-entries "${RING_TASK_ENTRIES}" --drain-flush-payload-ratio "${RING_FLUSH}" --drain-flush-task-ratio "${RING_FLUSH}")
            ;;
        proj_dmi_compile_ring10g)
            RING_MB=10240
            CMD=(python run_proj_dmi.py "${COMMON[@]}" --ring-payload-mb "${RING_MB}" --ring-pinned-mb "${RING_MB}" --ring-task-entries "${RING_TASK_ENTRIES}" --drain-flush-payload-ratio "${RING_FLUSH}" --drain-flush-task-ratio "${RING_FLUSH}")
            ;;
        *)
            echo "Unknown baseline ${baseline}"
            exit 1
            ;;
    esac
}

append_attempt() {
    local baseline="$1"
    local ring_mb="$2"
    local phase="$3"
    local bs="$4"
    local limit="$5"
    local status="$6"
    local target_toks="$7"
    local compute_toks="$8"
    local prompts_s="$9"
    local total_seconds="${10}"
    local log_file="${11}"
    local json_file="${12}"
    echo "${baseline},${ring_mb},${phase},${bs},${limit},${status},${target_toks},${compute_toks},${prompts_s},${total_seconds},${log_file},${json_file}" >> "${ATTEMPTS_CSV}"
}

append_summary() {
    local baseline="$1"
    local ring_mb="$2"
    local max_bs="$3"
    local low_ok="$4"
    local high_fail="$5"
    local measure_limit="$6"
    local measure_status="$7"
    local target_toks="$8"
    local compute_toks="$9"
    local prompts_s="${10}"
    local total_seconds="${11}"
    local search_total_seconds="${12}"
    local log_file="${13}"
    local json_file="${14}"
    echo "${baseline},${ring_mb},${max_bs},${low_ok},${high_fail},${measure_limit},${measure_status},${target_toks},${compute_toks},${prompts_s},${total_seconds},${search_total_seconds},${log_file},${json_file}" >> "${SUMMARY_CSV}"
}

run_attempt() {
    local baseline="$1"
    local phase="$2"
    local bs="$3"
    local limit="$4"

    local safe_label="${baseline// /_}"
    safe_label="${safe_label//\//_}"
    local tag="${safe_label}__${phase}__bs${bs}__limit${limit}"
    local run_dir="${ATTEMPT_DIR}/${tag}"
    local log_file="${LOG_DIR}/${tag}.log"
    mkdir -p "${run_dir}"

    build_command "${baseline}" "${bs}" "${limit}" "${run_dir}"

    echo ""
    echo "=== ${baseline} | phase=${phase} bs=${bs} limit=${limit} ring=${RING_MB:-na} ==="

    local status="FAIL"
    local target_toks=""
    local compute_toks=""
    local prompts_s=""
    local total_seconds=""
    local json_file=""

    if "${CMD[@]}" >"${log_file}" 2>&1; then
        status="OK"
    else
        if is_oom_log "${log_file}"; then
            status="OOM"
        else
            status="FAIL"
        fi
    fi

    cat "${log_file}"

    json_file=$(python - <<'PY2' "${run_dir}"
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
files = sorted(run_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
print(files[0] if files else '')
PY2
)

    if [ -n "${json_file}" ] && [ -f "${json_file}" ]; then
        read -r target_toks compute_toks prompts_s total_seconds <<EOF
$(python - <<'PY2' "${json_file}"
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
vals = [
    d.get('target_generated_tokens_per_s', ''),
    d.get('actual_generated_tokens_per_s', ''),
    d.get('prompts_per_s', ''),
    d.get('total_seconds', ''),
]
print(*vals)
PY2
)
EOF
    else
        target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
    fi

    append_attempt "${baseline}" "${RING_MB}" "${phase}" "${bs}" "${limit}" "${status}" "${target_toks}" "${compute_toks}" "${prompts_s}" "${total_seconds}" "${log_file}" "${json_file}"
    cleanup_gpu

    ATTEMPT_STATUS="${status}"
    ATTEMPT_TARGET_TOKS="${target_toks}"
    ATTEMPT_COMPUTE_TOKS="${compute_toks}"
    ATTEMPT_PROMPTS_S="${prompts_s}"
    ATTEMPT_TOTAL_SECONDS="${total_seconds}"
    ATTEMPT_LOG_FILE="${log_file}"
    ATTEMPT_JSON_FILE="${json_file}"
}

search_baseline() {
    local baseline="$1"
    local start_probe
    start_probe=$(get_start_probe "${baseline}")
    local fallbacks
    fallbacks=$(get_fallbacks "${baseline}")

    local search_start_ts
    search_start_ts=$(date +%s)
    local low_ok=0
    local high_fail=0

    run_attempt "${baseline}" "search" "${start_probe}" "${start_probe}"
    if [ "${ATTEMPT_STATUS}" = "OK" ]; then
        low_ok="${start_probe}"
        local next_probe
        next_probe=$(next_probe_up "${low_ok}")
        while [ "${next_probe}" -gt 0 ]; do
            run_attempt "${baseline}" "search" "${next_probe}" "${next_probe}"
            if [ "${ATTEMPT_STATUS}" = "OK" ]; then
                low_ok="${next_probe}"
                if [ "${low_ok}" -ge "${MAX_BS_CAP}" ]; then
                    high_fail=0
                    break
                fi
                next_probe=$(next_probe_up "${low_ok}")
            else
                high_fail="${next_probe}"
                break
            fi
        done
    else
        high_fail="${start_probe}"
        for bs in ${fallbacks}; do
            if [ "${bs}" -ge "${start_probe}" ]; then
                continue
            fi
            run_attempt "${baseline}" "search" "${bs}" "${bs}"
            if [ "${ATTEMPT_STATUS}" = "OK" ]; then
                low_ok="${bs}"
                break
            fi
            high_fail="${bs}"
        done
    fi

    if [ "${low_ok}" -gt 0 ] && [ "${high_fail}" -gt $((low_ok + 1)) ]; then
        while [ $((high_fail - low_ok)) -gt 1 ]; do
            local mid=$(((low_ok + high_fail) / 2))
            run_attempt "${baseline}" "search" "${mid}" "${mid}"
            if [ "${ATTEMPT_STATUS}" = "OK" ]; then
                low_ok="${mid}"
            else
                high_fail="${mid}"
            fi
        done
    fi

    local max_bs="${low_ok}"
    local measure_limit=""
    local measure_status=""
    local measure_target_toks=""
    local measure_compute_toks=""
    local measure_prompts_s=""
    local measure_total_seconds=""
    local measure_log_file=""
    local measure_json_file=""

    if [ "${MEASURE_AFTER_SEARCH}" = "1" ] && [ "${max_bs}" -gt 0 ]; then
        measure_limit=$((max_bs * 2))
        if [ "${measure_limit}" -gt "${LIMIT_BASE}" ]; then
            measure_limit="${LIMIT_BASE}"
        fi
        if [ "${measure_limit}" -lt "${max_bs}" ]; then
            measure_limit="${max_bs}"
        fi
        run_attempt "${baseline}" "measure" "${max_bs}" "${measure_limit}"
        measure_status="${ATTEMPT_STATUS}"
        measure_target_toks="${ATTEMPT_TARGET_TOKS}"
        measure_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
        measure_prompts_s="${ATTEMPT_PROMPTS_S}"
        measure_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
        measure_log_file="${ATTEMPT_LOG_FILE}"
        measure_json_file="${ATTEMPT_JSON_FILE}"
    fi

    local search_end_ts
    search_end_ts=$(date +%s)
    local search_total_seconds=$((search_end_ts - search_start_ts))

    append_summary "${baseline}" "${RING_MB}" "${max_bs}" "${low_ok}" "${high_fail}" "${measure_limit}" "${measure_status}" "${measure_target_toks}" "${measure_compute_toks}" "${measure_prompts_s}" "${measure_total_seconds}" "${search_total_seconds}" "${measure_log_file}" "${measure_json_file}"
}

echo "============================================================"
echo "Node: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"
echo "Model    : ${MODEL}"
echo "Dataset  : ${DATASET}"
echo "Capture  : ${CAPTURE_MODE}"
echo "Sample   : ${SAMPLE_FILE}"
echo "Input    : ${MAX_INPUT}"
echo "Output   : ${MAX_OUTPUT}"
echo "Max cap  : ${MAX_BS_CAP}"
echo "Results  : ${RESULTS_DIR}"
echo "Smoke    : ${SMOKE}"

prepare_model_cache

for baseline in "${BASELINES[@]}"; do
    echo ""
    echo "############################################################"
    echo "Searching max batch size for ${baseline}"
    echo "############################################################"
    search_baseline "${baseline}"
done

echo ""
echo "============================================================"
echo "Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Attempts CSV: ${ATTEMPTS_CSV}"
echo "============================================================"
