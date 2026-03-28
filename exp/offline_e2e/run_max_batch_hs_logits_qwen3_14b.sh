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
PAD_BUCKETS=${PAD_BUCKETS:-64,128,256,512}
MAX_BS_CAP=${MAX_BS_CAP:-500}
SEARCH_HIGH_NORMAL=${SEARCH_HIGH_NORMAL:-128}
SEARCH_HIGH_HF_MONITOR=${SEARCH_HIGH_HF_MONITOR:-32}
SMOKE=${SMOKE:-0}
RING_TASK_ENTRIES=${RING_TASK_ENTRIES:-131072}
RING_FLUSH=${RING_FLUSH:-0.15}

if [ "${SMOKE}" = "1" ]; then
    MAX_BS_CAP=${MAX_BS_CAP_SMOKE:-64}
    SEARCH_HIGH_NORMAL=${SEARCH_HIGH_NORMAL_SMOKE:-64}
    SEARCH_HIGH_HF_MONITOR=${SEARCH_HIGH_HF_MONITOR_SMOKE:-16}
fi

case "${DATASET}" in
    wildchat)
        MAX_INPUT=250
        MAX_OUTPUT=1000
        ;;
    sharegpt)
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
RESULTS_DIR="${PROJECT}/exp/offline_e2e/results/max_batch_${MODEL}_${DATASET}_${CAPTURE_MODE}_synthetic${TAG_SUFFIX}_${RUN_TAG}"
LOG_DIR="${RESULTS_DIR}/logs"
ATTEMPT_DIR="${RESULTS_DIR}/attempts"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
ATTEMPTS_CSV="${RESULTS_DIR}/attempts.csv"
SYNTHETIC_FILE="${RESULTS_DIR}/synthetic_${DATASET}_${MAX_INPUT}_${MAX_OUTPUT}_${MAX_BS_CAP}.jsonl"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${ATTEMPT_DIR}"

echo "baseline,ring_mb,max_batch_size,last_ok_bs,first_fail_bs,target_tok_s,compute_tok_s,prompts_s,total_seconds,search_total_seconds,search_log_file,search_json" > "${SUMMARY_CSV}"
echo "baseline,ring_mb,batch_size,status,target_tok_s,compute_tok_s,prompts_s,total_seconds,log_file,json_file" > "${ATTEMPTS_CSV}"

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

generate_synthetic_sample() {
    echo "Generating synthetic sample file: ${SYNTHETIC_FILE}"
    python - <<'PY2' "${SYNTHETIC_FILE}" "${MAX_BS_CAP}" "${DATASET}" "${MAX_INPUT}" "${MAX_OUTPUT}"
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
count = int(sys.argv[2])
dataset = sys.argv[3]
max_input = int(sys.argv[4])
max_output = int(sys.argv[5])
prompt_text = ("synthetic prompt token " * (max_input * 12)).strip()
target_text = ("synthetic target token " * (max_output * 12)).strip()
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    for i in range(count):
        row = {
            "dataset": dataset,
            "sample_id": i + 1,
            "entry_id": f"synthetic_{i}",
            "source_conversation_id": f"synthetic_{i}",
            "approx_prompt_tokens": max_input,
            "approx_target_tokens": max_output,
            "target_text": target_text,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        f.write(json.dumps(row) + "\n")
PY2
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
    local run_dir="$3"

    COMMON=(
        --model "${MODEL}"
        --batch-size "${bs}"
        --sample-file "${SYNTHETIC_FILE}"
        --local-files-only
        --max-input-tokens "${MAX_INPUT}"
        --max-new-tokens "${MAX_OUTPUT}"
        --limit "${bs}"
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
    local bs="$3"
    local status="$4"
    local target_toks="$5"
    local compute_toks="$6"
    local prompts_s="$7"
    local total_seconds="$8"
    local log_file="$9"
    local json_file="${10}"
    echo "${baseline},${ring_mb},${bs},${status},${target_toks},${compute_toks},${prompts_s},${total_seconds},${log_file},${json_file}" >> "${ATTEMPTS_CSV}"
}

append_summary() {
    local baseline="$1"
    local ring_mb="$2"
    local max_bs="$3"
    local low_ok="$4"
    local high_fail="$5"
    local target_toks="$6"
    local compute_toks="$7"
    local prompts_s="$8"
    local total_seconds="$9"
    local search_total_seconds="${10}"
    local log_file="${11}"
    local json_file="${12}"
    echo "${baseline},${ring_mb},${max_bs},${low_ok},${high_fail},${target_toks},${compute_toks},${prompts_s},${total_seconds},${search_total_seconds},${log_file},${json_file}" >> "${SUMMARY_CSV}"
}

run_attempt() {
    local baseline="$1"
    local bs="$2"

    local safe_label="${baseline// /_}"
    safe_label="${safe_label//\//_}"
    local tag="${safe_label}__bs${bs}"
    local run_dir="${ATTEMPT_DIR}/${tag}"
    local log_file="${LOG_DIR}/${tag}.log"
    mkdir -p "${run_dir}"

    build_command "${baseline}" "${bs}" "${run_dir}"

    echo ""
    echo "=== ${baseline} | bs=${bs} ring=${RING_MB:-na} ==="

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

    append_attempt "${baseline}" "${RING_MB}" "${bs}" "${status}" "${target_toks}" "${compute_toks}" "${prompts_s}" "${total_seconds}" "${log_file}" "${json_file}"
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
    local best_target_toks=""
    local best_compute_toks=""
    local best_prompts_s=""
    local best_total_seconds=""
    local best_log_file=""
    local best_json_file=""

    run_attempt "${baseline}" "${start_probe}"
    if [ "${ATTEMPT_STATUS}" = "OK" ]; then
        low_ok="${start_probe}"
        best_target_toks="${ATTEMPT_TARGET_TOKS}"
        best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
        best_prompts_s="${ATTEMPT_PROMPTS_S}"
        best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
        best_log_file="${ATTEMPT_LOG_FILE}"
        best_json_file="${ATTEMPT_JSON_FILE}"
        local next_probe
        next_probe=$(next_probe_up "${low_ok}")
        while [ "${next_probe}" -gt 0 ]; do
            run_attempt "${baseline}" "${next_probe}"
            if [ "${ATTEMPT_STATUS}" = "OK" ]; then
                low_ok="${next_probe}"
                best_target_toks="${ATTEMPT_TARGET_TOKS}"
                best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
                best_prompts_s="${ATTEMPT_PROMPTS_S}"
                best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
                best_log_file="${ATTEMPT_LOG_FILE}"
                best_json_file="${ATTEMPT_JSON_FILE}"
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
            run_attempt "${baseline}" "${bs}"
            if [ "${ATTEMPT_STATUS}" = "OK" ]; then
                low_ok="${bs}"
                best_target_toks="${ATTEMPT_TARGET_TOKS}"
                best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
                best_prompts_s="${ATTEMPT_PROMPTS_S}"
                best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
                best_log_file="${ATTEMPT_LOG_FILE}"
                best_json_file="${ATTEMPT_JSON_FILE}"
                break
            fi
            high_fail="${bs}"
        done
    fi

    if [ "${low_ok}" -gt 0 ] && [ "${high_fail}" -gt $((low_ok + 1)) ]; then
        while [ $((high_fail - low_ok)) -gt 1 ]; do
            local mid=$(((low_ok + high_fail) / 2))
            run_attempt "${baseline}" "${mid}"
            if [ "${ATTEMPT_STATUS}" = "OK" ]; then
                low_ok="${mid}"
                best_target_toks="${ATTEMPT_TARGET_TOKS}"
                best_compute_toks="${ATTEMPT_COMPUTE_TOKS}"
                best_prompts_s="${ATTEMPT_PROMPTS_S}"
                best_total_seconds="${ATTEMPT_TOTAL_SECONDS}"
                best_log_file="${ATTEMPT_LOG_FILE}"
                best_json_file="${ATTEMPT_JSON_FILE}"
            else
                high_fail="${mid}"
            fi
        done
    fi

    local search_end_ts
    search_end_ts=$(date +%s)
    local search_total_seconds=$((search_end_ts - search_start_ts))

    append_summary "${baseline}" "${RING_MB}" "${low_ok}" "${low_ok}" "${high_fail}" "${best_target_toks}" "${best_compute_toks}" "${best_prompts_s}" "${best_total_seconds}" "${search_total_seconds}" "${best_log_file}" "${best_json_file}"
}

echo "============================================================"
echo "Node: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"
echo "Model    : ${MODEL}"
echo "Dataset  : ${DATASET}"
echo "Capture  : ${CAPTURE_MODE}"
echo "Synthetic: ${SYNTHETIC_FILE}"
echo "Input    : ${MAX_INPUT}"
echo "Output   : ${MAX_OUTPUT}"
echo "Max cap  : ${MAX_BS_CAP}"
echo "Results  : ${RESULTS_DIR}"
echo "Smoke    : ${SMOKE}"

prepare_model_cache
generate_synthetic_sample

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
