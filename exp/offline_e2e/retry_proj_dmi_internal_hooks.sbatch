#!/bin/bash
#SBATCH --job-name=retry-dmi-hooks
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/ynn1999
ENV_DIR=${SCRATCH}/proj-dmx
PROJECT=${SCRATCH}/DMI/DMI

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

echo "============================================================"
echo "Node: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

LIMIT=128
PAD_BUCKETS="64,128,256,512"
SAMPLE_FILE_SHAREGPT="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
SAMPLE_FILE_WILDCHAT="../../benchmark/data/offline_e2e/wildchat_500_sample1.jsonl"
RING_STEP_MB=5120
RING_MIN_MB=5120
RING_TASK_ENTRIES=131072
RING_FLUSH=0.15
CAPTURE_MODE="hs"
HOOK_SELECTION="q,k,v,z,mlp_in,mlp_out,resid_mid"
HOOK_TAG="${HOOK_SELECTION//,/__}"
HOOK_TAG="${HOOK_TAG// /}"
HOOK_LABEL="internal_hooks:${HOOK_SELECTION}"

# Write into the ORIGINAL results dir from job 18689016
RESULTS_DIR="${PROJECT}/exp/offline_e2e/results/full_sweep_q__k__v__z__mlp_in__mlp_out__resid_mid_20260324_150039_18689016"
LOG_DIR="${RESULTS_DIR}/logs"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
mkdir -p "${LOG_DIR}"

LOCAL_NVME_HF="/tmp/${USER}_hf_cache_${SLURM_JOB_ID}"
LOCAL_HF="${LOCAL_NVME_HF}"
CLEANUP_DIR="${LOCAL_NVME_HF}"
mkdir -p "${LOCAL_NVME_HF}/hub"
SRC_HUB="${SCRATCH}/hf_cache/hub"

trap 'if [ -n "${CLEANUP_DIR}" ] && [ -d "${CLEANUP_DIR}" ]; then echo "Cleaning up ${CLEANUP_DIR}..."; rm -rf "${CLEANUP_DIR}"; fi' EXIT

verify_copy() {
    local src="$1"
    local dst="$2"
    local log="/tmp/verify_${SLURM_JOB_ID}.log"
    if ! diff -qr --no-dereference "$src" "$dst" >"$log" 2>&1; then
        echo "ERROR: copy verification failed!"
        head -100 "$log" || true
        return 1
    fi
    echo "Copy verified OK"
}

copy_model() {
    local model_dir="$1"
    local src="${SRC_HUB}/${model_dir}"
    local dst="${LOCAL_NVME_HF}/hub/${model_dir}"

    echo "  src: ${src}  exists=$([ -d "${src}" ] && echo yes || echo no)"
    echo "  dst: ${dst}  exists=$([ -d "${dst}" ] && echo yes || echo no)"

    if [ ! -d "${src}" ]; then
        echo "ERROR: source model not found: ${src}"
        echo "Falling back to scratch HF cache"
        LOCAL_HF="${SCRATCH}/hf_cache"
        return 0
    fi

    if [ -d "${dst}" ]; then
        echo "Model already copied, skipping"
        LOCAL_HF="${LOCAL_NVME_HF}"
        return 0
    fi

    echo "Copying ${model_dir} to local NVMe ..."
    cp -a "${src}" "${dst}"

    if ! verify_copy "${src}" "${dst}"; then
        rm -rf "${dst}"
        echo "Falling back to scratch HF cache"
        LOCAL_HF="${SCRATCH}/hf_cache"
        return 0
    fi

    echo "  Copy verified OK: $(du -sh "${dst}" | cut -f1)"
    LOCAL_HF="${LOCAL_NVME_HF}"
}

prepare_model_cache() {
    local model="$1"
    echo "Copying model '${model}' to local NVMe..."
    case "${model}" in
        qwen3-4b)
            copy_model "models--Qwen--Qwen3-4B"
            ;;
        qwen3-14b)
            copy_model "models--Qwen--Qwen3-14B"
            ;;
        *)
            echo "Unknown model '${model}', falling back to scratch HF cache"
            LOCAL_HF="${SCRATCH}/hf_cache"
            ;;
    esac
    export HF_HOME="${LOCAL_HF}"
    echo "HF_HOME=${HF_HOME}"
}

is_oom_log() {
    local log_file="$1"
    grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM' "${log_file}"
}

append_summary() {
    local model="$1"
    local dataset="$2"
    local bs="$3"
    local baseline="$4"
    local status="$5"
    local ring_mb="$6"
    local target_toks="$7"
    local compute_toks="$8"
    local log_file="$9"
    echo "${model},${dataset},${bs},${baseline},${CAPTURE_MODE},${HOOK_LABEL},${status},${ring_mb},${target_toks},${compute_toks},${log_file}" >> "${SUMMARY_CSV}"
}

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

estimate_proj_dmi_ring_mb() {
    local model="$1"
    local bs="$2"
    local max_input="$3"
    local max_output="$4"
    local total_mb
    total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

    local base_reserve
    case "${model}" in
        qwen3-4b) base_reserve=14336 ;;
        qwen3-14b) base_reserve=34816 ;;
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
    if [ "${total_seq}" -ge 1200 ]; then
        seq_penalty=5120
    elif [ "${total_seq}" -ge 900 ]; then
        seq_penalty=3072
    fi

    local slack_mb=4096
    local ring_mb=$(( total_mb - base_reserve - batch_penalty - seq_penalty - slack_mb ))
    if [ "${ring_mb}" -gt 56320 ]; then
        ring_mb=56320
    fi
    if [ "${ring_mb}" -lt "${RING_MIN_MB}" ]; then
        ring_mb="${RING_MIN_MB}"
    fi
    ring_mb=$(( (ring_mb / RING_STEP_MB) * RING_STEP_MB ))
    if [ "${ring_mb}" -lt "${RING_MIN_MB}" ]; then
        ring_mb="${RING_MIN_MB}"
    fi
    echo "${ring_mb}"
}

run_proj_dmi_with_retries() {
    local model="$1"
    local dataset="$2"
    local bs="$3"
    local baseline="$4"
    local max_input="$5"
    local max_output="$6"
    shift 6

    local ring_mb
    ring_mb=$(estimate_proj_dmi_ring_mb "${model}" "${bs}" "${max_input}" "${max_output}")
    local last_ring_mb="${ring_mb}"
    local last_log_file=""

    local safe_label="${baseline// /_}"
    safe_label="${safe_label//\//_}"

    while [ "${ring_mb}" -ge "${RING_MIN_MB}" ]; do
        local log_file="${LOG_DIR}/${model}__${dataset}__bs${bs}__${HOOK_TAG}__${safe_label}__ring${ring_mb}.log"
        last_log_file="${log_file}"
        last_ring_mb="${ring_mb}"
        echo ""
        echo "=== ${baseline} | model=${model} dataset=${dataset} bs=${bs} ring=${ring_mb}MB hooks=${HOOK_SELECTION} ==="

        if "$@" --ring-payload-mb "${ring_mb}" --ring-pinned-mb "${ring_mb}" \
            --ring-task-entries "${RING_TASK_ENTRIES}" \
            --drain-flush-payload-ratio "${RING_FLUSH}" --drain-flush-task-ratio "${RING_FLUSH}" \
            >"${log_file}" 2>&1; then
            cat "${log_file}"
            local target_toks compute_toks
            target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
            compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
            append_summary "${model}" "${dataset}" "${bs}" "${baseline}" "OK" "${ring_mb}" "${target_toks}" "${compute_toks}" "${log_file}"
            cleanup_gpu
            return 0
        fi

        cat "${log_file}"
        if is_oom_log "${log_file}"; then
            append_summary "${model}" "${dataset}" "${bs}" "${baseline}" "OOM_RETRY" "${ring_mb}" "" "" "${log_file}"
            echo "Proj-DMI OOM at ring=${ring_mb}MB; retrying with ${RING_STEP_MB}MB less."
            ring_mb=$((ring_mb - RING_STEP_MB))
            cleanup_gpu
            continue
        fi

        append_summary "${model}" "${dataset}" "${bs}" "${baseline}" "FAIL_SKIP" "${ring_mb}" "" "" "${log_file}"
        echo "Skipping Proj-DMI after non-OOM failure: ${baseline}"
        cleanup_gpu
        return 0
    done

    append_summary "${model}" "${dataset}" "${bs}" "${baseline}" "OOM_EXHAUSTED_SKIP" "${last_ring_mb}" "" "" "${last_log_file}"
    echo "Skipping Proj-DMI after exhausting ring buffer retries: ${baseline}"
    cleanup_gpu
    return 0
}

# ---- Only retry the 10 failed proj_dmi_compile configs from job 18689016 ----

echo ""
echo "Retry: proj_dmi_compile for internal hooks (10 failed configs)"
echo "Results appended to: ${RESULTS_DIR}"
echo ""

run_config() {
    local model="$1"
    local dataset="$2"
    local bs="$3"
    local max_input="$4"
    local max_output="$5"
    local sample_file="$6"

    echo ""
    echo "############################################################"
    echo "RETRY: model=${model} dataset=${dataset} bs=${bs} hooks=${HOOK_SELECTION}"
    echo "############################################################"

    COMMON=(--model "${model}" --batch-size "${bs}" --sample-file "${sample_file}" --local-files-only \
        --max-input-tokens "${max_input}" --max-new-tokens "${max_output}" --limit "${LIMIT}" \
        --pad-buckets "${PAD_BUCKETS}" --results-dir "${RESULTS_DIR}" --capture-mode "${CAPTURE_MODE}")

    run_proj_dmi_with_retries "${model}" "${dataset}" "${bs}" "proj_dmi_compile" "${max_input}" "${max_output}" \
        python run_proj_dmi.py "${COMMON[@]}" --hook-selection "${HOOK_SELECTION}"
}

# qwen3-4b (4 configs)
prepare_model_cache "qwen3-4b"
run_config "qwen3-4b" "sharegpt" 64  200 750  "${SAMPLE_FILE_SHAREGPT}"
run_config "qwen3-4b" "wildchat" 16  250 1000 "${SAMPLE_FILE_WILDCHAT}"
run_config "qwen3-4b" "wildchat" 32  250 1000 "${SAMPLE_FILE_WILDCHAT}"
run_config "qwen3-4b" "wildchat" 64  250 1000 "${SAMPLE_FILE_WILDCHAT}"

# qwen3-14b (6 configs)
prepare_model_cache "qwen3-14b"
run_config "qwen3-14b" "sharegpt" 16  200 750  "${SAMPLE_FILE_SHAREGPT}"
run_config "qwen3-14b" "sharegpt" 32  200 750  "${SAMPLE_FILE_SHAREGPT}"
run_config "qwen3-14b" "sharegpt" 64  200 750  "${SAMPLE_FILE_SHAREGPT}"
run_config "qwen3-14b" "wildchat" 16  250 1000 "${SAMPLE_FILE_WILDCHAT}"
run_config "qwen3-14b" "wildchat" 32  250 1000 "${SAMPLE_FILE_WILDCHAT}"
run_config "qwen3-14b" "wildchat" 64  250 1000 "${SAMPLE_FILE_WILDCHAT}"

echo ""
echo "============================================================"
echo "Retry done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "============================================================"
