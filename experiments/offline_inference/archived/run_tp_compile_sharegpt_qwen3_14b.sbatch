#!/bin/bash
#SBATCH --job-name=tp-hslogits-14b
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclusive
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

cd "${PROJECT}/experiments/offline_inference"

echo "============================================================"
echo "Node: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

MODEL="qwen3-14b"
DATASET="sharegpt"
BATCH_SIZE=64
LIMIT=128
MAX_INPUT=200
MAX_OUTPUT=750
PAD_BUCKETS="64,128,256,512"
CAPTURE_MODE="hs_logits"
HOOK_LABEL="capture_${CAPTURE_MODE}"
HOOK_TAG="${HOOK_LABEL// /_}"
SAMPLE_FILE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"

RING_INITIAL_MB=30720
RING_STEP_MB=5120
RING_MIN_MB=5120
RING_TASK_ENTRIES=131072
RING_FLUSH=0.15

RUN_TAG="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT}/experiments/offline_inference/results/tp_compile_sharegpt_qwen3_14b_${RUN_TAG}"
LOG_DIR="${RESULTS_DIR}/logs"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

echo "model,dataset,batch_size,baseline,tp_size,status,ring_mb,target_tok_s,compute_tok_s,log_file" > "${SUMMARY_CSV}"

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
    copy_model "models--Qwen--Qwen3-14B"
    export HF_HOME="${LOCAL_HF}"
    echo "HF_HOME=${HF_HOME}"
}

is_oom_log() {
    local log_file="$1"
    grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM' "${log_file}"
}

append_summary() {
    local baseline="$1"
    local tp_size="$2"
    local status="$3"
    local ring_mb="$4"
    local target_toks="$5"
    local compute_toks="$6"
    local log_file="$7"
    echo "${MODEL},${DATASET},${BATCH_SIZE},${baseline},${tp_size},${status},${ring_mb},${target_toks},${compute_toks},${log_file}" >> "${SUMMARY_CSV}"
}

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

run_baseline_allow_fail() {
    local baseline="$1"
    local tp_size="$2"
    shift 2

    local safe_label="${baseline// /_}"
    safe_label="${safe_label//\//_}"
    local log_file="${LOG_DIR}/${MODEL}__${DATASET}__bs${BATCH_SIZE}__${HOOK_TAG}__${safe_label}.log"
    local status="FAIL"
    local target_toks=""
    local compute_toks=""

    echo ""
    echo "=== ${baseline} | tp=${tp_size} ==="

    if "$@" >"${log_file}" 2>&1; then
        cat "${log_file}"
        target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        status="OK"
    else
        cat "${log_file}"
        if is_oom_log "${log_file}"; then
            status="OOM_SKIP"
        else
            status="FAIL_SKIP"
        fi
        echo "Skipping baseline after failure: ${baseline}"
    fi

    append_summary "${baseline}" "${tp_size}" "${status}" "" "${target_toks}" "${compute_toks}" "${log_file}"
    cleanup_gpu
    return 0
}

run_proj_dmi_retry() {
    local baseline="$1"
    local tp_size="$2"
    shift 2

    local ring_mb="${RING_INITIAL_MB}"
    local safe_label="${baseline// /_}"
    safe_label="${safe_label//\//_}"

    while [ "${ring_mb}" -ge "${RING_MIN_MB}" ]; do
        local log_file="${LOG_DIR}/${MODEL}__${DATASET}__bs${BATCH_SIZE}__${HOOK_TAG}__${safe_label}__ring${ring_mb}.log"
        echo ""
        echo "=== ${baseline} | tp=${tp_size} ring=${ring_mb}MB ==="

        if "$@" --ring-payload-mb "${ring_mb}" --ring-pinned-mb "${ring_mb}" \
            --ring-task-entries "${RING_TASK_ENTRIES}" \
            --drain-flush-payload-ratio "${RING_FLUSH}" --drain-flush-task-ratio "${RING_FLUSH}" \
            >"${log_file}" 2>&1; then
            cat "${log_file}"
            local target_toks compute_toks
            target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
            compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
            append_summary "${baseline}" "${tp_size}" "OK" "${ring_mb}" "${target_toks}" "${compute_toks}" "${log_file}"
            cleanup_gpu
            return 0
        fi

        cat "${log_file}"
        if is_oom_log "${log_file}"; then
            append_summary "${baseline}" "${tp_size}" "OOM_RETRY" "${ring_mb}" "" "" "${log_file}"
            ring_mb=$((ring_mb - RING_STEP_MB))
            cleanup_gpu
            continue
        fi

        append_summary "${baseline}" "${tp_size}" "FAIL_SKIP" "${ring_mb}" "" "" "${log_file}"
        echo "Skipping DMI baseline after non-OOM failure: ${baseline}"
        cleanup_gpu
        return 0
    done

    append_summary "${baseline}" "${tp_size}" "OOM_EXHAUSTED_SKIP" "${ring_mb}" "" "" "${LOG_DIR}/${safe_label}__exhausted.log"
    return 0
}

prepare_model_cache

COMMON=(--model "${MODEL}" --batch-size "${BATCH_SIZE}" --sample-file "${SAMPLE_FILE}" --local-files-only \
    --max-input-tokens "${MAX_INPUT}" --max-new-tokens "${MAX_OUTPUT}" --limit "${LIMIT}" \
    --pad-buckets "${PAD_BUCKETS}" --results-dir "${RESULTS_DIR}" --capture-mode "${CAPTURE_MODE}")

echo ""
echo "Planned runs:"
echo "  hf_compile_single"
echo "  hf_compile_tp2"
echo "  hf_compile_tp4"
echo "  dmi_compile_single"
echo "  dmi_compile_tp2"
echo "  dmi_compile_tp4"
echo ""

run_baseline_allow_fail "hf_compile_single" "1" \
    env CUDA_VISIBLE_DEVICES=0 python scripts/run_hf_upper_bound.py "${COMMON[@]}"

run_baseline_allow_fail "hf_compile_tp2" "2" \
    env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/run_hf_upper_bound.py "${COMMON[@]}" --tp-size 2

run_baseline_allow_fail "hf_compile_tp4" "4" \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/run_hf_upper_bound.py "${COMMON[@]}" --tp-size 4

run_proj_dmi_retry "dmi_compile_single" "1" \
    env CUDA_VISIBLE_DEVICES=0 python scripts/run_proj_dmi.py "${COMMON[@]}" --proj-dmi-mode ring_null

run_proj_dmi_retry "dmi_compile_tp2" "2" \
    env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/run_proj_dmi.py "${COMMON[@]}" --proj-dmi-mode ring_null --tp-size 2

run_proj_dmi_retry "dmi_compile_tp4" "4" \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/run_proj_dmi.py "${COMMON[@]}" --proj-dmi-mode ring_null --tp-size 4

echo ""
echo "============================================================"
echo "TP compile sharegpt sweep done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "============================================================"
