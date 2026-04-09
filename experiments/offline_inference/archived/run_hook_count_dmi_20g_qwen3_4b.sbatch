#!/bin/bash
#SBATCH --job-name=hook-count-dmi-20g
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclusive

set -uo pipefail

SCRATCH="${SCRATCH:-YOUR_SCRATCH_PATH}"
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

MODEL="qwen3-4b"
DATASET="${DATASET:-sharegpt}"
BATCH_SIZE=64
LIMIT=512
PAD_BUCKETS="128,256,384,512"
CAPTURE_MODE="hs_logits"
RING_MB=20480
RING_TASK_ENTRIES=131072
RING_FLUSH_TIMEOUT_US=200

RUN_TAG="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT}/experiments/offline_inference/results/hook_count_dmi_20g_qwen3_4b_${DATASET}_${RUN_TAG}"
LOG_DIR="${RESULTS_DIR}/logs"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

case "${DATASET}" in
    sharegpt)
        SAMPLE_FILE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
        MAX_INPUT=384
        MAX_OUTPUT=64
        ;;
    wildchat)
        SAMPLE_FILE="../../benchmark/data/offline_e2e/wildchat_500_sample1.jsonl"
        MAX_INPUT=384
        MAX_OUTPUT=64
        ;;
    *)
        echo "ERROR: unsupported DATASET=${DATASET}; use sharegpt or wildchat"
        exit 1
        ;;
esac

HOOK_SELECTIONS=(
    "logits"
    "q,k,logits"
    "q,k,pattern,attn_scores,logits"
    "q,k,v,z,pattern,attn_scores,resid_mid,logits"
    "q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits"
)
HOOK_COUNTS=(1 73 145 253 325)

echo "dataset,model,batch_size,hook_count,hook_selection,baseline,status,ring_mb,target_tok_s,compute_tok_s,log_file" > "${SUMMARY_CSV}"

TMP_ROOT="${TMP_ROOT:-${PWD}/.tmp}"
mkdir -p "${TMP_ROOT}"
LOCAL_NVME_HF="${TMP_ROOT}/${USER}_hf_cache_${SLURM_JOB_ID}"
LOCAL_HF="${LOCAL_NVME_HF}"
CLEANUP_DIR="${LOCAL_NVME_HF}"
mkdir -p "${LOCAL_NVME_HF}/hub"
SRC_HUB="${SCRATCH}/hf_cache/hub"

trap 'if [ -n "${CLEANUP_DIR}" ] && [ -d "${CLEANUP_DIR}" ]; then echo "Cleaning up ${CLEANUP_DIR}..."; rm -rf "${CLEANUP_DIR}"; fi' EXIT

verify_copy() {
    local src="$1"
    local dst="$2"
    local log="${TMP_ROOT}/verify_${SLURM_JOB_ID}.log"
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

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

append_summary() {
    local hook_count="$1"
    local hook_selection="$2"
    local baseline="$3"
    local status="$4"
    local ring_mb="$5"
    local target_toks="$6"
    local compute_toks="$7"
    local log_file="$8"
    echo "${DATASET},${MODEL},${BATCH_SIZE},${hook_count},${hook_selection},${baseline},${status},${ring_mb},${target_toks},${compute_toks},${log_file}" >> "${SUMMARY_CSV}"
}

is_oom_log() {
    local log_file="$1"
    grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|oom_kill|Killed' "${log_file}"
}

run_proj_dmi_allow_fail() {
    local hook_count="$1"
    local hook_sel="$2"
    local log_file="${LOG_DIR}/${MODEL}__${DATASET}__bs${BATCH_SIZE}__hc${hook_count}__${hook_sel//,/__}__proj_dmi_compile.log"
    local status="FAIL_SKIP"
    local target_toks=""
    local compute_toks=""

    echo ""
    echo "=== proj_dmi_compile | dataset=${DATASET} hook_count=${hook_count} ring=${RING_MB}MB hooks=${hook_sel} ==="

    if python scripts/run_proj_dmi.py \
        --model "${MODEL}" \
        --batch-size "${BATCH_SIZE}" \
        --sample-file "${SAMPLE_FILE}" \
        --local-files-only \
        --max-input-tokens "${MAX_INPUT}" \
        --max-new-tokens "${MAX_OUTPUT}" \
        --limit "${LIMIT}" \
        --pad-buckets "${PAD_BUCKETS}" \
        --capture-mode "${CAPTURE_MODE}" \
        --results-dir "${RESULTS_DIR}" \
        --hook-selection "${hook_sel}" \
        --ring-payload-mb "${RING_MB}" \
        --ring-pinned-mb "${RING_MB}" \
        --ring-task-entries "${RING_TASK_ENTRIES}" \
        --drain-flush-payload-ratio 0 \
        --drain-flush-task-ratio 0 \
        --drain-flush-timeout-us "${RING_FLUSH_TIMEOUT_US}" \
        >"${log_file}" 2>&1; then
        cat "${log_file}"
        target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        status="OK"
    else
        cat "${log_file}"
        if is_oom_log "${log_file}"; then
            status="OOM_SKIP"
            echo "Skipping Proj-DMI after OOM"
        else
            echo "Skipping Proj-DMI after failure"
        fi
    fi

    append_summary "${hook_count}" "${hook_sel}" "proj_dmi_compile" "${status}" "${RING_MB}" "${target_toks}" "${compute_toks}" "${log_file}"
    cleanup_gpu
}

copy_model "models--Qwen--Qwen3-4B"
export HF_HOME="${LOCAL_HF}"
echo "HF_HOME=${HF_HOME}"

for idx in "${!HOOK_SELECTIONS[@]}"; do
    run_proj_dmi_allow_fail "${HOOK_COUNTS[$idx]}" "${HOOK_SELECTIONS[$idx]}"
done

echo ""
echo "============================================================"
echo "Hook-count DMI-only sweep done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "============================================================"
