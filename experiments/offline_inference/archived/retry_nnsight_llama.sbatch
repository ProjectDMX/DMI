#!/bin/bash
#SBATCH --job-name=retry-nnsight-llama
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
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

LIMIT=128
PAD_BUCKETS="64,128,256,512"
SAMPLE_FILE_SHAREGPT="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
RING_STEP_MB=5120
RING_MIN_MB=5120

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

    if [ ! -d "${src}" ]; then
        echo "ERROR: source model not found: ${src}"
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
        LOCAL_HF="${SCRATCH}/hf_cache"
        return 0
    fi

    echo "  Copy verified OK: $(du -sh "${dst}" | cut -f1)"
    LOCAL_HF="${LOCAL_NVME_HF}"
}

echo "Copying llama3.1-8b to NVMe..."
copy_model "models--meta-llama--Llama-3.1-8B-Instruct"
export HF_HOME="${LOCAL_HF}"
echo "HF_HOME=${HF_HOME}"

is_oom_log() {
    local log_file="$1"
    grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM' "${log_file}"
}

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

run_nnsight() {
    local label="$1"
    local capture_mode="$2"
    local hook_label="$3"
    local hook_tag="$4"
    local results_dir="$5"
    shift 5

    local log_dir="${results_dir}/logs"
    local summary_csv="${results_dir}/summary.csv"
    local safe_label="nnsight_eager"
    local log_file="${log_dir}/llama3.1-8b__sharegpt__bs16__${hook_tag}__${safe_label}.log"

    echo ""
    echo "=== RETRY: ${label} ==="

    if "$@" >"${log_file}" 2>&1; then
        cat "${log_file}"
        local target_toks compute_toks
        target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        echo "llama3.1-8b,sharegpt,16,nnsight_eager,${capture_mode},${hook_label},OK,,${target_toks},${compute_toks},${log_file}" >> "${summary_csv}"
    else
        cat "${log_file}"
        local status="FAIL_SKIP"
        if is_oom_log "${log_file}"; then status="OOM_SKIP"; fi
        echo "llama3.1-8b,sharegpt,16,nnsight_eager,${capture_mode},${hook_label},${status},,,,${log_file}" >> "${summary_csv}"
    fi
    cleanup_gpu
}

COMMON_BASE=(--model llama3.1-8b --batch-size 16 --sample-file "${SAMPLE_FILE_SHAREGPT}" --local-files-only \
    --max-input-tokens 200 --max-new-tokens 750 --limit "${LIMIT}" --pad-buckets "${PAD_BUCKETS}")

# 1. hs_logits sweep — nnsight sharegpt/bs16
RESULTS_HS="${PROJECT}/experiments/offline_inference/results/full_sweep_llama31_8b_capture_hs_logits_20260324_214639_18691396"
run_nnsight "nnsight hs_logits sharegpt/bs16" "hs_logits" "capture_hs_logits" "capture_hs_logits" "${RESULTS_HS}" \
    python scripts/run_nnsight.py "${COMMON_BASE[@]}" --results-dir "${RESULTS_HS}" --capture-mode hs_logits

# 2. internal hooks sweep — nnsight sharegpt/bs16
HOOK_SELECTION="q,k,v,z,mlp_in,mlp_out,resid_mid"
HOOK_TAG="q__k__v__z__mlp_in__mlp_out__resid_mid"
RESULTS_IH="${PROJECT}/experiments/offline_inference/results/full_sweep_llama31_8b_q__k__v__z__mlp_in__mlp_out__resid_mid_20260324_222322_18691591"
run_nnsight "nnsight internal_hooks sharegpt/bs16" "hs" "internal_hooks:${HOOK_SELECTION}" "${HOOK_TAG}" "${RESULTS_IH}" \
    python scripts/run_nnsight.py "${COMMON_BASE[@]}" --results-dir "${RESULTS_IH}" --capture-mode hs --hook-selection "${HOOK_SELECTION}" --disable-compile

echo ""
echo "============================================================"
echo "Retry done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
