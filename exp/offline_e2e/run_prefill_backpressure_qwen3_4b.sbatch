#!/bin/bash
#SBATCH --job-name=prefill-backpressure
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclusive

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

MODEL="qwen3-4b"
DATASET="${DATASET:-sharegpt}"
BATCH_SIZE=64
LIMIT=128
PREFIX_LENGTHS="64,128,192,256,320,384,448"
HF_HOOK_SET="none"
DMI_LIGHT_HOOKS="logits"
DMI_HEAVY_HOOKS="q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits"
RING_PAYLOAD_MB=16384
RING_PINNED_MB=16384
RING_TASK_ENTRIES=65536

case "${DATASET}" in
    sharegpt)
        SAMPLE_FILE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
        ;;
    wildchat)
        SAMPLE_FILE="../../benchmark/data/offline_e2e/wildchat_500_sample1.jsonl"
        ;;
    *)
        echo "ERROR: unsupported DATASET=${DATASET}"
        exit 1
        ;;
esac

RUN_TAG="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT}/exp/offline_e2e/results/prefill_backpressure_${DATASET}_${RUN_TAG}"
LOG_DIR="${RESULTS_DIR}/logs"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
echo "baseline,status,hook_label,results_json,log_file" > "${SUMMARY_CSV}"

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

copy_model "models--Qwen--Qwen3-4B"
export HF_HOME="${LOCAL_HF}"
echo "HF_HOME=${HF_HOME}"

COMMON=(
  --model "${MODEL}"
  --sample-file "${SAMPLE_FILE}"
  --local-files-only
  --batch-size "${BATCH_SIZE}"
  --limit "${LIMIT}"
  --prefix-lengths "${PREFIX_LENGTHS}"
  --results-dir "${RESULTS_DIR}"
)

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

append_summary() {
    local baseline="$1"
    local status="$2"
    local hook_label="$3"
    local results_json="$4"
    local log_file="$5"
    echo "${baseline},${status},${hook_label},${results_json},${log_file}" >> "${SUMMARY_CSV}"
}

run_allow_fail() {
    local baseline="$1"
    local hook_label="$2"
    shift 2
    local log_file="${LOG_DIR}/${baseline}.log"
    local status="FAIL_SKIP"
    local results_json=""

    echo ""
    echo "############################################################"
    echo "${baseline} | hooks=${hook_label}"
    echo "############################################################"

    if "$@" >"${log_file}" 2>&1; then
        cat "${log_file}"
        results_json=$(grep -oP '^Saved \K.*json$' "${log_file}" | tail -1 || true)
        status="OK"
    else
        cat "${log_file}"
        if grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|OOM|oom_kill' "${log_file}"; then
            status="OOM_SKIP"
        fi
        echo "Skipping ${baseline} after failure"
    fi

    append_summary "${baseline}" "${status}" "${hook_label}" "${results_json}" "${log_file}"
    cleanup_gpu
}

run_allow_fail "hf_native" "none" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline hf_native \
    --baseline-label hf_native

run_allow_fail "torch_hooks" "${DMI_HEAVY_HOOKS}" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline torch_hooks \
    --baseline-label torch_hooks \
    --hook-selection "${DMI_HEAVY_HOOKS}"

run_allow_fail "dmi_light" "${DMI_LIGHT_HOOKS}" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline proj_dmi \
    --baseline-label dmi_light \
    --proj-dmi-mode ring_db \
    --hook-selection "${DMI_LIGHT_HOOKS}" \
    --ring-payload-mb "${RING_PAYLOAD_MB}" \
    --ring-pinned-mb "${RING_PINNED_MB}" \
    --ring-task-entries "${RING_TASK_ENTRIES}"

run_allow_fail "dmi_heavy" "${DMI_HEAVY_HOOKS}" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline proj_dmi \
    --baseline-label dmi_heavy \
    --proj-dmi-mode ring_db \
    --hook-selection "${DMI_HEAVY_HOOKS}" \
    --ring-payload-mb "${RING_PAYLOAD_MB}" \
    --ring-pinned-mb "${RING_PINNED_MB}" \
    --ring-task-entries "${RING_TASK_ENTRIES}"

echo ""
echo "============================================================"
echo "Prefill-backpressure sweep done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary: ${SUMMARY_CSV}"
echo "============================================================"
