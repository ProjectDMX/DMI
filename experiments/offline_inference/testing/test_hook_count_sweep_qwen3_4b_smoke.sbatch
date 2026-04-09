#!/bin/bash
#SBATCH --job-name=hook-count-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRATCH="${SCRATCH:-}"
if [ -z "${SCRATCH}" ] || [ "${SCRATCH}" = "YOUR_SCRATCH_PATH" ]; then
    echo "ERROR: SCRATCH is not set to a valid scratch directory." >&2
    echo "Please export SCRATCH=/path/to/your/scratch before running this script." >&2
    exit 1
fi
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
SAMPLE_FILE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
BATCH_SIZE=2
LIMIT=4
MAX_INPUT=128
MAX_OUTPUT=32
PAD_BUCKETS="64,128"
CAPTURE_MODE="hs_logits"
RESULTS_DIR="${PROJECT}/experiments/offline_inference/results/hook_count_smoke_qwen3_4b_$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

TMP_ROOT="${TMP_ROOT:-${TMPDIR:-/tmp}}"
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

copy_model "models--Qwen--Qwen3-4B"
export HF_HOME="${LOCAL_HF}"
echo "HF_HOME=${HF_HOME}"

COMMON=(
  --model "${MODEL}"
  --batch-size "${BATCH_SIZE}"
  --sample-file "${SAMPLE_FILE}"
  --local-files-only
  --max-input-tokens "${MAX_INPUT}"
  --max-new-tokens "${MAX_OUTPUT}"
  --limit "${LIMIT}"
  --pad-buckets "${PAD_BUCKETS}"
  --capture-mode "${CAPTURE_MODE}"
  --results-dir "${RESULTS_DIR}"
)

run_case() {
    local label="$1"
    shift
    echo ""
    echo "############################################################"
    echo "${label}"
    echo "############################################################"
    "$@"
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 3
}

HOOK_325="q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits"

run_case "Smoke: hook325 / hf_upper_bound_compile" \
  python scripts/run_hf_upper_bound.py "${COMMON[@]}"

run_case "Smoke: hook325 / torch_hooks_eager" \
  python scripts/run_torch_hooks.py "${COMMON[@]}" --disable-compile --hook-selection "${HOOK_325}"

run_case "Smoke: hook325 / proj_dmi_compile" \
  python scripts/run_proj_dmi.py "${COMMON[@]}" --hook-selection "${HOOK_325}" --proj-dmi-mode ring_null --ring-payload-mb 5120 --ring-pinned-mb 5120 --ring-task-entries 65536

echo ""
echo "============================================================"
echo "Hook-count smoke done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================================"
