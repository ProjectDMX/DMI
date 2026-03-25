#!/bin/bash
#SBATCH --job-name=prefill-backpressure-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

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
BATCH_SIZE=2
PREFILL_TOKENS=16
NUM_MICROBATCHES=4
HEAVY_HOOKS="q,k,v,z,mlp_in,mlp_out,resid_mid,pattern,attn_scores,logits"
LIGHT_HOOKS="${HEAVY_HOOKS}"
RING_FLUSH=0.15
RESULTS_DIR="${PROJECT}/exp/offline_e2e/results/prefill_backpressure_smoke_$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

LOCAL_NVME_HF="/tmp/${USER}_hf_cache_${SLURM_JOB_ID}"
LOCAL_HF="${LOCAL_NVME_HF}"
CLEANUP_DIR="${LOCAL_NVME_HF}"
mkdir -p "${LOCAL_NVME_HF}/hub"
SRC_HUB="${SCRATCH}/hf_cache/hub"

trap 'if [ -n "${CLEANUP_DIR}" ] && [ -d "${CLEANUP_DIR}" ]; then echo "Cleaning up ${CLEANUP_DIR}..."; rm -rf "${CLEANUP_DIR}"; fi' EXIT

copy_model() {
    local model_dir="$1"
    local src="${SRC_HUB}/${model_dir}"
    local dst="${LOCAL_NVME_HF}/hub/${model_dir}"
    if [ -d "${src}" ] && [ ! -d "${dst}" ]; then
        cp -a "${src}" "${dst}"
    fi
    if [ -d "${dst}" ]; then
        LOCAL_HF="${LOCAL_NVME_HF}"
    else
        LOCAL_HF="${SCRATCH}/hf_cache"
    fi
}

copy_model "models--Qwen--Qwen3-4B"
export HF_HOME="${LOCAL_HF}"
echo "HF_HOME=${HF_HOME}"

COMMON=(
  --model "${MODEL}"
  --local-files-only
  --batch-size "${BATCH_SIZE}"
  --prefill-tokens "${PREFILL_TOKENS}"
  --num-microbatches "${NUM_MICROBATCHES}"
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

run_case "Smoke: hf_native" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline hf_native \
    --baseline-label hf_native

run_case "Smoke: torch_hooks heavy" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline torch_hooks \
    --baseline-label torch_hooks \
    --hook-selection "${HEAVY_HOOKS}"

run_case "Smoke: dmi_light" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline proj_dmi \
    --baseline-label dmi_light \
    --proj-dmi-mode ring_null \
    --hook-selection "${LIGHT_HOOKS}" \
    --ring-payload-mb 8192 \
    --ring-pinned-mb 8192 \
    --drain-flush-payload-ratio "${RING_FLUSH}" \
    --drain-flush-task-ratio "${RING_FLUSH}"

run_case "Smoke: dmi_heavy" \
  python run_prefill_backpressure.py "${COMMON[@]}" \
    --baseline proj_dmi \
    --baseline-label dmi_heavy \
    --proj-dmi-mode ring_null \
    --hook-selection "${HEAVY_HOOKS}" \
    --ring-payload-mb 2048 \
    --ring-pinned-mb 2048 \
    --drain-flush-payload-ratio "${RING_FLUSH}" \
    --drain-flush-task-ratio "${RING_FLUSH}"

echo ""
echo "============================================================"
echo "Prefill-backpressure smoke done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================================"
