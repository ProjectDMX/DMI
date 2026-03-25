#!/bin/bash
#SBATCH --job-name=step-breakdown-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/ynn1999
ENV_DIR=${SCRATCH}/proj-dmx
PROJECT=${SCRATCH}/DMI

if [ -d "${PROJECT}/DMI" ]; then
    PROJECT=${PROJECT}/DMI
fi

if [ ! -d "${ENV_DIR}" ]; then
    echo "Extracting conda environment ..."
    tar xzf "${SCRATCH}/proj-dmx.tar.gz" -C "${SCRATCH}"
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "ERROR: ${ENV_DIR}/bin/python not found!"
    exit 1
fi

if [ -f "${SCRATCH}/export_env.sh" ]; then
    # shellcheck disable=SC1090
    source "${SCRATCH}/export_env.sh"
fi

export ENV_PYTHON="${ENV_DIR}/bin/python"
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${PROJECT}/transformers/src:${PROJECT}"
export TL_ENABLE_NVTX=1
export DMI_ROOT="${PROJECT}"

cd "${PROJECT}/exp/offline_e2e"

echo "============================================================"
echo "Node: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

if command -v g++-11 >/dev/null 2>&1; then
    CXX_BIN="$(command -v g++-11)"
else
    CXX_BIN="$(command -v g++)"
fi

NVCC_BIN=""
if command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
else
    for cand in \
        "${ENV_DIR}/bin/nvcc" \
        "/usr/local/cuda/bin/nvcc" \
        "/usr/local/cuda-12.4/bin/nvcc" \
        "/usr/local/cuda-12.3/bin/nvcc" \
        "/usr/local/cuda-12.2/bin/nvcc" \
        "/usr/local/cuda-12.1/bin/nvcc" \
        "/usr/local/cuda-12.0/bin/nvcc" \
        "/opt/cuda/bin/nvcc"
    do
        if [ -x "${cand}" ]; then
            NVCC_BIN="${cand}"
            break
        fi
    done
fi

if [ -z "${NVCC_BIN}" ]; then
    echo "ERROR: nvcc not found on compute node"
    exit 1
fi

export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
export PATH="$(dirname "${NVCC_BIN}"):${PATH}"
echo "CXX=${CXX_BIN}"
echo "NVCC=${NVCC_BIN}"
echo "CUDA_HOME=${CUDA_HOME}"

echo "Building monitoring extension inside job..."
make -C "${PROJECT}/monitoring" -j "${SLURM_CPUS_PER_TASK:-8}" \
    PYTHON="${ENV_DIR}/bin/python" \
    CXX="${CXX_BIN}" \
    NVCC="${NVCC_BIN} -ccbin ${CXX_BIN}"

MODEL="qwen3-4b"
PREFILL_TOKENS=128
WARMUP=1
ITERS=2
BATCH_SIZES=(1 8)
HOOK_SELECTION="hidden-states,final_ln,logits"
RESULTS_DIR="${PROJECT}/exp/offline_e2e/results/step_breakdown_smoke_$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
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

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    COMMON=(
      --model "${MODEL}"
      --local-files-only
      --batch-size "${BATCH_SIZE}"
      --prefill-tokens "${PREFILL_TOKENS}"
      --warmup "${WARMUP}"
      --iters "${ITERS}"
      --results-dir "${RESULTS_DIR}"
    )

    run_case "Smoke: hf_ideal bs=${BATCH_SIZE}" \
      python run_step_breakdown_microbench.py "${COMMON[@]}" \
        --baseline hf_ideal \
        --baseline-label hf_ideal

    run_case "Smoke: hf_api bs=${BATCH_SIZE}" \
      python run_step_breakdown_microbench.py "${COMMON[@]}" \
        --baseline hf_api \
        --baseline-label hf_api

    run_case "Smoke: torch_hooks bs=${BATCH_SIZE}" \
      python run_step_breakdown_microbench.py "${COMMON[@]}" \
        --baseline torch_hooks \
        --baseline-label torch_hooks \
        --hook-selection "${HOOK_SELECTION}"

    run_case "Smoke: dmi bs=${BATCH_SIZE}" \
      python run_step_breakdown_microbench.py "${COMMON[@]}" \
        --baseline proj_dmi \
        --baseline-label dmi \
        --proj-dmi-mode ring_db \
        --hook-selection "${HOOK_SELECTION}" \
        --ring-payload-mb 4096 \
        --ring-pinned-mb 4096
done

echo ""
echo "============================================================"
echo "Step-breakdown microbenchmark smoke done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================================"
