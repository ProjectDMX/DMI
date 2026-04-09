#!/bin/bash
#SBATCH --job-name=step-breakdown
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

SCRATCH="${SCRATCH:-YOUR_SCRATCH_PATH}"
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
    # Reuse the same compiler/CUDA environment as the interactive job.
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

cd "${PROJECT}/experiments/offline_inference"

echo "============================================================"
echo "Node: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

MODEL="qwen3-4b"
PREFILL_TOKENS=128
WARMUP=2
ITERS=5
BATCH_SIZES=(1 8 32 64)
HOOK_SELECTION="hidden-states,final_ln,logits"
DMI_RING_PAYLOAD_MB=8192
DMI_RING_PINNED_MB=8192
DMI_RING_TASK_ENTRIES=65536

RUN_TAG="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT}/experiments/offline_inference/results/step_breakdown_qwen3_4b_${RUN_TAG}"
LOG_DIR="${RESULTS_DIR}/logs"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
echo "baseline,batch_size,status,prefill_compute_ms,prefill_total_ms,decode_compute_ms,decode_total_ms,results_json,log_file" > "${SUMMARY_CSV}"

TMP_ROOT="${TMP_ROOT:-${PWD}/.tmp}"
mkdir -p "${TMP_ROOT}"
LOCAL_NVME_HF="${TMP_ROOT}/${USER}_hf_cache_${SLURM_JOB_ID}"
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

append_summary() {
    local baseline="$1"
    local batch_size="$2"
    local status="$3"
    local results_json="$4"
    local log_file="$5"
    local prefill_compute=""; local prefill_total=""; local decode_compute=""; local decode_total=""
    if [ -n "${results_json}" ] && [ -f "${results_json}" ]; then
        prefill_compute=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["prefill"]["compute_ms"])' "${results_json}" 2>/dev/null || true)
        prefill_total=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["prefill"]["total_ms"])' "${results_json}" 2>/dev/null || true)
        decode_compute=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); x=d.get("decode_last_step") or d.get("decode_1") or {}; print(x.get("compute_ms",""))' "${results_json}" 2>/dev/null || true)
        decode_total=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); x=d.get("decode_last_step") or d.get("decode_1") or {}; print(x.get("total_ms",""))' "${results_json}" 2>/dev/null || true)
    fi
    echo "${baseline},${batch_size},${status},${prefill_compute},${prefill_total},${decode_compute},${decode_total},${results_json},${log_file}" >> "${SUMMARY_CSV}"
}

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

run_allow_fail() {
    local baseline="$1"
    local batch_size="$2"
    shift 2
    local log_file="${LOG_DIR}/${baseline}__bs${batch_size}.log"
    local status="FAIL_SKIP"
    local results_json=""

    echo ""
    echo "############################################################"
    echo "${baseline} | bs=${batch_size}"
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

    append_summary "${baseline}" "${batch_size}" "${status}" "${results_json}" "${log_file}"
    cleanup_gpu
}

copy_model "models--Qwen--Qwen3-4B"
export HF_HOME="${LOCAL_HF}"
echo "HF_HOME=${HF_HOME}"

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

    run_allow_fail "hf_ideal" "${BATCH_SIZE}" \
      bash ./run_step_breakdown_baseline.sh hf_ideal "${COMMON[@]}" \
        --baseline-label hf_ideal

    run_allow_fail "hf_api" "${BATCH_SIZE}" \
      bash ./run_step_breakdown_baseline.sh hf_api "${COMMON[@]}" \
        --baseline-label hf_api

    run_allow_fail "torch_hooks" "${BATCH_SIZE}" \
      bash ./run_step_breakdown_baseline.sh torch_hooks "${COMMON[@]}" \
        --baseline-label torch_hooks \
        --hook-selection "${HOOK_SELECTION}"

    run_allow_fail "dmi_manual" "${BATCH_SIZE}" \
      bash ./run_step_breakdown_baseline.sh proj_dmi_manual "${COMMON[@]}" \
        --baseline-label dmi_manual \
        --proj-dmi-mode ring_db \
        --hook-selection "${HOOK_SELECTION}" \
        --ring-payload-mb "${DMI_RING_PAYLOAD_MB}" \
        --ring-pinned-mb "${DMI_RING_PINNED_MB}" \
        --ring-task-entries "${DMI_RING_TASK_ENTRIES}"

    run_allow_fail "dmi_manual_eager" "${BATCH_SIZE}" \
      bash ./run_step_breakdown_baseline.sh proj_dmi_manual "${COMMON[@]}" \
        --baseline-label dmi_manual_eager \
        --disable-compile \
        --proj-dmi-mode ring_db \
        --hook-selection "${HOOK_SELECTION}" \
        --ring-payload-mb "${DMI_RING_PAYLOAD_MB}" \
        --ring-pinned-mb "${DMI_RING_PINNED_MB}" \
        --ring-task-entries "${DMI_RING_TASK_ENTRIES}"
done

echo ""
echo "============================================================"
echo "Step-breakdown microbenchmark done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary: ${SUMMARY_CSV}"
echo "============================================================"
