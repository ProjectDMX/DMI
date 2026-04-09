#!/bin/bash
#SBATCH --job-name=sweep-internal-hooks
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

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

MODELS=("qwen3-4b" "qwen3-14b")
DATASETS=("sharegpt" "wildchat")
BATCHES=(16 32 64)
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

RUN_TAG="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT}/experiments/offline_inference/results/full_sweep_${HOOK_TAG}_${RUN_TAG}"
LOG_DIR="${RESULTS_DIR}/logs"
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

echo "model,dataset,batch_size,baseline,capture_mode,hook_label,status,ring_mb,target_tok_s,compute_tok_s,log_file" > "${SUMMARY_CSV}"

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
        qwen3-1.7b)
            copy_model "models--Qwen--Qwen3-1.7B"
            ;;
        llama3.1-8b)
            copy_model "models--meta-llama--Llama-3.1-8B-Instruct"
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

run_baseline_allow_fail() {
    local model="$1"
    local dataset="$2"
    local bs="$3"
    local baseline="$4"
    shift 4

    local safe_label="${baseline// /_}"
    safe_label="${safe_label//\//_}"
    local log_file="${LOG_DIR}/${model}__${dataset}__bs${bs}__${HOOK_TAG}__${safe_label}.log"
    local status="FAIL"
    local target_toks=""
    local compute_toks=""

    echo ""
    echo "=== ${baseline} | model=${model} dataset=${dataset} bs=${bs} hooks=${HOOK_SELECTION} ==="

    if "$@" >"${log_file}" 2>&1; then
        cat "${log_file}"
        target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
        status="OK"
    else
        cat "${log_file}"
        if is_oom_log "${log_file}"; then
            status="OOM_SKIP"
            echo "Skipping baseline after OOM: ${baseline}"
        else
            status="FAIL_SKIP"
            echo "Skipping baseline after failure: ${baseline}"
        fi
    fi

    append_summary "${model}" "${dataset}" "${bs}" "${baseline}" "${status}" "" "${target_toks}" "${compute_toks}" "${log_file}"
    cleanup_gpu
    return 0
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

echo ""
echo "Planned matrix:"
echo "  models   : ${MODELS[*]}"
echo "  datasets : ${DATASETS[*]}"
echo "  batches  : ${BATCHES[*]}"
echo "  limit    : ${LIMIT}"
echo "  capture  : ${CAPTURE_MODE}"
echo "  hooks    : ${HOOK_SELECTION}"
echo "  results  : ${RESULTS_DIR}"
echo ""
echo "Total configs: 2 models x 2 datasets x 3 batch sizes = 12"
echo "Total baseline invocations planned: 12 x 4 = 48 (plus Proj-DMI retries)"
echo ""

for model in "${MODELS[@]}"; do
    prepare_model_cache "${model}"

    for dataset in "${DATASETS[@]}"; do
        case "${dataset}" in
            sharegpt)
                SAMPLE_FILE="${SAMPLE_FILE_SHAREGPT}"
                MAX_INPUT=200
                MAX_OUTPUT=750
                ;;
            wildchat)
                SAMPLE_FILE="${SAMPLE_FILE_WILDCHAT}"
                MAX_INPUT=250
                MAX_OUTPUT=1000
                ;;
            *)
                echo "Unknown dataset: ${dataset}"
                exit 1
                ;;
        esac

        for bs in "${BATCHES[@]}"; do
            echo ""
            echo "############################################################"
            echo "Config: model=${model} dataset=${dataset} bs=${bs} input=${MAX_INPUT} output=${MAX_OUTPUT} limit=${LIMIT} hooks=${HOOK_SELECTION}"
            echo "############################################################"

            COMMON=(--model "${model}" --batch-size "${bs}" --sample-file "${SAMPLE_FILE}" --local-files-only \
                --max-input-tokens "${MAX_INPUT}" --max-new-tokens "${MAX_OUTPUT}" --limit "${LIMIT}" \
                --pad-buckets "${PAD_BUCKETS}" --results-dir "${RESULTS_DIR}" --capture-mode "${CAPTURE_MODE}")

            run_baseline_allow_fail "${model}" "${dataset}" "${bs}" "hf_upper_bound_compile" \
                python scripts/run_hf_upper_bound.py "${COMMON[@]}"

            run_baseline_allow_fail "${model}" "${dataset}" "${bs}" "torch_hooks_eager" \
                python scripts/run_torch_hooks.py "${COMMON[@]}" --hook-selection "${HOOK_SELECTION}" --disable-compile

            run_baseline_allow_fail "${model}" "${dataset}" "${bs}" "nnsight_eager" \
                python scripts/run_nnsight.py "${COMMON[@]}" --hook-selection "${HOOK_SELECTION}" --disable-compile

            run_proj_dmi_with_retries "${model}" "${dataset}" "${bs}" "proj_dmi_compile" "${MAX_INPUT}" "${MAX_OUTPUT}" \
                python scripts/run_proj_dmi.py "${COMMON[@]}" --hook-selection "${HOOK_SELECTION}"
        done
    done
done

echo ""
echo "============================================================"
echo "Full internal-hooks sweep done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Hook label : ${HOOK_LABEL}"
echo "============================================================"
