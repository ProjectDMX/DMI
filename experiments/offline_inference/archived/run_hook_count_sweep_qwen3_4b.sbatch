#!/bin/bash
#SBATCH --job-name=hook-count-qwen4b
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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

MODEL="qwen3-4b"
DATASET="${DATASET:-sharegpt}"
BATCH_SIZE=64
LIMIT=512
PAD_BUCKETS="128,256,384,512"
CAPTURE_MODE="hs_logits"
RING_STEP_MB=5120
RING_MIN_MB=1024
RING_MAX_MB=40960
RING_TASK_ENTRIES=131072
RING_FLUSH=0.15
RUN_TAG="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}"
RESULTS_DIR="${PROJECT}/experiments/offline_inference/results/hook_count_qwen3_4b_${DATASET}_${RUN_TAG}"
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
BASELINES=("hf_native_compile" "torch_hooks_eager" "proj_dmi_compile")

echo "dataset,model,batch_size,hook_count,hook_selection,baseline,status,ring_mb,target_tok_s,compute_tok_s,log_file" > "${SUMMARY_CSV}"

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
    echo "Copying model '${MODEL}' to local NVMe..."
    copy_model "models--Qwen--Qwen3-4B"
    export HF_HOME="${LOCAL_HF}"
    echo "HF_HOME=${HF_HOME}"
}

is_oom_log() {
    local log_file="$1"
    grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|oom_kill|Killed' "${log_file}"
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

cleanup_gpu() {
    echo "--- GPU cleanup ---"
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" || true
    sleep 5
}

common_args() {
    local hook_sel="$1"
    local include_hook="$2"
    local arr=(
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
    if [ "${include_hook}" = "1" ]; then
        arr+=(--hook-selection "${hook_sel}")
    fi
    printf '%q ' "${arr[@]}"
}

run_baseline_allow_fail() {
    local hook_count="$1"
    local hook_sel="$2"
    local baseline="$3"
    shift 3

    local hook_tag="${hook_sel//,/__}"
    local safe_label="${baseline// /_}"
    local log_file="${LOG_DIR}/${MODEL}__${DATASET}__bs${BATCH_SIZE}__hc${hook_count}__${hook_tag}__${safe_label}.log"
    local status="FAIL_SKIP"
    local target_toks=""
    local compute_toks=""

    echo ""
    echo "=== ${baseline} | dataset=${DATASET} hook_count=${hook_count} hook_selection=${hook_sel} ==="

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
            echo "Skipping baseline after failure: ${baseline}"
        fi
    fi

    append_summary "${hook_count}" "${hook_sel}" "${baseline}" "${status}" "" "${target_toks}" "${compute_toks}" "${log_file}"
    cleanup_gpu
    return 0
}

estimate_proj_dmi_ring_mb() {
    local hook_count="$1"
    local raw_mb
    case "${hook_count}" in
        1)
            raw_mb=1024
            ;;
        73)
            # q + k + logits at bs64 / prefill384 is ~8.5 GiB; use ~2x.
            raw_mb=17408
            ;;
        145)
            # q + k + pattern + attn_scores is ~49 GiB at worst-case cap.
            # 2x is not feasible once model/workspace are included, so cap.
            raw_mb=${RING_MAX_MB}
            ;;
        253)
            raw_mb=${RING_MAX_MB}
            ;;
        325)
            raw_mb=${RING_MAX_MB}
            ;;
        *)
            raw_mb=${RING_MAX_MB}
            ;;
    esac
    if [ "${raw_mb}" -lt "${RING_MIN_MB}" ]; then
        raw_mb="${RING_MIN_MB}"
    fi
    if [ "${raw_mb}" -gt "${RING_MAX_MB}" ]; then
        raw_mb="${RING_MAX_MB}"
    fi
    if [ "${raw_mb}" -gt "${RING_STEP_MB}" ]; then
        raw_mb=$(( raw_mb / RING_STEP_MB * RING_STEP_MB ))
    fi
    if [ "${raw_mb}" -lt "${RING_MIN_MB}" ]; then
        raw_mb="${RING_MIN_MB}"
    fi
    echo "${raw_mb}"
}

run_proj_dmi_with_retry() {
    local hook_count="$1"
    local hook_sel="$2"
    local ring_mb="$3"
    local hook_tag="${hook_sel//,/__}"
    local log_file="${LOG_DIR}/${MODEL}__${DATASET}__bs${BATCH_SIZE}__hc${hook_count}__${hook_tag}__proj_dmi_compile.log"
    local target_toks=""
    local compute_toks=""

    while true; do
        echo ""
        echo "=== proj_dmi_compile | dataset=${DATASET} hook_count=${hook_count} ring=${ring_mb}MB hooks=${hook_sel} ==="
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
            --ring-payload-mb "${ring_mb}" \
            --ring-pinned-mb "${ring_mb}" \
            --ring-task-entries "${RING_TASK_ENTRIES}" \
            --drain-flush-payload-ratio "${RING_FLUSH}" \
            --drain-flush-task-ratio "${RING_FLUSH}" \
            >"${log_file}" 2>&1; then
            cat "${log_file}"
            target_toks=$(grep -oP 'target_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
            compute_toks=$(grep -oP 'compute_tok/s=\K[0-9.]+' "${log_file}" | tail -1 || true)
            append_summary "${hook_count}" "${hook_sel}" "proj_dmi_compile" "OK" "${ring_mb}" "${target_toks}" "${compute_toks}" "${log_file}"
            cleanup_gpu
            return 0
        fi

        cat "${log_file}"
        if is_oom_log "${log_file}"; then
            if [ "${ring_mb}" -gt "${RING_MIN_MB}" ]; then
                append_summary "${hook_count}" "${hook_sel}" "proj_dmi_compile" "OOM_RETRY" "${ring_mb}" "" "" "${log_file}"
                ring_mb=$((ring_mb - RING_STEP_MB))
                if [ "${ring_mb}" -lt "${RING_MIN_MB}" ]; then
                    ring_mb="${RING_MIN_MB}"
                fi
                echo "Reducing ring to ${ring_mb} MB and retrying ..."
                cleanup_gpu
                continue
            fi
            echo "Skipping Proj-DMI after OOM exhaustion"
            append_summary "${hook_count}" "${hook_sel}" "proj_dmi_compile" "OOM_EXHAUSTED_SKIP" "${ring_mb}" "" "" "${log_file}"
        else
            echo "Skipping Proj-DMI after non-OOM failure"
            append_summary "${hook_count}" "${hook_sel}" "proj_dmi_compile" "FAIL_SKIP" "${ring_mb}" "" "" "${log_file}"
        fi
        cleanup_gpu
        return 0
    done
}

prepare_model_cache

for idx in "${!HOOK_SELECTIONS[@]}"; do
    hook_sel="${HOOK_SELECTIONS[$idx]}"
    hook_count="${HOOK_COUNTS[$idx]}"

    run_baseline_allow_fail "${hook_count}" "${hook_sel}" "hf_native_compile" \
        python scripts/run_hf_upper_bound.py \
        --model "${MODEL}" \
        --batch-size "${BATCH_SIZE}" \
        --sample-file "${SAMPLE_FILE}" \
        --local-files-only \
        --max-input-tokens "${MAX_INPUT}" \
        --max-new-tokens "${MAX_OUTPUT}" \
        --limit "${LIMIT}" \
        --pad-buckets "${PAD_BUCKETS}" \
        --capture-mode "${CAPTURE_MODE}" \
        --results-dir "${RESULTS_DIR}"

    run_baseline_allow_fail "${hook_count}" "${hook_sel}" "torch_hooks_eager" \
        python scripts/run_torch_hooks.py \
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
        --disable-compile \
        --hook-selection "${hook_sel}"

    initial_ring_mb=$(estimate_proj_dmi_ring_mb "${hook_count}")
    run_proj_dmi_with_retry "${hook_count}" "${hook_sel}" "${initial_ring_mb}"
done

echo ""
echo "============================================================"
echo "Hook-count sweep done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: ${RESULTS_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "============================================================"
