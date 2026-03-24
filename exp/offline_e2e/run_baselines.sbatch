#!/bin/bash
#SBATCH --job-name=offline-e2e
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -eo pipefail

SCRATCH=/scratch/zt1/project/zaoxing-prj/user/ynn1999
ENV_DIR=${SCRATCH}/proj-dmx
PROJECT=${SCRATCH}/DMI/DMI

# ---- 解包 conda 环境（如果还没解过就解） ----
if [ ! -d "${ENV_DIR}" ]; then
    echo "Extracting conda environment ..."
    tar xzf ${SCRATCH}/proj-dmx.tar.gz -C "${SCRATCH}"
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "ERROR: ${ENV_DIR}/bin/python not found!"
    exit 1
fi

export PATH=${ENV_DIR}/bin:${PATH}
export CUDA_HOME=${ENV_DIR}
export LD_LIBRARY_PATH=${ENV_DIR}/lib:${LD_LIBRARY_PATH}
export HF_HOME=${SCRATCH}/hf_cache
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=${PROJECT}/transformers/src:${PROJECT}
export TL_ENABLE_NVTX=1

cd ${PROJECT}/exp/offline_e2e

echo "============================================================"
echo "Node: $(hostname)  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python: $(python --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

# ---- 实验参数 ----
MODEL="qwen3-4b"
BS=32
SAMPLE_FILE="../../benchmark/data/offline_e2e/sharegpt_500_sample1.jsonl"
MAX_INPUT=250
MAX_OUTPUT=1000
LIMIT=100
RING_MB=56320        # 55GB
FLUSH=0.15
PAD_BUCKETS="64,128,256,512"

COMMON="--model ${MODEL} --batch-size ${BS} --sample-file ${SAMPLE_FILE} --local-files-only --max-input-tokens ${MAX_INPUT} --max-new-tokens ${MAX_OUTPUT} --limit ${LIMIT} --pad-buckets ${PAD_BUCKETS}"

# Collect results: "label|target_tok/s|compute_tok/s"
declare -a RESULTS=()

run_baseline() {
    local label="$1"; shift
    echo ""
    echo "=== ${label} ==="
    local output
    output=$("$@" 2>&1)
    echo "${output}"
    # Extract target_tok/s and compute_tok/s from output line
    local target_toks compute_toks
    target_toks=$(echo "${output}" | grep -oP 'target_tok/s=\K[0-9.]+' | tail -1)
    compute_toks=$(echo "${output}" | grep -oP 'compute_tok/s=\K[0-9.]+' | tail -1)
    RESULTS+=("${label}|${target_toks}|${compute_toks}")
    echo "--- GPU cleanup ---"
    python -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()"
    sleep 5
}

# ---- Run baselines ----
run_baseline "hf_upper_bound (compile)" \
    python run_hf_upper_bound.py ${COMMON}

run_baseline "hf_upper_bound (eager)" \
    python run_hf_upper_bound.py ${COMMON} --disable-compile

run_baseline "hf_monitor (generate, eager)" \
    python run_hf_monitor.py ${COMMON}

run_baseline "hf_monitor_manual (compile)" \
    python run_hf_monitor_manual.py ${COMMON}

run_baseline "proj_dmi (compile) ${RING_MB}MB" \
    python run_proj_dmi.py ${COMMON} \
    --ring-payload-mb ${RING_MB} --ring-pinned-mb ${RING_MB} \
    --ring-task-entries 131072 \
    --drain-flush-payload-ratio ${FLUSH} --drain-flush-task-ratio ${FLUSH}

run_baseline "proj_dmi (eager) ${RING_MB}MB" \
    python run_proj_dmi.py ${COMMON} --disable-compile \
    --ring-payload-mb ${RING_MB} --ring-pinned-mb ${RING_MB} \
    --ring-task-entries 131072 \
    --drain-flush-payload-ratio ${FLUSH} --drain-flush-task-ratio ${FLUSH}

# ---- Print summary table ----
echo ""
echo "============================================================"
echo "All baselines done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config: model=${MODEL} bs=${BS} input=${MAX_INPUT} output=${MAX_OUTPUT} samples=${LIMIT} buckets=${PAD_BUCKETS}"
echo "============================================================"
echo ""

# Extract upper_bound target_tok/s as reference
REF_TOKS=$(echo "${RESULTS[0]}" | cut -d'|' -f2)

# Column widths
W1=34; W2=14; W3=15; W4=17
SEP_LINE="├$(printf '─%.0s' $(seq 1 $W1))┼$(printf '─%.0s' $(seq 1 $W2))┼$(printf '─%.0s' $(seq 1 $W3))┼$(printf '─%.0s' $(seq 1 $W4))┤"
TOP_LINE="┌$(printf '─%.0s' $(seq 1 $W1))┬$(printf '─%.0s' $(seq 1 $W2))┬$(printf '─%.0s' $(seq 1 $W3))┬$(printf '─%.0s' $(seq 1 $W4))┐"
BOT_LINE="└$(printf '─%.0s' $(seq 1 $W1))┴$(printf '─%.0s' $(seq 1 $W2))┴$(printf '─%.0s' $(seq 1 $W3))┴$(printf '─%.0s' $(seq 1 $W4))┘"

echo "${TOP_LINE}"
printf "│ %-$((W1-2))s │ %-$((W2-2))s │ %-$((W3-2))s │ %-$((W4-2))s │\n" "Baseline" "target_tok/s" "compute_tok/s" "overhead"
echo "${SEP_LINE}"

for i in "${!RESULTS[@]}"; do
    IFS='|' read -r label ttoks ctoks <<< "${RESULTS[$i]}"
    if [ "$i" -eq 0 ]; then
        overhead="—"
    else
        overhead=$(python -c "
ref=${REF_TOKS}; cur=${ttoks}
ratio = ref / cur if cur > 0 else 0
if ratio < 1.0:
    print(f'{ratio:.2f}x (faster)')
else:
    print(f'{ratio:.2f}x')
")
    fi
    printf "│ %-$((W1-2))s │ %$((W2-2)).1f │ %$((W3-2)).1f │ %-$((W4-2))s │\n" \
        "${label}" "${ttoks}" "${ctoks}" "${overhead}"
    if [ "$i" -lt $((${#RESULTS[@]}-1)) ]; then
        echo "${SEP_LINE}"
    fi
done
echo "${BOT_LINE}"
