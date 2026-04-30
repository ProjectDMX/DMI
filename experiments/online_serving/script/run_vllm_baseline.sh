#!/bin/bash
# Run vLLM baseline (no monitoring) benchmark at user-specified rates.
#
# Usage:
#   ./run_vllm_baseline.sh --model qwen4b --rates "1 2 4 8 16 32 64 128 256"
#   ./run_vllm_baseline.sh --model llama8b --rates "1 2 4 8 16 32 64"
#   ./run_vllm_baseline.sh --model qwen14b --rates "1 2 4 8 16 32"
#
# Prerequisites:
#   - source setup_env.sh baseline
#   - HuggingFace models downloaded in $HF_HOME

set -eo pipefail

# ── Parse arguments ─────────────────────────────────────────────────
MODEL_TAG=""
RATES="1 2 4 8 16 32 64 128 256"
RESULT_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/vllm_wo_monitor"
PORT=8010
DURATION=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL_TAG="$2";    shift 2 ;;
        --rates)    RATES="$2";        shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --port)     PORT="$2";         shift 2 ;;
        --duration) DURATION="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_TAG" ]; then
    echo "Usage: $0 --model {qwen4b|llama8b|qwen14b} [--rates \"1 2 4 ...\"]"
    exit 1
fi

# ── Resolve model path ──────────────────────────────────────────────
WORK_DIR=${WORK_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$WORK_DIR"
export HF_HOME=${HF_HOME:-${WORK_DIR}/hf_cache}

case $MODEL_TAG in
    qwen4b)  MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::') ;;
    llama8b) MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::') ;;
    qwen14b) MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-14B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::') ;;
    *) echo "Unknown model: $MODEL_TAG (use qwen4b, llama8b, qwen14b)"; exit 1 ;;
esac

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model $MODEL_TAG not found in $HF_HOME"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Environment ─────────────────────────────────────────────────────
# Activate your conda env before running, then this script uses that python.
ENV_PYTHON=${ENV_PYTHON:-python}
export VLLM_TARGET_DEVICE=cuda
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export XDG_CACHE_HOME=${WORK_DIR}/.cache
export VLLM_CACHE_ROOT=${WORK_DIR}/vllm_cache
export VLLM_DISABLE_COMPILE_CACHE=1

mkdir -p "$RESULT_DIR"

echo "========== vLLM Baseline: $MODEL_TAG  port=$PORT =========="
echo "Model: $MODEL_PATH"
echo "Rates: $RATES"

# ── Start server ────────────────────────────────────────────────────
$ENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --port $PORT &
SERVER_PID=$!

echo "Waiting for server (PID=$SERVER_PID)..."
for i in $(seq 1 900); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died"; exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server failed to start"
    kill $SERVER_PID 2>/dev/null; exit 1
fi

# ── Benchmark ───────────────────────────────────────────────────────
DATASETS=(
    "experiments/online_serving/sampled_datasets/sharegpt_seed42_n500_n30.json:sharegpt_s42"
    "experiments/online_serving/sampled_datasets/sharegpt_seed123_n500_n30.json:sharegpt_s123"
    "experiments/online_serving/sampled_datasets/sharegpt_seed456_n500_n30.json:sharegpt_s456"
    "experiments/online_serving/sampled_datasets/wildchat_seed42_n500_n30.json:wildchat_s42"
    "experiments/online_serving/sampled_datasets/wildchat_seed123_n500_n30.json:wildchat_s123"
    "experiments/online_serving/sampled_datasets/wildchat_seed456_n500_n30.json:wildchat_s456"
)

for ds_entry in "${DATASETS[@]}"; do
    DS_PATH="${ds_entry%%:*}"
    DS_TAG="${ds_entry##*:}"
    echo ""
    echo "--- Dataset: $DS_TAG ---"

    for rate in $RATES; do
        NP=$((rate * DURATION))
        OUTFILE="${MODEL_TAG}_${DS_TAG}_rate${rate}.json"
        echo "  rate=$rate num_prompts=$NP -> $OUTFILE"
        $ENV_PYTHON "$SCRIPT_DIR/run_bench.py" \
            --dataset-name sharegpt \
            --dataset-path "$DS_PATH" \
            --backend openai \
            --base-url http://localhost:$PORT \
            --model "$MODEL_PATH" \
            --sharegpt-output-len 128 \
            --request-rate "$rate" \
            --num-prompts "$NP" \
            --num-warmups 50 \
            --save-result \
            --result-dir "$RESULT_DIR" \
            --result-filename "$OUTFILE"
    done
done

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "========== Done =========="
