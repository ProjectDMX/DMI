#!/bin/bash
# Run vLLM-Hook benchmark at user-specified rates.
# NOTE: --enforce-eager is REQUIRED (CUDA graphs bypass Python hooks).
#
# Usage:
#   ./run_vllm_hook.sh --model qwen4b --rates "1 2 4 8 16 32 64"
#   ./run_vllm_hook.sh --model llama8b --rates "1 2 4 8 16"

set -eo pipefail

# ── Parse arguments ─────────────────────────────────────────────────
MODEL_TAG=""
RATES="1 2 4 8 16 32 64"
RESULT_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/vllm_hook"
PORT=8020
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
WORK_DIR=${WORK_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$WORK_DIR"
export HF_HOME=${HF_HOME:-${WORK_DIR}/hf_cache}

case $MODEL_TAG in
    qwen4b)  MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::') ;;
    llama8b) MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::') ;;
    qwen14b) MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-14B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::') ;;
    *) echo "Unknown model: $MODEL_TAG"; exit 1 ;;
esac

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model $MODEL_TAG not found in $HF_HOME"; exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Environment ─────────────────────────────────────────────────────
# Activate your conda env (with vllm-hook-plugins installed) before running.
ENV_PYTHON=${ENV_PYTHON:-python}
export VLLM_TARGET_DEVICE=cuda
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export XDG_CACHE_HOME=${WORK_DIR}/.cache
export VLLM_CACHE_ROOT=${WORK_DIR}/vllm_cache
export TORCHDYNAMO_DISABLE=1

mkdir -p "$RESULT_DIR"

# ── Setup hooks ─────────────────────────────────────────────────────
HOOK_DIR=${WORK_DIR}/hook_bench_results/_qk_tmp_${MODEL_TAG}
mkdir -p "$HOOK_DIR"
export VLLM_HOOK_DIR="$HOOK_DIR"
export VLLM_HOOK_FLAG="$HOOK_DIR/EXTRACT.flag"
export VLLM_RUN_ID="$HOOK_DIR/RUN_ID.txt"
export VLLM_HOOKQ_MODE="last_token"

NUM_LAYERS=$($ENV_PYTHON -c "
import json, os
with open(os.path.join('$MODEL_PATH', 'config.json')) as f:
    print(json.load(f).get('num_hidden_layers', 32))
")
LAYER_HEADS=""
for ((l=0; l<NUM_LAYERS; l+=1)); do
    [ -n "$LAYER_HEADS" ] && LAYER_HEADS="${LAYER_HEADS};"
    LAYER_HEADS="${LAYER_HEADS}${l}:0"
done
export VLLM_HOOK_LAYER_HEADS="$LAYER_HEADS"
echo "hook_bench" > "$VLLM_RUN_ID"
touch "$VLLM_HOOK_FLAG"

echo "========== vLLM-Hook: $MODEL_TAG  port=$PORT =========="
echo "Model: $MODEL_PATH"
echo "Layers: $NUM_LAYERS, hooks: all"
echo "Rates: $RATES"

# ── Start server ────────────────────────────────────────────────────
$ENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --port $PORT \
    --enforce-eager \
    --worker-cls "vllm_hook_plugins.workers.probe_hidden_states_worker.ProbeHiddenStatesWorker" &
SERVER_PID=$!

echo "Waiting for server (PID=$SERVER_PID)..."
for i in $(seq 1 1800); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died!"; exit 1
    fi
    if curl -s --max-time 2 http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"; break
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server failed to start"
    kill $SERVER_PID 2>/dev/null; exit 1
fi
sleep 5

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
rm -f "$VLLM_HOOK_FLAG"
echo "========== Done =========="
