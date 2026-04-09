#!/bin/bash
#SBATCH --job-name=ahk-qwen14b
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --output=cur_sbatch_log/ahk-qwen14b-%j.out
#SBATCH --error=cur_sbatch_log/ahk-qwen14b-%j.err

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${WORK_ROOT}"

source vllm-hook-env/bin/activate
export PYTHONPATH=$(pwd)/vllm-0.17.0:$PYTHONPATH
export VLLM_TARGET_DEVICE=cuda
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$(pwd)/.cache
export VLLM_CACHE_ROOT=$(pwd)/vllm_cache
export TORCHDYNAMO_DISABLE=1

mkdir -p hook_bench_results

MODEL_PATH=$(ls -d hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/*/ | head -1 | sed 's:/$::')
PORT=8022

echo "========== Adaptive Hook: qwen14b  port=$PORT  bracket=[1,4] =========="

# Setup hooks
HOOK_DIR=$(pwd)/hook_bench_results/_qk_tmp_qwen14b
mkdir -p "$HOOK_DIR"
export VLLM_HOOK_DIR="$HOOK_DIR"
export VLLM_HOOK_FLAG="$HOOK_DIR/EXTRACT.flag"
export VLLM_RUN_ID="$HOOK_DIR/RUN_ID.txt"
export VLLM_HOOKQ_MODE="last_token"

NUM_LAYERS=$(python -c "
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
echo "hook_adaptive" > "$VLLM_RUN_ID"
touch "$VLLM_HOOK_FLAG"
echo "Layers: $NUM_LAYERS, hooks: all (stride=1)"

python -m vllm.entrypoints.openai.api_server \
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
        echo "Server process died!"
        exit 1
    fi
    if curl -s --max-time 2 http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
sleep 5

python adaptive_bench.py \
    --base-url http://localhost:$PORT \
    --model "$MODEL_PATH" \
    --tag "qwen14b" \
    --result-dir hook_bench_results \
    --low-rate 1 --high-rate 4 \
    --multiplier 5 \
    --resolution 0.25 \
    --duration 30 \
    --cooldown 30 \
    --fractional

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
rm -f "$VLLM_HOOK_FLAG"
echo "========== Done =========="
