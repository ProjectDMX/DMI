#!/bin/bash
#SBATCH --job-name=admi-qwen14b
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --output=cur_sbatch_log/admi-qwen14b-%j.out
#SBATCH --error=cur_sbatch_log/admi-qwen14b-%j.err

WORK_ROOT=""
cd ""

ENV_PYTHON=$(pwd)/envs/vllm-h100/bin/python3.10
SITE=$(pwd)/dmi-env/lib/python3.10/site-packages
DMI=$(pwd)/DMI
export PYTHONPATH=$DMI/integration/vllm:$DMI:$DMI/transformers/src:$SITE
export LD_LIBRARY_PATH=$(pwd)/dmi-env/lib:$(pwd)/DMI/libs/clickhouse-cpp/build/clickhouse${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VLLM_TARGET_DEVICE=cuda
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_MODULE_LOADING=EAGER
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$(pwd)/.cache
export VLLM_CACHE_ROOT=$(pwd)/vllm_cache

mkdir -p dmil_results

DMI_ARGS='{"dmx_hook_selection": "hidden-states", "dmx_ring_payload_mb": 4096, "dmx_ring_pinned_mb": 4096}'

MODEL_PATH=$(ls -d hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/*/ | head -1 | sed 's:/$::')
PORT=8042

echo "========== Adaptive DMI: qwen14b  port=$PORT  bracket=[32,64] =========="

$ENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --port $PORT \
    --worker-cls monitoring.vllm_integration.DMXGPUWorker \
    --additional-config "$DMI_ARGS" &
SERVER_PID=$!

echo "Waiting for server (PID=$SERVER_PID)..."
for i in $(seq 1 1800); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

$ENV_PYTHON adaptive_bench.py \
    --base-url http://localhost:$PORT \
    --model "$MODEL_PATH" \
    --tag "qwen14b" \
    --result-dir dmil_results \
    --result-prefix "dmil_" \
    --low-rate 32 --high-rate 64 \
    --multiplier 5 \
    --resolution 4 \
    --duration 30 \
    --cooldown 30

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "========== Done =========="
