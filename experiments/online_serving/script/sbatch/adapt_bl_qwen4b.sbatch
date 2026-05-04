#!/bin/bash
#SBATCH --job-name=abl-qwen4b
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --output=cur_sbatch_log/abl-qwen4b-%j.out
#SBATCH --error=cur_sbatch_log/abl-qwen4b-%j.err

WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${WORK_ROOT}"

ENV_PYTHON=$(pwd)/envs/vllm-h100/bin/python3.10
SITE=$(pwd)/dmi-env/lib/python3.10/site-packages
export PYTHONPATH=$(pwd)/vllm-0.17.0:$(pwd)/DMI/transformers/src:$SITE
export VLLM_TARGET_DEVICE=cuda
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$(pwd)/.cache
export VLLM_CACHE_ROOT=$(pwd)/vllm_cache
export VLLM_DISABLE_COMPILE_CACHE=1

mkdir -p ttft_results

MODEL_PATH=$(ls -d hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/ | head -1 | sed 's:/$::')
PORT=8010

echo "========== Adaptive baseline: qwen4b  port=$PORT  bracket=[128,256] =========="

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
    --tag "qwen4b" \
    --result-dir ttft_results \
    --low-rate 128 --high-rate 256 \
    --multiplier 5 \
    --resolution 4 \
    --duration 30 \
    --cooldown 30

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "========== Done =========="
