#!/bin/bash
# Run TRT-LLM Debug API (D2H) benchmark at user-specified rates.
# Requires pre-built engines with enable_debug_output=True.
#
# Usage:
#   ./run_trtllm_d2h.sh --model qwen4b --rates "1 2 4 8 16 32 64"
#   ./run_trtllm_d2h.sh --model llama8b --rates "1 2 4 8 16 32 64"
#
# Prerequisites:
#   - TRT-LLM engines built (see build_trtllm_engines.sh)
#   - TRT-LLM Python patches applied (see experiments/online_serving/TensorRT-LLM submodule)
#   - MPI available (mpirun)

set -eo pipefail

# ── Parse arguments ─────────────────────────────────────────────────
MODEL_TAG=""
RATES="1 2 4 8 16 32 64 128 256"
RESULT_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/trtllm_d2h"
PORT=8000
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

# ── Resolve model path and engine ───────────────────────────────────
WORK_DIR=${WORK_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$WORK_DIR"
export HF_HOME=${HF_HOME:-${WORK_DIR}/hf_cache}

case $MODEL_TAG in
    qwen4b)
        MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_DIR=${WORK_DIR}/trtllm_engines/qwen4b_v2
        NLAYERS=36
        ;;
    llama8b)
        MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_DIR=${WORK_DIR}/trtllm_engines/llama8b_v2
        NLAYERS=32
        ;;
    qwen14b)
        MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-14B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_DIR=${WORK_DIR}/trtllm_engines/qwen14b_v2
        NLAYERS=40
        ;;
    *) echo "Unknown model: $MODEL_TAG"; exit 1 ;;
esac

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model $MODEL_TAG not found in $HF_HOME"; exit 1
fi
if [ ! -d "$ENGINE_DIR" ]; then
    echo "ERROR: Engine directory not found: $ENGINE_DIR"
    echo "Run build_trtllm_engines.sh first."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Environment ─────────────────────────────────────────────────────
# TRT-LLM needs TWO Python environments:
#   ENV_PYTHON   - TRT-LLM env (tensorrt_llm installed) for the server
#   BENCH_PYTHON - vLLM env (vllm installed) for the benchmark client
# Example:
#   ENV_PYTHON=/path/to/trtllm-env/bin/python \
#   BENCH_PYTHON=/path/to/vllm-env/bin/python \
#   ./run_trtllm_d2h.sh --model qwen4b --rates "1 2 4"
ENV_PYTHON=${ENV_PYTHON:-python}
BENCH_PYTHON=${BENCH_PYTHON:-$ENV_PYTHON}
MPIRUN=${MPIRUN:-$(which mpirun 2>/dev/null || echo mpirun)}

export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_mca_base_env_list="LD_PRELOAD,LD_LIBRARY_PATH"
export TRTLLM_EXTRACT_NLAYERS=$NLAYERS
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export XDG_CACHE_HOME=${WORK_DIR}/.cache

mkdir -p "$RESULT_DIR"

echo "========== TRT-LLM D2H: $MODEL_TAG  port=$PORT =========="
echo "Engine: $ENGINE_DIR"
echo "Model: $MODEL_PATH"
echo "NLAYERS: $NLAYERS"
echo "Rates: $RATES"

# ── Start TRT-LLM server ───────────────────────────────────────────
$MPIRUN --oversubscribe -n 1 $ENV_PYTHON -u -m tensorrt_llm.commands.serve "$ENGINE_DIR" \
    --port $PORT \
    --backend tensorrt \
    --max_batch_size 256 \
    --tokenizer "$MODEL_PATH" &
SERVER_PID=$!

echo "Waiting for server (PID=$SERVER_PID)..."
for i in $(seq 1 1800); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"; break
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
        $BENCH_PYTHON "$SCRIPT_DIR/run_bench.py" \
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
