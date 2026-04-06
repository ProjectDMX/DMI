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
#   - TRT-LLM Python patches applied (see experiments/TensorRT-LLM submodule)
#   - MPI available (mpirun)

set -eo pipefail

# ── Parse arguments ─────────────────────────────────────────────────
MODEL_TAG=""
RATES="1 2 4 8 16 32 64 128 256"
RESULT_DIR="results/trtllm_d2h"
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
WORK_DIR=${WORK_DIR:-$(cd ~/scratch.zaoxing-prj && pwd)}
cd "$WORK_DIR"

case $MODEL_TAG in
    qwen4b)
        MODEL_PATH=$(ls -d hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_DIR=${WORK_DIR}/trtllm_engines/qwen4b_v2
        NLAYERS=36
        ;;
    llama8b)
        MODEL_PATH=$(ls -d hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_DIR=${WORK_DIR}/trtllm_engines/llama8b_v2
        NLAYERS=32
        ;;
    qwen14b)
        MODEL_PATH=$(ls -d hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_DIR=${WORK_DIR}/trtllm_engines/qwen14b_v2
        NLAYERS=40
        ;;
    *) echo "Unknown model: $MODEL_TAG"; exit 1 ;;
esac

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model $MODEL_TAG not found in hf_cache"; exit 1
fi
if [ ! -d "$ENGINE_DIR" ]; then
    echo "ERROR: Engine directory not found: $ENGINE_DIR"
    echo "Run build_trtllm_engines.sh first."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Environment ─────────────────────────────────────────────────────
# Find mpirun (adjust path for your cluster)
MPIRUN=${MPIRUN:-$(which mpirun 2>/dev/null || echo /cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-x86_64/gcc-11.3.0/openmpi-4.1.5-h3d4fsbq2zpfqyhmle4c44k35mvpw2bp/bin/mpirun)}

TRTLLM_PYTHON=${WORK_DIR}/envs/trtllm/bin/python3.10
BENCH_PYTHON=${WORK_DIR}/vllm-env/bin/python
BENCH_PYTHONPATH=${WORK_DIR}/vllm-0.17.0

NVIDIA_BASE=envs/trtllm/lib/python3.10/site-packages/nvidia
NVIDIA_LIBS=$(find $NVIDIA_BASE -maxdepth 2 -name lib -type d ! -path "*/cu13/*" 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH=$(dirname $MPIRUN)/../lib:${NVIDIA_LIBS}:${WORK_DIR}/envs/trtllm/lib/cublas13_only
export LD_PRELOAD=${WORK_DIR}/envs/trtllm/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_mca_base_env_list="LD_PRELOAD,LD_LIBRARY_PATH"
export TRTLLM_EXTRACT_NLAYERS=$NLAYERS
export HF_HOME=${WORK_DIR}/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=${WORK_DIR}/.cache

mkdir -p "$RESULT_DIR"

echo "========== TRT-LLM D2H: $MODEL_TAG  port=$PORT =========="
echo "Engine: $ENGINE_DIR"
echo "Model: $MODEL_PATH"
echo "NLAYERS: $NLAYERS"
echo "Rates: $RATES"

# ── Start TRT-LLM server ───────────────────────────────────────────
$MPIRUN --oversubscribe -n 1 $TRTLLM_PYTHON -u -m tensorrt_llm.commands.serve "$ENGINE_DIR" \
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
    "sampled_datasets/sharegpt_seed42_n500_n30.json:sharegpt_s42"
    "sampled_datasets/sharegpt_seed123_n500_n30.json:sharegpt_s123"
    "sampled_datasets/sharegpt_seed456_n500_n30.json:sharegpt_s456"
    "sampled_datasets/wildchat_seed42_n500_n30.json:wildchat_s42"
    "sampled_datasets/wildchat_seed123_n500_n30.json:wildchat_s123"
    "sampled_datasets/wildchat_seed456_n500_n30.json:wildchat_s456"
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
        env -u LD_PRELOAD PYTHONPATH="$BENCH_PYTHONPATH" $BENCH_PYTHON "$SCRIPT_DIR/run_bench.py" \
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
