#!/bin/bash
#SBATCH --job-name=d2h-qwen4b
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --output=d2h-serve-qwen4b-%j.out
#SBATCH --error=d2h-serve-qwen4b-%j.err
#SBATCH --exclude=gpu-a6-2

cd ~/scratch.zaoxing-prj
MPIRUN=/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-x86_64/gcc-11.3.0/openmpi-4.1.5-h3d4fsbq2zpfqyhmle4c44k35mvpw2bp/bin/mpirun

# ---------- Environment ----------
TRTLLM_PYTHON=$(pwd)/envs/trtllm/bin/python3.10
BENCH_PYTHON=$(pwd)/vllm-env/bin/python
BENCH_PYTHONPATH=$(pwd)/vllm-0.17.0

NVIDIA_BASE=envs/trtllm/lib/python3.10/site-packages/nvidia
NVIDIA_LIBS=$(find $NVIDIA_BASE -maxdepth 2 -name lib -type d ! -path "*/cu13/*" 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH=/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-x86_64/gcc-11.3.0/openmpi-4.1.5-h3d4fsbq2zpfqyhmle4c44k35mvpw2bp/lib:${NVIDIA_LIBS}:$(pwd)/envs/trtllm/lib/cublas13_only
export LD_PRELOAD=$(pwd)/envs/trtllm/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_mca_base_env_list="LD_PRELOAD,LD_LIBRARY_PATH"
export TRTLLM_EXTRACT_NLAYERS=36
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$(pwd)/.cache

PORT=8000
ENGINE_DIR=$(pwd)/trtllm_engines/qwen4b_v2
MODEL=$(ls -d hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/ | head -1 | sed 's:/$::')
TAG="trtllm_d2h_qwen4b"

echo "Engine: $ENGINE_DIR"
echo "Model: $MODEL"

if [ ! -d "$ENGINE_DIR" ]; then
    echo "ERROR: Engine directory not found: $ENGINE_DIR"
    exit 1
fi

# ---------- Start TRT-LLM server ----------
$MPIRUN --oversubscribe -n 1 $TRTLLM_PYTHON -u -m tensorrt_llm.commands.serve "$ENGINE_DIR" \
    --port $PORT \
    --backend tensorrt \
    --max_batch_size 256 \
    --tokenizer "$MODEL" &
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

# ---------- Benchmark ----------
mkdir -p trtllm_results

DURATION=30

DATASETS=(
    "sampled_datasets/sharegpt_seed42_n500_n30.json"
    "sampled_datasets/sharegpt_seed123_n500_n30.json"
    "sampled_datasets/sharegpt_seed456_n500_n30.json"
    "sampled_datasets/wildchat_seed42_n500_n30.json"
    "sampled_datasets/wildchat_seed123_n500_n30.json"
    "sampled_datasets/wildchat_seed456_n500_n30.json"
)

DATASET_TAGS=(
    "sharegpt_s42"
    "sharegpt_s123"
    "sharegpt_s456"
    "wildchat_s42"
    "wildchat_s123"
    "wildchat_s456"
)

for didx in "${!DATASETS[@]}"; do
    DS_PATH="${DATASETS[$didx]}"
    DS_TAG="${DATASET_TAGS[$didx]}"
    echo ""
    echo "--- Dataset: $DS_TAG ---"

    for rate in 1 2 4 8 16 32 64 128 256; do
        NP=$((rate * DURATION))
        OUTFILE="${TAG}_${DS_TAG}_rate${rate}.json"
        echo "  rate=$rate num_prompts=$NP -> $OUTFILE"
        env -u LD_PRELOAD PYTHONPATH="$BENCH_PYTHONPATH" $BENCH_PYTHON run_bench.py \
            --dataset-name sharegpt \
            --dataset-path "$DS_PATH" \
            --backend openai \
            --base-url http://localhost:$PORT \
            --model "$MODEL" \
            --sharegpt-output-len 128 \
            --request-rate "$rate" \
            --num-prompts "$NP" \
            --num-warmups 50 \
            --save-result \
            --result-dir trtllm_d2h_serve_results \
            --result-filename "$OUTFILE"
    done
done

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "=== Done ==="
