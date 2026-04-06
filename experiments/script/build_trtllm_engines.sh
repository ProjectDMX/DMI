#!/bin/bash
# Build TRT-LLM engines with enable_debug_output=True.
# This enables per-layer hidden_states as engine outputs.
#
# Usage:
#   ./build_trtllm_engines.sh --model qwen4b
#   ./build_trtllm_engines.sh --model llama8b
#   ./build_trtllm_engines.sh --model qwen14b
#
# Prerequisites:
#   - TRT-LLM conda environment (envs/trtllm/)
#   - TRT-LLM patches applied (modeling_utils.py register_network_output)
#   - HuggingFace models downloaded

set -eo pipefail

MODEL_TAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_TAG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_TAG" ]; then
    echo "Usage: $0 --model {qwen4b|llama8b|qwen14b}"
    exit 1
fi

WORK_DIR=${WORK_DIR:-$(cd ~/scratch.zaoxing-prj && pwd)}
cd "$WORK_DIR"

# ── Environment ─────────────────────────────────────────────────────
NVIDIA_BASE=envs/trtllm/lib/python3.10/site-packages/nvidia
NVIDIA_LIBS=$(find $NVIDIA_BASE -maxdepth 2 -name lib -type d ! -path "*/cu13/*" 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH=/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-x86_64/gcc-11.3.0/openmpi-4.1.5-h3d4fsbq2zpfqyhmle4c44k35mvpw2bp/lib:${NVIDIA_LIBS}:$(pwd)/envs/trtllm/lib/cublas13_only
export LD_PRELOAD=$(pwd)/envs/trtllm/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_mca_base_env_list="LD_PRELOAD,LD_LIBRARY_PATH"
export HF_HOME=$(pwd)/hf_cache
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=$(pwd)/.cache

PYTHON=envs/trtllm/bin/python3.10
ENGINE_ROOT=$(pwd)/trtllm_engines
mkdir -p "$ENGINE_ROOT"

# ── Resolve model ───────────────────────────────────────────────────
case $MODEL_TAG in
    qwen4b)
        MODEL_PATH=$(ls -d hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_NAME="qwen4b_v2"
        ;;
    llama8b)
        MODEL_PATH=$(ls -d hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_NAME="llama8b_v2"
        ;;
    qwen14b)
        MODEL_PATH=$(ls -d hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_NAME="qwen14b_v2"
        ;;
    *) echo "Unknown model: $MODEL_TAG"; exit 1 ;;
esac

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model $MODEL_TAG not found in hf_cache"; exit 1
fi

echo "=== $(date) === Building $MODEL_TAG engine ==="
echo "Model: $MODEL_PATH"
echo "Output: $ENGINE_ROOT/$ENGINE_NAME"

$PYTHON -u -c "
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig
import os

cfg = BuildConfig()
cfg.max_batch_size = 256
cfg.max_input_len = 4096
cfg.max_seq_len = 4096
cfg.enable_debug_output = True

dst = '$ENGINE_ROOT/$ENGINE_NAME'

llm = LLM(model='$MODEL_PATH', build_config=cfg)
llm.save(dst)
llm.close()
print(f'Engine saved to {dst}')
print('Contents:', os.listdir(dst), flush=True)
"

echo "=== $(date) === Done ==="
ls -lh "$ENGINE_ROOT/$ENGINE_NAME/"
