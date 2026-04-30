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

WORK_DIR=${WORK_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$WORK_DIR"

# ── Environment ─────────────────────────────────────────────────────
# Activate your TRT-LLM conda env before running.
PYTHON=${ENV_PYTHON:-python}
MPIRUN=${MPIRUN:-$(which mpirun 2>/dev/null || echo mpirun)}
export HF_HOME=${HF_HOME:-${WORK_DIR}/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export XDG_CACHE_HOME=${WORK_DIR}/.cache
ENGINE_ROOT=${WORK_DIR}/trtllm_engines
mkdir -p "$ENGINE_ROOT"

# ── Resolve model ───────────────────────────────────────────────────
case $MODEL_TAG in
    qwen4b)
        MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_NAME="qwen4b_v2"
        ;;
    llama8b)
        MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_NAME="llama8b_v2"
        ;;
    qwen14b)
        MODEL_PATH=$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-14B/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
        ENGINE_NAME="qwen14b_v2"
        ;;
    *) echo "Unknown model: $MODEL_TAG"; exit 1 ;;
esac

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model $MODEL_TAG not found in $HF_HOME"; exit 1
fi

echo "=== $(date) === Building $MODEL_TAG engine ==="
echo "Model: $MODEL_PATH"
echo "Output: $ENGINE_ROOT/$ENGINE_NAME"

$MPIRUN --oversubscribe -n 1 $PYTHON -u -c "
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
