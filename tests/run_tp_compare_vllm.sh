#!/bin/bash
# Transport correctness test: single run, compare .copy_() buffers vs ClickHouse.
# Usage: bash tests/run_tp_compare_vllm.sh [model] [mode] [tp]
#   model: qwen3 (default) or gpt2
#   mode:  eager (default) or cudagraph
#   tp:    1 (default) or 2
#
# Configurable env vars (all have defaults):
#   E2E_GPUS, E2E_NUM_PROMPTS, E2E_MAX_NEW_TOKENS, E2E_REF_MAX_LEN
#   E2E_RING_PAYLOAD_MB, E2E_RING_PINNED_MB
#   DMX_HOOK_SELECTION, DMX_DB_HOST, DMX_DB_PORT
#   DMX_DB_DATABASE, DMX_DB_TABLE, DMX_CH_PARALLELISM
#   DMX_CH_MAX_BATCH_ITEMS, DMX_CH_MAX_BATCH_BYTES
set -e

MODEL=${1:-qwen3}
MODE=${2:-eager}
TP=${3:-1}

# Default GPUs: "0" for tp=1, "0,1" for tp=2
if [ "$TP" = "1" ]; then
    DEFAULT_GPUS="0"
else
    DEFAULT_GPUS="0,1"
fi
export CUDA_VISIBLE_DEVICES=${E2E_GPUS:-${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPUS}}

export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-0}
export E2E_MODEL=$MODEL
export E2E_NUM_PROMPTS=${E2E_NUM_PROMPTS:-8}
export E2E_MAX_NEW_TOKENS=${E2E_MAX_NEW_TOKENS:-20}
export E2E_TP_SIZE=$TP
export E2E_REF_MAX_LEN=${E2E_REF_MAX_LEN:-8192}
export E2E_RING_PAYLOAD_MB=${E2E_RING_PAYLOAD_MB:-4096}
export E2E_RING_PINNED_MB=${E2E_RING_PINNED_MB:-4096}
export DMX_HOOK_SELECTION=${DMX_HOOK_SELECTION:-vllm-full}
export DMX_DB_HOST=${DMX_DB_HOST:-localhost}
export DMX_DB_PORT=${DMX_DB_PORT:-9000}

if [ "$MODE" = "eager" ]; then
    export E2E_ENFORCE_EAGER=1
elif [ "$MODE" = "cudagraph" ]; then
    export E2E_ENFORCE_EAGER=0
else
    echo "Unknown mode: $MODE (use eager or cudagraph)"
    exit 1
fi

echo "============================================"
echo "  Transport compare test"
echo "  model=$MODEL  mode=$MODE  tp=$TP"
echo "  prompts=$E2E_NUM_PROMPTS  tokens=$E2E_MAX_NEW_TOKENS"
echo "============================================"

python -m tests.vllm_compare_runner
