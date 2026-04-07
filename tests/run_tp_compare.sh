#!/bin/bash
# Transport correctness test: single run, compare .copy_() buffers vs ClickHouse.
# Usage: LD_PRELOAD=... CUDA_VISIBLE_DEVICES=0,1 bash tests/run_tp_compare.sh [model] [mode] [tp]
#   model: qwen3 (default) or gpt2
#   mode:  eager (default) or cudagraph
#   tp:    1 (default) or 2
set -e

MODEL=${1:-qwen3}
MODE=${2:-eager}
TP=${3:-1}

export VLLM_DISABLE_COMPILE_CACHE=1
export E2E_MODEL=$MODEL
export E2E_NUM_PROMPTS=${E2E_NUM_PROMPTS:-8}
export E2E_MAX_NEW_TOKENS=${E2E_MAX_NEW_TOKENS:-20}
export E2E_TP_SIZE=$TP
export E2E_REF_MAX_LEN=${E2E_REF_MAX_LEN:-8192}
export E2E_RING_PAYLOAD_MB=${E2E_RING_PAYLOAD_MB:-4096}
export E2E_RING_PINNED_MB=${E2E_RING_PINNED_MB:-4096}
export DMX_HOOK_SELECTION=vllm-full
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
