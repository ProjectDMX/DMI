#!/bin/bash
# HF transport correctness test: single run with compare model.
# Usage: bash tests/run_tp_compare_hf.sh [model] [mode]
#   model: gpt2 (default) or qwen3
#   mode:  eager (default) or cudagraph
#
# HF hooked models are single-GPU only (no TP support), so tp is always 1.
#
# Configurable env vars (all have defaults):
#   E2E_GPUS, E2E_BATCH_SIZE, E2E_MAX_NEW_TOKENS
#   E2E_RING_PAYLOAD_MB, E2E_RING_PINNED_MB
#   DMX_DB_HOST, DMX_DB_PORT, DMX_DB_DATABASE, DMX_DB_TABLE
set -e

MODEL=${1:-gpt2}
MODE=${2:-eager}

export CUDA_VISIBLE_DEVICES=${E2E_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
export E2E_MODEL=$MODEL
export E2E_BATCH_SIZE=${E2E_BATCH_SIZE:-4}
export E2E_MAX_NEW_TOKENS=${E2E_MAX_NEW_TOKENS:-8}
export DMX_DB_HOST=${DMX_DB_HOST:-localhost}
export DMX_DB_PORT=${DMX_DB_PORT:-9000}

if [ "$MODE" = "eager" ]; then
    export E2E_CUDA_GRAPHS=0
elif [ "$MODE" = "cudagraph" ]; then
    export E2E_CUDA_GRAPHS=1
else
    echo "Unknown mode: $MODE (use eager or cudagraph)"
    exit 1
fi

echo "============================================"
echo "  HF transport compare test"
echo "  model=$MODEL  mode=$MODE  tp=1"
echo "  batch=$E2E_BATCH_SIZE  tokens=$E2E_MAX_NEW_TOKENS"
echo "============================================"

python -m tests.hf_compare_runner
