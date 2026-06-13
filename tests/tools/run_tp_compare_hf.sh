#!/bin/bash
# HF transport correctness test: single run with compare model.
# Usage: bash tests/tools/run_tp_compare_hf.sh [model] [mode] [tp]
#   model: gpt2 (default) or qwen3
#   mode:  eager (default) or cudagraph
#   tp:    1 (default) or 2
#
# Configurable env vars (all have defaults):
#   E2E_GPUS, E2E_BATCH_SIZE, E2E_MAX_NEW_TOKENS
#   E2E_RING_PAYLOAD_MB, E2E_RING_PINNED_MB
#   DMX_DB_HOST, DMX_DB_PORT, DMX_DB_DATABASE, DMX_DB_TABLE
set -e

MODEL=${1:-gpt2}
MODE=${2:-eager}
TP=${3:-1}

if [ "$TP" = "1" ]; then
    DEFAULT_GPUS="0"
else
    DEFAULT_GPUS="0,1"
fi
export CUDA_VISIBLE_DEVICES=${E2E_GPUS:-${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPUS}}
export E2E_MODEL=$MODEL
export E2E_TP_SIZE=$TP
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
echo "  model=$MODEL  mode=$MODE  tp=$TP"
echo "  batch=$E2E_BATCH_SIZE  tokens=$E2E_MAX_NEW_TOKENS"
echo "============================================"

if [ "$TP" = "1" ]; then
    python -m tests.hf_compare_runner
else
    torchrun --nproc_per_node=$TP -m tests.hf_compare_runner
fi
