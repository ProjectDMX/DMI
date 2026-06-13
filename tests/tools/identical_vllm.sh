#!/bin/bash
# vLLM identical check: bitwise tensor comparison (ref model vs ring transport)
# 2 models x 2 modes x 4 ring sizes = 16 tests
#
# Ring sizes match verify_vllm.sh: 1, 4, 16, 4096 MB
# Small rings trigger cpu_direct fallback — verifies correctness in all paths.
#
# Requires:
#   - ClickHouse running on localhost:9000
#   - VLLM_DISABLE_COMPILE_CACHE=1 (set below)
#   - LD_PRELOAD for libstdc++ (caller's responsibility)
#
# Usage:
#   LD_PRELOAD=/path/to/libstdc++.so.6 bash tests/tools/identical_vllm.sh
set -e

export VLLM_DISABLE_COMPILE_CACHE=1

run_identical_test() {
    local model_name=$1
    local model_key=$2
    local eager=$3
    local ring_mb=$4

    local mode_name="eager"
    if [ "$eager" = "0" ]; then
        mode_name="CUDA-graph"
    fi

    echo "=== $model_name identical check ($mode_name) ring=${ring_mb}MB ==="
    rm -rf /tmp/torchinductor_$(whoami)/ 2>/dev/null
    rm -rf ~/.cache/vllm/ 2>/dev/null

    E2E_MODEL=$model_key \
    E2E_ENFORCE_EAGER=$eager \
    E2E_DTYPE=bfloat16 \
    E2E_HOOK_SELECTION=vllm-full \
    E2E_REF_MAX_LEN=8192 \
    E2E_RING_PAYLOAD_MB=$ring_mb \
    E2E_RING_PINNED_MB=$ring_mb \
    python -m pytest tests/test_vllm_identical.py -q -s
}

for model in gpt2 qwen3; do
    if [ "$model" = "gpt2" ]; then
        model_name="GPT-2"
    else
        model_name="Qwen3"
    fi

    for ring_mb in 1 4 16 4096; do
        run_identical_test "$model_name" "$model" "1" "$ring_mb"   # eager
        run_identical_test "$model_name" "$model" "0" "$ring_mb"   # CUDA graph
    done
done

echo ""
echo "=== All 16 identical tests passed ==="
