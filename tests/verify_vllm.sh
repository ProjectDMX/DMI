#!/bin/bash
# vLLM regression: 2 models x 2 modes x 4 ring sizes = 16 tests
#
# All tests use 8 prompts.  Ring sizes chosen so every combination of
# "decode exceeds / fits" and "prefill exceeds / fits" is covered for
# both GPT-2 and Qwen3-0.6B:
#
#   Ring   GPT-2 decode(~2.4MB) GPT-2 prefill(~9MB)  Qwen3 decode(~6.9MB) Qwen3 prefill(~24.5MB)
#   1MB    exceeds               exceeds               exceeds               exceeds
#   4MB    fits                  exceeds               exceeds               exceeds
#   16MB   fits                  fits                  fits                  exceeds
#   4096MB fits                  fits                  fits                  fits
#
# Requires:
#   - ClickHouse running on localhost:9000
#   - VLLM_DISABLE_COMPILE_CACHE=1 (set below)
#   - LD_PRELOAD for libstdc++ (caller's responsibility)
#
# Usage:
#   LD_PRELOAD=/path/to/libstdc++.so.6 bash tests/verify_vllm.sh
set -e

export VLLM_DISABLE_COMPILE_CACHE=1

run_test() {
    local model_name=$1
    local mode_name=$2
    local size_name=$3
    local model_key=$4
    local eager=$5
    local ring_mb=$6

    echo "=== $model_name $mode_name ring=${size_name} ==="
    rm -rf /tmp/torchinductor_$(whoami)/ 2>/dev/null
    rm -rf ~/.cache/vllm/ 2>/dev/null

    E2E_MODEL=$model_key \
    E2E_ENFORCE_EAGER=$eager \
    E2E_RING_PAYLOAD_MB=$ring_mb \
    E2E_RING_PINNED_MB=$ring_mb \
    python -m pytest tests/test_vllm_correctness.py -q -s
}

for model in gpt2 qwen3; do
    if [ "$model" = "gpt2" ]; then
        model_name="GPT-2"
    else
        model_name="Qwen3"
    fi

    for ring_mb in 1 4 16 4096; do
        size_name="${ring_mb}MB"
        run_test "$model_name" "eager"      "$size_name" "$model" "1" "$ring_mb"
        run_test "$model_name" "CUDA-graph" "$size_name" "$model" "0" "$ring_mb"
    done
done

echo ""
echo "=== All 16 tests passed ==="
