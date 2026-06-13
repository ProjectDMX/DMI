#!/bin/bash
# Full HF regression: 2 models x 2 modes x 4 ring sizes = 16 tests
#
# Ring sizes:
#   1MB  — both prefill and decode exceed → cpu_direct, CUDA graphs disabled
#   4MB  — GPT-2: prefill exceeds, decode fits (partial ring)
#          Qwen3: both exceed → cpu_direct
#   10MB — Qwen3: prefill exceeds, decode fits (partial ring)
#          GPT-2: nothing exceeds → full ring
#   4GB  — nothing exceeds → full ring + CUDA graphs
set -e

run_test() {
    local model_name=$1
    local mode_name=$2
    local size_name=$3
    local test_func=$4

    echo "=== $model_name $mode_name $size_name ==="
    python -m pytest tests/test_e2e_correctness_vs_hf.py::$test_func -q -s
}

for model in gpt2 qwen3; do
    if [ "$model" = "gpt2" ]; then
        unset E2E_MODEL
        model_name="GPT-2"
    else
        export E2E_MODEL=qwen3
        model_name="Qwen3"
    fi

    for size in 1 4 10 4096; do
        size_name="${size}MB"
        export E2E_RING_PAYLOAD_BYTES=$(($size * 1024 * 1024))
        export E2E_RING_PINNED_BYTES=$(($size * 1024 * 1024))

        run_test "$model_name" "eager"      "$size_name" "test_e2e_correctness_hf"
        run_test "$model_name" "CUDA-graph" "$size_name" "test_e2e_cuda_graphs_vs_eager_hf"
    done
done

echo ""
echo "=== All 16 tests passed ==="
