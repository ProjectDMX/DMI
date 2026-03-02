#!/bin/bash
set -e

BATCH_SIZES=( 8 16)

for bs in "${BATCH_SIZES[@]}"; do
    echo "=========================================="
    echo "Running batch_size=$bs"
    echo "=========================================="
    python -m benchmark.run_bench --batch-size "$bs" --tag "bs${bs}" --prompts "benchmark/data/openorca_prompts.txt"
    echo ""
done

echo "All runs completed."
