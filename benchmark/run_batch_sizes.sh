#!/bin/bash
set -e

BATCH_SIZES=(16 32 64 128)

for bs in "${BATCH_SIZES[@]}"; do
    echo "=========================================="
    echo "Running batch_size=$bs"
    echo "=========================================="
    python -m benchmark.run_bench --batch-size "$bs" --tag "bs${bs}"
    echo ""
done

echo "All runs completed."
