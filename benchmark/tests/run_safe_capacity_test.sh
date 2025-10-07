#!/bin/bash

# Safe capacity test with conservative parameters

echo "Running safe capacity test with conservative parameters..."
echo "This will test smaller ranges to avoid CUDA errors."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TL

# Set CUDA environment for better error reporting
export CUDA_LAUNCH_BLOCKING=1

# Run with conservative parameters
python benchmark/tests/max_capacity_test.py \
    --model gpt2 \
    --test-batch-sizes 1 2 4 8 \
    --test-seq-lengths 64 128 256 512 \
    --implementations "HuggingFace" "TransformerLens (no cache)" \
    "$@"

echo "Test complete. Check results/max_capacity/ for output files."