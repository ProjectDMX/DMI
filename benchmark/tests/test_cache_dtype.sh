#!/bin/bash
# Test cache_dtype functionality in both benchmark scripts

echo "=========================================="
echo "Testing cache_dtype in hf_modified_async_only.py"
echo "=========================================="

echo ""
echo "Test 1: Default (no cache_dtype)"
python benchmark/tests/hf_modified_async_only.py \
    --batch-size 2 \
    --steps 1 \
    --warmup 0 \
    --decode-steps 5 \
    --collect-hidden \
    --no-profile

echo ""
echo "Test 2: cache_dtype=fp16"
python benchmark/tests/hf_modified_async_only.py \
    --batch-size 2 \
    --steps 1 \
    --warmup 0 \
    --decode-steps 5 \
    --collect-hidden \
    --cache-dtype fp16 \
    --no-profile

echo ""
echo "Test 3: cache_dtype=bf16"
python benchmark/tests/hf_modified_async_only.py \
    --batch-size 2 \
    --steps 1 \
    --warmup 0 \
    --decode-steps 5 \
    --collect-hidden \
    --cache-dtype bf16 \
    --no-profile

echo ""
echo "=========================================="
echo "Testing cache_dtype in profile_decode.py"
echo "=========================================="

echo ""
echo "Test 4: profile_decode.py with cache_dtype=bf16"
python benchmark/tests/profile_decode.py \
    --batch-size 2 \
    --steps 1 \
    --warmup 0 \
    --decode-steps 5 \
    --collect-hidden \
    --cache-dtype bf16 \
    --no-profile

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
