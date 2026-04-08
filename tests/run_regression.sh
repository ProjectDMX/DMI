#!/bin/bash
# Full regression test suite.
# Usage: LD_PRELOAD=... CUDA_VISIBLE_DEVICES=0,1 bash tests/run_regression.sh
set -e

PASS=0
FAIL=0
RESULTS=""

run_test() {
    local name="$1"
    shift
    echo ""
    echo "============================================"
    echo "  $name"
    echo "============================================"
    if "$@"; then
        PASS=$((PASS + 1))
        RESULTS="$RESULTS\n  [PASS] $name"
    else
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n  [FAIL] $name"
    fi
}

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

# --- Unit tests ---
run_test "unit: test_tp_shapes + test_config" \
    python -m pytest tests/test_tp_shapes.py tests/test_config.py -q

# --- vLLM transport correctness (compare model) ---
for model in qwen3 gpt2; do
    for mode in eager cudagraph; do
        run_test "vllm: $model $mode tp=1" \
            bash tests/run_tp_compare.sh "$model" "$mode" 1
        run_test "vllm: $model $mode tp=2" \
            bash tests/run_tp_compare.sh "$model" "$mode" 2
    done
done

# --- HF E2E correctness ---
run_test "hf: gpt2+qwen3 eager tp=1" \
    python -m pytest tests/test_e2e_correctness_vs_hf.py -q -s

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo ""
echo "============================================"
echo "  REGRESSION SUMMARY: $PASS/$TOTAL passed, $FAIL failed"
echo "============================================"
echo -e "$RESULTS"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
