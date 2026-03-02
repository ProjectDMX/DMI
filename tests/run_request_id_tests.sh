#!/usr/bin/env bash
# Run all per-request-id tests: unit tests + E2E pipeline validation.
#
# Usage:
#   bash tests/run_request_id_tests.sh [--model gpt2] [--device cuda] [--batch-size 4]
#
# Options (passed through to E2E test):
#   --model       HF model name or alias (default: gpt2)
#   --device      cuda | cpu (default: cuda)
#   --batch-size  batch size for E2E test (default: 4)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="gpt2"
DEVICE="cuda"
BATCH_SIZE="4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0

run_step() {
    local label="$1"; shift
    echo ""
    echo "========================================"
    echo "  $label"
    echo "========================================"
    if "$@"; then
        echo "[PASS] $label"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $label"
        FAIL=$((FAIL + 1))
    fi
}

# ── 1. Unit tests ────────────────────────────────────────────────────────────
run_step "Unit tests (test_monitoring_engine_request_id)" \
    pytest tests/test_monitoring_engine_request_id.py -v

# ── 2. E2E: basic (final_logits only) ────────────────────────────────────────
run_step "E2E: basic (final_logits, no attn hook)" \
    python -m tests.validate_request_id_pipeline \
        --prompts benchmark/data/prompts_varlen_validation.txt \
        --model "$MODEL" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens 6

# ── 3. E2E: with attention hook ───────────────────────────────────────────────
run_step "E2E: with attn hook (is_attn narrowing path)" \
    python -m tests.validate_request_id_pipeline \
        --prompts benchmark/data/prompts_varlen_validation.txt \
        --model "$MODEL" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens 6 \
        --with-attn-hook

# ── 4. E2E: EOS early termination path ───────────────────────────────────────
run_step "E2E: EOS early termination (finished-request row count)" \
    python -m tests.validate_request_id_pipeline \
        --prompts benchmark/data/prompts_varlen_validation.txt \
        --model "$MODEL" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens 6 \
        --exercise-eos-path

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
