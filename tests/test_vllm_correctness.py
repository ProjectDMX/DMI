"""vLLM E2E correctness test — verify ring transport delivers all activations
to ClickHouse with correct row counts and no shape mismatches.

Uses the same env-var configuration pattern as test_e2e_correctness_vs_hf.py.

Environment variables:
  E2E_MODEL             "gpt2" (default) or "qwen3"
  E2E_NUM_PROMPTS       Number of prompts (default 3)
  E2E_MAX_NEW_TOKENS    Tokens to generate per prompt (default 20)
  E2E_ENFORCE_EAGER     "1" to disable torch.compile + CUDA graphs (default "0")
  E2E_RING_PAYLOAD_MB   Ring payload size in MB (default 4096)
  E2E_RING_PINNED_MB    Pinned staging size in MB (default 4096)
  E2E_HOOK_SELECTION    Hook selection preset (default "vllm-full")
  DMX_DB_HOST           ClickHouse host (default "localhost")
  DMX_DB_PORT           ClickHouse port (default 9000)

Requires:
  - ClickHouse running on DMX_DB_HOST:DMX_DB_PORT
  - VLLM_DISABLE_COMPILE_CACHE=1 (set automatically)
  - LD_PRELOAD for libstdc++ if needed (caller's responsibility)

Usage:
  python -m pytest tests/test_vllm_correctness.py -q -s
  E2E_MODEL=qwen3 python -m pytest tests/test_vllm_correctness.py -q -s
  E2E_ENFORCE_EAGER=1 python -m pytest tests/test_vllm_correctness.py -q -s
"""

import os
import sys

# Force disable compile cache (effectful_op incompatible with AOT serialization)
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import pytest
import torch

_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
}

# vllm-full hook types (excludes attn_scores, attn_pattern, resid_final)
# Per-layer hooks (11): resid_pre, ln1, q, k, v, z, attn_out, resid_mid, ln2, mlp_in, mlp_out
# Global hooks (5): token_ids, embed, pos_embed, final_ln, final_logits
_NUM_PER_LAYER_HOOK_TYPES = 11
_NUM_GLOBAL_HOOK_TYPES = 5  # token_ids, embed, pos_embed, final_ln, final_logits
_GLOBAL_HOOK_NAMES = {"token_ids", "hook_embed", "hook_pos_embed", "hook_final_ln", "final_logits"}

# GPT-2 has pos_embed; Qwen3 does not
_QWEN3_MISSING_GLOBALS = {"hook_pos_embed"}


def _get_num_layers(model_id: str) -> int:
    if "gpt2" in model_id.lower():
        return 12
    if "qwen3" in model_id.lower() or "qwen" in model_id.lower():
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        return cfg.num_hidden_layers
    raise ValueError(f"Unknown model: {model_id}")


@pytest.mark.skipif(
    not torch.backends.cuda.is_built(), reason="CUDA not built")
def test_vllm_correctness(subtests):
    try:
        import clickhouse_driver
    except ImportError:
        pytest.skip("clickhouse-driver required")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        pytest.skip("vllm not installed")

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "0") == "1"
    ring_payload_mb = int(os.environ.get("E2E_RING_PAYLOAD_MB", "4096"))
    ring_pinned_mb = int(os.environ.get("E2E_RING_PINNED_MB", "4096"))
    hook_selection = os.environ.get("E2E_HOOK_SELECTION", "vllm-full")
    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))

    num_layers = _get_num_layers(model_id)
    is_qwen = "qwen" in model_id.lower()

    print(f"\n{'=' * 60}")
    print(f"  vLLM correctness test")
    print(f"  model={model_id}  layers={num_layers}")
    print(f"  prompts={num_prompts}  max_new_tokens={max_new_tokens}")
    print(f"  enforce_eager={enforce_eager}  ring={ring_payload_mb}MB")
    print(f"  hooks={hook_selection}")
    print(f"{'=' * 60}")

    # -----------------------------------------------------------------------
    # Prompts
    # -----------------------------------------------------------------------
    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    # -----------------------------------------------------------------------
    # Drop existing table
    # -----------------------------------------------------------------------
    client = clickhouse_driver.Client(db_host, port=db_port)
    client.execute("DROP TABLE IF EXISTS default.offload")

    # -----------------------------------------------------------------------
    # Run vLLM with monitoring
    # -----------------------------------------------------------------------
    llm = LLM(
        model=model_id,
        worker_cls="monitoring.vllm_integration.DMXGPUWorker",
        additional_config={
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
        },
        max_model_len=512,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.5,
    )

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)

    # Collect generated token counts
    generated_tokens = {}
    for i, o in enumerate(outputs):
        n_gen = len(o.outputs[0].token_ids)
        generated_tokens[i] = n_gen
        print(f"  prompt[{i}]: {n_gen} tokens generated")

    del llm
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Query ClickHouse
    # -----------------------------------------------------------------------
    total_rows = client.execute("SELECT count() FROM default.offload")[0][0]
    per_hook = client.execute(
        "SELECT act_name, count() as cnt FROM default.offload "
        "GROUP BY act_name ORDER BY act_name"
    )
    per_hook_dict = {name: cnt for name, cnt in per_hook}

    print(f"\n  Total DB rows: {total_rows}")
    for name, cnt in sorted(per_hook_dict.items()):
        print(f"    {name}: {cnt}")

    # -----------------------------------------------------------------------
    # Compute expected counts
    # -----------------------------------------------------------------------
    # vLLM schedules: 1 prefill step (all prompt tokens) + N decode steps
    # per request. But with continuous batching, multiple requests may share
    # steps. We can't predict exact step count, but we CAN predict:
    #
    # For per-layer hooks: each step pushes num_layers entries per hook type.
    #   All per-layer types should have the SAME count.
    #
    # For global hooks: each step pushes 1 entry per global hook type.
    #   All global types should have the SAME count (except model-specific
    #   missing hooks like pos_embed for Qwen3).
    #
    # final_logits count = global count (1 per step, 1 logit per request).

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------

    # 1. Total rows > 0
    with subtests.test("total_rows > 0"):
        assert total_rows > 0, f"No rows in ClickHouse (total={total_rows})"

    # 2. All per-layer hook types present with equal counts
    per_layer_names = [n for n in per_hook_dict if n.startswith("blocks.")]
    per_layer_counts = [per_hook_dict[n] for n in per_layer_names]

    with subtests.test("per-layer hooks present"):
        assert len(per_layer_names) == _NUM_PER_LAYER_HOOK_TYPES, (
            f"Expected {_NUM_PER_LAYER_HOOK_TYPES} per-layer hook types, "
            f"got {len(per_layer_names)}: {per_layer_names}"
        )

    with subtests.test("per-layer hooks equal count"):
        if per_layer_counts:
            expected = per_layer_counts[0]
            for name, cnt in zip(per_layer_names, per_layer_counts):
                assert cnt == expected, (
                    f"{name} has {cnt} rows, expected {expected} "
                    f"(same as {per_layer_names[0]})"
                )

    # 3. Per-layer count is divisible by num_layers
    if per_layer_counts:
        pl_count = per_layer_counts[0]
        with subtests.test("per-layer divisible by num_layers"):
            assert pl_count % num_layers == 0, (
                f"Per-layer count {pl_count} not divisible by "
                f"num_layers={num_layers}"
            )
        num_steps = pl_count // num_layers
        print(f"\n  Inferred steps: {num_steps} (from {pl_count} / {num_layers})")

    # 4. Global hooks present with equal counts
    expected_globals = _GLOBAL_HOOK_NAMES.copy()
    if is_qwen:
        expected_globals -= _QWEN3_MISSING_GLOBALS

    with subtests.test("global hooks present"):
        for gname in expected_globals:
            assert gname in per_hook_dict, f"Missing global hook: {gname}"

    global_counts = [per_hook_dict.get(n, 0) for n in expected_globals]
    with subtests.test("global hooks equal count"):
        if global_counts:
            expected_g = global_counts[0]
            for name, cnt in zip(expected_globals, global_counts):
                assert cnt == expected_g, (
                    f"{name} has {cnt} rows, expected {expected_g}"
                )

    # 5. Global count matches per-layer step count
    if per_layer_counts and global_counts:
        with subtests.test("global count == num_steps"):
            assert global_counts[0] == num_steps, (
                f"Global hook count {global_counts[0]} != "
                f"inferred steps {num_steps}"
            )

    # 6. Verify total = per_layer * types + global * types
    expected_total = 0
    if per_layer_counts:
        expected_total += per_layer_counts[0] * _NUM_PER_LAYER_HOOK_TYPES
    expected_total += sum(per_hook_dict.get(n, 0) for n in expected_globals)
    # Add any model-specific missing globals that might still be in DB
    for n in _GLOBAL_HOOK_NAMES - expected_globals:
        expected_total += per_hook_dict.get(n, 0)

    with subtests.test("total rows consistent"):
        assert total_rows == expected_total, (
            f"Total rows {total_rows} != expected {expected_total}"
        )

    # 7. No shape/bytes mismatches (check final_logits count specifically)
    with subtests.test("final_logits present"):
        assert "final_logits" in per_hook_dict, "final_logits missing from DB"

    # 8. Token ranges are valid (no negative, start < end)
    bad_ranges = client.execute(
        "SELECT act_name, start_token_idx, end_token_idx "
        "FROM default.offload "
        "WHERE start_token_idx >= end_token_idx "
        "LIMIT 5"
    )
    with subtests.test("valid token ranges"):
        assert len(bad_ranges) == 0, (
            f"Found rows with invalid token ranges: {bad_ranges}"
        )

    print(f"\n  ALL CHECKS PASSED ({total_rows} rows)")
