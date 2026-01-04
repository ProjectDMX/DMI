"""Verify that hook_k and hook_v now capture only current token after the fix."""

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

def test_fixed_hooks():
    print("=" * 80)
    print("Testing Hook K/V Shapes After Fix")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    prefill_len = 5

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedGPT2Model.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float32)
    model.to(device)
    model.eval()

    prompts = ["Hello world"] * batch_size
    prefill_tokens = tokenizer(
        prompts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=prefill_len
    )["input_ids"].to(device)

    print(f"\nSetup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {prefill_len}")

    # Prefill
    with torch.no_grad():
        prefill_outputs, prefill_cache = model.run_with_cache(
            prefill_tokens, use_cache=True, return_dict=True
        )

    print(f"\n{'='*80}")
    print("PREFILL - Layer 0 Hook Shapes")
    print("=" * 80)
    print(f"  hook_q: {prefill_cache['blocks.0.attn.hook_q'].shape}")
    print(f"  hook_k: {prefill_cache['blocks.0.attn.hook_k'].shape}")
    print(f"  hook_v: {prefill_cache['blocks.0.attn.hook_v'].shape}")
    print(f"  Expected: All should be [2, 12, 5, 64]")

    # Decode Step 1
    next_token = torch.randint(0, 50257, (batch_size, 1), device=device)
    with torch.no_grad():
        decode_outputs, decode_cache = model.run_with_cache(
            next_token,
            use_cache=True,
            past_key_values=prefill_outputs.past_key_values,
            return_dict=True,
        )

    print(f"\n{'='*80}")
    print("DECODE STEP 1 - Layer 0 Hook Shapes")
    print("=" * 80)
    hook_q_shape = decode_cache['blocks.0.attn.hook_q'].shape
    hook_k_shape = decode_cache['blocks.0.attn.hook_k'].shape
    hook_v_shape = decode_cache['blocks.0.attn.hook_v'].shape

    print(f"  hook_q: {hook_q_shape}")
    print(f"  hook_k: {hook_k_shape}")
    print(f"  hook_v: {hook_v_shape}")

    # Decode Step 2
    next_token_2 = torch.randint(0, 50257, (batch_size, 1), device=device)
    with torch.no_grad():
        decode_outputs_2, decode_cache_2 = model.run_with_cache(
            next_token_2,
            use_cache=True,
            past_key_values=decode_outputs.past_key_values,
            return_dict=True,
        )

    print(f"\n{'='*80}")
    print("DECODE STEP 2 - Layer 0 Hook Shapes")
    print("=" * 80)
    hook_q_shape_2 = decode_cache_2['blocks.0.attn.hook_q'].shape
    hook_k_shape_2 = decode_cache_2['blocks.0.attn.hook_k'].shape
    hook_v_shape_2 = decode_cache_2['blocks.0.attn.hook_v'].shape

    print(f"  hook_q: {hook_q_shape_2}")
    print(f"  hook_k: {hook_k_shape_2}")
    print(f"  hook_v: {hook_v_shape_2}")

    # Verification
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print("=" * 80)

    expected_q = (batch_size, 12, 1, 64)
    expected_kv = (batch_size, 12, 1, 64)  # ← Should be 1, not seq_len!

    step1_pass = (
        hook_q_shape == expected_q and
        hook_k_shape == expected_kv and
        hook_v_shape == expected_kv
    )

    step2_pass = (
        hook_q_shape_2 == expected_q and
        hook_k_shape_2 == expected_kv and
        hook_v_shape_2 == expected_kv
    )

    print(f"\nDecode Step 1:")
    print(f"  Expected hook_q: {expected_q}")
    print(f"  Expected hook_k: {expected_kv}")
    print(f"  Expected hook_v: {expected_kv}")
    print(f"  Result: {'✅ PASS' if step1_pass else '❌ FAIL'}")

    print(f"\nDecode Step 2:")
    print(f"  Expected hook_q: {expected_q}")
    print(f"  Expected hook_k: {expected_kv}")
    print(f"  Expected hook_v: {expected_kv}")
    print(f"  Result: {'✅ PASS' if step2_pass else '❌ FAIL'}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    if step1_pass and step2_pass:
        print("""
✅ SUCCESS! The fix works correctly:

During decode (single token input):
  ✓ hook_q: [B, H, 1, D] - only current token
  ✓ hook_k: [B, H, 1, D] - only current token (FIXED!)
  ✓ hook_v: [B, H, 1, D] - only current token (FIXED!)

All hooks now capture consistent-sized data across decode steps.
No more memory growth issues!
        """)
    else:
        print("""
❌ FAIL: Hooks are still capturing cumulative data:

  hook_k and hook_v are still growing with sequence length.
  The fix may not be working as expected.
        """)
        print(f"\nStep 1 shapes: q={hook_q_shape}, k={hook_k_shape}, v={hook_v_shape}")
        print(f"Step 2 shapes: q={hook_q_shape_2}, k={hook_k_shape_2}, v={hook_v_shape_2}")

if __name__ == "__main__":
    test_fixed_hooks()
