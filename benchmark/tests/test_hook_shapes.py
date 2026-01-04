"""Test to verify shapes of hooked tensors during decode."""

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

def test_hook_shapes():
    print("=" * 80)
    print("Testing Hook Tensor Shapes During Decode")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    batch_size = 2
    prefill_len = 5

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedGPT2Model.from_pretrained(
        "gpt2",
        attn_implementation="eager",
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    # Create inputs
    prompts = ["Hello world"] * batch_size
    prefill_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=prefill_len,
    )["input_ids"].to(device)

    print(f"\nSetup:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {prefill_len}")
    print(f"  Prefill tokens shape: {prefill_tokens.shape}")

    # ========================================
    # PREFILL PHASE
    # ========================================
    print("\n" + "=" * 80)
    print("PREFILL PHASE (full prompt)")
    print("=" * 80)

    with torch.no_grad():
        prefill_outputs, prefill_cache = model.run_with_cache(
            prefill_tokens,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    print(f"\nPrefill Outputs:")
    print(f"  last_hidden_state: {prefill_outputs.last_hidden_state.shape}")
    print(f"  past_key_values type: {type(prefill_outputs.past_key_values)}")
    print(f"  num layers: {len(prefill_outputs.past_key_values.layers)}")

    # Check cache shapes
    print(f"\nLayer 0 KV Cache:")
    print(f"  keys shape: {prefill_outputs.past_key_values.layers[0].keys.shape}")
    print(f"  values shape: {prefill_outputs.past_key_values.layers[0].values.shape}")

    # Check hook captured data
    print(f"\nPrefill Hook Cache (selected keys):")
    for key in sorted(prefill_cache.keys()):
        if 'blocks.0' in key:  # Only show layer 0
            val = prefill_cache[key]
            if val is not None and hasattr(val, 'shape'):
                print(f"  {key:40s}: {str(val.shape)}")

    # ========================================
    # DECODE PHASE - Step 1
    # ========================================
    print("\n" + "=" * 80)
    print("DECODE PHASE - Step 1 (single new token)")
    print("=" * 80)

    # HookedGPT2Model doesn't have lm_head, just pick a random token for testing
    next_token = torch.randint(0, 50257, (batch_size, 1), device=device)

    print(f"\nDecode input:")
    print(f"  token shape: {next_token.shape}")
    print(f"  past_key_values from prefill")

    with torch.no_grad():
        decode_outputs, decode_cache = model.run_with_cache(
            next_token,
            use_cache=True,
            past_key_values=prefill_outputs.past_key_values,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    print(f"\nDecode Outputs:")
    print(f"  last_hidden_state: {decode_outputs.last_hidden_state.shape}")

    # Check updated cache
    updated_cache = decode_outputs.past_key_values
    print(f"\nUpdated Layer 0 KV Cache:")
    print(f"  keys shape: {updated_cache.layers[0].keys.shape}")
    print(f"  values shape: {updated_cache.layers[0].values.shape}")
    print(f"  ^^^ Note: sequence dimension grew from {prefill_len} to {prefill_len + 1}")

    # Check hook captured data during decode
    print(f"\nDecode Hook Cache (Layer 0):")
    for key in sorted(decode_cache.keys()):
        if 'blocks.0' in key:
            val = decode_cache[key]
            if val is not None and hasattr(val, 'shape'):
                print(f"  {key:40s}: {str(val.shape)}")

    # ========================================
    # DECODE PHASE - Step 2
    # ========================================
    print("\n" + "=" * 80)
    print("DECODE PHASE - Step 2 (another token)")
    print("=" * 80)

    next_token_2 = torch.randint(0, 50257, (batch_size, 1), device=device)

    print(f"\nDecode step 2 input:")
    print(f"  token shape: {next_token_2.shape}")

    with torch.no_grad():
        decode_outputs_2, decode_cache_2 = model.run_with_cache(
            next_token_2,
            use_cache=True,
            past_key_values=decode_outputs.past_key_values,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    updated_cache_2 = decode_outputs_2.past_key_values
    print(f"\nUpdated Layer 0 KV Cache (after step 2):")
    print(f"  keys shape: {updated_cache_2.layers[0].keys.shape}")
    print(f"  values shape: {updated_cache_2.layers[0].values.shape}")
    print(f"  ^^^ Note: sequence dimension grew from {prefill_len + 1} to {prefill_len + 2}")

    print(f"\nDecode Step 2 Hook Cache (Layer 0):")
    for key in sorted(decode_cache_2.keys()):
        if 'blocks.0' in key and 'attn' in key:
            val = decode_cache_2[key]
            if val is not None and hasattr(val, 'shape'):
                print(f"  {key:40s}: {str(val.shape)}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nDuring DECODE (single token input):")
    print("\n1. HuggingFace model outputs (output_attentions=True):")
    if decode_outputs.attentions is not None:
        print(f"   outputs.attentions[0]: {decode_outputs.attentions[0].shape}")
        print(f"   ^^^ [batch, num_heads, query_len=1, key_len={prefill_len+1}]")

    print("\n2. HuggingFace model outputs (output_hidden_states=True):")
    if decode_outputs.hidden_states is not None:
        print(f"   outputs.hidden_states[0]: {decode_outputs.hidden_states[0].shape}")
        print(f"   ^^^ [batch, seq_len=1, hidden_dim]")

    print("\n3. Hook-captured data (via cache_dict):")
    hook_q_key = 'blocks.0.attn.hook_q'
    hook_k_key = 'blocks.0.attn.hook_k'
    hook_v_key = 'blocks.0.attn.hook_v'
    hook_pattern_key = 'blocks.0.attn.hook_pattern'

    if hook_q_key in decode_cache:
        print(f"   hook_q: {decode_cache[hook_q_key].shape if decode_cache[hook_q_key] is not None else 'None'}")
        print(f"   ^^^ [batch, num_heads, query_len=1, head_dim]")

    if hook_k_key in decode_cache:
        print(f"   hook_k: {decode_cache[hook_k_key].shape if decode_cache[hook_k_key] is not None else 'None'}")
        print(f"   ^^^ [batch, num_heads, key_len={prefill_len+1}, head_dim] - CUMULATIVE!")

    if hook_v_key in decode_cache:
        print(f"   hook_v: {decode_cache[hook_v_key].shape if decode_cache[hook_v_key] is not None else 'None'}")
        print(f"   ^^^ [batch, num_heads, key_len={prefill_len+1}, head_dim] - CUMULATIVE!")

    if hook_pattern_key in decode_cache:
        print(f"   hook_pattern: {decode_cache[hook_pattern_key].shape if decode_cache[hook_pattern_key] is not None else 'None'}")
        print(f"   ^^^ [batch, num_heads, query_len=1, key_len={prefill_len+1}]")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("""
During decode with single token input:

  ✓ outputs.attentions[i]: [B, H, 1, seq_len] - only current token's attention
  ✓ outputs.hidden_states[i]: [B, 1, D] - only current token's hidden state

  ✓ hook_q: [B, H, 1, D] - only current token's query
  ✗ hook_k: [B, H, seq_len, D] - CUMULATIVE (includes all history)
  ✗ hook_v: [B, H, seq_len, D] - CUMULATIVE (includes all history)
  ✓ hook_pattern: [B, H, 1, seq_len] - current token attending to all history

The hook_k and hook_v capture the FULL KV cache (after concatenation),
NOT just the current token!
    """)

if __name__ == "__main__":
    test_hook_shapes()
