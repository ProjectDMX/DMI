"""Demonstrate the memory growth problem with hook_k and hook_v during long decode."""

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

def test_kv_growth():
    print("=" * 80)
    print("Testing KV Cache Growth During Long Decode Sequence")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    prefill_len = 5
    decode_steps = 20  # Simulate 20 decode steps

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedGPT2Model.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float32)
    model.to(device)
    model.eval()

    # Prefill
    prompts = ["Hello world"] * batch_size
    prefill_tokens = tokenizer(prompts, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=prefill_len)["input_ids"].to(device)

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {prefill_len}")
    print(f"  Decode steps: {decode_steps}")
    print(f"  Device: {device}")

    with torch.no_grad():
        prefill_outputs, prefill_cache = model.run_with_cache(
            prefill_tokens, use_cache=True, return_dict=True
        )

    # Track sizes during decode
    print("\n" + "=" * 80)
    print("Tracking K/V sizes during decode")
    print("=" * 80)

    layer_0_k_sizes = []
    layer_0_v_sizes = []
    layer_0_k_bytes = []
    layer_0_v_bytes = []

    past_kv = prefill_outputs.past_key_values

    print(f"\n{'Step':<6} {'Seq Len':<10} {'hook_k Shape':<25} {'hook_v Shape':<25} {'Total KB':<12}")
    print("-" * 90)

    # Prefill step
    if 'blocks.0.attn.hook_k' in prefill_cache:
        k_shape = prefill_cache['blocks.0.attn.hook_k'].shape
        v_shape = prefill_cache['blocks.0.attn.hook_v'].shape
        k_bytes = prefill_cache['blocks.0.attn.hook_k'].numel() * prefill_cache['blocks.0.attn.hook_k'].element_size()
        v_bytes = prefill_cache['blocks.0.attn.hook_v'].numel() * prefill_cache['blocks.0.attn.hook_v'].element_size()
        total_kb = (k_bytes + v_bytes) / 1024
        print(f"{'Prefill':<6} {k_shape[2]:<10} {str(k_shape):<25} {str(v_shape):<25} {total_kb:<12.2f}")
        layer_0_k_sizes.append(k_shape[2])
        layer_0_v_sizes.append(v_shape[2])
        layer_0_k_bytes.append(k_bytes)
        layer_0_v_bytes.append(v_bytes)

    # Decode steps
    for step in range(decode_steps):
        next_token = torch.randint(0, 50257, (batch_size, 1), device=device)

        decode_outputs, decode_cache = model.run_with_cache(
            next_token,
            use_cache=True,
            past_key_values=past_kv,
            return_dict=True,
        )

        past_kv = decode_outputs.past_key_values

        if 'blocks.0.attn.hook_k' in decode_cache:
            k_shape = decode_cache['blocks.0.attn.hook_k'].shape
            v_shape = decode_cache['blocks.0.attn.hook_v'].shape
            k_bytes = decode_cache['blocks.0.attn.hook_k'].numel() * decode_cache['blocks.0.attn.hook_k'].element_size()
            v_bytes = decode_cache['blocks.0.attn.hook_v'].numel() * decode_cache['blocks.0.attn.hook_v'].element_size()
            total_kb = (k_bytes + v_bytes) / 1024

            layer_0_k_sizes.append(k_shape[2])
            layer_0_v_sizes.append(v_shape[2])
            layer_0_k_bytes.append(k_bytes)
            layer_0_v_bytes.append(v_bytes)

            print(f"{step+1:<6} {k_shape[2]:<10} {str(k_shape):<25} {str(v_shape):<25} {total_kb:<12.2f}")

    # Analysis
    print("\n" + "=" * 80)
    print("Memory Growth Analysis (Layer 0 only)")
    print("=" * 80)

    initial_kb = (layer_0_k_bytes[0] + layer_0_v_bytes[0]) / 1024
    final_kb = (layer_0_k_bytes[-1] + layer_0_v_bytes[-1]) / 1024
    growth_factor = final_kb / initial_kb

    print(f"\nInitial (Prefill):  {initial_kb:.2f} KB")
    print(f"Final (Step {decode_steps}):   {final_kb:.2f} KB")
    print(f"Growth Factor:      {growth_factor:.2f}x")
    print(f"Absolute Growth:    +{final_kb - initial_kb:.2f} KB")

    # Extrapolate for full model
    num_layers = 12
    total_initial_mb = (initial_kb * num_layers) / 1024
    total_final_mb = (final_kb * num_layers) / 1024

    print(f"\n{'='*80}")
    print("Extrapolated for Full GPT-2 Model (12 layers)")
    print("=" * 80)
    print(f"\nPer-step K+V storage:")
    print(f"  Prefill:    {total_initial_mb:.2f} MB")
    print(f"  Step {decode_steps}:     {total_final_mb:.2f} MB")
    print(f"  Growth:     +{total_final_mb - total_initial_mb:.2f} MB")

    # Project to longer sequences
    print(f"\n{'='*80}")
    print("Projection for Longer Sequences")
    print("=" * 80)
    print(f"\n{'Decode Steps':<15} {'Seq Len':<12} {'Total MB':<15} {'vs Prefill':<15}")
    print("-" * 60)

    for steps in [50, 100, 200, 512, 1024, 2048]:
        projected_seq_len = prefill_len + steps
        # Linear growth: bytes_per_token = (final - initial) / decode_steps
        bytes_per_token = (layer_0_k_bytes[-1] + layer_0_v_bytes[-1] - layer_0_k_bytes[0] - layer_0_v_bytes[0]) / decode_steps
        projected_bytes = layer_0_k_bytes[0] + layer_0_v_bytes[0] + bytes_per_token * steps
        projected_mb = (projected_bytes * num_layers) / (1024 * 1024)
        vs_prefill = projected_mb / total_initial_mb
        print(f"{steps:<15} {projected_seq_len:<12} {projected_mb:<15.2f} {vs_prefill:.2f}x")

    print("\n" + "=" * 80)
    print("CRITICAL ISSUES")
    print("=" * 80)
    print("""
1. **LINEAR MEMORY GROWTH**: Every decode step increases K/V size by 1 token
   - Prefill:     [B, H, 5, D]
   - Decode@50:   [B, H, 55, D]  (11x larger!)
   - Decode@512:  [B, H, 517, D] (103x larger!)

2. **IMPACT ON MONITORING SYSTEM**:
   - ❌ Inconsistent data sizes between steps
   - ❌ Growing async transfer overhead
   - ❌ Increasing storage per step
   - ❌ Memory pressure accumulates

3. **SOLUTIONS**:

   Option A: Extract only the NEW token from K/V
   ```python
   # After getting cache_dict from run_with_cache
   if 'blocks.0.attn.hook_k' in cache_dict:
       full_k = cache_dict['blocks.0.attn.hook_k']  # [B, H, seq_len, D]
       current_k = full_k[:, :, -1:, :]  # [B, H, 1, D] - only new token
   ```

   Option B: Don't collect hook_k/hook_v during decode
   ```python
   # Use a names_filter to exclude K/V during decode
   names_filter = lambda name: 'hook_k' not in name and 'hook_v' not in name
   outputs, cache_dict = model.run_with_cache(
       token,
       names_filter=names_filter,  # Skip K/V
       ...
   )
   ```

   Option C: Use outputs.attentions instead
   ```python
   # outputs.attentions[i] has shape [B, H, 1, seq_len]
   # This is more compact and doesn't include the full K/V
   ```

4. **RECOMMENDATION**:
   For decode monitoring, DO NOT collect hook_k and hook_v.
   Use hook_q, hook_pattern, and outputs.hidden_states instead.
    """)

if __name__ == "__main__":
    test_kv_growth()
