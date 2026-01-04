"""Demonstrate correct ways to collect K/V data during decode without memory explosion."""

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

def compare_collection_methods():
    print("=" * 80)
    print("Comparing Different K/V Collection Methods")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    prefill_len = 5
    decode_steps = 10

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedGPT2Model.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float32)
    model.to(device)
    model.eval()

    prompts = ["Hello world"] * batch_size
    prefill_tokens = tokenizer(prompts, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=prefill_len)["input_ids"].to(device)

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {prefill_len}")
    print(f"  Decode steps: {decode_steps}")

    # Prefill
    with torch.no_grad():
        prefill_outputs, _ = model.run_with_cache(
            prefill_tokens, use_cache=True, return_dict=True
        )

    past_kv = prefill_outputs.past_key_values

    print("\n" + "=" * 80)
    print("Method Comparison During Decode")
    print("=" * 80)

    methods_data = {
        "❌ Wrong: Full K/V": [],
        "✅ Fix A: Extract Last": [],
        "✅ Fix B: Filter K/V": [],
        "✅ Fix C: Use Attention": [],
    }

    for step in range(decode_steps):
        next_token = torch.randint(0, 50257, (batch_size, 1), device=device)

        with torch.no_grad():
            # Method 1: ❌ WRONG - Collect full K/V (growing)
            outputs1, cache1 = model.run_with_cache(
                next_token,
                use_cache=True,
                past_key_values=past_kv,
                return_dict=True,
            )
            if 'blocks.0.attn.hook_k' in cache1:
                full_k = cache1['blocks.0.attn.hook_k']
                full_v = cache1['blocks.0.attn.hook_v']
                size_mb = (full_k.numel() + full_v.numel()) * 4 / (1024*1024) * 12  # 12 layers
                methods_data["❌ Wrong: Full K/V"].append((full_k.shape, size_mb))

                # Method 2: ✅ FIX A - Extract only last token
                current_k = full_k[:, :, -1:, :]  # [B, H, 1, D]
                current_v = full_v[:, :, -1:, :]  # [B, H, 1, D]
                size_mb_fix = (current_k.numel() + current_v.numel()) * 4 / (1024*1024) * 12
                methods_data["✅ Fix A: Extract Last"].append((current_k.shape, size_mb_fix))

            # Method 3: ✅ FIX B - Filter out K/V entirely
            names_filter = lambda name: 'hook_k' not in name and 'hook_v' not in name
            outputs2, cache2 = model.run_with_cache(
                next_token,
                use_cache=True,
                past_key_values=past_kv,
                names_filter=names_filter,
                return_dict=True,
            )
            # Only collect hook_q, hook_pattern, etc.
            if 'blocks.0.attn.hook_q' in cache2:
                hook_q = cache2['blocks.0.attn.hook_q']
                hook_pattern = cache2.get('blocks.0.attn.hook_pattern')
                size_mb_alt = hook_q.numel() * 4 / (1024*1024) * 12
                if hook_pattern is not None:
                    size_mb_alt += hook_pattern.numel() * 4 / (1024*1024) * 12
                methods_data["✅ Fix B: Filter K/V"].append((hook_q.shape, size_mb_alt))

            # Method 4: ✅ FIX C - Use outputs.attentions
            outputs3, _ = model.run_with_cache(
                next_token,
                use_cache=True,
                past_key_values=past_kv,
                output_attentions=True,
                names_filter=lambda name: False,  # Don't collect any hooks
                return_dict=True,
            )
            if outputs3.attentions is not None:
                attn = outputs3.attentions[0]  # [B, H, 1, seq_len]
                size_mb_attn = attn.numel() * 4 / (1024*1024) * 12
                methods_data["✅ Fix C: Use Attention"].append((attn.shape, size_mb_attn))

        past_kv = outputs1.past_key_values

    # Print comparison
    print(f"\n{'Step':<6} {'Method':<25} {'Shape':<30} {'MB/step':<12}")
    print("-" * 80)

    for step_idx in range(min(5, decode_steps)):  # Show first 5 steps
        for method_name, data_list in methods_data.items():
            if step_idx < len(data_list):
                shape, size_mb = data_list[step_idx]
                if step_idx == 0:
                    print(f"{step_idx+1:<6} {method_name:<25} {str(shape):<30} {size_mb:.2f}")
                else:
                    print(f"{step_idx+1:<6} {'':25} {str(shape):<30} {size_mb:.2f}")

    print("\n" + "..." * 20)

    # Show last step
    last_idx = decode_steps - 1
    print(f"\n{last_idx+1:<6} Final Step:")
    for method_name, data_list in methods_data.items():
        if last_idx < len(data_list):
            shape, size_mb = data_list[last_idx]
            print(f"       {method_name:<25} {str(shape):<30} {size_mb:.2f}")

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for method_name, data_list in methods_data.items():
        total_mb = sum(size for _, size in data_list)
        avg_mb = total_mb / len(data_list)
        first_mb = data_list[0][1]
        last_mb = data_list[-1][1]
        growth = last_mb / first_mb if first_mb > 0 else 1.0

        print(f"\n{method_name}:")
        print(f"  First step:    {first_mb:.2f} MB")
        print(f"  Last step:     {last_mb:.2f} MB")
        print(f"  Average:       {avg_mb:.2f} MB")
        print(f"  Total (all):   {total_mb:.2f} MB")
        print(f"  Growth factor: {growth:.2f}x")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
For your monitoring system during DECODE:

1. **NEVER collect hook_k and hook_v directly** ❌
   - They grow linearly with sequence length
   - Wastes bandwidth and storage

2. **Best Option - Extract only new tokens** ✅
   ```python
   outputs, cache_dict = model.run_with_cache(...)

   # Post-process to extract only new K/V
   for layer_idx in range(num_layers):
       if f'blocks.{layer_idx}.attn.hook_k' in cache_dict:
           full_k = cache_dict[f'blocks.{layer_idx}.attn.hook_k']
           full_v = cache_dict[f'blocks.{layer_idx}.attn.hook_v']

           # Replace with only the new token
           cache_dict[f'blocks.{layer_idx}.attn.hook_k'] = full_k[:, :, -1:, :]
           cache_dict[f'blocks.{layer_idx}.attn.hook_v'] = full_v[:, :, -1:, :]
   ```

3. **Alternative - Filter during collection** ✅
   ```python
   # Don't collect K/V at all during decode
   names_filter = lambda name: 'hook_k' not in name and 'hook_v' not in name

   outputs, cache_dict = model.run_with_cache(
       token,
       names_filter=names_filter,
       ...
   )
   ```

4. **Alternative - Use HF native outputs** ✅
   ```python
   # outputs.attentions is compact and doesn't grow
   outputs, _ = model.run_with_cache(
       token,
       output_attentions=True,
       names_filter=lambda name: False,  # Skip all hooks
       ...
   )
   # outputs.attentions[i]: [B, H, 1, seq_len]
   # Contains attention patterns without full K/V
   ```

5. **For your async monitoring engine**:
   - ✅ Implement post-processing to slice K/V to last token
   - ✅ Add this in MonitoringEngine before async submission
   - ✅ Ensures consistent memory usage per step
    """)

def demonstrate_extraction():
    """Show how to properly extract current-token K/V."""
    print("\n\n" + "=" * 80)
    print("PRACTICAL EXAMPLE: Post-Processing K/V Extraction")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedGPT2Model.from_pretrained("gpt2", attn_implementation="eager", dtype=torch.float32)
    model.to(device)
    model.eval()

    # Simulate decode with past_key_values
    batch_size = 2
    past_len = 10
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = ["Test"] * batch_size
    tokens = tokenizer(prompts, return_tensors="pt", padding="max_length",
                       max_length=past_len)["input_ids"].to(device)

    with torch.no_grad():
        outputs, _ = model.run_with_cache(tokens, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values

        # Decode step
        next_token = torch.randint(0, 50257, (batch_size, 1), device=device)
        decode_outputs, cache_dict = model.run_with_cache(
            next_token,
            use_cache=True,
            past_key_values=past_kv,
            return_dict=True,
        )

    print("\nBefore post-processing:")
    print(f"  blocks.0.attn.hook_k: {cache_dict['blocks.0.attn.hook_k'].shape}")
    print(f"  blocks.0.attn.hook_v: {cache_dict['blocks.0.attn.hook_v'].shape}")

    # POST-PROCESSING: Extract only new tokens
    num_layers = 12
    for layer_idx in range(num_layers):
        k_key = f'blocks.{layer_idx}.attn.hook_k'
        v_key = f'blocks.{layer_idx}.attn.hook_v'

        if k_key in cache_dict and cache_dict[k_key] is not None:
            full_k = cache_dict[k_key]
            full_v = cache_dict[v_key]

            # Extract only the LAST token (the new one)
            cache_dict[k_key] = full_k[:, :, -1:, :]
            cache_dict[v_key] = full_v[:, :, -1:, :]

    print("\nAfter post-processing:")
    print(f"  blocks.0.attn.hook_k: {cache_dict['blocks.0.attn.hook_k'].shape}")
    print(f"  blocks.0.attn.hook_v: {cache_dict['blocks.0.attn.hook_v'].shape}")
    print("\n✅ Now K/V tensors are consistent size across all decode steps!")

if __name__ == "__main__":
    compare_collection_methods()
    demonstrate_extraction()
