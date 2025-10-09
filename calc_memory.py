#!/usr/bin/env python3
"""Calculate theoretical GPU memory usage for profile_decode.py benchmark."""

# Benchmark configuration
BATCH_SIZE = 64
PREFILL_TOKENS = 1  # Default
DECODE_STEPS = 64
DTYPE = "fp32"  # Default
COLLECT_HIDDEN = True
COLLECT_ATTENTION = True

# GPT-2 model config
N_LAYER = 12
N_HEAD = 12
HIDDEN_SIZE = 768
HEAD_DIM = HIDDEN_SIZE // N_HEAD  # 64
VOCAB_SIZE = 50257
MAX_POS = 1024

# Bytes per element
BYTES_PER_ELEMENT = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
}
BYTES = BYTES_PER_ELEMENT[DTYPE]

def format_mb(bytes_val):
    """Convert bytes to MB."""
    return bytes_val / (1024 ** 2)

def format_gb(bytes_val):
    """Convert bytes to GB."""
    return bytes_val / (1024 ** 3)

print("=" * 70)
print("GPU Memory Calculation for profile_decode.py Benchmark")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Prefill tokens: {PREFILL_TOKENS}")
print(f"  Decode steps: {DECODE_STEPS}")
print(f"  Data type: {DTYPE} ({BYTES} bytes/element)")
print(f"  Collect hidden states: {COLLECT_HIDDEN}")
print(f"  Collect attention: {COLLECT_ATTENTION}")
print(f"\nModel: GPT-2")
print(f"  Layers: {N_LAYER}")
print(f"  Heads: {N_HEAD}")
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  Head dim: {HEAD_DIM}")

# ========== 1. Model Parameters ==========
print("\n" + "=" * 70)
print("1. MODEL PARAMETERS (Static)")
print("=" * 70)

# Token embeddings: [vocab_size, hidden_size]
wte_params = VOCAB_SIZE * HIDDEN_SIZE
# Position embeddings: [max_pos, hidden_size]
wpe_params = MAX_POS * HIDDEN_SIZE

# Per transformer block:
# - c_attn (Q,K,V combined): [hidden_size, 3*hidden_size]
# - c_proj: [hidden_size, hidden_size]
# - mlp.c_fc: [hidden_size, 4*hidden_size]
# - mlp.c_proj: [4*hidden_size, hidden_size]
# - 2 LayerNorms: 2 * hidden_size (weight) + 2 * hidden_size (bias)
per_block_params = (
    HIDDEN_SIZE * (3 * HIDDEN_SIZE) +  # c_attn
    HIDDEN_SIZE * HIDDEN_SIZE +         # c_proj
    HIDDEN_SIZE * (4 * HIDDEN_SIZE) +   # mlp.c_fc
    (4 * HIDDEN_SIZE) * HIDDEN_SIZE +   # mlp.c_proj
    4 * HIDDEN_SIZE                     # 2 LayerNorms (weight + bias)
)

total_params = wte_params + wpe_params + N_LAYER * per_block_params + 2 * HIDDEN_SIZE  # final LN
model_memory = total_params * BYTES

print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  Model weights memory: {format_mb(model_memory):.2f} MB ({format_gb(model_memory):.3f} GB)")

# ========== 2. KV Cache ==========
print("\n" + "=" * 70)
print("2. KV CACHE (Dynamic - grows with decode steps)")
print("=" * 70)

# KV cache shape per layer: [batch, heads, seq_len, head_dim]
# We have K and V, so 2x
# During decode, seq_len grows from (prefill_tokens) to (prefill_tokens + decode_steps)

# Final sequence length after all decode steps
final_seq_len = PREFILL_TOKENS + DECODE_STEPS

# KV cache memory per layer at final state
kv_per_layer_final = 2 * BATCH_SIZE * N_HEAD * final_seq_len * HEAD_DIM * BYTES

# Total KV cache for all layers
kv_cache_total = N_LAYER * kv_per_layer_final

print(f"  Final sequence length: {final_seq_len}")
print(f"  KV shape per layer: [2, {BATCH_SIZE}, {N_HEAD}, {final_seq_len}, {HEAD_DIM}]")
print(f"  KV cache per layer: {format_mb(kv_per_layer_final):.2f} MB")
print(f"  Total KV cache ({N_LAYER} layers): {format_mb(kv_cache_total):.2f} MB ({format_gb(kv_cache_total):.3f} GB)")

# ========== 3. Activations During Forward Pass ==========
print("\n" + "=" * 70)
print("3. ACTIVATIONS (Per decode step, reused)")
print("=" * 70)

# During decode, sequence length is 1 (generating one token at a time)
decode_seq_len = 1

# Input embeddings: [batch, seq_len, hidden]
input_emb = BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES

# Per layer intermediate activations (rough estimate):
# - Attention scores: [batch, heads, 1, past_seq_len] (at final step)
# - Attention output: [batch, 1, hidden]
# - MLP intermediate: [batch, 1, 4*hidden]
# - Residuals and norms
attn_scores_final = BATCH_SIZE * N_HEAD * 1 * final_seq_len * BYTES
attn_output = BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES
mlp_intermediate = BATCH_SIZE * decode_seq_len * (4 * HIDDEN_SIZE) * BYTES

# Peak activation per layer (conservative estimate)
peak_activation_per_layer = (
    attn_scores_final +    # Attention scores
    attn_output +          # Attention output
    mlp_intermediate +     # MLP activation
    2 * BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES  # Residuals
)

# Assuming some layers may have overlapping activations
total_activation_memory = 3 * peak_activation_per_layer  # Conservative estimate

print(f"  Decode sequence length: {decode_seq_len}")
print(f"  Attention scores (final step): [batch={BATCH_SIZE}, heads={N_HEAD}, 1, {final_seq_len}]")
print(f"  Attention scores memory: {format_mb(attn_scores_final):.2f} MB")
print(f"  Peak activation per layer: {format_mb(peak_activation_per_layer):.2f} MB")
print(f"  Estimated total activation memory: {format_mb(total_activation_memory):.2f} MB ({format_gb(total_activation_memory):.3f} GB)")

# ========== 4. Collected Hidden States (--collect-hidden) ==========
print("\n" + "=" * 70)
print("4. COLLECTED HIDDEN STATES (if --collect-hidden)")
print("=" * 70)

if COLLECT_HIDDEN:
    # HF returns hidden_states as tuple of (n_layer + 1) tensors
    # Each: [batch, seq_len, hidden]
    # During decode: seq_len = 1

    hidden_per_step = (N_LAYER + 1) * BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES

    # Accumulated over all decode steps (if kept in memory)
    hidden_total_all_steps = hidden_per_step * DECODE_STEPS

    print(f"  Hidden states per step: {N_LAYER + 1} tensors × [{BATCH_SIZE}, {decode_seq_len}, {HIDDEN_SIZE}]")
    print(f"  Memory per step: {format_mb(hidden_per_step):.2f} MB")
    print(f"  If accumulated ({DECODE_STEPS} steps): {format_mb(hidden_total_all_steps):.2f} MB ({format_gb(hidden_total_all_steps):.3f} GB)")
    print(f"\n  Note: Benchmark likely doesn't keep all steps, so actual usage ≈ {format_mb(hidden_per_step):.2f} MB")
else:
    hidden_total_all_steps = 0
    print(f"  Not collecting hidden states (disabled)")

# ========== 5. Collected Attention Weights (--collect-attention) ==========
print("\n" + "=" * 70)
print("5. COLLECTED ATTENTION WEIGHTS (if --collect-attention)")
print("=" * 70)

if COLLECT_ATTENTION:
    # HF returns attentions as tuple of n_layer tensors
    # Each: [batch, heads, seq_len, past_seq_len]
    # During decode at step k: [batch, heads, 1, prefill_tokens + k]

    # Average past_seq_len over all decode steps
    avg_past_seq_len = PREFILL_TOKENS + (DECODE_STEPS / 2)

    attn_per_step_avg = N_LAYER * BATCH_SIZE * N_HEAD * decode_seq_len * avg_past_seq_len * BYTES

    # Total if accumulated (sum of arithmetic sequence)
    # Step 1: past_len = prefill_tokens + 1
    # Step k: past_len = prefill_tokens + k
    # Sum = N_LAYER × batch × heads × 1 × Σ(prefill + k) for k=1..decode_steps
    sum_past_lens = sum(PREFILL_TOKENS + k for k in range(1, DECODE_STEPS + 1))
    attn_total_all_steps = N_LAYER * BATCH_SIZE * N_HEAD * 1 * sum_past_lens * BYTES

    print(f"  Attention weights per step (avg): {N_LAYER} tensors × [{BATCH_SIZE}, {N_HEAD}, {decode_seq_len}, {avg_past_seq_len:.1f}]")
    print(f"  Memory per step (avg): {format_mb(attn_per_step_avg):.2f} MB")
    print(f"  If accumulated ({DECODE_STEPS} steps): {format_mb(attn_total_all_steps):.2f} MB ({format_gb(attn_total_all_steps):.3f} GB)")
    print(f"\n  Note: Benchmark likely doesn't keep all steps, so actual usage ≈ {format_mb(attn_per_step_avg):.2f} MB")
else:
    attn_total_all_steps = 0
    print(f"  Not collecting attention weights (disabled)")

# ========== 6. Output Logits ==========
print("\n" + "=" * 70)
print("6. OUTPUT LOGITS")
print("=" * 70)

# Logits: [batch, seq_len, vocab_size]
logits_memory = BATCH_SIZE * decode_seq_len * VOCAB_SIZE * BYTES

print(f"  Logits shape: [{BATCH_SIZE}, {decode_seq_len}, {VOCAB_SIZE}]")
print(f"  Logits memory: {format_mb(logits_memory):.2f} MB")

# ========== TOTAL MEMORY ESTIMATE ==========
print("\n" + "=" * 70)
print("TOTAL GPU MEMORY ESTIMATE")
print("=" * 70)

# Base memory (always required)
base_memory = (
    model_memory +           # Model parameters
    kv_cache_total +         # KV cache (final state)
    total_activation_memory + # Forward pass activations
    logits_memory            # Output logits
)

# Peak memory (if all collections kept in memory - worst case)
peak_with_collections = (
    base_memory +
    hidden_per_step +        # Hidden states (one step worth)
    attn_per_step_avg        # Attention weights (one step worth)
)

# If benchmark accumulates everything (unrealistic but theoretical max)
peak_accumulated_all = (
    base_memory +
    hidden_total_all_steps +  # All hidden states
    attn_total_all_steps      # All attention weights
)

print(f"\nBase memory (no collections):")
print(f"  Model + KV cache + Activations + Logits")
print(f"  = {format_mb(base_memory):.2f} MB ({format_gb(base_memory):.3f} GB)")

print(f"\nPeak memory (with collections, per-step):")
print(f"  Base + Hidden (1 step) + Attention (1 step)")
print(f"  = {format_mb(peak_with_collections):.2f} MB ({format_gb(peak_with_collections):.3f} GB)")

print(f"\nTheoretical max (if accumulating ALL steps - unlikely):")
print(f"  Base + Hidden (all) + Attention (all)")
print(f"  = {format_mb(peak_accumulated_all):.2f} MB ({format_gb(peak_accumulated_all):.3f} GB)")

# Add PyTorch overhead and fragmentation
pytorch_overhead_factor = 1.2  # 20% overhead for memory management
realistic_peak = peak_with_collections * pytorch_overhead_factor

print(f"\n{'='*70}")
print(f"REALISTIC ESTIMATE (with 20% PyTorch overhead):")
print(f"  {format_gb(realistic_peak):.3f} GB")
print(f"{'='*70}")

print(f"\nMemory breakdown:")
print(f"  Model parameters:     {format_gb(model_memory):.3f} GB ({format_gb(model_memory)/format_gb(realistic_peak)*100:.1f}%)")
print(f"  KV cache:             {format_gb(kv_cache_total):.3f} GB ({format_gb(kv_cache_total)/format_gb(realistic_peak)*100:.1f}%)")
print(f"  Activations:          {format_gb(total_activation_memory):.3f} GB ({format_gb(total_activation_memory)/format_gb(realistic_peak)*100:.1f}%)")
print(f"  Logits:               {format_gb(logits_memory):.3f} GB ({format_gb(logits_memory)/format_gb(realistic_peak)*100:.1f}%)")
if COLLECT_HIDDEN:
    print(f"  Hidden states:        {format_gb(hidden_per_step):.3f} GB ({format_gb(hidden_per_step)/format_gb(realistic_peak)*100:.1f}%)")
if COLLECT_ATTENTION:
    print(f"  Attention weights:    {format_gb(attn_per_step_avg):.3f} GB ({format_gb(attn_per_step_avg)/format_gb(realistic_peak)*100:.1f}%)")

print(f"\n{'='*70}")
print(f"GPU COMPATIBILITY:")
if realistic_peak < 8:
    print(f"  ✅ Fits on 8GB GPU (RTX 3070, RTX 4060 Ti)")
elif realistic_peak < 12:
    print(f"  ✅ Fits on 12GB GPU (RTX 3060, RTX 4070)")
elif realistic_peak < 16:
    print(f"  ✅ Fits on 16GB GPU (RTX 4080)")
elif realistic_peak < 24:
    print(f"  ✅ Fits on 24GB GPU (RTX 3090, RTX 4090, A5000)")
else:
    print(f"  ⚠️  Requires > 24GB GPU (A100, H100)")

print(f"{'='*70}")

# Optimization suggestions
print(f"\n{'='*70}")
print(f"OPTIMIZATION SUGGESTIONS:")
print(f"{'='*70}")

if realistic_peak > 23:
    print(f"\n⚠️  Current config may cause OOM on 24GB GPU!")
    print(f"\nTo reduce memory usage:")

    # Calculate alternative configs
    bs_16 = realistic_peak * (16 / BATCH_SIZE)
    bs_32 = realistic_peak * (32 / BATCH_SIZE)

    print(f"  1. Reduce batch size:")
    print(f"     --batch-size 32 → ~{format_gb(bs_32):.2f} GB")
    print(f"     --batch-size 16 → ~{format_gb(bs_16):.2f} GB")

    fp16_mem = realistic_peak * 0.5
    print(f"  2. Use fp16:")
    print(f"     --dtype fp16 → ~{format_gb(fp16_mem):.2f} GB")

    combined = bs_32 * 0.5
    print(f"  3. Combined (batch=32, fp16):")
    print(f"     → ~{format_gb(combined):.2f} GB")
