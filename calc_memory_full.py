#!/usr/bin/env python3
"""
Complete memory calculation for profile_decode.py benchmark.
This benchmark loads 3 GPT-2 models simultaneously!
"""

# Benchmark configuration
BATCH_SIZE = 64
PREFILL_TOKENS = 1
DECODE_STEPS = 64
DTYPE = "fp32"
COLLECT_HIDDEN = True
COLLECT_ATTENTION = True
NUM_MODELS = 3  # hf_model, hf_hooked_model, tl_model

# GPT-2 config
N_LAYER = 12
N_HEAD = 12
HIDDEN_SIZE = 768
HEAD_DIM = 64
VOCAB_SIZE = 50257
MAX_POS = 1024

BYTES_PER_ELEMENT = {"fp32": 4, "fp16": 2, "bf16": 2}
BYTES = BYTES_PER_ELEMENT[DTYPE]

def format_mb(b):
    return b / (1024 ** 2)

def format_gb(b):
    return b / (1024 ** 3)

print("=" * 70)
print("COMPLETE Memory Calculation for profile_decode.py")
print("=" * 70)
print(f"\n⚠️  IMPORTANT: This benchmark loads {NUM_MODELS} separate GPT-2 models!")
print(f"    1. hf_model (AutoModelForCausalLM)")
print(f"    2. hf_hooked_model (HookedGPT2Model)")
print(f"    3. tl_model (HookedTransformer)")
print(f"\nConfiguration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Decode steps: {DECODE_STEPS}")
print(f"  Data type: {DTYPE} ({BYTES} bytes/element)")

# ========== 1. Model Parameters (×3 models!) ==========
print("\n" + "=" * 70)
print("1. MODEL PARAMETERS (×3 models in memory simultaneously!)")
print("=" * 70)

wte = VOCAB_SIZE * HIDDEN_SIZE
wpe = MAX_POS * HIDDEN_SIZE
per_block = (
    HIDDEN_SIZE * (3 * HIDDEN_SIZE) +  # c_attn
    HIDDEN_SIZE * HIDDEN_SIZE +         # c_proj
    HIDDEN_SIZE * (4 * HIDDEN_SIZE) +   # mlp.c_fc
    (4 * HIDDEN_SIZE) * HIDDEN_SIZE +   # mlp.c_proj
    4 * HIDDEN_SIZE                     # LayerNorms
)

params_per_model = wte + wpe + N_LAYER * per_block + 2 * HIDDEN_SIZE
memory_per_model = params_per_model * BYTES
total_model_memory = memory_per_model * NUM_MODELS

print(f"  Parameters per model: {params_per_model:,} ({params_per_model/1e6:.2f}M)")
print(f"  Memory per model: {format_gb(memory_per_model):.3f} GB")
print(f"  Total for {NUM_MODELS} models: {format_gb(total_model_memory):.3f} GB ⚠️")

# ========== 2. KV Cache (per model) ==========
print("\n" + "=" * 70)
print("2. KV CACHE (Each model has its own)")
print("=" * 70)

final_seq_len = PREFILL_TOKENS + DECODE_STEPS
kv_per_layer = 2 * BATCH_SIZE * N_HEAD * final_seq_len * HEAD_DIM * BYTES
kv_cache_per_model = N_LAYER * kv_per_layer

# Only the active model needs KV cache at any time (they run sequentially)
# But during transitions, both old and new might be in memory
kv_cache_active = kv_cache_per_model
kv_cache_peak = 2 * kv_cache_per_model  # Conservative: 2 models worth during transition

print(f"  KV cache per model: {format_gb(kv_cache_per_model):.3f} GB")
print(f"  Active (1 model running): {format_gb(kv_cache_active):.3f} GB")
print(f"  Peak (during transition): {format_gb(kv_cache_peak):.3f} GB")

# ========== 3. Activations ==========
print("\n" + "=" * 70)
print("3. ACTIVATIONS (Per forward pass)")
print("=" * 70)

decode_seq_len = 1
attn_scores = BATCH_SIZE * N_HEAD * decode_seq_len * final_seq_len * BYTES
attn_output = BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES
mlp_intermediate = BATCH_SIZE * decode_seq_len * (4 * HIDDEN_SIZE) * BYTES

peak_activation_per_layer = (
    attn_scores +
    attn_output +
    mlp_intermediate +
    2 * BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES
)

total_activation = 3 * peak_activation_per_layer

print(f"  Peak activation per layer: {format_mb(peak_activation_per_layer):.2f} MB")
print(f"  Total activation memory: {format_mb(total_activation):.2f} MB ({format_gb(total_activation):.3f} GB)")

# ========== 4. Collected Data ==========
print("\n" + "=" * 70)
print("4. COLLECTED HIDDEN STATES & ATTENTION")
print("=" * 70)

if COLLECT_HIDDEN:
    hidden_per_step = (N_LAYER + 1) * BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES
    print(f"  Hidden states per step: {format_mb(hidden_per_step):.2f} MB")
else:
    hidden_per_step = 0

if COLLECT_ATTENTION:
    avg_past_len = PREFILL_TOKENS + (DECODE_STEPS / 2)
    attn_per_step = N_LAYER * BATCH_SIZE * N_HEAD * decode_seq_len * avg_past_len * BYTES
    print(f"  Attention per step (avg): {format_mb(attn_per_step):.2f} MB")
else:
    attn_per_step = 0

collected_data = hidden_per_step + attn_per_step

# ========== 5. Output Logits ==========
logits_memory = BATCH_SIZE * decode_seq_len * VOCAB_SIZE * BYTES

# ========== 6. Hook Caches (for HF hook variants) ==========
print("\n" + "=" * 70)
print("5. HOOK CACHES (setup_hf_decode_hook)")
print("=" * 70)

# The setup_hf_decode_hook creates many cache lists (line 296-306)
# Each cache stores N_LAYER tensors per step
hook_cache_per_layer = (
    BATCH_SIZE * N_HEAD * decode_seq_len * final_seq_len * BYTES +  # attn_cache
    3 * BATCH_SIZE * N_HEAD * decode_seq_len * HEAD_DIM * BYTES +   # q,k,v cache
    BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES +              # attn_output
    2 * BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES +          # resid pre/post
    2 * BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES +          # ln1/ln2
    2 * BATCH_SIZE * decode_seq_len * HIDDEN_SIZE * BYTES            # mlp in/out
)

hook_cache_total = N_LAYER * hook_cache_per_layer

print(f"  Hook cache per layer: {format_mb(hook_cache_per_layer):.2f} MB")
print(f"  Total hook cache: {format_mb(hook_cache_total):.2f} MB ({format_gb(hook_cache_total):.3f} GB)")

# ========== TOTAL CALCULATION ==========
print("\n" + "=" * 70)
print("PEAK GPU MEMORY ESTIMATE")
print("=" * 70)

# Base: all 3 models always in memory
base_models = total_model_memory

# Runtime: one model active at a time, but KV cache can overlap during transition
runtime_per_model = (
    kv_cache_active +      # KV cache for active model
    total_activation +      # Activations
    collected_data +        # Hidden/attention collections
    logits_memory          # Logits
)

# Peak during hook benchmarks (they add extra caches)
peak_with_hooks = (
    base_models +
    kv_cache_peak +        # KV cache (with transition overlap)
    total_activation +
    collected_data +
    logits_memory +
    hook_cache_total       # Hook caches
)

# Peak without hooks (most benchmarks)
peak_without_hooks = (
    base_models +
    kv_cache_peak +
    total_activation +
    collected_data +
    logits_memory
)

pytorch_overhead = 1.25  # 25% for memory fragmentation, overhead, etc.

realistic_peak_with_hooks = peak_with_hooks * pytorch_overhead
realistic_peak_without_hooks = peak_without_hooks * pytorch_overhead

print(f"\nBase (3 models loaded):")
print(f"  {format_gb(base_models):.3f} GB")

print(f"\nPeak WITHOUT hooks (most benchmarks):")
print(f"  Models + KV + Activations + Collections + Logits")
print(f"  = {format_gb(peak_without_hooks):.3f} GB (before overhead)")
print(f"  = {format_gb(realistic_peak_without_hooks):.3f} GB (with 25% PyTorch overhead)")

print(f"\nPeak WITH hooks (huggingface_hook variants):")
print(f"  Above + Hook caches")
print(f"  = {format_gb(peak_with_hooks):.3f} GB (before overhead)")
print(f"  = {format_gb(realistic_peak_with_hooks):.3f} GB (with 25% PyTorch overhead)")

print(f"\n{'='*70}")
print(f"ESTIMATED PEAK MEMORY USAGE:")
print(f"  WITHOUT hooks: {format_gb(realistic_peak_without_hooks):.2f} GB")
print(f"  WITH hooks:    {format_gb(realistic_peak_with_hooks):.2f} GB")
print(f"{'='*70}")

print(f"\nDetailed breakdown (peak with hooks):")
print(f"  3× Model parameters:  {format_gb(base_models):.2f} GB ({base_models/realistic_peak_with_hooks*100:.1f}%)")
print(f"  KV cache (peak):      {format_gb(kv_cache_peak):.2f} GB ({kv_cache_peak/realistic_peak_with_hooks*100:.1f}%)")
print(f"  Activations:          {format_gb(total_activation):.2f} GB ({total_activation/realistic_peak_with_hooks*100:.1f}%)")
print(f"  Hook caches:          {format_gb(hook_cache_total):.2f} GB ({hook_cache_total/realistic_peak_with_hooks*100:.1f}%)")
print(f"  Collections:          {format_gb(collected_data):.2f} GB ({collected_data/realistic_peak_with_hooks*100:.1f}%)")
print(f"  Logits:               {format_gb(logits_memory):.2f} GB ({logits_memory/realistic_peak_with_hooks*100:.1f}%)")

print(f"\n{'='*70}")
print(f"GPU COMPATIBILITY:")
gpu_sizes = [
    (8, "RTX 3070, RTX 4060 Ti"),
    (12, "RTX 3060, RTX 4070"),
    (16, "RTX 4080, RTX 4070 Ti SUPER"),
    (24, "RTX 3090, RTX 4090, A5000, A6000"),
    (40, "A100 40GB"),
    (80, "A100 80GB, H100"),
]

for size_gb, name in gpu_sizes:
    if realistic_peak_with_hooks <= size_gb:
        print(f"  ✅ Fits on {size_gb}GB GPU ({name})")
        break
else:
    print(f"  ⚠️  Requires > 80GB GPU")

if realistic_peak_with_hooks > 23.5:
    print(f"\n⚠️  WILL CAUSE OOM ON 24GB GPU!")
    print(f"    Peak usage ({format_gb(realistic_peak_with_hooks):.2f} GB) exceeds available memory")

print(f"{'='*70}")

# ========== OPTIMIZATION SUGGESTIONS ==========
print(f"\n{'='*70}")
print(f"OPTIMIZATION SUGGESTIONS:")
print(f"{'='*70}")

if realistic_peak_with_hooks > 23:
    print(f"\n⚠️  Current config WILL OOM on 24GB GPU!")
    print(f"\nOptions to reduce memory:")

    # Option 1: Reduce batch size
    bs_32 = realistic_peak_with_hooks * (32 / BATCH_SIZE)
    bs_16 = realistic_peak_with_hooks * (16 / BATCH_SIZE)
    bs_8 = realistic_peak_with_hooks * (8 / BATCH_SIZE)

    print(f"\n  1. Reduce batch size (linear scaling):")
    print(f"     --batch-size 32 → {format_gb(bs_32):.2f} GB {'✅' if bs_32 <= 23 else '❌'}")
    print(f"     --batch-size 16 → {format_gb(bs_16):.2f} GB {'✅' if bs_16 <= 23 else '❌'}")
    print(f"     --batch-size 8  → {format_gb(bs_8):.2f} GB ✅")

    # Option 2: Use fp16
    fp16_mem = realistic_peak_with_hooks * 0.5
    print(f"\n  2. Use fp16 (halves most memory):")
    print(f"     --dtype fp16 → {format_gb(fp16_mem):.2f} GB {'✅' if fp16_mem <= 23 else '❌'}")

    # Option 3: Combined
    combined = bs_16 * 0.5
    print(f"\n  3. Combined (batch=16, fp16):")
    print(f"     → {format_gb(combined):.2f} GB ✅")

    # Option 4: Skip hook benchmarks
    print(f"\n  4. Skip hook benchmarks (comment out huggingface_hook*):")
    print(f"     → {format_gb(realistic_peak_without_hooks):.2f} GB {'✅' if realistic_peak_without_hooks <= 23 else '❌'}")

    print(f"\n{'='*70}")
    print(f"RECOMMENDED for 24GB GPU:")
    if fp16_mem <= 23:
        print(f"  --dtype fp16 --batch-size 64")
        print(f"  (Estimated: {format_gb(fp16_mem):.2f} GB)")
    elif bs_16 <= 23:
        print(f"  --batch-size 16 --dtype fp32")
        print(f"  (Estimated: {format_gb(bs_16):.2f} GB)")
    else:
        print(f"  --batch-size 16 --dtype fp16")
        print(f"  (Estimated: {format_gb(combined):.2f} GB)")
    print(f"{'='*70}")
elif realistic_peak_with_hooks > 20:
    print(f"\n⚠️  Current config is close to 24GB limit")
    print(f"    Recommend --dtype fp16 for safety margin")
else:
    print(f"\n✅ Current config should fit on 24GB GPU with room to spare")
