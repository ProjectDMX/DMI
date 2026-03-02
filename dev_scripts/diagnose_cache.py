"""Diagnose: DynamicCache vs tuple past vs StaticCache under torch.compile.

Usage:
    CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include \
    PYTHONPATH=. MON_NATIVE_FORCE_BUILD=1 MON_NATIVE_UNIFIED=1 \
    python diagnose_cache.py
"""
import time
import torch
from transformers import AutoModelForCausalLM, StaticCache
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

device = torch.device("cuda")
batch_size = 4
num_decode = 64
num_warmup = 3
max_cache_len = 128  # for StaticCache


def to_tuple_past(past):
    if isinstance(past, tuple):
        return past
    result = []
    for layer_idx in range(len(past)):
        kv = past[layer_idx]
        if isinstance(kv, (list, tuple)) and len(kv) == 2:
            result.append((kv[0], kv[1]))
        else:
            # DynamicCache layer
            result.append((past.key_cache[layer_idx], past.value_cache[layer_idx]))
    return tuple(result)


def measure_tuple_past(label, model, project_fn):
    """Pass tuple past_key_values — forces DynamicCache.from_legacy_cache each step."""
    token = torch.randint(0, 50257, (batch_size, 1), device=device)
    torch._dynamo.reset()

    def fwd(tok, past_kv):
        out = model(tok, use_cache=True, past_key_values=past_kv, return_dict=True)
        return project_fn(out), out.past_key_values

    compiled = torch.compile(fwd, mode="reduce-overhead", fullgraph=False)

    with torch.no_grad():
        for _ in range(num_warmup):
            out0 = model(token, use_cache=True, return_dict=True)
            past = tuple((k.clone(), v.clone()) for k, v in to_tuple_past(out0.past_key_values))
            for _ in range(num_decode):
                torch.compiler.cudagraph_mark_step_begin()
                logits, new_past = compiled(token, past)
                past = tuple((k.clone(), v.clone()) for k, v in to_tuple_past(new_past))
    torch.cuda.synchronize()

    with torch.no_grad():
        out0 = model(token, use_cache=True, return_dict=True)
        past = tuple((k.clone(), v.clone()) for k, v in to_tuple_past(out0.past_key_values))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_decode):
            torch.compiler.cudagraph_mark_step_begin()
            logits, new_past = compiled(token, past)
            past = tuple((k.clone(), v.clone()) for k, v in to_tuple_past(new_past))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms = (t1 - t0) / num_decode * 1000
    print(f"  {label:50s}: {(t1-t0)*1000:.1f} ms total, {ms:.2f} ms/step")


def measure_dynamic_cache(label, model, project_fn):
    """Pass DynamicCache directly — same pattern as benchmark TorchCompileDecodeRunner."""
    token = torch.randint(0, 50257, (batch_size, 1), device=device)
    torch._dynamo.reset()

    def fwd(tok, past_kv):
        out = model(tok, use_cache=True, past_key_values=past_kv, return_dict=True)
        return project_fn(out), out.past_key_values

    compiled = torch.compile(fwd, mode="reduce-overhead", fullgraph=False)

    with torch.no_grad():
        for _ in range(num_warmup):
            out0 = model(token, use_cache=True, return_dict=True)
            past = out0.past_key_values  # DynamicCache
            for _ in range(num_decode):
                torch.compiler.cudagraph_mark_step_begin()
                logits, past = compiled(token, past)
    torch.cuda.synchronize()

    with torch.no_grad():
        out0 = model(token, use_cache=True, return_dict=True)
        past = out0.past_key_values
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_decode):
            torch.compiler.cudagraph_mark_step_begin()
            logits, past = compiled(token, past)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms = (t1 - t0) / num_decode * 1000
    print(f"  {label:50s}: {(t1-t0)*1000:.1f} ms total, {ms:.2f} ms/step")


def measure_static_cache(label, model, project_fn):
    """Use StaticCache — pre-allocated, no torch.cat, ideal for CUDA Graphs."""
    token = torch.randint(0, 50257, (batch_size, 1), device=device)
    torch._dynamo.reset()

    def fwd(tok, past_kv):
        out = model(tok, use_cache=True, past_key_values=past_kv, return_dict=True)
        return project_fn(out), out.past_key_values

    compiled = torch.compile(fwd, mode="reduce-overhead", fullgraph=False)

    config = model.config if hasattr(model, 'config') else model.transformer.config if hasattr(model, 'transformer') else None

    with torch.no_grad():
        for _ in range(num_warmup):
            cache = StaticCache(
                config=config,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                device=device,
                dtype=torch.float32,
            )
            # prefill
            model(token, use_cache=True, past_key_values=cache, return_dict=True)
            for _ in range(num_decode):
                torch.compiler.cudagraph_mark_step_begin()
                logits, cache = compiled(token, cache)
    torch.cuda.synchronize()

    with torch.no_grad():
        cache = StaticCache(
            config=config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.float32,
        )
        model(token, use_cache=True, past_key_values=cache, return_dict=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_decode):
            torch.compiler.cudagraph_mark_step_begin()
            logits, cache = compiled(token, cache)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms = (t1 - t0) / num_decode * 1000
    print(f"  {label:50s}: {(t1-t0)*1000:.1f} ms total, {ms:.2f} ms/step")


print("Loading models...")
hooked = HookedGPT2Model.from_pretrained("gpt2", attn_implementation="eager").to(device).eval()
vanilla = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").to(device).eval()

lm_head = vanilla.lm_head
def hooked_project(out): return lm_head(out.last_hidden_state)
def vanilla_project(out): return out.logits

print(f"\nbatch_size={batch_size}, decode_steps={num_decode}\n")

print("=== torch.compile + tuple past (clone outside compiled fn) ===")
measure_tuple_past("vanilla GPT2 (tuple past)", vanilla, vanilla_project)
measure_tuple_past("HookedGPT2 (tuple past)", hooked, hooked_project)

print("\n=== torch.compile + DynamicCache (no clone, pass directly) ===")
try:
    measure_dynamic_cache("vanilla GPT2 (DynamicCache)", vanilla, vanilla_project)
except Exception as e:
    print(f"  vanilla GPT2 (DynamicCache) FAILED: {e}")
try:
    measure_dynamic_cache("HookedGPT2 (DynamicCache)", hooked, hooked_project)
except Exception as e:
    print(f"  HookedGPT2 (DynamicCache) FAILED: {e}")

print("\n=== torch.compile + StaticCache (pre-allocated, fixed size) ===")
try:
    measure_static_cache("vanilla GPT2 (StaticCache)", vanilla, vanilla_project)
except Exception as e:
    print(f"  vanilla GPT2 (StaticCache) FAILED: {e}")
try:
    measure_static_cache("HookedGPT2 (StaticCache)", hooked, hooked_project)
except Exception as e:
    print(f"  HookedGPT2 (StaticCache) FAILED: {e}")

print("\nDone.")
