"""Compare per-step timing: vanilla GPT2 vs HookedGPT2Model, eager vs compiled.

Usage:
    CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include \
    PYTHONPATH=. MON_NATIVE_FORCE_BUILD=1 MON_NATIVE_UNIFIED=1 \
    python diagnose_slowdown.py
"""
import time
import torch
from transformers import AutoModelForCausalLM
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model

device = torch.device("cuda")
batch_size = 4
num_decode = 64
num_warmup = 3


def to_tuple_past(past):
    """Convert DynamicCache to tuple format for CUDA Graph compatibility."""
    if isinstance(past, tuple):
        return past
    # DynamicCache — extract key/value tensors
    result = []
    for layer in past.layers if hasattr(past, 'layers') else past:
        if hasattr(layer, 'keys') and hasattr(layer, 'values'):
            result.append((layer.keys, layer.values))
        elif isinstance(layer, (list, tuple)) and len(layer) == 2:
            result.append((layer[0], layer[1]))
    if result:
        return tuple(result)
    # Fallback: try iterating
    return tuple((k, v) for k, v in past)


def fresh_prefill(model, token):
    with torch.no_grad():
        out = model(token, use_cache=True, return_dict=True)
    past = out.past_key_values
    # Clone to detach from any computation graph
    past_t = to_tuple_past(past)
    return tuple((k.clone(), v.clone()) for k, v in past_t)


def measure_eager(label, model, project_fn, collect_outputs=False):
    token = torch.randint(0, 50257, (batch_size, 1), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            past = fresh_prefill(model, token)
            for i in range(num_decode):
                out = model(
                    token, use_cache=True, past_key_values=past,
                    output_hidden_states=collect_outputs,
                    output_attentions=collect_outputs,
                    return_dict=True,
                )
                logits = project_fn(out)
                past = to_tuple_past(out.past_key_values)
    torch.cuda.synchronize()

    # Measure
    with torch.no_grad():
        past = fresh_prefill(model, token)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(num_decode):
            out = model(
                token, use_cache=True, past_key_values=past,
                output_hidden_states=collect_outputs,
                output_attentions=collect_outputs,
                return_dict=True,
            )
            logits = project_fn(out)
            past = to_tuple_past(out.past_key_values)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed = t1 - t0
    per_step = elapsed / num_decode * 1000
    print(f"  {label:50s}: {elapsed*1000:.1f} ms total, {per_step:.2f} ms/step")
    return per_step


def measure_compiled(label, model, project_fn, collect_outputs=False):
    token = torch.randint(0, 50257, (batch_size, 1), device=device)

    torch._dynamo.reset()

    def forward_step(tok, past_key_values):
        out = model(
            tok, use_cache=True, past_key_values=past_key_values,
            output_hidden_states=collect_outputs,
            output_attentions=collect_outputs,
            return_dict=True,
        )
        logits = project_fn(out)
        return logits, out.past_key_values

    compiled_fn = torch.compile(forward_step, mode="reduce-overhead", fullgraph=False)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            past = fresh_prefill(model, token)
            for i in range(num_decode):
                torch.compiler.cudagraph_mark_step_begin()
                logits, new_past = compiled_fn(token, past)
                past = tuple((k.clone(), v.clone()) for k, v in to_tuple_past(new_past))
    torch.cuda.synchronize()

    # Measure
    with torch.no_grad():
        past = fresh_prefill(model, token)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(num_decode):
            torch.compiler.cudagraph_mark_step_begin()
            logits, new_past = compiled_fn(token, past)
            past = tuple((k.clone(), v.clone()) for k, v in to_tuple_past(new_past))
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed = t1 - t0
    per_step = elapsed / num_decode * 1000
    print(f"  {label:50s}: {elapsed*1000:.1f} ms total, {per_step:.2f} ms/step")
    return per_step


# Load models
print("Loading models...")
vanilla = AutoModelForCausalLM.from_pretrained(
    "gpt2", attn_implementation="eager"
).to(device).eval()
hooked = HookedGPT2Model.from_pretrained(
    "gpt2", attn_implementation="eager"
).to(device).eval()

# Vanilla GPT2 has lm_head built-in
def vanilla_project(out):
    return out.logits

# HookedGPT2 needs external lm_head
lm_head = vanilla.lm_head
def hooked_project(out):
    return lm_head(out.last_hidden_state)

print(f"\nbatch_size={batch_size}, decode_steps={num_decode}\n")

# Test matrix
print("=== Eager mode ===")
measure_eager("vanilla GPT2 (eager)", vanilla, vanilla_project)
measure_eager("vanilla GPT2 (eager, +outputs)", vanilla, vanilla_project, collect_outputs=True)
measure_eager("HookedGPT2 (eager)", hooked, hooked_project)
measure_eager("HookedGPT2 (eager, +outputs)", hooked, hooked_project, collect_outputs=True)

print("\n=== torch.compile(mode='reduce-overhead') ===")
measure_compiled("vanilla GPT2 (compiled)", vanilla, vanilla_project)
measure_compiled("vanilla GPT2 (compiled, +outputs)", vanilla, vanilla_project, collect_outputs=True)
measure_compiled("HookedGPT2 (compiled)", hooked, hooked_project)
measure_compiled("HookedGPT2 (compiled, +outputs)", hooked, hooked_project, collect_outputs=True)

print("\nDone.")
