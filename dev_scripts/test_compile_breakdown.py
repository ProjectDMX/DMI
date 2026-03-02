"""Breakdown timing of torch.compile decode with monitoring.

Measures each component separately to identify bottleneck.

Usage:
    CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include \
    PYTHONPATH=. MON_NATIVE_FORCE_BUILD=1 MON_NATIVE_UNIFIED=1 \
    MON_NATIVE_TO_CPU=1 MON_NATIVE_CALLBACK=1 \
    python test_compile_breakdown.py
"""
import time
import torch
from transformers import AutoModelForCausalLM

device = torch.device("cuda")
batch_size = 64
num_decode_steps = 64
num_warmup = 2  # enough for torch.compile to cache all shapes


def fresh_prefill(model, token):
    """Run a fresh prefill to get clean past_key_values (not tainted by graph)."""
    with torch.no_grad():
        out = model(token, use_cache=True, return_dict=True)
    # Convert to legacy tuple format to avoid DynamicCache issues with graph
    past = out.past_key_values
    if not isinstance(past, tuple):
        past = tuple((layer.keys, layer.values) for layer in past.layers if layer.is_initialized)
    # Clone to detach from any graph buffers
    return tuple((k.clone(), v.clone()) for k, v in past)


# ── Load model ──────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device=device, dtype=torch.float32).eval()

token = torch.randint(0, 50257, (batch_size, 1), device=device)


# ── Test 1: Vanilla torch.compile (no hooks, no monitoring) ─────
print("\n=== Test 1: Vanilla torch.compile (no hooks) ===")
torch._dynamo.reset()

def forward_step(token, past_key_values):
    outputs = model(
        token, use_cache=True, past_key_values=past_key_values,
        output_hidden_states=True, output_attentions=True, return_dict=True,
    )
    return outputs.logits, outputs.past_key_values

compiled_forward = torch.compile(forward_step, mode="reduce-overhead", fullgraph=False)

# Warmup
print("  Warming up...")
with torch.no_grad():
    for w in range(num_warmup):
        p = fresh_prefill(model, token)
        for i in range(num_decode_steps):
            torch.compiler.cudagraph_mark_step_begin()
            _, new_past = compiled_forward(token, p)
            p = tuple((k.clone(), v.clone()) for k, v in new_past)

# Measure
torch.cuda.synchronize()
with torch.no_grad():
    p = fresh_prefill(model, token)
    t0 = time.perf_counter()
    for i in range(num_decode_steps):
        torch.compiler.cudagraph_mark_step_begin()
        _, new_past = compiled_forward(token, p)
        p = tuple((k.clone(), v.clone()) for k, v in new_past)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
elapsed = t1 - t0
print(f"  Total: {elapsed*1000:.1f} ms | Per step: {elapsed/num_decode_steps*1000:.2f} ms")


# ── Test 2: torch.compile + monitoring hooks (no drain) ─────────
print("\n=== Test 2: torch.compile + monitoring hooks (NO drain) ===")
from monitoring.graph_engine import GraphSafeEngine

engine = GraphSafeEngine(
    module_filter=lambda name, module: True,
    max_slots=4096, device=device, graph_mode="compile",
)
engine.prepare_for_model(model)
num_hooks = engine._monitor.num_slots() if engine._monitor else 0
print(f"  Registered {num_hooks} hooks")

torch._dynamo.reset()
compiled_forward2 = torch.compile(forward_step, mode="reduce-overhead", fullgraph=False)

# Warmup
print("  Warming up...")
with torch.no_grad():
    for w in range(num_warmup):
        p = fresh_prefill(model, token)
        for i in range(num_decode_steps):
            torch.compiler.cudagraph_mark_step_begin()
            engine.start_step()
            _, new_past = compiled_forward2(token, p)
            engine.end_step()
            p = tuple((k.clone(), v.clone()) for k, v in new_past)
        engine.resolve_all()

# Measure: forward + hooks, but NO drain_ready_results
torch.cuda.synchronize()
with torch.no_grad():
    p = fresh_prefill(model, token)
    t0 = time.perf_counter()
    for i in range(num_decode_steps):
        torch.compiler.cudagraph_mark_step_begin()
        engine.start_step()
        _, new_past = compiled_forward2(token, p)
        engine.end_step()
        p = tuple((k.clone(), v.clone()) for k, v in new_past)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
elapsed = t1 - t0
print(f"  Total: {elapsed*1000:.1f} ms | Per step: {elapsed/num_decode_steps*1000:.2f} ms")
# Drain leftover
engine.resolve_all()


# ── Test 3: torch.compile + monitoring + drain per step ─────────
print("\n=== Test 3: torch.compile + monitoring + drain_ready_results(wait=True) per step ===")

torch._dynamo.reset()
compiled_forward3 = torch.compile(forward_step, mode="reduce-overhead", fullgraph=False)

# Warmup
print("  Warming up...")
with torch.no_grad():
    for w in range(num_warmup):
        p = fresh_prefill(model, token)
        for i in range(num_decode_steps):
            torch.compiler.cudagraph_mark_step_begin()
            engine.start_step()
            _, new_past = compiled_forward3(token, p)
            engine.end_step()
            drained = False
            while not drained:
                drained = engine.drain_ready_results(wait=True)
            p = tuple((k.clone(), v.clone()) for k, v in new_past)

# Measure: full pipeline
torch.cuda.synchronize()
with torch.no_grad():
    p = fresh_prefill(model, token)
    t0 = time.perf_counter()
    for i in range(num_decode_steps):
        torch.compiler.cudagraph_mark_step_begin()
        engine.start_step()
        _, new_past = compiled_forward3(token, p)
        engine.end_step()
        drained = False
        while not drained:
            drained = engine.drain_ready_results(wait=True)
        p = tuple((k.clone(), v.clone()) for k, v in new_past)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
elapsed = t1 - t0
print(f"  Total: {elapsed*1000:.1f} ms | Per step: {elapsed/num_decode_steps*1000:.2f} ms")


# ── Test 4: Per-step breakdown of Test 3 ───────────────────────
print("\n=== Test 4: Per-step breakdown (last 10 steps) ===")

torch.cuda.synchronize()
with torch.no_grad():
    p = fresh_prefill(model, token)
    for i in range(num_decode_steps):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        torch.compiler.cudagraph_mark_step_begin()
        engine.start_step()
        t_after_start = time.perf_counter()

        _, new_past = compiled_forward3(token, p)
        torch.cuda.synchronize()
        t_after_forward = time.perf_counter()

        engine.end_step()
        t_after_end = time.perf_counter()

        drained = False
        while not drained:
            drained = engine.drain_ready_results(wait=True)
        t_after_drain = time.perf_counter()

        p = tuple((k.clone(), v.clone()) for k, v in new_past)
        torch.cuda.synchronize()
        t_after_clone = time.perf_counter()

        if i >= num_decode_steps - 10:
            print(f"  step {i}: "
                  f"start={1000*(t_after_start-t_start):.2f}ms "
                  f"forward={1000*(t_after_forward-t_after_start):.2f}ms "
                  f"end_step={1000*(t_after_end-t_after_forward):.2f}ms "
                  f"drain={1000*(t_after_drain-t_after_end):.2f}ms "
                  f"clone={1000*(t_after_clone-t_after_drain):.2f}ms "
                  f"TOTAL={1000*(t_after_clone-t_start):.2f}ms")

print("\nDone.")
