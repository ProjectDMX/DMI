"""Compare torch.compile(reduce-overhead) with and without monitoring hooks.

Usage:
    PYTHONPATH=. python test_compile_warmup.py
"""
import time
import torch
from transformers import AutoModelForCausalLM

device = torch.device("cuda")
batch_size = 4
num_decode_steps = 10

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device=device, dtype=torch.float32).eval()

def forward_step(input_ids, past_key_values):
    outputs = model(
        input_ids, use_cache=True, past_key_values=past_key_values,
        output_hidden_states=True, output_attentions=True, return_dict=True,
    )
    return outputs.logits, outputs.past_key_values


# ============================================================
# Test 1: Vanilla (no hooks)
# ============================================================
print("\n=== Test 1: Vanilla GPT-2 + reduce-overhead (no hooks) ===")
compiled_vanilla = torch.compile(forward_step, mode="reduce-overhead", fullgraph=False)

token = torch.randint(0, 50257, (batch_size, 1), device=device)
with torch.no_grad():
    prefill = model(token, use_cache=True, return_dict=True)
past = prefill.past_key_values

with torch.no_grad():
    for i in range(num_decode_steps):
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.time()
        logits, new_past = compiled_vanilla(token, past)
        torch.cuda.synchronize()
        t1 = time.time()
        past = tuple((k.clone(), v.clone()) for k, v in new_past)
        print(f"  step {i}: {(t1-t0)*1000:.1f} ms  kv_seq_len={past[0][0].shape[-2]}")

# ============================================================
# Test 2: With monitoring hooks (compile mode)
# ============================================================
print("\n=== Test 2: GPT-2 + reduce-overhead + monitoring hooks ===")

# Need fresh model to avoid hook conflicts
from monitoring.graph_engine import GraphSafeEngine

engine = GraphSafeEngine(
    module_filter=lambda name, module: True,
    max_slots=4096, device=device, graph_mode="compile",
)
engine.prepare_for_model(model)
num_hooks = engine._monitor.num_slots() if engine._monitor else 0
print(f"  Registered {num_hooks} hooks")

# Reset compile cache for fair comparison
torch._dynamo.reset()
compiled_hooked = torch.compile(forward_step, mode="reduce-overhead", fullgraph=False)

with torch.no_grad():
    prefill = model(token, use_cache=True, return_dict=True)
past = prefill.past_key_values

with torch.no_grad():
    for i in range(num_decode_steps):
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.time()
        logits, new_past = compiled_hooked(token, past)
        torch.cuda.synchronize()
        t1 = time.time()
        past = tuple((k.clone(), v.clone()) for k, v in new_past)
        print(f"  step {i}: {(t1-t0)*1000:.1f} ms  kv_seq_len={past[0][0].shape[-2]}")

print("\nDone.")
