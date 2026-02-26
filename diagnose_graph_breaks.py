"""Diagnose graph breaks in HookedGPT2Model under torch.compile.

Usage:
    CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include \
    PYTHONPATH=. MON_NATIVE_FORCE_BUILD=1 MON_NATIVE_UNIFIED=1 \
    TORCH_LOGS="graph_breaks" \
    python diagnose_graph_breaks.py 2>&1 | head -200
"""
import os
import logging
import torch
from transformers import AutoModelForCausalLM

# Reduce noise
logging.disable(logging.WARNING)

device = torch.device("cuda")

print("=== Loading HookedGPT2Model ===")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device=device, dtype=torch.float32).eval()
print(f"Model class: {type(model).__name__}")

token = torch.randint(0, 50257, (2, 1), device=device)

# Prefill to get past_key_values
with torch.no_grad():
    out = model(token, use_cache=True, return_dict=True)
past = out.past_key_values

def forward_step(input_ids, past_kv):
    outputs = model(
        input_ids,
        use_cache=True,
        past_key_values=past_kv,
        return_dict=True,
    )
    return outputs.logits, outputs.past_key_values

print("\n=== Running torch._dynamo.explain ===")
try:
    explanation = torch._dynamo.explain(forward_step)(token, past)
    print(f"\nExplanation result type: {type(explanation)}")
    print(f"Explanation dir: {[a for a in dir(explanation) if not a.startswith('_')]}")
    print(f"\nFull explanation:\n{explanation}")
except Exception as e:
    print(f"explain() failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Counting compiled graph regions ===")
torch._dynamo.reset()
cnt = torch._dynamo.testing.CompileCounter()
compiled = torch.compile(forward_step, backend=cnt)
with torch.no_grad():
    try:
        compiled(token, past)
        print(f"Number of graph captures (frames): {cnt.frame_count}")
        print(f"Number of ops in graphs: {cnt.op_count}")
    except Exception as e:
        print(f"CompileCounter failed: {e}")
