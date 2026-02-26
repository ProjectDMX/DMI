"""Diagnose: verify monitoring tensors actually land on CPU after drain."""
import os, sys, torch

os.environ.setdefault("MON_NATIVE_FORCE_BUILD", "1")
os.environ.setdefault("MON_NATIVE_UNIFIED", "1")
os.environ.setdefault("MON_NATIVE_TO_CPU", "1")
os.environ.setdefault("MON_NATIVE_CALLBACK", "1")

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, StaticCache
from transformers.models.gpt2_p.modeling_gpt2 import HookedGPT2Model
from monitoring import GraphSafeEngine, GraphSlotConsumer, _native_engine
from monitoring.config import CaptureSchedule, HookSelection, MonitoringConfig
from monitoring.monitor_native import create_graph_delegate

device = torch.device("cuda")
dtype = torch.float32

model = HookedGPT2Model.from_pretrained("gpt2", attn_implementation="eager", torch_dtype=dtype)
model.to(device).eval()

config = MonitoringConfig(
    hooks=HookSelection(mode="full"),
    schedule=CaptureSchedule(),
)
engine = GraphSafeEngine(
    config=config, module_filter=lambda n, m: True,
    max_slots=4096, device=device, graph_mode="compile",
)
consumer = GraphSlotConsumer(delay_steps=0)
engine.attach_consumer(consumer)

native_backend = _native_engine.create_engine(queue_size=0, cache_dtype=None, delay_steps=0)
delegate = create_graph_delegate(native_backend)
engine.attach_backend_delegate(delegate)

engine.prepare_for_model(model)

BATCH = 64
MAX_CACHE_LEN = 1 + 64 + 16  # = 81, same as benchmark

cache = StaticCache(config=model.config, max_cache_len=MAX_CACHE_LEN)
token = torch.full((BATCH, 1), 50256, device=device, dtype=torch.long)

with torch.no_grad():
    out = model(token, use_cache=True, past_key_values=cache, return_dict=True,
                output_hidden_states=False, output_attentions=False)
    next_token = out.last_hidden_state[:, -1:, :].argmax(dim=-1)

# Run a few decode steps to advance cache_pos, then capture one with monitoring
NUM_EAGER_STEPS = 30  # advance to seq_len ~31
with torch.no_grad():
    for i in range(NUM_EAGER_STEPS):
        cache_pos = torch.tensor([1 + i], device=device, dtype=torch.long)
        out = model(next_token, use_cache=True, past_key_values=cache,
                    cache_position=cache_pos, return_dict=True,
                    output_hidden_states=False, output_attentions=False)
        next_token = out.last_hidden_state[:, -1:, :].argmax(dim=-1)

print(f"Batch={BATCH}, max_cache_len={MAX_CACHE_LEN}, "
      f"actual cache_pos={1 + NUM_EAGER_STEPS}")

# One monitored decode step
with torch.no_grad():
    engine.start_step()
    cache_pos = torch.tensor([1 + NUM_EAGER_STEPS], device=device, dtype=torch.long)
    out = model(next_token, use_cache=True, past_key_values=cache,
                cache_position=cache_pos, return_dict=True,
                output_hidden_states=False, output_attentions=False)
    engine.end_step()

# Drain via proper pipeline (drain_ready_results → delegate)
print("\n=== Drain via proper pipeline ===")
drained = engine.drain_ready_results(wait=True)
print(f"drain_ready_results returned: {drained}")
engine.resolve_all()

stats = native_backend.get_stats()
print(f"total_tasks: {stats['total_tasks']}")
print(f"host_memcpy_mb: {stats['host_memcpy_mb']}")
print(f"pending_notifies: {stats['pending_notifies']}")

# Retrieve actual result tensors via future tokens
print(f"\n=== Checking future results ===")
# The native backend assigns token IDs starting from 1
# Try to retrieve results for token IDs 1..N
found = 0
cpu_count = 0
cuda_count = 0
sample_results = []
for token_id in range(1, stats['total_tasks'] + 1):
    try:
        if native_backend.future_ready(token_id):
            tensor = native_backend.future_result(token_id)
            found += 1
            if tensor.device.type == 'cpu':
                cpu_count += 1
            else:
                cuda_count += 1
            if found <= 10:
                sample_results.append((token_id, tensor))
    except Exception:
        pass

print(f"Retrieved: {found}/{stats['total_tasks']}")
print(f"  on CPU: {cpu_count}")
print(f"  on CUDA: {cuda_count}")

print(f"\nSample tensors (first 10):")
for token_id, t in sample_results:
    pinned = t.is_pinned() if t.device.type == 'cpu' else 'N/A'
    print(f"  token={token_id}: device={t.device}, shape={t.shape}, "
          f"dtype={t.dtype}, pinned={pinned}, "
          f"mean={t.float().mean().item():.6f}")

engine.close()
native_backend.close()
print("\nDone.")
