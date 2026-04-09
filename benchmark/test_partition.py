"""Quick diagnostic: check if ring::producer causes CUDA graph partitions."""
import os
import torch
import torch._inductor.config as inductor_config

# Enable output code logging
os.environ["TORCH_LOGS"] = "+output_code"

# Now run a simple model compile
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

# Set up ring transport
from monitoring.ring_transport import (
    RingTransport, ModelShapeConfig, HookSpec, activate, install_ring_hooks,
    _hook_type_from_name, _hook_id_from_name,
)
from monitoring import _native_engine as _ne
_ne._load_extension()

ring_config = _ne.RingConfig()
ring_config.payload_ring_bytes = 64 * 1024 * 1024
ring_engine = _ne.RingEngine(ring_config)

transport = RingTransport(ring_engine)
transport.null_offload = True
ring_engine.set_null_mode(True)

cfg = ModelShapeConfig(
    hidden_dim=model.config.hidden_size,
    num_heads=model.config.num_attention_heads,
    num_kv_heads=getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
    head_dim=model.config.hidden_size // model.config.num_attention_heads,
    dtype=torch.float16,
    vocab_size=model.config.vocab_size,
)
transport.set_model_cfg(cfg)

# Install hooks
from monitoring.hook_points import HookPoint
specs = []
for name, mod in model.named_modules():
    if isinstance(mod, HookPoint) and mod._name:
        ht = _hook_type_from_name(mod._name)
        ln = _hook_id_from_name(mod._name)
        specs.append(HookSpec(hook_type=ht, module=mod, layer_no=ln))

install_ring_hooks(specs)
transport._active_specs = specs
transport._using_forward_hooks = True
activate(transport)

# Compile
print(f"\nCompiling with mode='reduce-overhead'...")
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

# Warmup
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    transport.set_step_context("test", ["req1"], [(0, input_ids.shape[1])])
    out = model(input_ids)
    # Decode step
    next_token = out.logits[:, -1:].argmax(dim=-1)
    transport.set_step_context("test", ["req1"], [(0, 1)])
    out2 = model(next_token, past_key_values=out.past_key_values)

print("\nDone. Check partition output above.")

# Cleanup
for h in handles:
    h.remove()
from monitoring.ring_transport import deactivate
deactivate()
ring_engine.stop()
