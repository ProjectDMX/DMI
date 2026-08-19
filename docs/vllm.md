# vLLM Usage

Run DMI through the vLLM path after completing [`install.md`](install.md) and
installing the `integration/vllm/` submodule.

DMI plugs into vLLM through:

```text
integration.vllm_adapter.DMXGPUWorker
```

This branch pins vLLM 0.27.1. The exact upstream base is
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`; DMI's vLLM patch branch is
`dmi-v0.27.1`, currently at
`5c00e7369bfcf8cbbc2a770024ebe520d4c9cddc`. The root repository's submodule
gitlink is the authoritative DMI/vLLM commit pair. Older ports remain on their
versioned branches rather than being overwritten by this branch.

The validation scope covers offline GPT-2 (`gpt2`) and Qwen3
(`Qwen/Qwen3-0.6B`) at TP1/PP1, TP2/PP1, and TP1/PP2, including the TP1/PP1
prefix-cache probe. Online Qwen3 storage validation covers the same topology
matrix through the OpenAI-compatible completions endpoint, including
prefix-cache token ranges, topology-specific ClickHouse inventory, and
bitwise comparison with `Qwen3Ref`.

The bundled monitored model classes are registered by the patched submodule;
the official vLLM 0.27.1 wheel alone does not contain them.

vLLM 0.27.1 defaults eligible dense models to its V2 model runner. DMI's
0.27.1 port currently supports the V1 runner only because its request-layout
and dispatch boundaries are different. Set this before importing `vllm`:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

`DMXGPUWorker` checks this at startup and fails before device initialization
with a corrective error if V2 was selected; it never silently runs without
monitoring.

## Model architecture coverage

The patched fork contains monitored variants for GPT-2, Qwen2/Qwen2.5, Qwen3,
Qwen2-MoE, and Llama. The adapter automatically remaps GPT-2, Qwen3,
Qwen2-MoE, and Llama. Dense Qwen2 remains in the fork but is not automatically
remapped by the minimum adapter.

Model support is architecture-based, so different checkpoint sizes in the
same family do not require another DMI model class. Quantized checkpoints and
nonstandard remote-code implementations still require separate validation.

Pass it through `worker_cls=` in the offline `LLM(...)` API or `--worker-cls`
in `vllm serve`.

## Required runtime environment

DMI's capture op is registered as a void+ordered-effect op, which the vLLM
AOT compile cache cannot serialize correctly. Set
`VLLM_DISABLE_COMPILE_CACHE=1` before importing `vllm`:

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V2_MODEL_RUNNER=0
```

## Offline API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=512,
    enforce_eager=False,
    gpu_memory_utilization=0.5,
    worker_cls="integration.vllm_adapter.DMXGPUWorker",
    additional_config={
        "dmx_hook_selection": "vllm-full",
        "dmx_ring_payload_mb": 4096,
        "dmx_ring_pinned_mb": 4096,
        "dmx_null_mode": False,
        "dmx_db_host": "",
    },
)

params = SamplingParams(temperature=0.0, max_tokens=32)
for o in llm.generate(["The answer is"], params):
    print(o.outputs[0].text)
```

With `"dmx_null_mode": False` and no database host, capture and transport remain
active without persistence. Configure `dmx_db_*` fields to persist captures to
ClickHouse. Setting `"dmx_null_mode": True` disables DMI planning, metadata,
and payload copying.

## Reading internals back: `DMILLM`

`DMILLM` is a drop-in subclass of `LLM`: it injects the DMI worker for you, and
every `RequestOutput` from `generate` carries a lazy `.dmi_internal` that reads
the captured internals back from the store -- the same object the HuggingFace
path's `out.dmi_internal` gives you.

```python
from vllm import SamplingParams
from integration.vllm_adapter import DMILLM

llm = DMILLM(
    "Qwen/Qwen3-0.6B",
    additional_config={
        "dmx_model_id": "demo_vllm",
        "dmx_hook_selection": "resid_pre",
        "dmx_db_host": "localhost",
        "dmx_db_port": 9000,
    },
    max_model_len=512, enforce_eager=True, gpu_memory_utilization=0.5,
)

out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8))

out[0].outputs[0].text             # native vLLM output, unchanged
out[0].dmi_internal.hidden_states  # tuple indexed by layer, each [1, seq, hidden]
out[0].dmi_internal.available      # ['hidden_states']
```

`DMILLM` only injects `worker_cls`; pass DMI settings through `additional_config`
exactly as with plain `LLM`. A nonempty `dmx_db_host` and
`dmx_null_mode=False` are required to read internals back. Each `RequestOutput`
exposes only its own request's internals; for the whole batch as one
`[batch, seq, hidden]` tensor use `get_internal(model_id)` from
`monitoring.internal_mapper`.

Persistence is asynchronous. For a complete read before an explicit
`stop_monitoring`, enable a bounded drain timeout and use the public retry
contract instead of sleeping for an assumed duration:

```python
from transformers import AutoConfig

# additional_config={..., "dmx_drain_flush_timeout_us": 100_000}
expected_layers = AutoConfig.from_pretrained(
    "Qwen/Qwen3-0.6B"
).num_hidden_layers
out[0].dmi_internal.require(
    "hidden_states",
    count=expected_layers,
    retry=True,
    timeout_s=30.0,
    poll_s=0.25,
)
hidden_states = out[0].dmi_internal.hidden_states
```

Without a drain timeout, call
`llm.collective_rpc("stop_monitoring")` before the authoritative read.

## vLLM serve

```bash
vllm serve Qwen/Qwen3-8B \
    --worker-cls integration.vllm_adapter.DMXGPUWorker \
    --additional-config '{
        "dmx_hook_selection": "vllm-full",
        "dmx_ring_payload_mb": 4096,
        "dmx_ring_pinned_mb": 4096,
        "dmx_db_host": "localhost",
        "dmx_db_port": 9000
    }'
```

Run the online Qwen3 storage regression with a local ClickHouse instance and
the checkpoint available in the Hugging Face cache:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tests.vllm_online_storage --tp 1 --pp 1
CUDA_VISIBLE_DEVICES=0,1 python -m tests.vllm_online_storage --tp 2 --pp 1
CUDA_VISIBLE_DEVICES=0,1 python -m tests.vllm_online_storage --tp 1 --pp 2
```

Each cell starts reference and monitored `vllm serve` processes, sends cold and
prefix-cache-hit HTTP completion requests, validates the exact topology-specific
ClickHouse row, shard, and token-range inventory, and compares every stored
tensor with the same-topology `Qwen3Ref` disk dump. It uses an isolated
temporary table and removes that table after the run. These online cells use
eager sequential requests; the offline GPU oracle covers CUDA-graph continuous
batching and scheduler-versus-packed-order divergence.

To guarantee that the final captured rows reach storage, call
`stop_monitoring` on every worker before terminating the server. The regression
test enables vLLM's developer RPC on localhost and invokes this method before
shutdown. Do not expose the developer RPC on a public interface.

## Common configuration

| Field | Meaning |
|---|---|
| `dmx_hook_selection` | Hook preset, usually `vllm-full` |
| `dmx_null_mode` | `True` disables DMI planning, metadata, and payload copying; `False` enables capture/transport |
| `dmx_ring_payload_mb` | GPU payload ring size |
| `dmx_ring_pinned_mb` | Host-side pinned payload staging buffer (D2H copy target). `0` = match `dmx_ring_payload_mb`. |
| `dmx_drain_flush_timeout_us` | Max time a completed tensor waits before GPU-to-CPU drain flush. Default `0` (disabled). |
| `dmx_db_host`, `dmx_db_port` | ClickHouse connection |

## Troubleshooting

- **Baseline vLLM** — remove `worker_cls` and `additional_config`.
- **Transport-only run** — set `"dmx_null_mode": False` and leave `dmx_db_host` empty.
- **`libstdc++` mismatch** — preload the conda libstdc++:
  `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python your_script.py`.
