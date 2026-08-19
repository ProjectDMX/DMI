# vLLM Usage

DMI currently supports vLLM 0.27.1 through the pinned
`integration/vllm/` submodule. Complete [`install.md`](install.md) before using
the vLLM integration.

DMI plugs into vLLM through:

```text
integration.vllm_adapter.DMXGPUWorker
```

The validation scope covers offline GPT-2 (`gpt2`) and Qwen3
(`Qwen/Qwen3-0.6B`) at TP1/PP1, TP2/PP1, and TP1/PP2, including the TP1/PP1
prefix-cache probe. Online Qwen3 storage validation covers the same topology
matrix through the OpenAI-compatible completions endpoint, including
prefix-cache token ranges, topology-specific ClickHouse inventory, and
bitwise comparison with `Qwen3Ref`.

The official vLLM 0.27.1 wheel does not include DMI's monitored model classes;
install the pinned submodule described above.

DMI currently supports the V1 model runner only. Set this before starting
vLLM:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

`DMXGPUWorker` checks this at startup and fails before device initialization
with a corrective error if V2 was selected; it never silently runs without
monitoring.

## Model architecture coverage

- GPT-2
- Qwen3
- Qwen2-MoE
- Llama
- Qwen2/Qwen2.5

Use `DMXGPUWorker` through `worker_cls=` in the offline `LLM(...)` API or
`--worker-cls` in `vllm serve`.

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

Persistence is asynchronous. To read while the engine remains active, enable a
bounded drain timeout and use the public retry contract instead of sleeping for
an assumed duration:

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

The online regression calls `stop_monitoring` on every monitored worker after
the final HTTP response, waits for and validates the complete ClickHouse
inventory, and only then terminates the server. The reference server has no DMI
engine and does not need this call. The test enables vLLM's developer RPC on
localhost for this terminal flush; do not expose that RPC on a public
interface.

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
