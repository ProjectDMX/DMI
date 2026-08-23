# vLLM usage

DMI supports official vLLM 0.27.1 through the separately installed
`DMI-vLLM-Integration` 0.27.1 package. The source checkout pins that package at
`third_party/vllm-integration/`; it does not contain a vLLM fork.

Install the matching releases:

```bash
pip install 'DMI>=1.1.0,<2.0'
pip install 'vllm==0.27.1'
pip install 'DMI-vLLM-Integration==0.27.1'
```

For a source checkout, install DMI and then the integration submodule:

```bash
pip install -e .
pip install -e third_party/vllm-integration/
```

DMI supports vLLM's V1 model runner only. Set this before importing or starting
vLLM:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

The integration fails before device initialization when it detects an
unsupported runner, version, architecture, or parallel mode. Its supported
model architectures are:

- GPT-2
- Llama
- Qwen2/Qwen2.5
- Qwen2-MoE
- Qwen3

The exact vLLM behavior assumed by this release is documented in the
[vLLM contract](https://github.com/ProjectDMX/DMI-vLLM-Integration/blob/v0.27.1/docs/vllm_contract.md).

## Offline API

Select the DMI worker through vLLM's Python API:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=512,
    enforce_eager=False,
    gpu_memory_utilization=0.5,
    worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
    additional_config={
        "dmx_hook_selection": "vllm-full",
        "dmx_ring_payload_mb": 4096,
        "dmx_ring_pinned_mb": 4096,
        "dmx_null_mode": False,
        "dmx_db_host": "",
    },
)

params = SamplingParams(temperature=0.0, max_tokens=32)
for output in llm.generate(["The answer is"], params):
    print(output.outputs[0].text)
```

With `dmx_null_mode=False` and an empty database host, capture and transport are
active without persistence. Set the `dmx_db_*` fields to persist captures to
ClickHouse. Setting `dmx_null_mode=True` disables DMI planning, metadata, and
payload copying.

### Persisted readback with `DMILLM`

`DMILLM` injects the DMI worker and attaches a lazy `.dmi_internal` handle to
each completed `RequestOutput` from `generate`, `chat`, or
`wait_for_completion`:

```python
from dmi_vllm_integration.llm import DMILLM
from transformers import AutoConfig
from vllm import SamplingParams

llm = DMILLM(
    model="Qwen/Qwen3-0.6B",
    additional_config={
        "dmx_hook_selection": "resid_pre",
        "dmx_db_host": "localhost",
        "dmx_db_port": 9000,
        "dmx_drain_flush_timeout_us": 100_000,
    },
    max_model_len=512,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
)

outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0.0, max_tokens=8),
)

print(outputs[0].outputs[0].text)
expected_layers = AutoConfig.from_pretrained(
    "Qwen/Qwen3-0.6B"
).num_hidden_layers
hidden_states = outputs[0].dmi_internal.require(
    "hidden_states",
    count=expected_layers,
    retry=True,
    timeout_s=30.0,
    poll_s=0.25,
).hidden_states
```

Persistence is asynchronous. Configure a nonzero
`dmx_drain_flush_timeout_us` and use the lazy handle's `require(...,
retry=True)` contract to wait for the expected layer inventory instead of
sleeping for an assumed duration.

## Online serving

Online serving needs both the model-registration plugin and the opt-in DMI
finalization endpoint. Setting `VLLM_PLUGINS` is an allowlist, so include both:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_PLUGINS=dmi_models,dmi_stop_monitoring

vllm serve Qwen/Qwen3-8B \
    --worker-cls dmi_vllm_integration.worker.DMXGPUWorker \
    --additional-config '{
        "dmx_hook_selection": "vllm-full",
        "dmx_ring_payload_mb": 4096,
        "dmx_ring_pinned_mb": 4096,
        "dmx_db_host": "localhost",
        "dmx_db_port": 9000
    }'
```

Send requests through vLLM's OpenAI-compatible API:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model":"Qwen/Qwen3-8B",
        "prompt":"The answer is",
        "max_tokens":32,
        "temperature":0
    }'
```

Before terminating the server, stop external request intake and call the DMI
endpoint:

```bash
curl --fail-with-body -X POST \
    'http://127.0.0.1:8000/v1/dmi/stop_monitoring?timeout=30'
```

Wait for `{"status":"stopped"}` before terminating vLLM. The endpoint pauses
generation in wait mode, runs `stop_monitoring` on every worker, and leaves the
engine terminally paused. Do not submit more requests afterward. If vLLM was
started with `--api-key`, include the same `Authorization: Bearer ...` header
used for other `/v1` endpoints.

## Common configuration

| Field | Meaning |
|---|---|
| `dmx_hook_selection` | Hook preset, usually `vllm-full` |
| `dmx_null_mode` | `True` disables DMI planning, metadata, and payload copying; `False` enables capture and transport |
| `dmx_ring_payload_mb` | GPU payload ring size |
| `dmx_ring_pinned_mb` | Host-side pinned payload staging size; `0` matches `dmx_ring_payload_mb` |
| `dmx_drain_flush_timeout_us` | Maximum time a completed tensor waits before a GPU-to-CPU drain flush; `0` disables the timer |
| `dmx_db_host`, `dmx_db_port` | ClickHouse connection; an empty host disables persistence |
| `dmx_db_database`, `dmx_db_table` | ClickHouse destination |

## Troubleshooting

- **Baseline vLLM** — remove the DMI worker and `additional_config`.
- **Transport-only run** — set `dmx_null_mode=False` and leave `dmx_db_host`
  empty.
- **`libstdc++` mismatch** — preload the active environment's library, for
  example `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python your_script.py`.
