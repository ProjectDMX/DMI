# vLLM usage

DMI supports official vLLM 0.27.1 through the version-matched integration
checkout pinned at `third_party/vllm-integration/`. It does not contain a vLLM
fork.

## Install the vLLM backend

Use a dedicated environment and DMI checkout for vLLM. Do not install the
modified HuggingFace integration from `third_party/transformers/` in this
environment. Complete the [core installation](install.md), then install the
version-matched integration editable. Its dependency metadata installs the
matching official vLLM release.

```bash
pip install -e third_party/vllm-integration/
make -C native clean
make -C native -j
python -c "from dmi.transport.native import RingConfig; print(RingConfig())"
```

On a checkout shared with another DMI environment (for example a
Python 3.10 install alongside a Python 3.12 vLLM environment), drop the
`clean` step: the native build emits ABI-suffixed extensions
(`_native_backend.cpython-310-*.so`, `...cpython-312-*.so`) side by side, so
one build tree serves both interpreters. Running `clean` deletes every
environment's extensions, not just the current one's.

DMI supports both vLLM 0.27.1 GPU model runners. Use vLLM's normal
architecture-dependent default, set `VLLM_USE_V2_MODEL_RUNNER=1` to require
V2, or set it to `0` to require V1. V2 speculative decoding is not yet
supported and is rejected before CUDA initialization.

The integration fails before device initialization when it detects an
unsupported runner, version, architecture, or parallel mode. See the
versioned integration's [model support list](https://github.com/ProjectDMX/DMI-vLLM-Integration/blob/v0.27.1-r3/README.md#model-support)
for available model families and their qualification status.

The exact vLLM behavior assumed by this release is documented in the
[vLLM contract](https://github.com/ProjectDMX/DMI-vLLM-Integration/blob/v0.27.1-r3/docs/vllm_contract.md).

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

## Verifying the configurator runtime attach (live)

A configuration authored in the configurator reaches the vLLM runtime as a
saved `.dmi.yaml` artifact driven through
`dmi.configuration.attach_config(adapter, model, config)` — not through the
worker's own attach. To verify that path end to end on real hardware:

1. Build the environment: one venv with `vllm==0.27.1`, then
   `pip install --no-deps -e <DMI checkout> -e third_party/vllm-integration/`
   (DMI borrows vLLM's dependency set, exactly as the integration CI does),
   plus a native build for that interpreter (see the ABI note above).
2. Write a worker subclass of the **V1** `DMXGPUWorker`
   (`dmi_vllm_integration.adapter.DMXGPUWorker`) whose `load_model` temporarily
   sets `self.adaptor = None` around `super().load_model()` — suppressing the
   worker's self-attach — then loads the artifact and calls
   `attach_config(self.adaptor, self.model_runner.model, config)`. V2 refuses
   subclasses of the dynamic entry point entirely.
3. Launch offline `LLM(...)` with that worker class and
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process engine core, so the attach
   and its evidence stay reachable), `enforce_eager=True`, and a
   `gpu_memory_utilization` that coexists with other tenants on the device.
4. Evidence to assert: `attach_config` owns the model
   (`model._dmi_active_adapter`), the adaptor's active spec layers sit inside
   the configured range, out-of-range `HookPoint`s are disabled, and
   `generate()` completes with capture enabled (real ring traffic).

The one-owner invariant is what makes the suppression step mandatory: the
worker's self-attach marks the model, and a second attach by the configurator
runtime would be refused.

## Troubleshooting

- **Baseline vLLM** — remove the DMI worker and `additional_config`.
- **Transport-only run** — set `dmx_null_mode=False` and leave `dmx_db_host`
  empty.
- **`libstdc++` mismatch** — preload the active environment's library, for
  example `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python your_script.py`.
