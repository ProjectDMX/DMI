# vLLM Usage

Run DMI through the vLLM path after completing [`install.md`](install.md) and
installing the `integration/vllm/` submodule.

DMI plugs into vLLM through:

```text
integration.vllm_adapter.DMXGPUWorker
```

Pass it through `worker_cls=` in the offline `LLM(...)` API or `--worker-cls`
in `vllm serve`.

## Required: disable the vLLM compile cache

DMI's capture op is registered as a void+ordered-effect op, which the vLLM
AOT compile cache cannot serialize correctly. Set
`VLLM_DISABLE_COMPILE_CACHE=1` before importing `vllm`:

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
```

## Offline API

```python
import os
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"  # required (see above)

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
        "dmx_null_mode": True,
    },
)

params = SamplingParams(temperature=0.0, max_tokens=32)
for o in llm.generate(["The answer is"], params):
    print(o.outputs[0].text)
```

Set `"dmx_null_mode": False` and configure `dmx_db_*` fields to persist captures
to ClickHouse.

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

## Common configuration

| Field | Meaning |
|---|---|
| `dmx_hook_selection` | Hook preset, usually `vllm-full` |
| `dmx_null_mode` | `True` drops captures after transport; `False` persists |
| `dmx_ring_payload_mb` | GPU payload ring size |
| `dmx_ring_pinned_mb` | Host-side pinned payload staging buffer (D2H copy target). `0` = match `dmx_ring_payload_mb`. |
| `dmx_drain_flush_timeout_us` | Max time a completed tensor waits before GPU-to-CPU drain flush. Default `100000` (100 ms). `0` disables timeout-based flushing. |
| `dmx_db_host`, `dmx_db_port` | ClickHouse connection |

## Troubleshooting

- **Baseline vLLM** — remove `worker_cls` and `additional_config`.
- **Transport-only run** — set `"dmx_null_mode": True`.
- **`libstdc++` mismatch** — preload the conda libstdc++:
  `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python your_script.py`.
