# SGLang usage

DMI supports official SGLang 0.5.19 through the version-matched integration
checkout pinned at `third_party/sglang-integration/`. It does not contain an
SGLang fork: the integration is an out-of-tree package that registers through
SGLang's general-plugin entry point (`sglang.srt.plugins`) and its external
model package mechanism.

## Install the SGLang backend

Use a dedicated environment and DMI checkout for SGLang. Do not install the
modified HuggingFace integration from `third_party/transformers/` or the vLLM
integration in this environment. Complete the [core installation](install.md)
first.

SGLang 0.5.19 is not published on PyPI (wheels stop at 0.5.10, which predates
the plugin framework), so install it from the pinned source tag, then install
the integration editable and rebuild the native backend against that
environment's PyTorch (2.13.0 / CUDA 13.0 for this release):

```bash
git clone --branch v0.5.19 --depth 1 https://github.com/sgl-project/sglang.git
pip install -e sglang/python
pip install -e third_party/sglang-integration/
make -C native clean
make -C native -j
python -c "from dmi.transport.native import RingConfig; print(RingConfig())"
```

The integration checkout's
[`docs/env-setup.md`](https://github.com/ProjectDMX/DMI-SGLang-Integration/blob/main/docs/env-setup.md)
records the exact dependency resolution used for qualification (uv, pinned
torch constraints, no Rust extensions).

The integration fails before device initialization when it detects an
unsupported SGLang version, architecture, or execution mode (TP/PP/DP,
speculative decoding, LoRA, torch.compile, PD disaggregation, and others). See
the integration's
[port audit](https://github.com/ProjectDMX/DMI-SGLang-Integration/blob/main/docs/v0519-port.md)
for the qualified compatibility cells.

## Offline API

`DMIEngine` is a drop-in replacement for `sglang.Engine`. It forwards every
`ServerArgs` keyword unchanged, carries DMI settings to the spawned worker
processes, and tags each output with a lazy `dmi_internal` readback handle:

```python
from dmi_sglang_integration import DMIEngine

engine = DMIEngine(
    model_path="Qwen/Qwen3-0.6B",
    mem_fraction_static=0.5,
    disable_prefill_cuda_graph=True,   # DMI runs prefill eagerly anyway
    dmx_model_id="my-run",
    dmx_db_host="localhost",
    dmx_hook_selection="resid_pre,final_ln,token_ids,final_logits",
    dmx_ring_payload_mb=1024,
    dmx_ring_pinned_mb=1024,
)
try:
    outputs = engine.generate(
        prompt=["The capital of France is"],
        sampling_params={"temperature": 0, "max_new_tokens": 8},
        rid=["request-0"],
    )
    print(outputs[0]["text"])
    engine.dmi_stop_monitoring()          # authoritative flush to ClickHouse
    hidden = outputs[0]["dmi_internal"].hidden_states
finally:
    engine.shutdown()
```

Call `dmi_stop_monitoring()` before `shutdown()`: SGLang terminates its worker
processes without a graceful path, so the explicit RPC is the only durable
flush point. Monitoring is terminal for the engine; create a new `DMIEngine`
for another capture.

Plain `sglang.Engine` also works when the integration package is installed:
set the `DMX_*` environment variables before constructing it. Set
`SGLANG_PLUGINS=none` (or `DMX_SGLANG_ENABLE=0`) to run an unmonitored
baseline in the same environment.

## Common configuration

| Setting | Environment variable | Default |
| --- | --- | --- |
| Capture identifier | `DMX_MODEL_ID` | model path |
| Hook selection | `DMX_HOOK_SELECTION` | `sglang-full` (every hook except attention matrices) |
| ClickHouse host / port / database / table | `DMX_DB_HOST`, `DMX_DB_PORT`, `DMX_DB_DATABASE`, `DMX_DB_TABLE` | unset (no persistence), `9000`, `default`, `offload` |
| GPU payload ring / pinned staging | `DMX_RING_PAYLOAD_MB`, `DMX_RING_PINNED_MB` | `1024`, `1024` |
| Device-side padding strip | `DMX_GPU_PADDING_STRIP` | `1` |
| System `libstdc++` preload for conda interpreters | `DMX_SGLANG_PRELOAD_LIBSTDCXX` | `auto` |

## Troubleshooting

- **`GLIBCXX_3.4.30 not found` in the scheduler process** -- a conda-provided
  interpreter binds conda's older `libstdc++` before DMI's native backend
  loads. `DMIEngine` preloads the system library in worker processes
  automatically; for plain `sglang.Engine` or `sglang serve`, export
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.
- **`DMI monitoring is not active in this scheduler`** -- the plugin did not
  register. Check `pip show DMI-SGLang-Integration`, `SGLANG_PLUGINS`, and
  `DMX_SGLANG_ENABLE`.
- **Overlap scheduler rows** -- with the default overlap scheduler SGLang runs
  one extra decode forward per request after its final decision; DMI captures
  that forward too (one extra `token_ids` row holding the last generated token
  and one unused `final_logits` decision). Pass `disable_overlap_schedule=True`
  for a capture that matches the public output exactly.
