# HuggingFace Usage

Run DMI through the HuggingFace path after completing
[`install.md`](install.md). The HF path uses the modified Transformers submodule
and DMI's generation wrapper.

## Sanity check

Run vanilla HF generation first:

```bash
python benchmark/scripts/hf_generate.py \
    --model gpt2 --device cuda --batch-size 8 --max-new-tokens 16
```

## DMI monitoring (transport only or with persistence)

Both flows go through `benchmark.bench_hf_transport`. Pick the mode that
matches whether you want to persist captures:

- `ring_null` — Ring² capture + transport, drop on the host. Isolates transport
  overhead without ClickHouse setup.
- `ring_db` — Ring² capture + transport + ClickHouse insert. Start ClickHouse
  per [`install.md`](install.md) first.

Inspect captured rows after a `ring_db` run:

```bash
clickhouse-client --query "SELECT count() FROM default.offload"
```

## Reading internals from HF generation output

`generate_with_monitoring(...)` preserves Hugging Face's normal return behavior.
Use `generate_with_monitoring_dict(...)` when you want a dict-style generation
output with DMI internals attached:

```python
from integration.hf_adapter import generate_with_monitoring_dict
from monitoring.internal_mapper import InternalRequirements

requirements = InternalRequirements().require(
    "hidden_states",
    count=model.config.num_hidden_layers,
)

out = generate_with_monitoring_dict(
    model,
    **inputs,
    max_new_tokens=8,
    do_sample=False,
    internal_requirements=requirements,
)

hidden_states = out.dmi_internal.hidden_states
```

Plain field access such as `out.dmi_internal.hidden_states` is lazy and
field-cached. It reads the backing store on first successful access and then
reuses the cached value. If ClickHouse is still receiving rows, plain access can
cache a partial result. Use a requirement when you know the expected number of
entries and want incomplete reads to fail explicitly.

Per-output requirements are also supported:

```python
out.dmi_internal.require("hidden_states", count=model.config.num_hidden_layers)
hidden_states = out.dmi_internal.hidden_states
```

If more rows arrive after a field has been cached, clear that field before
reading again:

```python
out.dmi_internal.clear_cache("hidden_states")
hidden_states = out.dmi_internal.hidden_states
```

## Ring-transport benchmark

The benchmark compares:

| Mode | What runs |
|---|---|
| `baseline` | Vanilla HF generate, no observation |
| `ring_null` | DMI capture + Ring² transport, drain to `/dev/null` |
| `ring_db` | DMI capture + Ring² transport + ClickHouse write |
| `hf_offload` | HF's `output_hidden_states=True` path |

```bash
python -m benchmark.bench_hf_transport \
    --model qwen3 --batch-size 4 \
    --prefill-len 1 --decode-len 16 \
    --warmup 1 --iters 3 \
    --modes baseline,ring_null,ring_db,hf_offload \
    --cuda-graphs \
    --csv results/ring_transport.csv
```

Useful flags:

- `--model gpt2 | qwen3 | llama`
- `--hook-selection full | hf-only | hidden-states | logits | <individual hook short name>` (individual hooks like `q`, `k`, `v`, `attn_scores`, `pattern`, `attn_out`, `mlp_post` can be passed comma-separated)
- `--ring-payload-mb`, `--ring-pinned-mb`
- `--cuda-graphs`
- `--csv path.csv`
