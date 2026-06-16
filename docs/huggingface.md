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
    retry=True,
    timeout_s=30.0,
)

out = generate_with_monitoring_dict(
    model,
    **inputs,
    max_new_tokens=8,
    do_sample=False,
    internal_requirements=requirements,
)

hidden_states = out.dmi_internal.hidden_states
token_mask = out.dmi_internal.token_mask
```

Supported mapped fields are:

| Field | DMI act_name |
|---|---|
| `hidden_states` | `blocks.hook_resid_pre` |
| `attentions` | `blocks.attn.hook_pattern` |
| `logits` | `final_logits` |
| `token_mask` | Derived from this generate call's token ranges |

These fields are reassembled into one value per field. For example,
`hidden_states` is a tuple ordered by layer, with each tensor shaped
`[batch, seq, hidden]`; `attentions` is a tuple ordered by layer, with each
tensor shaped `[batch, heads, seq, seq]`; `logits` is shaped
`[batch, seq, vocab]`; `token_mask` is shaped `[batch, seq]` with bool dtype.
For `generate_with_monitoring_dict(...)`, tensor reads are scoped to the request
IDs from that generate call rather than loading every row for the model_id.

This is intentionally not the raw nested layout returned by
`generate(..., output_hidden_states=True)`. Hugging Face returns generation
internals in step-first form, roughly `step -> layer -> tensor`. DMI internals
are returned in analysis-friendly layer-first form, `layer -> full-sequence
tensor`, after prefill and decode chunks have been stitched together on CPU.
The original Hugging Face generation output is still available directly on
`out`, for example `out.sequences`.

When DMI reassembles ragged per-request sequences into a batch, shorter
requests are left-padded with synthetic zeros. These zeros are not Hugging Face
pad-token activations. Use `token_mask` to ignore them:

```python
hidden_states = out.dmi_internal.hidden_states
token_mask = out.dmi_internal.token_mask
norm = hidden_states[0].float().norm(dim=-1)[token_mask].mean()
```

`dmi_internal` does not currently expose `scores` or `past_key_values`.
Hugging Face `scores` are generation scores after logits processors/warpers,
so they are not always the same as raw model logits; DMI exposes raw captured
`final_logits` as `out.dmi_internal.logits` instead. `past_key_values` is not
mapped because HF cache objects are cache-implementation-specific and DMI does
not reconstruct that object today. If Hugging Face itself returned these fields,
they remain available on the original generation output, for example
`out.scores` or `out.past_key_values`.

Plain field access such as `out.dmi_internal.hidden_states` is lazy and
field-cached. It reads the backing store on first successful access and then
reuses the cached value. If ClickHouse is still receiving rows, plain access can
cache a partial result. Use a requirement when you know the expected number of
entries.

By default, requirements are strict one-shot checks:

```python
out.dmi_internal.require("hidden_states", count=model.config.num_hidden_layers)
hidden_states = out.dmi_internal.hidden_states
```

If the field is missing or incomplete, access raises immediately. To wait for
asynchronous ClickHouse writes, opt into retry explicitly:

```python
out.dmi_internal.require(
    "hidden_states",
    count=model.config.num_hidden_layers,
    retry=True,
    timeout_s=30.0,
    poll_s=0.25,
)
hidden_states = out.dmi_internal.hidden_states
```

`retry=True` retries missing or incomplete fields until complete. `timeout_s`
defaults to 30 seconds; pass `timeout_s=None` only when you intentionally want
to wait forever. `poll_s` controls the interval between reads. Incomplete reads
are not cached; successful complete reads are cached.

Per-output requirements are also supported:

```python
out.dmi_internal.require(
    "hidden_states",
    count=model.config.num_hidden_layers,
    retry=True,
)
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
