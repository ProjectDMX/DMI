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
