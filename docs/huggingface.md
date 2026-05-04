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

## DMI monitoring without DB writes

This captures internal states through Ring² and drops them on the host. Use this
mode to isolate transport overhead without persistence cost.

```bash
python benchmark/scripts/hf_monitoring_generate.py \
    --model qwen3 --device cuda --batch-size 8 --max-new-tokens 16 --no-db
```

## DMI monitoring with persistence

Start ClickHouse as described in [`install.md`](install.md), then run:

```bash
python benchmark/scripts/hf_monitoring_generate.py \
    --model qwen3 --device cuda --batch-size 8 --max-new-tokens 16
```

Inspect captured rows:

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
python -m benchmark.bench_ring_transport \
    --model qwen3 --batch-size 4 \
    --prefill-len 1 --decode-len 16 \
    --warmup 1 --iters 3 \
    --modes baseline,ring_null,ring_db,hf_offload \
    --cuda-graphs \
    --csv results/ring_transport.csv
```

Useful flags:

- `--model gpt2 | qwen3`
- `--hook-selection full | hf-only | hidden-states | logits | attention`
- `--ring-payload-mb`, `--ring-pinned-mb`
- `--cuda-graphs`
- `--csv path.csv`
