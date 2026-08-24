# Benchmarks

End-to-end results for DMI's capture + transport pipeline against
observation-enabled baselines.

## Host-to-ClickHouse throughput

`benchmark.bench_clickhouse_host` isolates the host sink from model execution
and CUDA. It prepares a deterministic pool of CPU tensors, submits a fixed row
count through `DMXHostEngine`, drains the native queue, and verifies the stored
row and payload-byte counts.

For local testing, download the official precompiled ClickHouse binary for the
host platform and keep it outside this repository:

- [Linux x86_64](https://builds.clickhouse.com/master/amd64/clickhouse)
- [Linux ARM64](https://builds.clickhouse.com/master/aarch64/clickhouse)
- [macOS Intel](https://builds.clickhouse.com/master/macos/clickhouse)
- [macOS Apple Silicon](https://builds.clickhouse.com/master/macos-aarch64/clickhouse)
- [FreeBSD x86_64](https://builds.clickhouse.com/master/freebsd/clickhouse)
- [Docker image](https://hub.docker.com/r/clickhouse/clickhouse-server/)

See ClickHouse's [supported-platform matrix](https://clickhouse.com/support/platforms)
for support tiers. Build the native backend, install `clickhouse-driver`, and
run:

```bash
python -m benchmark.bench_clickhouse_host \
  --rows 10000 \
  --payload-bytes 64KiB \
  --min-batch-bytes 16MiB \
  --max-batch-bytes 64MiB \
  --compression lz4 \
  --json-output host-clickhouse.json
```

Use `--dry-run` to validate the workload without loading the native backend or
connecting to ClickHouse. The benchmark creates a uniquely named table and
drops it after collection; `--keep-table` preserves it for inspection. Set the
password with `DMI_CLICKHOUSE_PASSWORD` to keep it out of shell history.

The JSON separates enqueue time from total drain time. It also reports logical
payload throughput, sampled submit latency, DMI process CPU and peak RSS, active
MergeTree parts, compression size, primary-key size, and insert batching from
`system.query_log`. Query-log metrics degrade to a warning when the account
lacks access. The process measurements include the Python-to-C++ call used by
this synthetic driver; they do not include a ClickHouse server running in a
different process.

Compare one setting at a time with the same row count, payload pattern, seed,
and pool size. Useful sweeps are `--parallelism`, batch byte limits,
`--compression`, and `--async-insert`. Synchronous batching remains the default.
ClickHouse recommends batching synchronous inserts, commonly at least 1,000
rows and ideally 10,000–100,000 where row size permits; tensor workloads may
reach practical byte limits earlier. See ClickHouse's
[insert strategy](https://clickhouse.com/docs/concepts/best-practices/selecting-an-insert-strategy),
[`system.query_log`](https://clickhouse.com/docs/reference/system-tables/query_log),
and [`system.parts`](https://clickhouse.com/docs/reference/system-tables/parts)
documentation when interpreting results.

Baselines:

- **HuggingFace Ideal** — vanilla HF `generate`, no observation (used as 1.0)
- **HF Built-in Extraction** — HF's `output_hidden_states=True` / `output_attentions=True`
- **HF Stepwise Extraction** — HF returning internals one step at a time
- **Torch Hooks** — Python `register_forward_hook` instrumentation
- **NNsight** — [NNsight](https://nnsight.net/) tracing layer
- **vLLM Hook**, **TRT-LLM (Debug API)** — synchronous observation baselines
- **vLLM w/o Monitor** — vanilla vLLM, no observation

## Offline throughput

Setup: 1 hidden-state hook per layer + final-LN + logits (38 / 34 / 42 hooks
total on Qwen3-4B / Llama-3.1-8B / Qwen3-14B). Normalized to HuggingFace Ideal.

<p align="center">
  <img src="../Figures/offline_hs_logits_real.png" alt="Offline throughput with limited hooks, normalized to HF ideal" width="100%" />
</p>

DMI stays close to the HF-ideal line across all configurations. Python-callback
baselines (NNsight, Torch Hooks) collapse as hook count or batch size grows;
several configurations go OOM (red ×) at large batch sizes because they retain
captured tensors in the inference memory pool. HF's built-in extraction path
is bottlenecked similarly — it materializes internals on the hot path.

## Online serving — TPOT

Setup: vLLM serve, varying request rate on ShareGPT and WildChat.

<p align="center">
  <img src="../Figures/tpot_comparison.png" alt="Online TPOT vs request rate" width="100%" />
</p>

DMI tracks the no-monitor baseline closely. The synchronous hook/debug baselines
(vLLM Hook, TRT-LLM Debug API) saturate at substantially lower request rates
because their capture paths block the hot stream.

For local setup of this repo's native backend and ClickHouse sink, see
[`install.md`](install.md). For simple API examples, see
[`huggingface.md`](huggingface.md) and [`vllm.md`](vllm.md).
