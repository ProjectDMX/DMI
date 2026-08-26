# Benchmarks

End-to-end results for DMI's capture + transport pipeline against
observation-enabled baselines.

## Host-to-ClickHouse throughput

`benchmarks.bench_clickhouse_host` isolates the host sink from model execution
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
for support tiers. Build the CPU-only host backend, install
`clickhouse-driver`, and run:

```bash
make -C native host -j
```

```bash
python -m benchmarks.bench_clickhouse_host \
  --rows 100000 \
  --payload-bytes 64KiB \
  --parallelism-sweep 1,2,4,8 \
  --trials 5 \
  --min-batch-bytes 16MiB \
  --max-batch-bytes 64MiB \
  --compression lz4 \
  --socket-timeout-seconds 30 \
  --json-output host-clickhouse.json
```

Use `--dry-run` to validate the workload without loading the native backend or
connecting to ClickHouse. The benchmark creates a uniquely named table and
drops it after collection; `--keep-table` preserves it for inspection. Set the
password with `DMI_CLICKHOUSE_PASSWORD` to keep it out of shell history.

The scaling report retains every raw trial and summarizes median throughput,
variance, speedup over the reported baseline, and gain over the preceding
worker count. `speedup_vs_one` is populated only when the sweep includes one
worker. Trial order is deterministically shuffled. Startup ends only after
every native worker has connected and initialized, so steady-state throughput
does not mix in connection setup.

Each trial reports enqueue and total drain time, sampled submit latency, DMI
process CPU, process-lifetime peak RSS, per-worker batches/rows/bytes/insert
time, and peak simultaneous native inserts. The RSS value is not an isolated
per-trial peak, so do not use later values in an in-process sweep for memory
scaling. A 50 ms sampler records server-side active inserts,
query/merge/connection gauges, normalized CPU and I/O wait, resident memory,
part pressure, and block/network counter deltas over the same steady-state
interval as throughput. Set
`--server-sample-interval-ms 0` when the benchmark account cannot read those
tables. Query-log and server metrics degrade to warnings when unavailable.

Treat saturation as evidence from several signals, not the first flat number:
throughput gain should fall below the configured plateau threshold across
repeated trials, realized insert concurrency should reach the requested worker
count, and client CPU/backpressure plus ClickHouse query/merge metrics should
identify which side is limiting progress. The process measurements include the
Python-to-C++ synthetic producer but not ClickHouse server CPU when the server
runs in another process.

Queue admission uses a finite timeout. The drain deadline is not restarted
during abort cleanup, and native connect/send/receive calls use the configured
socket timeout. An in-flight native call cannot be cancelled, so failure
cleanup can extend past the drain deadline by up to the socket timeout.

Use `--parallelism-sweep` for worker scaling, then compare one additional
setting at a time with the same row count, payload pattern, seed, and pool size.
Useful follow-up sweeps are batch byte limits, `--compression`, and
`--async-insert`. The benchmark explicitly sets `async_insert=0` for
synchronous trials instead of inheriting the server or user profile; the JSON
records the effective client settings. This matters on releases such as
[ClickHouse 26.3](https://clickhouse.com/blog/clickhouse-release-26-03), which
changed the server default.
ClickHouse recommends batching synchronous inserts, commonly at least 1,000
rows and ideally 10,000–100,000 where row size permits; tensor workloads may
reach practical byte limits earlier. See ClickHouse's
[insert strategy](https://clickhouse.com/docs/concepts/best-practices/selecting-an-insert-strategy),
[`system.query_log`](https://clickhouse.com/docs/reference/system-tables/query_log),
[`system.processes`](https://clickhouse.com/docs/reference/system-tables/processes),
[`system.metrics`](https://clickhouse.com/docs/reference/system-tables/metrics),
[`system.asynchronous_metrics`](https://clickhouse.com/docs/reference/system-tables/asynchronous_metrics),
and [`system.parts`](https://clickhouse.com/docs/reference/system-tables/parts)
documentation when interpreting results.

See the [native build layout](native-build-layout.html) for the host/full
extension boundary and loader behavior.

## Derived catalog throughput

`benchmarks.bench_capture_catalog` measures the opt-in metadata projection; it
does not exercise CUDA or write tensor payloads to ClickHouse. It creates
temporary raw tables plus logically deduplicated views, inserts deterministic
capture descriptors in bounded batches, reports every trial, and drops the
objects afterward.

```bash
python -m benchmarks.bench_capture_catalog \
  --rows 100000 --batch-rows 10000 --trials 3
```

On local ClickHouse 26.9.1, median throughput rose from 13,954 rows/s with
1,000-row batches to 88,458 rows/s at 10,000 and 157,567 rows/s at 50,000.
These loopback results validate client batching, not a production capacity
claim. Repeat the sweep on the target server while observing part creation,
merge load, CPU, and catalog lag.

## Bounded catalog search

`benchmarks.bench_capture_search` measures the read side of the same opt-in
catalog: snapshot cost, page latency, filter selectivity, and core summary
throughput. It creates temporary tables, indexes a corpus several times to
create duplicate versions, and drops the objects afterward.

```bash
PYTHONPATH=src python -m benchmarks.bench_capture_search \
  --rows 50000 --replays 2 --trials 3
```

On local ClickHouse 26.9.1 with 50,000 rows written twice:

| Measurement | Median |
|---|---:|
| `argMax` snapshot read | 22.3 ms |
| `FINAL` read (not a snapshot) | 12.1 ms |
| `max(index_version)` watermark | 1.7 ms |
| Page, `limit=100` | 21.4 ms |
| Page, `limit=1000` | 78.8 ms |
| Page, `limit=5000` | 145.2 ms |
| Page 1 vs page 25 at `limit=100` | 21.5 ms vs 24.4 ms |

The two snapshot rows are not alternatives. `FINAL` drops any capture
re-indexed above the watermark, so the 1.85x gap is what correctness costs, not
a choice between query shapes. Page cost is flat with depth -- the property
keyset pagination exists to provide -- and grows with page size, not position.

Core summaries run at roughly 140–210 M elements/s depending on dtype
(`int64` fastest, `bfloat16` slowest because of the widening shift). As with
the insert sweep, these are loopback numbers on a laptop build: repeat them on
representative hardware and duplicate ratios before treating any as a capacity
claim.

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
  <img src="assets/images/offline_hs_logits_real.png" alt="Offline throughput with limited hooks, normalized to HF ideal" width="100%" />
</p>

DMI stays close to the HF-ideal line across all configurations. Python-callback
baselines (NNsight, Torch Hooks) collapse as hook count or batch size grows;
several configurations go OOM (red ×) at large batch sizes because they retain
captured tensors in the inference memory pool. HF's built-in extraction path
is bottlenecked similarly — it materializes internals on the hot path.

## Online serving — TPOT

Setup: vLLM serve, varying request rate on ShareGPT and WildChat.

<p align="center">
  <img src="assets/images/tpot_comparison.png" alt="Online TPOT vs request rate" width="100%" />
</p>

DMI tracks the no-monitor baseline closely. The synchronous hook/debug baselines
(vLLM Hook, TRT-LLM Debug API) saturate at substantially lower request rates
because their capture paths block the hot stream.

For local setup of this repo's native backend and ClickHouse sink, see
[`install.md`](install.md). For simple API examples, see
[`huggingface.md`](huggingface.md) and [`vllm.md`](vllm.md).
