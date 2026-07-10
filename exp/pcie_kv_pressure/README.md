# KV On/Offload PCIe Pressure Experiments

This directory contains a conservative experiment scaffold for stressing PCIe
while running DMI capture. The synthetic pressure, trace generation, monitors,
and summary scripts are self-contained under `exp/`; the optional replay command
can borrow the adjacent HiddenCache client.

The scaffold supports two pressure sources:

- Synthetic D2H/H2D pressure from repeated CUDA copies between GPU memory and
  pinned host memory.
- Real-prompt warm/evict/replay traces from HiddenCache RealReentry datasets.
- Synthetic KV store/retrieve request traces with long reusable prefixes.
- LMCache/vLLM co-run hooks, left as explicit environment commands so existing
  serving launchers can be plugged in without hard-coding local paths.

The default run is a dry-run smoke test. It creates a run directory, writes a
manifest, starts only requested lightweight monitors, and emits synthetic
placeholder samples. Set `DRY_RUN=0` explicitly before using a GPU.

## Layout

- `run_synthetic_pressure.sh`: main launcher for synthetic PCIe pressure.
- `scripts/synthetic_pcie_hog.py`: CUDA D2H/H2D hog, with dry-run mode.
- `scripts/synthetic_governor_contention.py`: native RingEngine benchmark for
  chunk capping and lifecycle-hint protection.
- `scripts/synthetic_feedback_activation.py`: native sustained-pressure check
  for passive feedback activation.
- `scripts/build_real_reentry_trace.py`: wrapper around HiddenCache
  `RealReentry` builders for real dataset traces.
- `scripts/generate_kv_pressure_trace.py`: writes `prefix_bank.jsonl`,
  `trace.jsonl`, and `metadata.json` for synthetic high KV on/offload runs.
- `scripts/monitor_prometheus.py`: generic Prometheus endpoint scraper for
  vLLM, LMCache, and DMI/governor metrics.
- `scripts/summarize_run.py`: summary writer for synthetic, serving, and
  generic DMI-like metrics.
- `configs/synthetic_smoke.json`: no-GPU dry-run config.
- `configs/synthetic_balanced.json`: conservative GPU copy config.
- `configs/kv_trace_smoke.json`: tiny trace generator smoke config.
- `configs/kv_trace_high_reuse.json`: synthetic high KV store/retrieve pressure
  config.
- `configs/lmcache_corun.env.example`: LMCache/vLLM co-run hook template.

## Smoke Test

```bash
cd HF_Prometheus
bash exp/pcie_kv_pressure/run_synthetic_pressure.sh
```

This writes results under:

```text
exp/pcie_kv_pressure/results/<run_id>/
```

Expected files include `run_manifest.json`, `synthetic_pcie_hog.jsonl`, logs,
and `summary.json`.

## Synthetic PCIe Pressure

Run a conservative GPU pressure job only after choosing GPU IDs and disabling
dry-run:

```bash
DRY_RUN=0 \
GPUS=0 \
DURATION_S=30 \
DIRECTION=bidirectional \
BLOCK_MB=64 \
INFLIGHT=2 \
TARGET_MB_S=2048 \
bash exp/pcie_kv_pressure/run_synthetic_pressure.sh
```

Direction choices:

- `h2d`: host-to-device pressure.
- `d2h`: device-to-host pressure.
- `bidirectional`: alternates H2D and D2H copies per iteration.

`TARGET_MB_S` is a per-GPU soft cap. Leave it at `0` for uncapped pressure.

## Scheduler Effectiveness

Use the native contention benchmark first. It creates real RingEngine backlog,
then times a serving-critical D2H copy with no control, chunk capping only, and
chunk capping plus a lifecycle hint:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n agent-dmi \
  python exp/pcie_kv_pressure/scripts/synthetic_governor_contention.py \
  --out-dir exp/pcie_kv_pressure/results/native_contention \
  --repetitions 50
```

This benchmark intentionally isolates the fast hint path; its condition is named
`hint_and_cap` and it does not claim to exercise feedback. Verify the passive
feedback path separately with sustained external D2H pressure. This check fails
unless real `on_step()` windows drive the governor into its feedback avoid state:

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n agent-dmi \
  python exp/pcie_kv_pressure/scripts/synthetic_feedback_activation.py \
  --out-dir exp/pcie_kv_pressure/results/native_feedback
```

For an end-to-end vLLM + DMI + LMCache run, use isolated high-intensity store
bursts. The long gap is intentional: the scheduler can defer DMI during the KV
burst and drain the backlog before the next serving step. The default `formal`
profile runs three repetitions with two warmup stores and 32 measured stores per
condition, then fails if a `gov_on` trial has no paired lifecycle hints.

```bash
GPU_INDEX=2 \
DMI_HOOK_SELECTION=resid_pre,ln1,attn_out,resid_mid,ln2,mlp_in,mlp_out \
DMI_RING_PAYLOAD_MB=4096 \
DMI_RING_PINNED_MB=2048 \
GOVERNOR_MAX_DEFER_US=1000000 \
GOVERNOR_HARD_WATERMARK_RATIO=0.95 \
bash exp/pcie_kv_pressure/run_scheduler_effectiveness.sh
```

For a wiring-only check, use `BENCHMARK_PROFILE=smoke`; smoke output is not a
paper-facing effectiveness result.

`GOVERNOR_MAX_DEFER_US` is a stale-hint watchdog. Normal DMI drain resumption
comes from the connector's end hint, so this value must exceed the longest
expected KV operation rather than approximate its usual duration.

Do not use an indefinitely saturated request stream as the only effectiveness
test. A scheduler can prioritize KV D2H but cannot create PCIe capacity; under
continuous overload, deferred DMI work moves into later serving steps. Keep a
saturation run as the overload boundary and report it separately.

## Real Dataset KV Trace

For paper-facing experiments, prefer the HiddenCache RealReentry trace builder.
It samples real prompts from `../hiddencache/DeepSpec/eval_datasets`, uses the
real tokenizer for prompt length, and emits warm/evict/replay phases:

```bash
python exp/pcie_kv_pressure/scripts/build_real_reentry_trace.py \
  --mode length-bucket \
  --tokenizer Qwen/Qwen3-8B \
  --out-dir exp/pcie_kv_pressure/traces/real_length_bucket
```

The default `length-bucket` mode stratifies prompt lengths, which is better for
showing when KV mobility becomes large enough to matter. `reentry` mode matches
the simpler warm target -> evict fillers -> replay target flow:

```bash
python exp/pcie_kv_pressure/scripts/build_real_reentry_trace.py \
  --mode reentry \
  --tokenizer Qwen/Qwen3-8B \
  --target-count 150 \
  --evict-count 300 \
  --min-prompt-tokens 512 \
  --out-dir exp/pcie_kv_pressure/traces/real_reentry
```

By default the wrapper resolves model ids such as `Qwen/Qwen3-8B` to the local
HuggingFace snapshot path, passes `--tokenizer-local-files-only`, and sets
offline env vars so it will not download anything. Add
`--allow-tokenizer-download` only when that is intended. The generated files
include `prefix_bank.jsonl`, `trace.jsonl`,
`trace_warm.jsonl`, `trace_evict.jsonl`, `trace_replay.jsonl`,
`selected_prompts.jsonl`, and `trace_summary.json`.

## Synthetic KV Trace

Generate a tiny trace first:

```bash
python exp/pcie_kv_pressure/scripts/generate_kv_pressure_trace.py \
  --config exp/pcie_kv_pressure/configs/kv_trace_smoke.json \
  --out-dir exp/pcie_kv_pressure/traces/smoke
```

Generate the high-pressure trace:

```bash
python exp/pcie_kv_pressure/scripts/generate_kv_pressure_trace.py \
  --config exp/pcie_kv_pressure/configs/kv_trace_high_reuse.json \
  --out-dir exp/pcie_kv_pressure/traces/high_reuse
```

The synthetic trace has two phases. The first wave sends unique long prefixes,
which should force KV store/offload. Later waves reuse the same prefixes after
`reuse_gap_s`, which should force KV retrieve/load if the serving stack uses
LMCache/offload. Token counts are approximate because this generator avoids a
tokenizer dependency; use the RealReentry wrapper above for exact token lengths
and real prompts. Both formats are compatible with the HiddenCache
`replay_trace.py` client:

```bash
python ../hiddencache/workload_generator/replay_trace.py \
  --trace exp/pcie_kv_pressure/traces/high_reuse/trace.jsonl \
  --prefix-bank exp/pcie_kv_pressure/traces/high_reuse/prefix_bank.jsonl \
  --results exp/pcie_kv_pressure/results/<run_id>/client_results.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --endpoint completions \
  --mode open-loop \
  --max-workers 256
```

## Co-Run Hooks

The launcher can start optional commands in the same run envelope:

```bash
source exp/pcie_kv_pressure/configs/lmcache_corun.env.example

DRY_RUN=0 \
GPUS=0 \
SERVE_CMD="$VLLM_LMCACHE_SERVE_CMD" \
SERVE_HEALTH_URL=http://127.0.0.1:8000/health \
VLLM_METRICS_URL=http://127.0.0.1:8000/metrics \
LMCACHE_METRICS_URLS=http://127.0.0.1:6999/metrics \
DMI_METRICS_URLS=http://127.0.0.1:9100/metrics \
EXTRA_CORUN_CMD="$CLIENT_REPLAY_CMD" \
bash exp/pcie_kv_pressure/run_synthetic_pressure.sh
```

Useful optional hooks:

- `PRE_HOOK_CMD`: run once before monitors and pressure start. Use it to prepare
  LMCache disk directories or clear stale state.
- `SERVE_CMD`: long-running vLLM/LMCache/DMI serving command. It is started in
  the background and stopped on exit.
- `EXTRA_CORUN_CMD`: another long-running command, for example an external DMI
  governor or client replay. For the generated trace, point it at a replay
  command that writes `${RUN_DIR}/client_results.jsonl`.
- `POST_HOOK_CMD`: run after pressure and monitors stop.

The script does not download models or dependencies and does not start serving
unless a command is provided.

## Metrics Targets

The summary script looks for these artifacts if present:

- `client_results.jsonl`: serving latency/TTFT P99, compatible with the
  HiddenCache workload generator client output.
- `synthetic_pcie_hog.jsonl`: generated D2H/H2D copy throughput.
- `vllm_metrics.jsonl`, `lmcache_metrics.jsonl`, `dmi_metrics.jsonl`: generic
  Prometheus scrapes.

For DMI governor evaluation, expose or log metric names containing these
concepts so the summary can pick them up:

- Drain bandwidth: metric names containing `dmi` or `dmx`, plus `drain`, plus
  `byte`, `bandwidth`, or `throughput`.
- Capture to consumable latency: metric names containing `capture`, plus
  `consumable`, `consumer`, or `latency`.
- Ring occupancy: metric names containing `ring`, plus `occupancy`, `fill`, or
  `util`.

The raw JSONL files are always preserved, so stricter post-processing can be
added later without changing the launcher.

## Borrowed Patterns

This scaffold borrows the run envelope from `../hiddencache`:

- `workload_generator/run_lmcache_offload_pressure.sh`: run directory layout,
  LMCache env/config pattern, background monitors, cleanup traps.
- `workload_generator/monitoring/monitor_lmcache_metrics.py` and
  `monitor_vllm_metrics.py`: Prometheus scrape-to-JSONL pattern.
- `workload_generator/monitoring/summarize_kv_wait_trace.py`: JSONL summary
  style and p50/p95/p99 reporting.

The core scripts here do not import `../hiddencache`; only the optional replay
command template shells into its existing OpenAI-compatible client.
