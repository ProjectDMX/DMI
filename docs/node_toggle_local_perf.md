# Node-Toggle Local Performance (Qwen3-0.6B, RTX 4090)

> Local validation of the runtime CUDA-graph node-toggle feature
> (`feature/dmi_kernel_node_toggle`). Measures per-step decode overhead of
> enabling/disabling producer (hook) nodes in an already-captured graph.
>
> **Setup:** Qwen3-0.6B (28 layers, hidden 1024, GQA kv_heads 8), vLLM fork
> `0.17.2.dev11+cu130` (editable), `cudagraph_mode=FULL`, batch=1, decode 200
> tokens, TPOT = median over 50 iterations. GPU 2 (RTX 4090). Hook selection
> `hidden-states` (one producer per layer). DMXGPUWorker.
>
> **Why not gpt2:** gpt2 (12 layers, ~0.77 ms/step) is misleading — its small
> per-node transport (~1 us/node) understates node-toggle's value and its small
> TPOT inflates the relative weight of the fixed host floor. Qwen3-0.6B gives
> tight variance (<1%) and a realistic decode shape. Numbers below are Qwen3-0.6B.

## Table 1 — Overhead overview

Baseline TPOT = 2.0043 ms (no DMI). All overheads vs baseline.

| Config | TPOT (ms) | Overhead | us/step | Meaning |
|--------|----------:|---------:|--------:|---------|
| baseline | 2.0043 | — | — | plain vLLM, no DMI |
| **toggle OFF** (0/28 enabled) | 2.0188 | **+0.72%** | +14.5 | all producer nodes disabled; only host floor remains |
| **toggle partial** (4/28) | 2.0286 | **+1.21%** | +24.3 | realistic online case |
| **toggle ON** (28/28) | 2.0632 | **+2.94%** | +58.9 | all nodes enabled (toggle machinery active) |
| **ring_null** (no toggle, full transport, no DB) | 2.0696 | **+3.26%** | +65.3 | producers fire + D2D copy + drain + p2p, null sink |
| **ring_full** (ring_db, + ClickHouse) | 2.0676 | **+3.16%** | +63.3 | full transport + async DB write |

Key conclusions:

1. **toggle OFF ≈ baseline (+0.72%).** Monitoring armed-but-idle is nearly free.
2. **toggle cost is ~linear in #enabled hooks:** 0/4/28 → +0.72/+1.21/+2.94%
   (~1.6–2.5 us/hook). Cost tracks how much you capture, not a fixed tax.
3. **toggle ON ≈ ring_null (+2.94% vs +3.26%, within noise).** Node-toggle is
   "free to have" — full-on costs the same as plain full transport, but you gain
   the ability to dial down.
4. **ring_full ≈ ring_null (+3.16% vs +3.26%).** Writing to ClickHouse adds ~0
   TPOT — export is fully async (drain→p2p→submit→DB off the critical path).
   Confirms DMI's async-export design.
5. **Node-toggle's value:** turns a fixed "+3.26% always-on transport tax" into
   "+0.72% (idle) … +2.94% (full)", paid in proportion to active hooks. The
   online steady state (mostly off) pays 0.72% instead of being stuck at ~3.3%.

Note: profiled host-floor CPU time (Table 2) overstates TPOT impact — the
end-to-end OFF overhead is 14.5 us/step while raw host CPU is 28.85 us/step,
because ~half the host work overlaps with the GPU replay of the prior step.
The host-floor optimizations therefore matter most for CPU-bound throughput
(online serving under concurrency), less for single-stream TPOT.

## Table 2 — Where the "toggle OFF" overhead goes (per-step host path)

CPU-time attribution of the `execute_model` host path, toggle OFF (0 enabled),
Qwen3-0.6B, 11000 steps. Each block runs *before* the real graph replay.

| Block | What it does | us/step | % of floor | Needed when OFF? | Removed by |
|-------|--------------|--------:|-----------:|------------------|------------|
| **5_push_meta_fifo** | loop active specs → `is_hook_enabled` per spec → push surviving metas to C++ FIFO (p2p pops these to slice payloads). OFF: 28 `is_hook_enabled` C-calls, push nothing. | 10.60 | 36.7% | ❌ | Opt B |
| **2_capacity_check** | compute step_bytes + `prepare_step()` C-call (ring/staging room; else cpu_direct fallback). Shapes are cached. | 8.74 | 30.3% | ❌ (0 bytes) | Opt A |
| **3_build_req_meta** | build per-request meta: req_ids, token_ranges, dim0_offsets; strip vLLM UUID suffix (regex) | 6.34 | 22.0% | ❌ | Opt A |
| **1_predict_padding** | map total_tokens → vLLM CUDA-graph padding bucket (`padded_q`) so DMI shapes match the padded tensor | 1.81 | 6.3% | ✅ | — |
| **4_store_step_ctx** | store the per-request meta into transport fields for pre_push to read | 1.36 | 4.7% | ❌ | Opt A |
| **HOST FLOOR total** | | **28.85** | 100% | | |
| (6_cudagraph_replay) | `super().execute_model()` — the actual vLLM step + graph replay | ~512 | — | ✅ | — |

When OFF, **94% (27/28.85 us) is wasted work** — producer nodes are disabled, so
no data is produced, yet the path still runs capacity/meta/push. Only
`1_predict_padding` is genuinely needed.

**IMPORTANT — this host CPU is NOT what drives toggle-OFF TPOT.** It overlaps with
the GPU replay of the prior step. Skipping the entire host path (the reverted
Opt A) left toggle-OFF TPOT unchanged. The actual TPOT driver is GPU-side; see
"Toggle-OFF overhead — what actually drives TPOT" below.

## Optimizations

Done (this round):
- **Gated the per-step debug `print`** behind `DMX_STEP_DEBUG` — it ran every
  decode step (flushed stdout). On gpt2 this alone cut full-on overhead ~13%→~5%.
- **Cached the capacity-check shape computation** keyed by `(padded_q, num_reqs)`
  (identical every decode step) → `2_capacity_check` 20→8.6 us (gpt2), host floor
  −31% (OFF) / −46% (full).

Done (reserve-invariant fix, commit `8d91350`):
- **Opt B (subsumed):** one `effective_specs` set, recomputed once per
  `set_active_hooks`, now drives capacity-reserve + meta-push + device-toggle. No
  per-step `is_hook_enabled` C-calls; capacity no longer over-counts disabled
  hooks (this was also a correctness fix — see "Reserve invariant" below).

Tried and reverted:
- **Opt A — empty-enabled fast path:** short-circuit `execute_model` to `super()`
  when `effective_specs` is empty. Implemented and confirmed firing (no host-path
  profile output), but **single-stream TPOT was unchanged** (toggle-OFF +0.68%
  before and after) — the host path overlaps the GPU replay, so removing it doesn't
  shorten the step. Also flagged unsafe by review (entrenches "OFF = skip the host
  protocol", which only holds on bound-FULL-graph steps; an eager/piecewise step
  fires producers untoggled → 0 metas → desync). Reverted. The host floor matters
  only for CPU-bound / high-concurrency throughput, not single-stream latency.

## Toggle-OFF overhead — what actually drives TPOT

Decomposed on Qwen3-0.6B (28 hidden-states hooks). Three components, each pinned
by a separate experiment:

| component | size | how pinned | optimizable? |
|-----------|------|-----------|--------------|
| host per-step path | ~14–28 us CPU | overlaps GPU → ~0 TPOT (Opt A removed it, no change) | n/a (already hidden) |
| producer kernel launch | ~1.07 us/node | `null_mode` (launch+no-op) = +1.49%; toggle (disabled) = +0.80% | **already removed by node-toggle** |
| disabled-node graph traversal | **~0.34 us/node** | linearity test below | only via dual-graph |

**Launch is already removed.** Leaving the 28 producers enabled-but-no-op
(`null_mode`) costs +1.49%; node-toggle disabling them costs +0.80%. So toggle-OFF
is ~half of `null_mode`-OFF — node-toggle saves ~14 us/step of kernel-launch
overhead (the real win over the old `null_mode` approach; invisible on gpt2 where
launch is ~free, clear on a 28-layer model).

**The residual is per-disabled-node graph traversal — linear, through the origin.**
A disabled node (`cudaGraphNodeSetEnabled(0)` → grid 0) stays an entry in the
executable graph; the replay engine still walks/skips it each step. Varying the
number of captured-but-disabled nodes (same model, 50 iters):

| disabled nodes N | overhead vs baseline | us/node |
|-----------------:|---------------------:|--------:|
| 0 | — | — |
| 7 | +2.6 us | 0.37 |
| 14 | +4.7 us | 0.34 |
| 21 | +7.0 us | 0.33 |
| 28 | +9.6 us | 0.34 |

Dead linear, slope ≈ **0.34 us/node**, intercept ≈ 0 → no fixed toggle/graph-launch
cost; the overhead is purely "how many nodes still live in the graph."

**Predictive use:** toggle-OFF overhead ≈ 0.34 us × (#captured nodes). E.g. vllm-full
(~40+ hooks) OFF ≈ 14 us ≈ 0.7%; a 4-layer residual monitor OFF ≈ 1.4 us ≈ 0.07%.

**Only dual-graph removes the residual.** Since the cost is ∝ nodes-in-graph (not
enabled count), the sole way to reach exactly baseline when OFF is to replay a
separate hook-free captured graph — 2× graph memory + re-capture, which defeats
node-toggle's single-graph premise. Poor trade for ~0.3–0.8%; not recommended.

## Reserve invariant (correctness, commit `8d91350`)

The ring's flow control requires `reserve(step_bytes, n_hooks)` to equal what the
producers actually write: the host pre-claims ring space in `reserve()` (advancing
`cpu_payload_head_`), the drain advances `cpu_*_tail_committed` as it consumes the
REAL task entries. node-toggle introduced "only enabled hooks fire" but capacity
still counted all `_active_specs` → `reserve > produced` → the head-tail gap drifts
monotonically (long-run probe: toggle_0 payload-gap → 344 MB, after which a uint64
underflow in `prepare_step`'s `pcap-(head-tail)` silently disabled the capacity
check). Same class of bug on the byte axis: `token_ids` was reserved at model dtype
but written/pushed at the real `input_ids` dtype. Fixed by the single
`effective_specs` source + matched dtype + saturating capacity math. Verified by the
probe: `resv_hooks == pushed` for toggle off/partial/full and token_ids-inclusive.

## Low-volume drain latency (general DMI, NOT toggle-specific)

**Symptom.** With few enabled hooks the per-step payload is small, so the ring
fills slowly. `DrainThread::should_flush()` with default thresholds (all 0) only
flushes when the ring is FULL (`pending_bytes >= payload_cap` / `pending_entries >=
task_cap`). So low-volume data sits in the GPU payload ring, uncommitted and
un-exported, until the ring fills (or shutdown). Probe: `toggle_4` payload-gap
climbs +8.2 MB/win with `task_tail` frozen — data produced but not drained.

**Not a node-toggle issue.** `should_flush()` is purely volume-driven; it doesn't
care HOW the hook set was made small — runtime node-toggle OR a static
`dmx_hook_selection` (e.g. residual on the last few layers for hallucination
monitoring) hit it identically.

**Impact (concrete).** Qwen3-0.6B, 4 hidden-states hooks ≈ 8.2 KB/step. With the
default 4 GB payload ring: fill time ≈ 4 GB / 8.2 KB ≈ 5e5 steps ≈ ~17 min at
~2 ms/step. So a sparse-layer monitor's data could be ~17 min stale before reaching
ClickHouse. Counter-intuitively, the LIGHTER the monitoring, the WORSE the latency.
Data is not lost or desynced — it flushes at ring-full or shutdown.

**Fix (commit `05bd6e8`).** Default `RingConfig.drain_flush_timeout_us = 50000`
(50 ms) in `DMXGPUWorker`, configurable via `dmx_drain_flush_timeout_us` /
`DMX_DRAIN_FLUSH_TIMEOUT_US`. `should_flush()` then flushes when the oldest pending
entry is older than the timeout, regardless of volume — bounding export latency to
~50 ms. The drain-thread timeout flush runs on the drain's OWN D2H stream (does NOT
stall the decode/compute stream, unlike `prepare_step`'s ring-full force_flush),
and idle periods have nothing pending (no flush). `0` restores legacy
flush-when-full.

Same commit also clamps `prepare_step`'s `used` to `[0, cap]` on BOTH ends: a small
constant `tail > head` skew exists because producers that fire during CUDA-graph
capture get drained (advancing the tail) with no matching `reserve()`. Once the
timeout flush keeps the tail caught up, this surfaced and `head - tail` underflowed
uint64 → `avail = 0` → a spurious ring-full flush (main-stream sync) every step
(+16% TPOT). `tail >= head` now reads as `used = 0` (ring drained).

### Potential issue considered: does 50 ms flushing hurt throughput?

"Flush only when full" is partly a batching optimization — larger D2H transfers
amortize per-transfer overhead and approach peak PCIe bandwidth. A 50 ms timeout
makes smaller, more frequent transfers, which *could* reduce export efficiency.

**Measured — it does not.** High-volume `all_on` (28 hooks), 2 GB ring, 4000-step
decode, `timeout=0` vs `50ms`:

| setting | gap behavior | mean TPOT | total |
|---------|--------------|----------:|------:|
| `timeout=0` (flush-when-full) | grows to ~115 MB, big-batch/at-end | 2.332 ms | 9.33 s |
| `timeout=50ms` | bounded ~0.5 MB, continuous | 2.328 ms | 9.31 s |

Identical (within noise), `resv==pushed=56000` both. Why the batching loss cancels:
- **Self-balancing:** `should_flush()` fires on ring-full OR timeout, whichever
  first. When export is bandwidth-bound the ring fills within 50 ms → ring-full wins
  → big batches preserved *exactly when they matter*. The timeout only shrinks
  batches when the ring isn't filling (bandwidth headroom — small batches are free).
- Drain D2H is async on its own stream → TPOT unaffected either way.
- PCIe batching saturates by ~1–16 MB, not GB; the old multi-GB-full batches were
  past diminishing returns.
- Per-flush fixed cost at ≤20 flushes/s is trivial CPU.

**Not locally measured:** export-bandwidth saturation with real ClickHouse ingestion
+ high concurrency (above was a null sink, batch=1, 4090). But that is precisely the
regime where the ring fills fast → ring-full wins the race → large batches preserved
by construction, so the design argument covers it.

## Caveats
- RTX 4090, batch=1, single stream. Paper-grade TPOT numbers require H100 +
  Qwen3-4B/14B (Zaratan). This validates direction and mechanism, not absolutes.
- TPOT here (~2.0 ms) is dominated by vLLM per-step framework overhead, not just
  the 512 us model replay; the host floor is a small fraction either way.
