# Node-Toggle — Phase 0 Results

> Phase 0 = productionize the dual-ring node-toggle probe into a `tests/ring`
> regression test and measure performance. Status: **done, all assertions pass.**
> Design constraints: `node_toggle_design_notes.md`. Plan: `ring2_toggle_implementation_plan.html`.

## What was built

- `tests/ring/test_node_toggle.cu` — links the real `monitoring/csrc/ring/producer.cu`
  + `AllocatedRing` (payload ring + task/meta ring), captures N producer launches
  into a CUDA graph, and uses `cudaGraphNodeSetEnabled` to disable/enable producer
  nodes between replays.
- `tests/ring/Makefile` — `test_node_toggle` target, parameterizable:
  `make -C tests/ring test_node_toggle NPROD=145 SRC_KB=512`.
- Bakes in design-note §3: builds an **explicit node→producer map** (enable one
  node at a time, observe the published writer-id) — never assumes
  `cudaGraphGetNodes()` returns nodes in capture order.

## Environment

- GPU: NVIDIA GeForce RTX 4090 (sm_89); CUDA runtime 13.0 / driver 13.0.
- Isolated microbenchmark: synthetic tensors, self-captured graph, **no model
  compute**, single stream + synced. See Scope below.

## Performance results

| metric | **N=16** | **N=145** (≈ backpressure hook count) |
|---|---|---|
| (a) all enabled | 180.64 µs | 2942.77 µs |
| (b) all true-disabled (node off) | **3.62 µs** | **28.04 µs** |
| (c) null_mode soft (kernel still launched) | 13.38 µs | 119.34 µs |
| (d) half node-disabled | 89.80 µs | 1500.17 µs |
| true-disable saves vs full | 177.02 µs (98.0%) | 2914.73 µs (99.0%) |
| **true-disable saves vs null_mode** | **9.75 µs** | **91.30 µs** |
| reconfigure: single node | 0.187 µs/call | 0.190 µs/call |
| reconfigure: full-set flip | 3.04 µs (0.190 µs/node) | 29.97 µs (0.207 µs/node) |

(per-producer tensor = 1024 KiB; payload ring 512 MiB; task ring 4096; IT=2000 replays.)

## Key takeaways

1. **True node-disable removes essentially all producer cost** (98–99%). This is
   the *isolated producer cost* (the work toggle removes), **not** serving overhead
   — there is no model compute in the graph. Serving-% needs Phase 1.
2. **The null_mode residual scales with hook count.** `null_mode` still *launches*
   every producer (then early-returns), so it leaves launch overhead that
   true-disable removes: **9.75 µs at 16 nodes, 91.30 µs at 145 nodes** (~0.6 µs/node).
   Measured on identical graph/data, so this delta is clean (uncontaminated by the
   "no model compute" caveat). This is the concrete, scale-dependent value of
   true-disable over null_mode.
3. **Reconfigure is cheap and linear, no re-instantiation.** ~0.19 µs/node;
   full-set flip ~3 µs at 16 nodes, ~30 µs at 145 nodes (host wall-clock).

## Correctness verified (cheap, included)

- Disabling a producer subset post-capture leaves the dual-ring aligned: remaining
  producers publish a contiguous, gap-free run of task entries with
  `ready_seq == slot index`, `payload_head == #published × ALLOC`. Re-enable restores.
- Assertions: 168/168 (N=16), 1458/1458 (N=145).

## Scope (what Phase 0 does NOT cover)

- **Stream-ordering / exec-mutation-lifecycle invariants** (design-notes §1, #1–#5):
  the test is single-stream + synced, so host reconfigure never overlaps in-flight
  GPU work — the race class that bit `set_null_mode` (d0a99a5) cannot appear here.
  Deferred to the **Phase 1** HF end-to-end prototype (concurrent non-blocking stream).
- **Serving-relative overhead (% of TPOT)**: needs producers riding on real model
  compute — Phase 1 / Zaratan H100.
- **Host meta-FIFO lockstep**: not exercised (no Python meta path here).

## Reproduce

```bash
make -C tests/ring test_node_toggle              # N=16
make -C tests/ring test_node_toggle NPROD=145    # heavy config
# vary tensor size: SRC_KB=512
```
