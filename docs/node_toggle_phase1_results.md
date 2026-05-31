# Node-Toggle — Phase 1 Results

> Phase 1 = end-to-end lockstep prototype: prove that node-toggle + lockstep
> meta-push keeps the **real consumer pipeline** aligned across reconfigures,
> under a real (non-blocking) compute stream. Status: **done (mechanism proof).**
> Invariants: `node_toggle_design_notes.md` §1. Phase 0: `node_toggle_phase0_results.md`.

## What was built

`tests/ring/test_node_toggle_e2e.cu` (Makefile target `test_node_toggle_e2e`) drives
the **full** consumer, not just the device ring:

```
captured producer graph  →  RingEngine (drain thread → p2p thread)
                         →  TensorMetaFifo (host meta lane)  →  SubmitFn
```

- Links `ring_engine.cu + drain_thread.cpp + p2p_thread.cpp + producer.cu` + torch
  (ATen) — no ClickHouse. Replays run on a `cudaStreamNonBlocking` stream.
- Reconfigure follows the design-note protocol: `cudaStreamSynchronize` (prior
  replay done, #1/#2) → `cudaGraphNodeSetEnabled` for the new set → push matching
  metas → `cudaGraphLaunch` (#4).
- Node→producer map read **statically** from each kernel node's `hook_type` arg via
  `cudaGraphKernelNodeGetParams` (no execution; no `cudaGraphGetNodes` order
  assumption — design-note §3).

**Ground-truth trick:** producer `j` fills its payload with byte `j`, and its meta is
pushed with `layer_no = j`. So under correct lockstep, for every submitted slice
`layer_no == first_byte(slice)`, and the delivered sequence equals the concatenation
of the per-step enabled sets. Any desync breaks one or both.

## Results (RTX 4090, N=12, 5-step reconfigure schedule)

| scenario | delivered | desync (`layer≠first_byte`) | mismatch (vs enabled seq) |
|---|---|---|---|
| **(1) lockstep** — metas only for enabled producers | 37 / 37 | **0** | **0** |
| **(2) lockstep violation** — metas for ALL producers | 37 | **22** | **22** |

8/8 assertions pass. Scenario (1) proves the invariant is **sufficient**; scenario (2)
proves it is **necessary** (and that the test has teeth — desync is really observed,
not assumed). This is the empirical confirmation of the FIFO-alignment hazard the
investigation flagged, now through the real drain/p2p/meta-FIFO consumer.

## What Phase 1 establishes

- Node-toggle + lockstep meta-push keeps `p2p_thread.cpp`'s positional
  `fifo_.pop(meta)` matching aligned across reconfigures — end-to-end to SubmitFn.
- The step-boundary reconfigure protocol (sync → SetEnabled → push metas → launch)
  works on a non-blocking stream.
- Violating lockstep desyncs exactly as predicted.

## What Phase 1 does NOT yet cover (→ Phase 1b / later)

- **Wiring into the real HF `generate()` forward.** Here the metas are hand-built;
  the real integration must gate `RingTransport.pre_push_all_metas`
  (`ring_transport.py:604`) by the enabled-node set so the lockstep is produced by
  the actual shape-computing meta path. That is the next step.
- **DMI owning the graph in the real HF path.** HF uses `torch.compile`
  cudagraph_trees, so DMI does not hold the exec graph there (same class of blocker
  as vLLM). This test uses a DMI-owned explicit capture as the sandbox; production
  wiring needs a way to obtain node handles from the framework's captured graph.
- **Whether the decode loop provides a natural per-step barrier for #2** (so the
  reconfigure sync is free vs an added cost) — open; verify in Phase 3.
- **Concurrency stress:** single reconfigure per step here; not a torn/contended
  reconfigure under heavy overlap.

## Reproduce

```bash
make -C tests/ring test_node_toggle_e2e
```
