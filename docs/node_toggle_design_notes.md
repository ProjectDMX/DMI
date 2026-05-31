# Node-Toggle — Design Notes & Invariants

> Authoritative constraints for implementing the Axis-A post-capture node toggle
> (`cudaGraphNodeSetEnabled`) on top of Ring². The implementation and its tests
> MUST honor every invariant below. Background: `node_toggle_investigation_report.md`;
> staged plan: `ring2_toggle_implementation_plan.html`.

## 0. Two race classes — what toggle avoids vs what it still requires

`null_mode` and node-toggle disable the producer differently, and they have
**different** stream-ordering hazards:

| | `null_mode` (device flag) | node-toggle (`cudaGraphNodeSetEnabled`) |
|---|---|---|
| State lives in | `__device__ bool g_ring_null_mode` | the exec graph (host object), node `enabled` bit |
| Consumed when | kernel reads it **at runtime** | launch dispatch decides if the node runs |
| Mutated via | `cudaMemcpyToSymbol` → **legacy default stream** | host-side exec edit — **not** a stream op, **not** a memcpy |
| The d0a99a5 race (default-stream vs non-blocking compute stream) | **present** → needs `cudaDeviceSynchronize` | **absent** (no memcpy / no device-global read) |

So switching the *disable path* from `null_mode` to node-toggle removes the
specific race d0a99a5 fixed (an in-flight kernel reading a device flag that an
async memcpy changes underneath it). **It does NOT make reconfigure
"synchronization-free."** It converts the problem into a host-side **exec-graph
mutation lifecycle** constraint plus a **meta-FIFO lockstep** constraint. That
constraint is cleaner — but it must be written into the design and the tests.

## 1. Hard invariants (the reconfigure protocol)

Every runtime reconfigure of the enabled-node set MUST satisfy ALL of:

1. **Reconfigure only at a step boundary.** Never mid-step / mid-forward.
2. **Prior replay must be complete before mutating the exec graph.** Calling
   `cudaGraphNodeSetEnabled` while the `cudaGraphExec_t` may be executing is
   undefined behavior. Ensure the previous graph replay has finished (piggyback
   on the decode loop's existing per-step barrier if one exists — **verify, do
   not assume** — otherwise add an explicit stream/device sync).
3. **Node-enabled set and host meta-push set are one and the same, same version,
   applied in the same step.** `pre_push_all_metas` must push metadata **only**
   for the nodes that are enabled for *that* step. The two sets are derived from
   a single source of truth, never independently.
4. **Mutate the exec graph, THEN launch the next step.** Order is: prior replay
   done → apply `cudaGraphNodeSetEnabled` for the new set → push matching metas →
   launch. Not interleaved.
5. **Never modify the `enabled` bit while the exec graph is running.** (Corollary
   of #2, stated explicitly because it is the UB that bites silently.)

Violating #3 desynchronizes the positional meta↔payload matching in
`p2p_thread.cpp` (`fifo_.pop(meta)`), corrupting every subsequent hook's
association — the same failure mode the report calls out, just per-node.

## 2. `null_mode` is retained (not fully replaced)

Node-toggle can only act **after** the graph is captured and instantiated. The
warmup / profiling / capture phase runs producer kernels eagerly *before* node
handles exist, so `null_mode` is still needed there to keep those runs no-ops.
Therefore:

- **Keep `null_mode` for the warmup/capture phase.** It still flips via
  `cudaMemcpyToSymbol`, so the d0a99a5 stream-ordering fix
  (`cudaDeviceSynchronize` before/after, now baked into
  `RingEnginePy::set_null_mode`, `ring_engine_py.cu`) **must stay**.
- **Use node-toggle for runtime reconfiguration** once the exec graph exists.
- The two coexist; they are not an either/or.

## 3. Node↔producer mapping must be explicit

`cudaGraphGetNodes()` does **not** guarantee the returned node order matches
capture/add order. Do NOT index nodes by enumeration order. Build an explicit
`node → (hook_type, layer)` map (e.g. by enabling one node at a time and
observing which producer publishes, as `probe_dualring_toggle.cu` does, or via a
capture-time registry that records each node handle as it is added). Toggle by
hook identity, never by raw `knodes[i]`.

## 4. Where each invariant is verified

- **Device-side ring consistency under toggle** (ring closes up, no desync): the
  standalone probe already proves this; Phase 0 productionizes it as a
  `tests/ring` regression test. NOTE: the probe runs single-stream + synced, so
  it does **not** exercise invariants #1–#5 (the stream-ordering / lifecycle
  hazards) — those only appear under concurrent non-blocking streams.
- **Reconfigure lifecycle + meta lockstep (#1–#5)**: must be verified in the
  Phase 1 HF-path end-to-end prototype, under a real compute stream, asserting
  that disabled hooks produce nothing AND remaining hooks stay aligned through
  drain/p2p across reconfigures.
- **vLLM has (or lacks) a usable per-step barrier for #2**: open question; verify
  in Phase 3.
