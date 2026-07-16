# Node-Toggle Migration Scope — Layer 1 (C++ engine) vs current `origin/main`

> Target: re-apply the node-toggle feature onto `feature/node-toggle-v2`
> (branched from `origin/main` @ `5c3906e`). This doc scopes the **C++ engine
> layer** conflicts — the layer with the highest risk because main's #40 (Ring²)
> and #51 (GPU-strip + MoE) rewrote the same files the toggle modifies.
>
> Source branch (old lineage): `feature/dmi_kernel_node_toggle`.
> Status: scoping only — no code ported yet.

## TL;DR

The C++ toggle delta is **mostly pure-add** (registry struct members + ~12 new
methods + bindings + a new header). There are **three real conflict points**:

1. **🔴 CRITICAL — producer fan-out (#51).** Main now has **3** producer ops
   (`producer` / `producer_prefix` / `producer_chunked`). With the default
   `gpu_padding_strip=True`, hidden-state hooks dispatch to **`producer_prefix`**,
   not the basic `producer` the toggle records. **Out of the box, toggle would
   record 0 nodes → `set_active_hooks` raises.** Must resolve before anything works.
2. **🟠 prepare_step diverged.** Main rewrote `prepare_step` (device
   `actual_bytes_counter` delta snapshot, `STEP_OVERSIZED` rename, **no two-sided
   clamp**). The branch's clamp fix (reserve-invariant safety) must be *merged*
   into main's version, not pasted.
3. **🟡 enum rename.** `STEP_CPU_DIRECT` → `STEP_OVERSIZED`. Touches the toggle's
   Python references.

Everything else slots in cleanly. `null_mode` (toggle's dependency) already
exists on main.

---

## Per-file scope

### `csrc/ring/node_toggle.h` — PURE-ADD
New file. No main counterpart. Drops in unchanged.

### `csrc/ring/ring_engine_py.cu`
main 316 lines → branch 458 (different lineage).

| Toggle piece | Port type | Notes |
|---|---|---|
| `Impl` registry members (`reg_nodes`, `reg_exec`, `enabled_hooks`, `registered_hooks`, `target_version`, `applied_version`, `replay_event`, `toggle_mu`) | **PURE-ADD** | slot into main's `Impl` (which already carries `payload_view`, `last_counter_read`) |
| `~Impl` event cleanup | PURE-ADD | main's Impl has no dtor; add one |
| ~12 methods (`enable_toggle_capture`, `register_capture_node`, `bind_graph_exec`, `set_enabled_hooks`, `apply_toggle`, `is_hook_enabled`, `effective_enabled_mask`, `ensure_graph_current`, `record_replay_event`, `toggle_registry_uniform`, `clear_toggle_registry`, `bound_graph_count`/`toggle_node_count`, `get_stats`) | **PURE-ADD** | no main equivalents (all 8 toggle symbols = 0 hits on main) |
| `prepare_step` two-sided clamp + reserve | **🟠 SEMANTIC-MERGE** | see below |

**`prepare_step` merge detail.** Main's version (lines 184–236) computes
`payload_avail = pcap - (head - tail)` and `task_avail = tcap - (head - tail)`
**raw, unclamped** — the exact underflow the branch fixed. Main also added a
device `actual_bytes_counter` delta read (for future strip reclamation) and
renamed the oversize return to `STEP_OVERSIZED`. Port = re-apply the branch's
**two-sided saturating clamp** onto main's `payload_avail`/`task_avail` lines,
keeping main's counter-delta block intact. The "reserve from the *effective* set"
half is driven from Python (the `step_total_bytes`/`num_hooks` args already carry
the effective counts), so `prepare_step` itself only needs the clamp.

### `csrc/ring/ring_engine_py.h`
| Piece | Port type | Notes |
|---|---|---|
| toggle method declarations | PURE-ADD | mirror the .cu additions |
| `STEP_CPU_DIRECT` → `STEP_OVERSIZED` | **🟡 RENAME** | toggle's Python (`vllm` capacity loop) must use main's name (result `== 2`) |
| `hook_no_notify` (3 variants on main) | no change | toggle's `current_hook_idx` reset stays in `prepare_step` |

### `csrc/ring/ring_torch_op.cpp` — 🔴 EXPANDED
main 156 lines → branch 128 (main is *longer*: it has 3 producer impls).

| Toggle piece | Port type | Notes |
|---|---|---|
| `g_toggle_capture` static + `ring_set_toggle_capture()` | PURE-ADD | module-global; clean |
| capture-record block (`cudaStreamGetCaptureInfo` + `register_capture_node` after the producer launch) | **🔴 EXPAND ×3** | branch added it to the *one* `ring_producer_impl`. Main has **`ring_producer_impl` / `ring_producer_prefix_impl` / `ring_producer_chunked_impl`** — each launches a kernel node. The block must be replicated into all three (or strip disabled, see decision). |
| op schema gained `Tensor(a!) ring_payload` first arg | no toggle change | lineage detail; toggle reads node via `GetCaptureInfo`, not the args |

### `csrc/bindings.cpp` — PURE-ADD
Add `.def(...)` for each new method into main's `RingEnginePy` class binding
(main already binds `prepare_step`, `push_all_metas`, `set_null_mode`). The toggle
meta-gate calls `push_all_metas`, which exists on main with a compatible
signature.

### `csrc/ring/producer.cu` — NO TOGGLE CHANGE
Toggle depends only on `null_mode` (`g_ring_null_mode` / `set_ring_null_mode`),
which **already exists on main**. #51's heavy `producer.cu` rewrite (+335) is a
lineage concern for the strip path, not for toggle.

---

## 🔴 The producer fan-out decision (blocks everything)

`hook_points.py::_dispatch_producer` on main:
```
_strip_tensor is None                    -> torch.ops.ring.producer          (basic)
_strip_tensor set, _strip_row_bytes > 0  -> torch.ops.ring.producer_prefix   (strip)
_strip_tensor set, _strip_row_bytes == 0 -> torch.ops.ring.producer_chunked
```
`VLLMAdaptor.attach_model` sets `_strip_tensor` + `_strip_row_bytes>0` on every
spec with `dim0_is_actual_tokens` (hidden-states in vLLM flat layout) **when
`gpu_padding_strip=True` (the default)**. So under default config the producer
that actually fires for hidden-states is `producer_prefix` — whose node the
current toggle capture recording never sees.

**Two resolutions:**

- **(A) Extend capture recording to all 3 impls** *(keeps the strip optimization)*.
  Add the same `if (g_toggle_capture) { GetCaptureInfo + register_capture_node }`
  block to `ring_producer_prefix_impl` and `ring_producer_chunked_impl`. The node
  recorded is still "the producer kernel for (hook_type, layer)", so the
  registry/apply/meta-gate logic is unchanged — only the recording sites grow
  from 1 to 3. Modest, and the *correct* long-term answer.
- **(B) Run toggle with `gpu_padding_strip=False`** *(simplest, smaller patch)*.
  Forces every hook through the basic `producer`, so the existing 1-site recording
  works as-is. Cost: producers capture padded bytes (the strip optimization is
  off) while toggle is in use. Clean combination; good for a first landing.

Recommendation: **start with (B)** to get the migration green end-to-end (the
toggle + strip-off combo is independently valid), then do **(A)** as a follow-up
so toggle composes with the strip optimization.

---

## Layer-1 conflict summary

| Conflict | Severity | Effort |
|---|---|---|
| producer fan-out (record prefix/chunked, or strip-off) | 🔴 blocks | (B) trivial / (A) ~1 block ×2 |
| `prepare_step` clamp semantic-merge | 🟠 | small, careful |
| `STEP_CPU_DIRECT`→`STEP_OVERSIZED` rename | 🟡 | trivial (Python side) |
| registry + 12 methods + bindings + header | ✅ pure-add | mechanical |
| `producer.cu` / `null_mode` | ✅ none | — |

**Verdict:** Layer 1 is tractable. No deep redesign required — the only design
decision is the producer fan-out (A vs B), and (B) unblocks immediately. After
this layer builds + `tests/ring/test_node_toggle*.cu` pass, Layers 2–4
(transport / adaptor_base / vLLM) are the mechanical re-targeting already scoped
in `docs/node_toggle_implementation.md` and the prior migration analysis.
</content>
