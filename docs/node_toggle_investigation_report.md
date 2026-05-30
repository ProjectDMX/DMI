# Toggle-Node Feature — Investigation & Scoping Report

> **Task:** Line 1 — investigate/scope/de-risk the post-capture node-toggle feature (Axis A #1–#4).
> **Branch:** `feature/dmi_kernel_node_toggle` · **Date:** 2026-05-29
> **Inputs:** `docs/line1_toggle_node_briefing.md`, `docs/dmi_system_upgrades.html`
> **Status of deliverable:** investigation complete; a standalone GPU prototype of the core primitive was **built and run locally** (3× RTX 4090, CUDA 13.0). Full-pipeline FIFO test deferred to Zaratan (plan below).

---

## 0. Bottom line up front

1. **The branch name is aspirational.** `cudaGraphNodeSetEnabled`-based node toggling is **not implemented** — there is no `cudaGraph*` call anywhere in `monitoring/`. The branch's commits are all `offline_inference` benchmarks/READMEs. Confirmed Task-A finding from the briefing.
2. **What exists are two unrelated mechanisms**, neither of which is post-capture node toggling:
   - `null_mode` — a **global, soft** producer disable (device flag; kernel still launches, early-returns). It is **FIFO-safe** because the metadata push is gated by the *same* flag (symmetric).
   - `set_enabled_hooks` — a **static name filter on the legacy non-ring D2H engine**. It does *not* touch the ring/CUDA-graph path at all. Less relevant than the briefing assumed.
3. **Both core questions are answered YES, empirically, against the real dual-ring.** Two probes were built and run locally:
   - **Q1 (does disabling reduce overhead?) — yes, and reconfigure is ~free.** On the *real* `producer.cu` + dual-ring (16 nodes × 1 MB): all-enabled replay = **183.6 µs**, all node-disabled = **3.4 µs** (−98.1%), null_mode-soft = **13.4 µs**. Reconfigure via `cudaGraphNodeSetEnabled` = **0.19 µs/call, no re-instantiation**. True-disable also beats null_mode by ~0.6 µs/node (it removes the launch entirely; null_mode still launches and reads the flag).
   - **Q2 (can it be done under the dual-ring?) — yes on the device side, structurally.** Disabling a subset of producer nodes post-capture leaves the dual-ring **consistent and aligned**: the remaining producers publish a *contiguous, gap-free, correctly-offset* run of task entries (verified by content + offsets + `payload_head`), through 50 randomized toggle reconfigs with zero corruption. This works because each producer advances `task_head`/`payload_head` *itself at runtime* (heads are not host-pre-reserved) — a skipped node leaves no hole; the ring "closes up." The one thing an implementer must add is **lockstep host-side meta push** (push metas only for enabled nodes), else the host `TensorMetaFifo` desyncs (§2).
4. **The real engineering blocker is graph ownership, not the primitive.** In the vLLM path, **vLLM owns the CUDA graph**; DMI never holds a `cudaGraphExec_t` or any node handle. DMI's entire current strategy is to install producer kernels *before* warmup so they get captured as fixed nodes, then steer them only via the device-global `null_mode`. To call `cudaGraphNodeSetEnabled` on a *real* DMI deployment we must obtain handles to DMI's nodes inside vLLM's captured exec graph — which DMI currently cannot do. This is almost certainly why the feature stalled.
5. **For the online hallucination monitor v0, you do not need this feature.** `null_mode` + a static superset of captured hooks covers the v0 needs. Node-toggle is a production-grade overhead/adaptivity optimization (#1/#2/#4), and per-request heterogeneity (#3) is *not* achievable with node-toggle at all (nodes are batch-shared). Recommendation: **do not build #1–#4 now**; ship v0 on the static path; keep the probe as the de-risk artifact.

---

## 1. Current state (Task A)

### 1.1 The feature is absent
- `grep -rn "KernelNodeSetEnabled\|GraphNodeSetEnabled\|cudaGraphExec" monitoring/` → **nothing**. No `cudaGraph` reference of any kind in `monitoring/`.
- Branch `feature/dmi_kernel_node_toggle` vs `main`: every commit is `offline_inference`/README/benchmark work. None touch kernel-node toggling.

### 1.2 `null_mode` — global soft disable (the mechanism that *does* gate capture)
- Device side: `monitoring/csrc/ring/producer.cu:13` `__device__ bool g_ring_null_mode`; `:61` `if (g_ring_null_mode) return;` — the producer kernel **still launches** but writes nothing.
- Toggle: `producer.cu:21` `set_ring_null_mode()` → `cudaMemcpyToSymbol`. Exposed up through `ring_engine_py.cu:93` → `bindings.cpp:482`.
- **Host side is gated by the same flag** — `ring_transport.py:613-614`:
  ```python
  if self.null_offload:
      return  # kernel launches happen; metas are intentionally skipped
  ```
  So when null_mode is on: **no payload written AND no meta pushed**. Symmetric ⇒ the FIFO stays empty on both sides ⇒ **no desync**. (Comment at `ring_transport.py:526-529` documents exactly this.)
- **Used between replays today:** yes — `vllm_integration.py:172` sets `null_mode=True` for warmup (so vLLM captures the producer kernels into its graph as no-ops), then `compile_or_warm_up_model` (`:240-245`) flips it to `False` after warmup. This is already a working "toggle between graph replays" — but **global**.

### 1.3 `set_enabled_hooks` — static, legacy, non-ring
- `native_engine.cpp:243` → `hooks.cpp:207` (`NativeMonitoringEngine::Impl::set_enabled_hooks`) just fills an `enabled_hooks_` string set on the **legacy `NativeMonitoringEngine`** (the old D2H path), under a mutex. Callers `hook_points.py:871/879/1105/1114`.
- **It is not part of the ring/CUDA-graph path.** In the ring path, hook selection is decided at install time by `apply_hook_selection` + `install_ring_hooks` (`ring_transport.py`, called from `vllm_integration.load_model`) — i.e. *which producer kernels even exist in the graph*. Once captured, that set is frozen. `set_enabled_hooks` cannot change which nodes run under a captured graph.

### 1.4 vLLM graph ownership (the feasibility gate)
- `DMXGPUWorker` (`vllm_integration.py`) never creates or instantiates a CUDA graph. It:
  1. remaps the arch to the hooked variant (`load_model`, `_ARCH_REMAP`),
  2. installs HookPoints/producer kernels **before** `super().compile_or_warm_up_model()` so vLLM's own capture pulls them in,
  3. controls them **only** via the global `null_mode` device flag.
- **DMI holds no `cudaGraphExec_t` and no node handles.** vLLM owns capture/instantiation/replay. → To use `cudaGraphNodeSetEnabled` per-node in production, DMI must reach into vLLM's captured exec graph and locate its producer nodes. No such plumbing exists. **This is the real blocker.**

---

## 2. The correctness crux: FIFO alignment (Task B, analysis)

Metadata is matched to payloads **positionally**, by FIFO order:
- `tensor_meta.h:124` `TensorMetaFifo` is a host `std::deque<TensorMeta>`; Python pushes the full active-spec set per step via `push_all_metas` (`ring_transport.py:647`).
- `p2p_thread.cpp:158` `do_post_processing` matches the *i-th drained payload* to the *i-th* `fifo_.pop(meta)`. The only guard is a byte-size check (`p2p_thread.cpp:185`).

**Implication for naïve node-toggle:** if you disable the k-th producer node *post-capture* but Python still pushes all N metas, every payload after k is matched to the wrong meta (off-by-one cascade) → wrong `act_name`/`layer_no`/shape for all subsequent hooks. The size check only *sometimes* catches it (when adjacent shapes differ); same-size neighbors desync **silently**.

⇒ Any real node-toggle implementation must **toggle the meta-push set in lockstep with the node-enable set** — exactly the symmetric discipline `null_mode` already follows globally. The hard part of #1 is not the CUDA call; it's keeping host meta-push and device node-enable consistent per step, per node.

---

## 3. Prototype results (Task B, empirical)

Environment: local box has **3× RTX 4090 (sm_89), CUDA 13.0**, nvcc on PATH. (Not the H100 target, but the node-toggle primitive is architecture-independent.) Native backend + ClickHouse client are **not built** locally; the full Python/ring/CH pipeline was *not* exercised here — see Zaratan plan §5.

Two probes, both built and run locally:

### 3.1 Real dual-ring probe — `docs/node_toggle_probe/probe_dualring_toggle.cu` (the important one)

This links the **actual `ring::producer_kernel` + `AllocatedRing`** (payload ring + task/meta ring) and captures N=16 real producer launches into a CUDA graph the same way vLLM captures DMI's hooks. Each source `j` is filled with byte `j` so the published task entries can be traced back to their writer.

```
GPU: NVIDIA GeForce RTX 4090 sm_89 | CUDA rt 13000 drv 13000 | REAL dual-ring producer
captured: 16 graph nodes, 16 kernel nodes (expected 16)

=== Q2: dual-ring consistency under node-toggle ===
  [all-enabled] published=16  ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  [subset-off]  published=11  ids=[0,1,3,4,6,8,9,10,12,14,15]      (dropped {2,5,7,11,13})
  [re-enabled]  published=16  ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
  + 50 randomized toggle reconfigs: all stay aligned, no cumulative corruption

=== Q1: per-replay overhead (real producer, 16 nodes x 1MB) ===
  (a) all enabled            :   183.58 us
  (b) all node-disabled      :     3.43 us   <- true disable
  (c) all null_mode soft     :    13.44 us   <- launches+early-return
  (d) half node-disabled     :    89.80 us
  true-disable saves vs full :   180.15 us (98.1%)
  true-disable saves vs null :    10.02 us (5.5% of full)
  reconfigure cost (cudaGraphNodeSetEnabled): 0.189 us/call (host wall-clock, 10000 iters), no re-instantiate
ALL CHECKS PASSED (0 checks failed)
```

**Q2 — feasibility under the dual-ring (device side): proven.** Disabling a subset leaves the ring fully consistent:
- The remaining producers publish a **contiguous, gap-free** run of task entries (`[0,1,3,4,6,…]` — exactly the enabled set, in order).
- Every published entry is well-formed (`ready_seq` set, correct `tensor_total_bytes`), payload offsets are contiguous, and `payload_head == #published × ALLOC` exactly.
- **Why it works:** the producer reads `*task_head`/`*payload_head` at runtime and the last block advances them (`producer.cu:96-117`). Heads are **not** host-pre-reserved, so a disabled node never touches them and the next enabled node writes the next slot — the ring "closes up" with no hole. 50 randomized reconfigs confirm no cumulative corruption.
- **Rigor (post code-review):** the subset toggle is driver-independent — the probe first builds an explicit `kernel-node → producer-id` map by enabling one node at a time (since `cudaGraphGetNodes()` does **not** guarantee enumeration order) and then toggles by producer-id; the per-entry check is `ready_seq == slot index` (validates the sequence protocol, not just "published").
- **Caveat (the actual work item):** this is the *device* side. The host `TensorMetaFifo` is pushed independently by Python; to stay aligned the implementer must push metas **only for enabled nodes**, in lockstep with the node-enable set (§2). That's a software-discipline requirement, **not** a dual-ring architectural limitation.

**Q1 — does disabling reduce overhead: yes, decisively, and reconfigure is ~free.**
- True node-disable removes essentially the entire cost (183.6 → 3.4 µs, −98.1%): the real D2D copy + publish is gone, only graph-launch overhead remains.
- True-disable also beats `null_mode` by ~0.6 µs/node (10 µs over 16 nodes): null_mode still *launches* every producer and pays the early-return; node-disable removes the launch.
- Reconfiguring between replays costs **0.189 µs/call with no re-instantiation** (host wall-clock over 10000 iters) — negligible.
- Half-disabled scales ~linearly (89.8 µs ≈ half), as expected.

> **Caveat on absolute numbers:** these are isolated microbench timings with 1 MB tensors. In a real decode step the producer copies overlap with / are dwarfed by model compute, so 183 µs is *not* "the serving overhead" — the meaningful takeaways are the **relative** ones (disable ≈ free of the copy; null_mode pays the launch; reconfigure ≈ free).

### 3.2 Synthetic primitive probe — `docs/node_toggle_probe/probe_node_toggle.cu`

A simpler probe (synthetic kernel, not the ring) that isolates the primitive and additionally times the `null_mode` `cudaMemcpyToSymbol` toggle path (host wall-clock, 10000 iters): `cudaGraphNodeSetEnabled` = 0.194 µs/call vs `cudaMemcpyToSymbol` = 3.99 µs/call (but the latter flips *all* producers in one global call). Subset disable exact, re-enable restores. Consistent with §3.1.

**What neither probe proves:** that DMI can obtain node handles from *vLLM's* captured graph (§1.4) — both probes capture their own graph. That plumbing is the real blocker (§4 Q3).

---

## 4. De-risk answers (Task C)

**Q1 — Is full `cudaGraphNodeSetEnabled` necessary, or does `null_mode` + a static superset suffice?**
For v0 of the hallucination monitor: **`null_mode` + static superset suffices.** Static full (or a chosen superset) capture already meets the core claim. Node-toggle is an *overhead/adaptivity* optimization, not a correctness requirement. Necessary only if (a) idle/selective overhead must approach zero, or (b) you need to escalate fidelity at runtime without a captured superset.

**Q2 — Granularity the monitor needs.** A probe-based online monitor typically needs **one hook type at one layer** (e.g. `resid_final` or a mid-layer residual) globally for the batch. That is **global / per-hook-type**, the *easy* granularity — already served by static selection + `null_mode`. **Per-request** heterogeneity (#3) is **not needed for v0** and is **not achievable via node-toggle** anyway: graph nodes are batch-shared, so a node is on for the whole batch or off for the whole batch.

**Q3 — Does DMI have access to vLLM's captured node handles?** **No.** vLLM owns the graph; DMI holds no exec/node handle (§1.4). This is the feasibility gate and the likely reason the feature stalled. Closing it requires new plumbing into vLLM's capture (tag/locate DMI's producer nodes in the exec graph).

**Q4 — Latency cost of toggling between replays.** Measured on the real exec graph (host wall-clock):
- *Single node:* `cudaGraphNodeSetEnabled` ≈ **0.19 µs/call, no re-instantiation**.
- *Full-set flip* (the realistic "full capture → fully off" — N serial host calls on the critical path): **linear at ~0.2 µs/node** — measured **3.1 µs at N=16** and **30.3 µs at N=145** (≈ the backpressure hook count). Per-node cost is constant across N, i.e. **no hidden re-instantiation/re-validation penalty** — which is exactly the point: it avoids `cudaGraphInstantiate` (hundreds of µs–ms).
- *`null_mode` flip:* ≈ **4 µs**, single global `cudaMemcpyToSymbol` (flips all producers at once, but soft).
- **Takeaway:** reconfigure is cheap *because it does not re-instantiate*. The full-set flip cost grows with hook count (~30 µs at 145 hooks, pure host/serial) — small vs a multi-ms decode step but non-zero and recurring if you flip the whole set every step. Adaptive escalation that touches only a few nodes costs `(#changed)×0.2 µs`, far less than a full flip. If a prior experiment saw "large" toggle overhead, that was almost certainly **re-instantiation/re-capture**, which this path avoids.

**Q5 — Overhead: toggled-off node vs `null_mode` vs statically-absent.** Measured on the real dual-ring producer (16×1 MB, §3.1):
- All-enabled **183.6 µs** → all node-disabled **3.4 µs** (**−98.1%**): true disable removes the real D2D copy + publish, leaving only graph-launch overhead. Statically-absent ≈ toggled-off.
- `null_mode` **13.4 µs**: kernel **still launches** and reads the flag → +10 µs over true-disable (~0.6 µs/node of pure launch+early-return waste).
- So **"true disable" buys a real saving over `null_mode`** — the launch cost itself (~0.6 µs/node). Small per node, meaningful at high hook counts / request rates — exactly the "near-zero idle overhead" production claim. Note absolute numbers are isolated-microbench, not serving overhead (see §3.1 caveat).

---

## 5. Zaratan test plan (full-pipeline FIFO-alignment, deferred)

The local probes already validate (i) the primitive and (ii) **device-side dual-ring consistency under toggle** (§3.1). What remains to prove on the target HW is the **end-to-end host path** — node-enable kept in lockstep with `pre_push_all_metas`, drained through the real meta FIFO + p2p to ClickHouse — plus target-HW overhead numbers. Run on Zaratan H100. (`--account=zaoxing-prj-cmsc`, `gpu-h100`; env per CLAUDE.md.)

1. **Build:** ClickHouse C++ client, then `make -C monitoring -j`; also add a `test_null_mode`/`test_node_toggle` target to `tests/ring/Makefile` (the file `tests/ring/test_null_mode.cu` exists but has **no Makefile target** — add one).
2. **Baseline alignment test (existing mechanism):** HF or vLLM path, full hook set, `null_mode` off → capture a graph, replay K steps, assert every hook's data is present and `act_name`/`layer_no`/shape correct (reuse `tests/test_e2e_correctness_vs_hf.py` machinery).
3. **Global toggle test:** flip `null_mode` on between replays → assert 0 tensors arrive and FIFO stays empty; flip off → assert full set returns aligned. (This is what warmup already does; make it an explicit assertion.)
4. **Node-toggle prototype (the new thing):** only if pursuing #1 — requires the §1.4 plumbing. Smallest version: in the **HF path** (where DMI could own its own graph capture, unlike vLLM), capture N producer nodes, `cudaGraphNodeSetEnabled(disable subset)`, **and** suppress those specs in `pre_push_all_metas` for that step, replay → assert (a) disabled hooks produce nothing, (b) remaining hooks stay aligned, (c) re-enable restores. The test must prove the **lockstep meta/node discipline** of §2.
5. **Overhead numbers on H100:** repeat the probe's per-replay comparison at realistic hook counts (e.g. 145-hook config from the backpressure sweeps) to quantify true-disable vs null_mode on the target HW.

---

## 6. Value mapping & recommendation (Task D)

| Axis-A upgrade | Needed for hallucination-monitor v0? | Granularity | Status / blocker |
|---|---|---|---|
| **#1 Post-capture node toggle** | **No** (static + null_mode suffices) | per-hook-type | API proven locally; **blocked** on vLLM graph-handle access |
| **#2 True disable vs null_mode** | No (nice-to-have for idle overhead) | per-node | Saving proven real but small/node; matters at scale |
| **#3 Per-request heterogeneous** | No | per-request | **Not achievable via node-toggle** (nodes batch-shared) — different mechanism entirely |
| **#4 Adaptive observability** | No for v0; yes for the production differentiator | per-hook-type | Built on #1; same blocker |

**Recommended next steps, in order:**
1. **Ship v0 on the static path** — static hook superset + global `null_mode`. No node-toggle. This unblocks the actual hallucination-monitoring measurement immediately.
2. **Keep the probe** (`docs/node_toggle_probe/`) as the de-risk artifact; it answers "does the primitive work + what does it cost." Run §5 step 5 on H100 if you want target-HW overhead numbers for the paper's "near-zero idle" claim.
3. **Before investing in #1/#4, resolve the feasibility gate (Q3):** prototype DMI-owned graph capture in the **HF path** first (DMI can own the graph there), proving the §2 lockstep meta/node discipline end-to-end. Only then tackle the harder vLLM-graph-handle plumbing.
4. **Do not build #3** via node-toggle — it's the wrong tool (batch-shared nodes). If per-request selectivity is ever needed, it's a separate design (e.g. per-request masking inside the kernel + per-request meta), not a graph-node toggle.
5. **Drop or rename the branch's implied scope.** Either implement #1 deliberately (with the gate resolved) or stop implying it; current branch content is unrelated benchmark work.

---

## Appendix — files & line references

| Area | Reference |
|---|---|
| null_mode device flag / early-return | `monitoring/csrc/ring/producer.cu:13,21,61` |
| null_mode host gating (symmetric) | `monitoring/ring_transport.py:526-529, 613-614` |
| null_mode warmup toggle (between replays) | `monitoring/vllm_integration.py:172, 240-245` |
| set_enabled_hooks (legacy, non-ring) | `monitoring/csrc/hooks.cpp:207`, `native_engine.cpp:243` |
| FIFO positional match (desync crux) | `monitoring/csrc/ring/tensor_meta.h:124-155`, `p2p_thread.cpp:158,185` |
| vLLM graph ownership (no handle) | `monitoring/vllm_integration.py` (`DMXGPUWorker.load_model`, `compile_or_warm_up_model`) |
| Producer head-advance (why ring closes up) | `monitoring/csrc/ring/producer.cu:96-117` (`*ring.task_head`/`*ring.payload_head` read+advanced by kernel) |
| Ring test harness (add toggle target) | `tests/ring/Makefile`, `tests/ring/test_null_mode.cu` |
| Probe — real dual-ring toggle (primary) | `docs/node_toggle_probe/probe_dualring_toggle.cu` |
| Probe — synthetic primitive | `docs/node_toggle_probe/probe_node_toggle.cu` |
