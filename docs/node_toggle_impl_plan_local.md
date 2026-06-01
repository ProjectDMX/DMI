# Node-Toggle — Concrete Code-Change Plan (local-first)

> File-by-file plan to implement the runtime toggle, prototyped locally (RTX 4090,
> small models, local vLLM 0.19) before H100 perf runs. Builds on: Phase 0/1/2
> (done), Phase 3 feasibility (done). Invariants: `node_toggle_design_notes.md` §1.
>
> Identity key: the producer op already receives `(hook_type, hook_id)` and
> `hook_id == layer_no` (`_hook_id_from_name = _layer_no_from_name` in
> `ring_transport.py:269`). `HookSpec` carries the same `(hook_type, layer_no)`.
> So both lanes (device toggle + host meta) key on the same `(hook_type, layer_no)`.
>
> Single source of truth: the **C++ controller holds the enabled-set**; BOTH the
> device toggle (`apply`) and the host meta gate (`pre_push_all_metas`) read it.
> One copy, two readers → lockstep by construction (read both within one step, no
> mutation between — §1).

---

## Phase A — Build the backend (prerequisite)  ✓ DONE (commit 881bc6d)

Working local recipe (RTX 4090, CUDA 13, torch 2.10, py3.12):
```bash
git submodule update --init libs/clickhouse-cpp           # was not checked out
cmake -S libs/clickhouse-cpp -B libs/clickhouse-cpp/build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON   # PIC required for .so
cmake --build libs/clickhouse-cpp/build -j
make -C monitoring -j                                      # auto-detect handles NVTX
python -c "import monitoring_native_backend as m; print(hasattr(m,'RingEngine'))"  # True
```
Three issues hit + fixed (all now in-tree):
1. **clickhouse-cpp submodule not checked out** → `submodule update --init`.
2. **`-fPIC`**: the default CLAUDE.md cmake line builds non-PIC static libs → `ld: ...
   can not be used when making a shared object`. Add `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`.
3. **NVTX auto-detect was broken** (not a missing lib — `libnvToolsExt.so` *does* exist at
   `/usr/lib/x86_64-linux-gnu`). The Makefile's `NVTX_HEADER_FOUND := $(shell echo '\#if
   __has_include...')` used `\#` to hide `#` from *make's* comment parser, but that
   backslash broke the preprocessor test → it reported "not found" and skipped
   `-lnvToolsExt`, while the *compiler* found the header (via torch's bundled NVTX) and
   emitted `nvtxRangePushA` → `undefined symbol` at import. Fix: (a) added a
   `MON_NVTX_DISABLE` guard to `nvtx_shim.h` so emission and linking stay consistent; (b)
   rewrote auto-detect to enable NVTX only when it can positively confirm header+lib, else
   define `MON_NVTX_DISABLE` (no-op shim). Net: **default `make` now builds + imports** (no
   `NVTX=0` needed); NVTX ends up conservatively off here, which is harmless (profiling only).
4. **Pre-existing branch break**: `ring_engine_py.cu` referenced `force_flush_and_wait_timed`
   / `FlushStats` / `get_stats` absent from this branch's `drain_thread.{h,cpp}`. Stubbed
   the two (diagnostics only) to match the real API. TODO: reconcile the drain timed/stats
   API across branches.
**Gate met:** `.so` imports, all classes (`RingEngine`/`RingConfig`/`DMXHostEngine`) present.

---

## Phase B (= 2b) — Toggle-list registry + Python binding  ✓ DONE (commit 283fe79)

**As built** (test: `tests/ring/test_toggle_binding.py`, ALL PASS):
- Registry implemented **inline in `ring_engine_py.cu`'s `Impl`** (multi-graph: per
  `cudaGraph_t` node list + exec map + a single global enabled-set), rather than per-graph
  `NodeToggleController` instances — cleaner for the backend's multi-graph reality.
  `node_toggle.h` remains the standalone-test abstraction. The enabled-set is the single
  source read by both `apply_toggle()` (device) and `is_hook_enabled()` (host meta gate).
- `ring_torch_op.cpp`: after the producer launch, if `g_toggle_capture` and the stream is
  capturing, records the tail-dependency node via `cudaStreamGetCaptureInfo` (CUDA-13
  7-arg signature) — gated by `enable_toggle_capture(true)`, default off.
- Bindings: `enable_toggle_capture / bind_graph_exec / set_enabled_hooks / apply_toggle /
  is_hook_enabled / toggle_node_count / clear_toggle_registry` (handles as `uint64_t`).
- **Verified through the real backend**: real `torch.ops.ring.producer` registered all
  nodes during a torch capture; `apply_toggle()` succeeded on torch's `raw_cuda_graph_exec`
  with the capture-recorded handles (the Phase 3 mechanism, now via the actual producer op).
- **Not yet verified here**: the device *effect* (disabled node writes nothing) observed
  end-to-end — that needs the meta gate + a consumer, i.e. Phase C.

### Original plan (for reference)

Goal: capture-time node registration in C++, a host enabled-set, an `apply` that
toggles on a bound exec, and a Python API. No model needed to unit-test.

### B1. `monitoring/csrc/ring/node_toggle.h` (extend the existing controller)
- Key entries by `(hook_type, layer_no)` (already `HookId`).
- Add a **per-graph registry** holder: `std::unordered_map<cudaGraph_t, NodeToggleController>`
  (one controller per captured graph; the graph ptr is the correlation key with torch's
  `raw_cuda_graph()`).
- Add `bind_exec(cudaGraph_t, cudaGraphExec_t)` to attach the exec discovered post-capture.

### B2. `monitoring/csrc/ring/ring_torch_op.cpp` — capture-time registration
In `ring_producer_impl(tensor, hook_type, hook_id)`, after the existing
`g_active_engine->hook_no_notify(...)` launch:
```cpp
// If we're inside a stream capture, record THIS producer's node so it can be
// toggled later. cuda-13 sig: (stream, &status, &id, &graph, &deps, &ndeps).
cudaStreamCaptureStatus st; unsigned long long cid; cudaGraph_t g;
const cudaGraphNode_t* deps; size_t nd;
if (cudaStreamGetCaptureInfo(stream, &st, &cid, &g, &deps, &nd) == cudaSuccess
    && st == cudaStreamCaptureStatusActive && nd >= 1) {
    g_active_engine->toggle_registry().register_node(
        g, HookId{(int)hook_type, (int)hook_id}, deps[nd-1]);
}
```
(Guard behind a feature flag so the default path is unchanged.)

### B3. `monitoring/csrc/ring/ring_engine_py.{h,cu}` — expose on `RingEnginePy`
Mirror the `set_null_mode` chain. Add:
- `void bind_graph(uintptr_t graph_ptr, uintptr_t exec_ptr)` — associate exec (from
  torch `raw_cuda_graph_exec()`) with the registry entry for `graph_ptr` (torch
  `raw_cuda_graph()`).
- `void set_enabled_hooks(std::vector<std::pair<int,int>> enabled)` — set the enabled
  `(hook_type,layer)` set on all bound controllers (the single source).
- `int apply()` — for each bound graph, `controller.apply(exec)`; return first cuda err.
  (Caller guarantees prior replay complete — §1.)
- `bool is_enabled(int hook_type, int layer)` — read the enabled-set (for the meta gate).
- A `toggle_registry()` accessor for B2.

### B4. `monitoring/csrc/bindings.cpp` — pybind
Bind the four methods above with `py::call_guard<py::gil_scoped_release>()` (same as
`set_null_mode` at `bindings.cpp:482`).

### B5. Local test — `tests/ring/test_toggle_binding.py` (new)
Standalone capture in Python (cuda-python or a tiny torch graph), drive
`set_enabled_hooks` + `apply`, assert via a SubmitFn collector that only enabled hooks
deliver and stay aligned. (Python analog of `test_node_toggle_e2e.cu`.)
**Gate:** lockstep subset delivers aligned; bypass desyncs.

---

## Phase C (= 1b) — Per-spec meta gate in the real Python path

### C1. `monitoring/ring_transport.py` — gate `pre_push_all_metas` (:604-659)
Inside the `for spec in self._active_specs:` loop, before computing/pushing the meta:
```python
if self._toggle_enabled is not None and not self._ring_engine.is_enabled(
        spec.hook_type, spec.layer_no):
    continue                      # lockstep: skip metas for disabled nodes
```
- `self._toggle_enabled` defaults `None` (feature off → current behavior, all push).
- Keep the existing `null_offload` early-return (:613) untouched.

### C2. `monitoring/ring_transport.py` — driver API
Add `RingTransport.set_active_hooks(enabled_set)`:
1. `self._ring_engine.set_enabled_hooks(list(enabled_set))`  (updates C++ source)
2. `self._ring_engine.apply()`                                (device toggle)
3. set `self._toggle_enabled = enabled_set`                   (enables the gate)
Called at a step boundary, prior replay synced (§1). After this, `apply` (device) and
`pre_push_all_metas` (host) both read the same C++ enabled-set → lockstep.

### C3. Local test — HF path, small model
`gpt2` / `Qwen3-4B` (fits 24 GB). DMI-owned explicit capture (Phase-1 sandbox style) or
torch graph with `keep_graph=True`. Steps: all-on → `set_active_hooks(subset)` → replay →
assert (a) disabled hooks deliver nothing, (b) remaining hooks' `act_name/layer_no/shape`
aligned (SubmitFn collector, or ClickHouse if running), (c) re-enable realigns. Reuse
`tests/test_e2e_correctness_vs_hf.py` machinery.
**Gate:** correct per-hook content + zero desync across multiple flips.

---

## Phase D (= 3b) — Real vLLM wiring (local small model first, then H100)

### D1. vLLM patch — `vllm/compilation/cuda_graph.py:283`
`torch.cuda.CUDAGraph()` → `torch.cuda.CUDAGraph(keep_graph=True)` (REQUIRED — proven in
`probe_phase3_handle_survival.py`; else node handles dangle). Since `keep_graph=True`
defers instantiation, ensure `cudagraph.instantiate()` is called after `capture_end`
(else first replay pays it). Apply as a vendored patch or a monkeypatch from
`vllm_integration.py` (don't fork vLLM yet).

### D2. `monitoring/vllm_integration.py` — `DMXGPUWorker`
- After warmup capture, for each captured `CUDAGraph` (one per batch size, FULL mode):
  `engine.bind_graph(g.raw_cuda_graph(), g.raw_cuda_graph_exec())`. (Locating vLLM's
  `CUDAGraph` objects: via `CUDAGraphWrapper`'s stored entries — `cuda_graph.py` keeps a
  per-batch-descriptor dict.)
- Between decode steps: `transport.set_active_hooks(...)` at the boundary where vLLM
  already syncs (verify such a barrier exists — §1 #2).
- Restrict to FULL cudagraph mode (default decode path via `FULL_AND_PIECEWISE`);
  piecewise/inductor pieces are out of scope (attention runs eager / inductor-managed).

### D3. Local test — vLLM 0.19 + small model, FULL cudagraph
Launch the hooked model with `cudagraph_mode=FULL` (or FULL_DECODE_ONLY), toggle a hook
mid-run, assert disabled hook stops landing while others stay aligned. Then move to H100
for the same + overhead numbers.
**Gate (local):** toggle works on a real vLLM decode graph with meta lockstep, no desync.
**Gate (H100):** TPOT overhead quantified vs the paper's ~6% anchor.

---

## Sequencing, risks, what NOT to do

- Order: **A → B → C → D** (each unit-testable; B/C need only A + a small model).
- Everything behind a **feature flag**; default path byte-identical to today.
- Risks: (A) backend build friction on CUDA 13; (B) correlating torch graph↔C++ registry
  by `cudaGraph_t` ptr assumes `keep_graph=True` so `raw_cuda_graph()` is valid; (D) does
  vLLM's decode loop give a free per-step barrier for §1 #2 — verify, else add a sync.
- Do NOT: touch the piecewise/inductor path; add any re-instantiate/re-capture path;
  change the legacy `set_enabled_hooks` (hooks.cpp) — unrelated.
- Fallback if D stalls: static superset + global `null_mode` (v0 already sufficient).
