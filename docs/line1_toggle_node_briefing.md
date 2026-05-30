# Line 1 — Toggle-Node Feature: Investigation & Scoping Briefing

> **For:** a fresh agent picking up this task with no prior conversation context.
> **Type:** investigation + scoping + minimal prototype/test. **NOT** "productize the feature."
> **Repo:** `/home/yibo/DMI` · **Branch:** `feature/dmi_kernel_node_toggle`
> **Related doc:** `docs/dmi_system_upgrades.html` (Axis A, upgrades #1–#4).

---

## 0. TL;DR / Mission

DMI is a high-performance LLM internal-observability system (SOSP submission). It captures internal tensors during inference via a CUDA-graph-compatible **Ring²** pipeline. Today its hook filtering is **static** — which hooks fire is decided **before** CUDA-graph capture. We want **runtime reconfiguration**: toggle which producer-kernel nodes fire **inside an already-captured graph**, between replays, **without re-capturing** — via `cudaGraphKernelNodeSetEnabled`.

**Your job:** find out what actually exists on this branch, whether the full node-toggle is needed at all, validate the hard correctness risk (FIFO alignment), and report how much this helps the new application (online hallucination monitoring). Produce a written report. Do **not** try to fully build/ship the feature.

---

## 1. Why this matters (Axis A framing)

This feature is upgrade **#1–#4** in `docs/dmi_system_upgrades.html`:

- **#1 Post-capture node toggle** — runtime enable/disable of producer-kernel nodes via `cudaGraphKernelNodeSetEnabled`, no re-capture.
- **#2 True node disable vs `null_mode` (dual-exec)** — `null_mode` is a *soft* disable (kernel still launched, early-returns); a "true disable" removes the node so there's *zero launch* — a pure path with no capture overhead.
- **#3 Per-request heterogeneous monitoring** — different requests in one batch capturing different tensors (likely NOT implemented; harder, since graph nodes are batch-shared).
- **#4 Adaptive observability** — escalate capture fidelity on demand (cheap default → full on suspicion); built on #1.

**Value context (be honest):** for the *core* hallucination-detection claim (run a probe online, measure serving cost), this is **NOT on the critical path** — static full capture works. Its value is the *production-grade* differentiator: near-zero idle overhead, selective + adaptive capture. So the goal here is to **scope what the application actually needs** and **de-risk**, not to gold-plate.

---

## 2. Current-state finding (verify this first — it reframes everything)

An initial grep suggests **the `cudaGraphKernelNodeSetEnabled`-based node toggling is NOT yet implemented**, despite the branch name. The recent branch commits are all about `offline_inference` benchmarks/READMEs, not kernel-node toggling. What actually exists:

1. **`set_enabled_hooks(names)`** — *static* hook selection (which hooks are active). Files:
   - `monitoring/csrc/native_engine.cpp:243`, `native_engine.h:78`, `native_engine_internal.h:208`, `hooks.cpp:207`
   - binding: `monitoring/csrc/bindings.cpp:74`
   - callers: `monitoring/hook_points.py:871, 879, 1105, 1114`
   - **Verify:** is this pre-capture only, or can it change anything after a CUDA graph is captured? (Under a captured graph it almost certainly does NOT change which nodes run.)
2. **`null_mode` (`g_ring_null_mode`)** — *global, soft* disable of the producer kernel. Files:
   - `monitoring/csrc/ring/producer.cu:13` (`__device__ bool g_ring_null_mode`), `:21` (`set_ring_null_mode`), `:61` (`if (g_ring_null_mode) return;`)
   - `producer.cuh:49`, `ring_engine_py.cu:93` (`RingEnginePy::set_null_mode`), `ring_engine_py.h:80`
   - binding: `bindings.cpp:482`
   - wiring: `monitoring/vllm_integration.py:114,166,172,240,245`; comment at `ring_transport.py:529`
3. **No** `cudaGraph(Exec)KernelNodeSetEnabled` anywhere (confirm with your own grep).

**So Task A below is largely: confirm the feature is unimplemented, and characterize the two mechanisms that DO exist.**

---

## 3. The core non-trivial challenge: FIFO alignment

This is the crux of why node-toggling is hard, and what any test MUST verify.

Metadata flows through `TensorMetaFifo` (`monitoring/csrc/ring/tensor_meta.h`): Python pushes metas in order before the forward (`pre_push_all_metas`), and the P2P thread pops them in order. **The FIFO order implicitly matches the producer-kernel firing order.**

If you disable a node post-capture but Python still pushes the full set of metas, the FIFO **desynchronizes** — every subsequent hook's data gets mis-associated. (SOSP context note: "Changing enabled hooks after capture would break this alignment unless carefully synchronized.")

**=> The first thing a correctness test checks is not "can we turn a node off" but "after turning it off, are the *remaining* hooks' data still correctly aligned (FIFO not corrupted)?"**

### Second complication: who owns the CUDA graph?

In the vLLM path, **vLLM captures the CUDA graph**, and DMI's producer kernels get captured *into* it. To toggle a node you need a **handle to that graph node** inside vLLM's captured exec graph. Determine: does DMI currently have any access to vLLM's `cudaGraphExec`/node handles? If not, that's a major feasibility gap to report (it may be the real reason the feature stalled). Look at `monitoring/vllm_integration.py` (`DMXGPUWorker`, `execute_model`) and how vLLM captures graphs.

---

## 4. Tasks

### A. Confirm & characterize current state
- Verify node-toggle (`cudaGraphKernelNodeSetEnabled`) is absent; document what `set_enabled_hooks` and `null_mode` actually do and when they take effect (pre-capture vs runtime).
- Trace how `null_mode` is toggled at runtime (`cudaMemcpyToSymbol`) and whether it's used between graph replays today (warmup uses it — see `vllm_integration.py:240`).

### B. Minimal correctness test / prototype
- Check the environment: is there a local GPU (`nvidia-smi`)? Can the native backend build (`make -C monitoring -j`, needs ClickHouse client built first — see CLAUDE.md)? Are there ring tests (`make -C tests/ring run`, `tests/` dir)?
- If GPU available, design+run a **minimal FIFO-alignment test**: capture a graph with N hooks; toggle a subset off (via whatever mechanism exists — likely `null_mode` only at first); replay; verify (1) disabled hooks produce no data, (2) **remaining hooks' data stays correct & aligned**, (3) re-enable works.
- If a true `cudaGraphKernelNodeSetEnabled` path doesn't exist, sketch the smallest prototype that would (and note the vLLM-graph-ownership blocker if real).
- **If no local GPU:** write a concrete Zaratan test plan (sbatch; see CLAUDE.md "Zaratan cluster" section — `--account=zaoxing-prj-cmsc`, `gpu-h100`).

### C. De-risk questions (answer these explicitly — they decide whether to invest)
1. **Is full `cudaGraphKernelNodeSetEnabled` even necessary**, or does `null_mode` (global soft) + a *static superset* of captured hooks get us most of the way for the application's needs?
2. What **granularity** is actually required by the hallucination monitor: global / per-hook-type / per-hook / **per-request**? (Per-request = #3, the hard one — graph nodes are batch-shared.)
3. Does DMI have access to vLLM's captured **graph-node handles** at all? (Feasibility gate.)
4. What's the **latency cost** of toggling between replays (the `cudaGraphExecKernelNodeSetEnabled` call, or the `cudaMemcpyToSymbol` for null_mode)?
5. Overhead comparison: toggled-off node vs `null_mode` (kernel still launched) vs statically-absent. Does "true disable" buy meaningfully less overhead than `null_mode`?

### D. Value mapping to the application
Map findings to Axis A #1–#4 and state, for the online hallucination monitor specifically: which of these we actually need, at what granularity, and whether `null_mode` + static superset suffices for v0. Cross-reference `docs/dmi_system_upgrades.html`.

---

## 5. Key file pointers (starting points — verify, don't trust blindly)

| Area | Files |
|---|---|
| Producer kernel + null_mode | `monitoring/csrc/ring/producer.cu` (`g_ring_null_mode` @13/21/61), `producer.cuh:49` |
| Ring engine Py wrapper | `monitoring/csrc/ring/ring_engine_py.cu` (`set_null_mode` @93, `prepare_step`), `ring_engine_py.h` |
| Meta FIFO (alignment crux) | `monitoring/csrc/ring/tensor_meta.h` (`TensorMetaFifo`); `pre_push_all_metas` (find in Python) |
| Static hook selection | `monitoring/csrc/native_engine.cpp:243`, `hooks.cpp:207`, `hook_points.py:871/879/1105/1114` |
| Bindings | `monitoring/csrc/bindings.cpp` (`set_null_mode` @482, `set_enabled_hooks` @74) |
| vLLM integration / graph capture | `monitoring/vllm_integration.py` (`DMXGPUWorker`, `execute_model`, null_mode @114/166/172/240/245) |
| Hook install / selection | `monitoring/ring_transport.py` (`install_ring_hooks`, `apply_hook_selection`, null_mode comment @529) |
| Build | `monitoring/Makefile`; CLAUDE.md "Build" section |

---

## 6. Environment notes
- Local platform: linux, repo at `/home/yibo/DMI`. Check `nvidia-smi` for GPU.
- Build needs ClickHouse C++ client built first, then `make -C monitoring -j`; runtime needs `CUDA_MODULE_LOADING=EAGER`.
- Real GPU experiments run on **Zaratan** (H100). See CLAUDE.md "Zaratan cluster" section for env vars, paths, sbatch template (`--account=zaoxing-prj-cmsc`).

## 7. Deliverable
A written report (markdown) covering:
1. **Current state** — what's implemented (`set_enabled_hooks`, `null_mode`) vs the absent node-toggle; the vLLM-graph-ownership situation.
2. **Test results** (if GPU) or **Zaratan test plan** (if not) — focused on FIFO-alignment correctness.
3. **Answers to the 5 de-risk questions** in §4C.
4. **Value assessment** — which of Axis A #1–#4 the hallucination monitor needs, at what granularity, and whether `null_mode` + static superset suffices for v0.
5. **Recommended next steps** — what to build, in what order, and what to *not* build yet.
