# Phase 3b — vLLM Wiring Investigation (vLLM 0.19)

> Goal: nail the bind path before writing `DMXGPUWorker` wiring. Findings below are
> from reading vLLM 0.19 source (`vllm/compilation/cuda_graph.py`,
> `vllm/v1/worker/gpu_model_runner.py`). No system code changed yet.

## The four questions, answered

**Q1 — Do DMI's hooks run INSIDE vLLM's capture?  YES.**
`CUDAGraphWrapper.__call__` captures with `with torch.cuda.graph(cudagraph, pool=...):
output = self.runnable(*args)` (`cuda_graph.py:308-314`). `self.runnable` is the model
forward, which contains the (remapped) HookPoints → `torch.ops.ring.producer` fires
during capture. So `cudaStreamGetCaptureInfo` in the producer op will see an active
capture and record the node. (This is the same reason DMI's producers already land in
vLLM's graph today.)

**Q2 — Where are the per-batch-size graphs, and can DMXGPUWorker reach them?  YES.**
- For FULL cudagraph, `gpu_model_runner.load_model` wraps the model at the end:
  `self.model = CUDAGraphWrapper(self.model, ..., runtime_mode=CUDAGraphMode.FULL)`
  (`gpu_model_runner.py:4874`). So after `super().load_model()`, `self.model_runner.model`
  **is** a `CUDAGraphWrapper`.
- The wrapper stores one entry per batch size:
  `wrapper.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry]`
  (`cuda_graph.py:207`); each `CUDAGraphEntry` has `.cudagraph` (a `torch.cuda.CUDAGraph`,
  set at `cuda_graph.py:332`).
- `DMXGPUWorker` already touches `self.model_runner.model` in `load_model`; after warmup
  the entries are populated, so it can iterate them and bind each graph's exec.

**Q3 — `keep_graph`?  NOT set → must patch.**
`cuda_graph.py:283` is `cudagraph = torch.cuda.CUDAGraph()` (no `keep_graph`). Phase 3b
precursor proved `keep_graph=True` is required (else the template is freed after
instantiate and the recorded node handles dangle). → patch to
`torch.cuda.CUDAGraph(keep_graph=True)` and ensure `instantiate()` is called (with
`keep_graph=True`, `capture_end` no longer auto-instantiates).

**Q4 — Multiple graphs / non-uniform hooks?  Handled.**
One capture per `BatchDescriptor` → each runs the model forward → `register_capture_node`
records that graph's nodes keyed by its own `cudaGraph_t`. The per-graph registry (Phase B)
keeps them separate; every graph captures the same hooks → `toggle_registry_uniform()`
passes; bind each entry's exec. Default mode is `FULL_AND_PIECEWISE` →
`has_full_cudagraphs()` true → decode uses the FULL wrapper (good — decode is where online
overhead matters). Prefill/mixed use the piecewise/inductor path → **out of scope** here.

## Concrete wiring (what Phase 3b will write in `vllm_integration.py`)

1. **Before warmup** (in `init_device` or `load_model`, before `compile_or_warm_up_model`):
   `engine.enable_toggle_capture(True)` so the producer op records nodes during capture.
2. **The keep_graph patch** (monkeypatch from `vllm_integration.py`, not a vLLM fork):
   make `CUDAGraphWrapper` create `torch.cuda.CUDAGraph(keep_graph=True)`.
3. **After warmup** (in `compile_or_warm_up_model`, after `super()` + `null_mode` off):
   ```python
   wrapper = self.model_runner.model           # CUDAGraphWrapper (FULL)
   if isinstance(wrapper, CUDAGraphWrapper):
       for entry in wrapper.concrete_cudagraph_entries.values():
           g = entry.cudagraph
           if g is None: continue
           g.instantiate()                      # once (keep_graph defers it)
           engine.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
   ```
4. **Per step** (in `execute_model`, at a step boundary, prior replay complete):
   `transport.set_active_hooks(<enabled set>)` → device toggle + meta gate in lockstep.

## Open items / risks to resolve during wiring

- **`instantiate()` timing.** With `keep_graph=True`, `capture_end` doesn't instantiate;
  `replay()` auto-instantiates on first call. If warmup replays before we bind, the graph
  is already instantiated → calling `instantiate()` again *destroys* the prior exec
  (`raw_cuda_graph_exec` docstring). Need to instantiate exactly once and bind that exec.
  Cleanest: patch the wrapper to `instantiate()` right after capture, and bind that.
- **Step-boundary barrier for §1.** Does vLLM's decode loop give a natural point where the
  prior replay is complete before the next `execute_model` reconfigure? Verify; else add a
  stream sync around `set_active_hooks`.
- **Piecewise path** (prefill/mixed) is not covered — restrict toggle to the FULL decode
  path, or treat piecewise as always-on.
- **`raw_cuda_graph()` ptr == capture `getCaptureInfo` graph ptr** (the registry key) — must
  hold for the bind to correlate; verify on a real run.
- **Local validation** is possible (vLLM 0.19 + small model + `cudagraph_mode=FULL` on a free
  GPU); only TPOT numbers need H100.

## Bottom line
The bind path is concrete and reachable: `self.model_runner.model` (a `CUDAGraphWrapper`
in FULL mode) → `concrete_cudagraph_entries[*].cudagraph` → `raw_cuda_graph()/_exec()`.
DMI's producers already run inside the capture. The only vLLM change needed is
`keep_graph=True` (+ instantiate handling). Everything else is `DMXGPUWorker` glue +
local validation.
