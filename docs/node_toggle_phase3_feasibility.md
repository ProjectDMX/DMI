# Node-Toggle — Phase 3 Feasibility Probe

> Phase 3 question (the "real blocker"): can DMI obtain handles to ITS producer
> nodes inside a CUDA graph that the framework (vLLM / torch.compile) captured,
> and call `cudaGraphNodeSetEnabled` on that foreign exec graph?
>
> **Verdict: feasible in principle — the blocker is cleared, with named conditions.**
> Proven at two levels on RTX 4090 / CUDA 13 / torch 2.10 / vLLM 0.19 (local).

## The feasibility chain (both parts now have evidence)

**(a) Recover DMI's node handles inside a capture it does not own.**
`docs/node_toggle_probe/probe_phase3_capture_handles.cu`: a separate "owner" drives
`cudaStreamBeginCapture/EndCapture/Instantiate`; DMI only launches its producers into
the stream and, right after each launch, calls `cudaStreamGetCaptureInfo` to read the
in-progress graph and the current tail dependency — exactly the node it just added.
Result: with compute kernels interleaved, DMI recorded all N producer nodes, then
disabled the subset `{1,4,6}` on the owner's exec → producers ran `[1,0,1,1,0,1,0,1]`,
and model compute still ran fully. **The mechanism works.**

This mirrors DMI's real position: its producer kernels are already launched *during*
the framework's warmup capture (that is how they enter the graph today), so a
producer-side hook can query capture info at that exact point.

**(b) Get the framework's exec handle and toggle on it.**
`torch.cuda.CUDAGraph` exposes `raw_cuda_graph()` and `raw_cuda_graph_exec()`. Confirmed
end-to-end in Python via `cuda-python`: capture two `add_` ops into a keep_graph graph,
enumerate nodes from `raw_cuda_graph()`, then `cudaGraphNodeSetEnabled(raw_cuda_graph_exec, node0, 0)`
→ replay → `a=0, b=1` (the disabled op did not run); re-enable → `a=1, b=1`.
**`cudaGraphNodeSetEnabled` works on torch's own exec.**

## Required conditions (concrete, must be satisfied for the real path)

1. **`keep_graph=True` on the framework's `CUDAGraph` — CONFIRMED REQUIRED.**
   `probe_phase3_handle_survival.py` settles this directly: a node handle recorded via
   `cudaStreamGetCaptureInfo` *inside a real `with torch.cuda.graph(g)` capture*
   - with `keep_graph=False` (vLLM's default): after torch instantiates and frees the
     template, `cudaGraphNodeSetEnabled(exec, recorded_node, 0)` **returns err 1** (the
     handle dangles) — toggle impossible.
   - with `keep_graph=True`: the handle stays valid and toggling works (`a` stays 0).
   vLLM 0.19 creates `torch.cuda.CUDAGraph()` *without* `keep_graph` (`vllm/compilation/cuda_graph.py:283`),
   so **DMI needs vLLM to pass `keep_graph=True`** — a small, concrete vLLM-side patch.
   (Also confirmed by the same probe: `cudaStreamGetCaptureInfo` works *inside* torch's
   capture context, i.e. the DMI-producer-during-capture hook point is real.)
2. **`instantiate()` must have run** before `raw_cuda_graph_exec()` — normal (it is
   instantiated for replay anyway).
3. **Capture mode matters.** vLLM 0.19 has both a full-cudagraph path
   (`vllm/compilation/cuda_graph.py`, uses `torch.cuda.CUDAGraph` → raw handles
   reachable) and a piecewise/inductor path (`piecewise_backend.py`, graphs managed by
   inductor's cudagraph_trees → raw exec access is murkier). The toggle is feasible on
   the **full-cudagraph** path; the piecewise path needs separate investigation.
4. **Per-graph registry.** vLLM captures one graph per batch size (piecewise = many).
   DMI must key its recorded node handles per capture (the `id_out` from
   `cudaStreamGetCaptureInfo` identifies the capture) and fetch each exec.
5. **Lifecycle / ownership (design-notes §1).** `SetEnabled` only between replays with
   the prior replay complete + meta-FIFO lockstep; and vLLM must tolerate its exec graph
   being mutated between replays.

## Resolved by the local probes
- `cudaStreamGetCaptureInfo` works inside a real `torch.cuda.graph` capture (the DMI hook
  point during capture is valid). ✓
- Node-handle validity through instantiate: **requires `keep_graph=True`** (proven). ✓

## Still to verify on the real vLLM path (next, on H100 with the backend)
- Land the `keep_graph=True` change in vLLM's `CUDAGraphWrapper` (or confirm a config knob),
  and confirm DMI's producer runs inside that capture under FULL mode (default decode path).
- The piecewise/inductor path (attention runs eager, graphs are inductor-managed): whether
  any toggle is possible there, or DMI restricts to the FULL (decode) path.
- Reaching `raw_cuda_graph_exec` for the specific graph(s) vLLM holds, per batch size.

## Bottom line

The thing the investigation called "the real blocker" — DMI holding no graph/node handle
— is **not a hard wall**: handles are recoverable (`cudaStreamGetCaptureInfo`) and the
exec is exposed by torch (`raw_cuda_graph_exec`), and toggling works on both a foreign
raw-CUDA exec and torch's own exec. The one hard requirement is now pinned down:
**the framework's CUDAGraph must be created with `keep_graph=True`** (proven — handles
dangle otherwise). That plus confirming the FULL capture path is the remaining
integration work — not a fundamental impossibility. This materially de-risks the feature.

## Reproduce

```bash
cd docs/node_toggle_probe
# (a) raw-CUDA mechanism: capture-info node recovery + toggle on a foreign exec
nvcc -std=c++17 -arch=native -O2 probe_phase3_capture_handles.cu -o probe_p3 && ./probe_p3
# (b) torch-level: cudaGraphNodeSetEnabled on torch's raw_cuda_graph_exec
python probe_phase3_torch_handle.py
# (c) handle survival: capture-info inside torch capture + keep_graph requirement
python probe_phase3_handle_survival.py
```
