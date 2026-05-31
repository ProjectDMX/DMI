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

1. **`keep_graph=True` on the framework's `CUDAGraph`.** By default torch destroys the
   template `cudaGraph_t` after `instantiate()` (and `raw_cuda_graph()` then raises).
   The node handles DMI records live in that template; they must stay valid against the
   exec. So vLLM's capture must be created with `keep_graph=True` (a vLLM-side setting),
   or the handles risk dangling. **Verify node-handle validity after instantiate on the
   actual vLLM path.**
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

## Still to verify on the real vLLM path (next, on H100 with the backend)

- Does DMI's producer op actually run *inside* vLLM's capture (so `cudaStreamGetCaptureInfo`
  is valid there), under both full and piecewise modes?
- Node-handle validity through vLLM's instantiate (condition 1).
- Reaching `raw_cuda_graph_exec` for the specific graph(s) vLLM holds, per batch size.

## Bottom line

The thing the investigation called "the real blocker" — DMI holding no graph/node handle
— is **not a hard wall**: handles are recoverable (`cudaStreamGetCaptureInfo`) and the
exec is exposed by torch (`raw_cuda_graph_exec`), and toggling works on both a foreign
raw-CUDA exec and torch's own exec. What remains is integration work + a `keep_graph=True`
requirement + confirming the vLLM capture mode — not a fundamental impossibility. This
materially de-risks the whole feature.

## Reproduce

```bash
# (a) raw-CUDA mechanism
cd docs/node_toggle_probe
nvcc -std=c++17 -arch=native -O2 probe_phase3_capture_handles.cu -o probe_p3 && ./probe_p3
# (b) torch-level: see the cuda-python snippet in this doc's commit message / history.
```
