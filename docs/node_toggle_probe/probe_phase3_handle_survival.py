"""Phase 3b precursor: does a capture-info-recorded node handle survive
vLLM-style instantiation?

Two questions, both local (no DMI backend, no H100):
  (1) Inside a real `with torch.cuda.graph(g):` capture, can we call
      cudaStreamGetCaptureInfo on the capturing stream and recover the node just
      added? (= the moment a DMI producer would run inside vLLM's capture.)
  (2) After torch instantiates and (with keep_graph=False, vLLM's default) FREES
      the template graph, is that recorded cudaGraphNode_t still usable with
      cudaGraphNodeSetEnabled on the exec?

Outcome decides whether DMI needs vLLM to pass keep_graph=True (a vLLM patch) or
can work with vLLM as-is. See ../node_toggle_phase3_feasibility.md.

Run: python probe_phase3_handle_survival.py
"""
import torch
from cuda.bindings import runtime as cudart


def run(keep_graph: bool):
    a = torch.zeros(256, device="cuda")
    b = torch.zeros(256, device="cuda")
    g = torch.cuda.CUDAGraph(keep_graph=keep_graph)

    recorded = None
    capture_ok = False
    with torch.cuda.graph(g):
        a.add_(1.0)  # the "DMI producer" op -> becomes a kernel node
        # DMI-side hook timing: query the capturing stream right after launch.
        s = torch.cuda.current_stream().cuda_stream
        # cuda-python returns (err, status, id, graph, deps_list, numDeps)
        err, status, _id, _graph, deps, ndeps = cudart.cudaStreamGetCaptureInfo(s)
        capture_ok = (int(err) == 0
                      and status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
                      and ndeps >= 1)
        if capture_ok:
            recorded = deps[-1]   # tail dependency = the a.add_ node just launched
        b.add_(1.0)  # a second op, so disabling a's node is observable

    # torch has now EndCaptured + instantiated; for keep_graph=False the template
    # graph is freed here. Make sure it's instantiated, then grab the exec.
    g.replay(); torch.cuda.synchronize()
    raw_exec = g.raw_cuda_graph_exec()

    def replay():
        a.zero_(); b.zero_(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()

    replay()
    base = (int(a[0].item()), int(b[0].item()))

    # The decisive call: toggle the RECORDED node on the exec AFTER instantiate.
    set_err = int(cudart.cudaGraphNodeSetEnabled(raw_exec, recorded, 0)[0])
    toggled = None
    if set_err == 0:
        replay()
        toggled = (int(a[0].item()), int(b[0].item()))
        cudart.cudaGraphNodeSetEnabled(raw_exec, recorded, 1)  # restore

    # success = recorded node identified a.add_ AND disabling it worked post-instantiate
    ok = (set_err == 0 and toggled == (0, 1))
    return dict(keep_graph=keep_graph, capture_query_ok=capture_ok,
               baseline=base, set_enabled_err=set_err, after_disable=toggled, survives=ok)


def main():
    for kg in (True, False):
        try:
            r = run(kg)
        except Exception as e:
            r = dict(keep_graph=kg, error=repr(e))
        print(r)
    print("\nInterpretation:")
    print("  keep_graph=True survives  + keep_graph=False survives  -> DMI needs NO vLLM change.")
    print("  keep_graph=True survives  + keep_graph=False fails      -> DMI needs vLLM keep_graph=True.")


if __name__ == "__main__":
    main()
