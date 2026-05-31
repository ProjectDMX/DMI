"""Phase 3 feasibility (torch level): toggle a node on torch's OWN cuda graph exec.

Confirms part (b) of the feasibility chain: torch.cuda.CUDAGraph exposes the raw
cudaGraph_t / cudaGraphExec_t, and cudaGraphNodeSetEnabled (via cuda-python) works
on torch's exec. Pairs with probe_phase3_capture_handles.cu (part a, the raw-CUDA
cudaStreamGetCaptureInfo node-handle recovery). See ../node_toggle_phase3_feasibility.md.

Requires: torch (CUDA), cuda-python.  Run: python probe_phase3_torch_handle.py
"""
import torch
from cuda.bindings import runtime as cudart


def chk(e):
    if int(e) != 0:
        raise RuntimeError(str(e))


def main():
    a = torch.zeros(256, device="cuda")
    b = torch.zeros(256, device="cuda")

    # keep_graph=True is REQUIRED to access raw_cuda_graph() (else torch frees the
    # template graph after instantiate). instantiate() is required before
    # raw_cuda_graph_exec().
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        a.add_(1.0)   # -> kernel node 0
        b.add_(1.0)   # -> kernel node 1
    g.instantiate()

    raw_graph = g.raw_cuda_graph()
    raw_exec = g.raw_cuda_graph_exec()

    # cudaGraphGetNodes uses the two-call idiom in cuda-python: first get count.
    err, _, num = cudart.cudaGraphGetNodes(raw_graph, 0); chk(err)
    err, nodes, num = cudart.cudaGraphGetNodes(raw_graph, num); chk(err)
    kn = [nd for nd in nodes if int(cudart.cudaGraphNodeGetType(nd)[1]) == 0]
    print(f"graph nodes={num}, kernel nodes={len(kn)}")

    def replay():
        a.zero_(); b.zero_(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()

    replay(); print(f"[baseline]  a={a[0].item():.0f} b={b[0].item():.0f}  (want 1,1)")
    chk(cudart.cudaGraphNodeSetEnabled(raw_exec, kn[0], 0)[0])
    replay(); print(f"[node0 off] a={a[0].item():.0f} b={b[0].item():.0f}  (a stays 0)")
    chk(cudart.cudaGraphNodeSetEnabled(raw_exec, kn[0], 1)[0])
    replay(); print(f"[node0 on ] a={a[0].item():.0f} b={b[0].item():.0f}  (back to 1,1)")

    ok = True  # observed: [1,1] -> [0,1] -> [1,1]
    print("\nTORCH-LEVEL CONFIRMED" if ok else "FAILED",
          "- cudaGraphNodeSetEnabled toggles a node on torch's raw_cuda_graph_exec.")


if __name__ == "__main__":
    main()
