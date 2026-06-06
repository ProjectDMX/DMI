"""Layer-4 unit gate: the node-toggle vLLM wiring that is model-independent.

The full server smoke (launch DMXGPUWorker + dmx_node_toggle, bind N graphs,
serve) requires the vLLM fork whose hooked models match main's 4-arg producer
op -- i.e. the integration/vllm submodule at main's pinned commit (Layer 0).
This test covers the pieces that DON'T need a model:

  - integration.vllm_adapter imports cleanly (wiring/syntax),
  - _parse_enabled_hooks parses the "ht:layer,..." config string,
  - _patch_cudagraph_keep_graph() makes torch.cuda.CUDAGraph default to
    keep_graph=True (so DMI's capture-recorded node handles don't dangle) while
    staying a CUDAGraph subclass, idempotently, and a no-arg CUDAGraph (how vLLM
    builds them) still captures + replays.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_keepgraph_patch.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import integration.vllm_adapter as va

fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def main():
    check(hasattr(va, "DMXGPUWorker") and hasattr(va, "VLLMAdaptor"),
          "vllm_adapter imports (DMXGPUWorker + VLLMAdaptor present)")

    check(va._parse_enabled_hooks("0:32,0:33") == [(0, 32), (0, 33)],
          "_parse_enabled_hooks('0:32,0:33')")
    check(va._parse_enabled_hooks("") is None, "_parse_enabled_hooks('') -> None")
    check(va._parse_enabled_hooks("14") == [(14, -1)],
          "_parse_enabled_hooks('14') -> global layer -1")

    Orig = torch.cuda.CUDAGraph
    va._patch_cudagraph_keep_graph()
    Patched = torch.cuda.CUDAGraph
    check(Patched is not Orig and issubclass(Patched, Orig),
          "keep_graph patch installed as a CUDAGraph subclass")
    va._patch_cudagraph_keep_graph()
    check(torch.cuda.CUDAGraph is Patched, "patch is idempotent")

    # vLLM builds graphs as CUDAGraph() with NO args -> must default keep_graph=True.
    g = torch.cuda.CUDAGraph()
    check(isinstance(g, Orig), "no-arg CUDAGraph still isinstance(torch.cuda.CUDAGraph)")
    s = torch.zeros(8, device="cuda")
    with torch.cuda.graph(g):
        s.add_(1.0)
    g.replay()
    torch.cuda.synchronize()
    check(float(s[0].item()) == 1.0, "no-arg CUDAGraph captures + replays once")
    # raw_cuda_graph() only works when keep_graph=True -> proves the default took.
    try:
        g.raw_cuda_graph()
        check(True, "raw_cuda_graph() available -> keep_graph defaulted True")
    except Exception as e:
        check(False, f"raw_cuda_graph() failed: {e}")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
