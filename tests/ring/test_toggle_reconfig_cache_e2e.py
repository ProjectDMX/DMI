"""Reconfigure caches: correctness of the guard-verdict and effective-specs memos.

set_active_hooks[_lazy] memoize two things on the engine's registry version:
the registry-guard PASS verdict, and the (enabled set -> effective_specs)
recompute. Both caches trade repeated work for a staleness risk, so this gate
pins down the two ways they could go wrong:

  Equivalence -- cycling presets must produce effective_specs identical to an
  uncached recompute, every round (hit or miss), and a changed _active_specs
  list must miss the cache (identity check).

  Invalidation -- a cached guard PASS must NOT survive a registry mutation:
  after note_capture_anomaly() the next reconfigure must RAISE even though the
  same call just succeeded; after clear_toggle + an incomplete re-bind it must
  RAISE the completeness error.

Run:  CUDA_VISIBLE_DEVICES=<free gpu> CUDA_MODULE_LOADING=EAGER \
      python tests/ring/test_toggle_reconfig_cache_e2e.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from monitoring import _native_engine as ne
from monitoring import ring_transport as rt
from monitoring.ring_transport import RingTransport, HookSpec, ModelShapeConfig

HT = rt.HOOK_TYPE_RESID_PRE
N = 8
HID = 8
PRESETS = [[0, 2, 4, 6], [1, 3, 5, 7], [], list(range(N)), [0, 2, 4, 6], [1, 3, 5, 7]]
fails = 0


def check(cond, msg):
    global fails
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        fails += 1


def fresh_effective(transport, keep):
    """Uncached ground truth: active specs whose layer is enabled AND registered."""
    eng = transport._ring_engine
    mask = eng.effective_enabled_mask(
        [(s.hook_type, s.layer_no) for s in transport._active_specs])
    return [s for s, m in zip(transport._active_specs, mask) if m]


def setup():
    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 32 * 1024 * 1024
    eng = ne.RingEngine(cfg, None)
    eng.init(0)
    ne.ring_set_active_engine(eng)
    eng.set_null_mode(True)
    eng.start()
    transport = RingTransport(eng)
    transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2, num_kv_heads=2,
                                             head_dim=HID // 2, dtype=torch.float32))
    transport._active_specs = [HookSpec(hook_type=HT, module=None, layer_no=j)
                               for j in range(N)]
    payload = eng.payload_tensor()
    eng.enable_toggle_capture(True)
    src = [torch.full((4, HID), float(j), device="cuda", dtype=torch.float32)
           for j in range(N)]
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        for j in range(N):
            torch.ops.ring.producer(payload, src[j], HT, j)
    g.instantiate()
    eng.enable_toggle_capture(False)
    eng.bind_graph_exec(g.raw_cuda_graph(), g.raw_cuda_graph_exec())
    return eng, transport, g, src


def main():
    eng, transport, g, _src = setup()

    print("[equivalence: cached presets == fresh recompute]")
    for r_i, keep in enumerate(PRESETS):
        transport.set_active_hooks_lazy([(HT, j) for j in keep])
        got = [(s.hook_type, s.layer_no) for s in transport.effective_specs]
        want = [(s.hook_type, s.layer_no) for s in fresh_effective(transport, keep)]
        check(got == want and sorted(ln for _, ln in got) == sorted(keep),
              f"round {r_i} keep={keep}: effective {sorted(ln for _, ln in got)}")
    check(len(transport._toggle._cache) > 0, "preset cache populated")

    print("[equivalence: _active_specs swap must miss the cache]")
    transport._active_specs = transport._active_specs[:4]   # new list object
    transport.set_active_hooks_lazy([(HT, j) for j in range(N)])  # all-on preset
    got = sorted(s.layer_no for s in transport.effective_specs)
    check(got == [0, 1, 2, 3],
          f"effective follows the NEW active list (got {got}, want [0, 1, 2, 3])")

    print("[invalidation: anomaly after a cached PASS]")
    transport.set_active_hooks([(HT, 0)])     # eager passes; verdict now cached
    eng.note_capture_anomaly()                # registry mutation -> version bump
    raised = False
    try:
        transport.set_active_hooks([(HT, 0)])
    except RuntimeError as e:
        raised = "cannot manage" in str(e)
    check(raised, "set_active_hooks RAISES after anomaly despite cached verdict")

    print("[invalidation: clear + incomplete re-bind]")
    transport.clear_toggle()                  # drops registry + transport caches
    g2 = torch.cuda.CUDAGraph(keep_graph=True)   # captured WITHOUT recording
    payload = eng.payload_tensor()
    src2 = torch.zeros(4, HID, device="cuda")
    with torch.cuda.graph(g2):
        torch.ops.ring.producer(payload, src2, HT, 0)
    g2.instantiate()
    eng.bind_graph_exec(g2.raw_cuda_graph(), g2.raw_cuda_graph_exec())  # bound, no nodes
    raised = False
    try:
        transport.set_active_hooks_lazy([(HT, 0)])
    except RuntimeError as e:
        raised = "no producer nodes registered" in str(e) or "incomplete" in str(e)
    check(raised, "reconfigure after clear+incomplete bind RAISES (no stale PASS)")

    ne.ring_clear_active_engine()
    eng.stop()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
