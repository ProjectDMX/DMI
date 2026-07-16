"""vLLM-side wiring for runtime node-toggle (low-overhead hook configurability).

Everything node-toggle needs from the vLLM integration lives here, so
``vllm_adapter`` keeps only config reads and two call sites behind the
``dmx_node_toggle`` flag:

  - the ``keep_graph`` CUDAGraph patch + the replay-time graph guard,
  - the ``dmx_enabled_hooks`` config parser,
  - the captured-graph bind walker,
  - the post-warmup static activation.
"""
from __future__ import annotations

from typing import Any

import torch

from monitoring import ring_transport

# Armed (True) whenever a node-toggle gate is active (eager OR lazy). The
# keep_graph CUDAGraph.replay() override then runs the replay-time graph guard:
#   - validate the graph is registered+bound (else a runtime-captured graph
#     would replay with default-ON producers while the meta gate filters ->
#     desync); FATAL if not.
#   - lazy mode only: apply the deferred toggle (ensure_graph_current) and check
#     its result; eager mode: validation only (apply happened at config
#     time -- read-only, per the eager/lazy separation).
_DMX_TOGGLE_REPLAY_GUARD = False
_CUDAGRAPH_KEEP_PATCHED = False


def set_replay_guard(on: bool) -> None:
    global _DMX_TOGGLE_REPLAY_GUARD
    _DMX_TOGGLE_REPLAY_GUARD = bool(on)


def _parse_enabled_hooks(s: Any):
    """Parse the dmx_enabled_hooks config into an enabled (ht, layer) list.

    Three distinct meanings (no overloading of the empty string, no
    nonexistent-hook hack):
      - not provided / "" -> None  : toggle gate INACTIVE (all hooks fire)
      - "none"            -> []    : explicit EMPTY set -> toggle-0 (all off)
      - "0:1,0:2"         -> [(0,1),(0,2)] : that specific set

    layer omitted ("14") -> layer_no -1 (global hook).
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None                 # unconfigured -> gate inactive
    if s.lower() == "none":
        return []                   # explicit all-off
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        ht, _, ln = tok.partition(":")
        out.append((int(ht), int(ln if ln != "" else -1)))
    return out or None


def _patch_cudagraph_keep_graph() -> None:
    """Monkeypatch torch.cuda.CUDAGraph to default keep_graph=True.

    vLLM's CUDAGraphWrapper creates ``torch.cuda.CUDAGraph()`` (keep_graph=False),
    which frees the captured template graph after instantiate -> the node handles
    DMI records via cudaStreamGetCaptureInfo would dangle. keep_graph=True keeps
    the template alive (host memory only; no device/latency cost). Idempotent,
    process-global; only applied when node-toggle is enabled.
    """
    global _CUDAGRAPH_KEEP_PATCHED
    if _CUDAGRAPH_KEEP_PATCHED:
        return
    Orig = torch.cuda.CUDAGraph
    if getattr(Orig, "_dmx_keep_graph", False):
        _CUDAGRAPH_KEEP_PATCHED = True
        return

    class _KeepGraphCUDAGraph(Orig):
        # Force keep_graph=True in BOTH __new__ and __init__: vLLM calls
        # ``CUDAGraph()`` with no args, so __init__ (builtin) would otherwise
        # reset keep_graph to its False default after __new__ set it. Subclass
        # (not a factory) so isinstance(x, torch.cuda.CUDAGraph) still holds.
        _dmx_keep_graph = True
        _dmx_lazy_logged = False

        def __new__(cls, *args, **kwargs):
            kwargs["keep_graph"] = True
            return Orig.__new__(cls, **kwargs)

        def __init__(self, *args, **kwargs):
            kwargs["keep_graph"] = True
            super().__init__(**kwargs)

        def replay(self):
            # Replay-time node-toggle guard (eager + lazy). Armed by
            # _DMX_TOGGLE_REPLAY_GUARD when a gate is active; otherwise stock
            # replay (the common case, incl. all non-toggle runs).
            if _DMX_TOGGLE_REPLAY_GUARD:
                t = ring_transport.get_active()
                if t is not None and t.toggle_gate_active:
                    raw = self.raw_cuda_graph()
                    # The graph about to replay MUST be a registered+bound
                    # toggle graph. A graph vLLM captured at RUNTIME (new
                    # batch_descriptor, after the warmup capture window closed)
                    # has no recorded producer nodes and no bound exec -> its
                    # producers run default-ON while the meta gate pushes only the
                    # enabled subset -> desync. before_forward already pushed this
                    # step's metas, so this is FATAL and unrecoverable: raise and
                    # let the worker die. Read-only validation (eager + lazy).
                    if not t.is_graph_ready(raw):
                        raise RuntimeError(
                            f"[DMX] node-toggle FATAL: graph {raw:#x} is about to replay "
                            f"but is NOT registered+bound in the toggle registry -- it was "
                            f"almost certainly captured by vLLM at RUNTIME after DMI closed "
                            f"its capture window at warmup. Its producers run all-ON while "
                            f"the meta gate filters to the enabled subset, which desyncs the "
                            f"ring. The worker MUST terminate; do NOT catch and continue. "
                            f"(Fix: ensure all decode batch_descriptors are captured during "
                            f"warmup, or add runtime graph registration.)")
                    if t.toggle_lazy_active:
                        # Lazy: apply the deferred toggle now; a nonzero
                        # result is FATAL (metas already pushed for the new set).
                        err = t.ensure_graph_current(raw)
                        if err != 0:
                            raise RuntimeError(
                                f"[DMX] node-toggle FATAL: lazy apply (ensure_graph_current) "
                                f"failed with CUDA error {err} for graph {raw:#x}. The meta "
                                f"gate already advanced this step, so replaying would desync "
                                f"the ring irrecoverably. The worker MUST terminate; do NOT "
                                f"catch and continue serving.")
                        super().replay()
                        # Record the replay event; if recording fails the
                        # next ensure could mutate an executing exec (UB) -> FATAL.
                        rerr = t.record_replay_event(raw)
                        if rerr != 0:
                            raise RuntimeError(
                                f"[DMX] node-toggle FATAL: record_replay_event failed with "
                                f"CUDA error {rerr} for graph {raw:#x}. Without a valid replay "
                                f"event the next lazy reconfigure could mutate a still-executing "
                                f"graph exec (UB). The worker MUST terminate.")
                    else:
                        # eager: device apply happened at config time; here we
                        # only validated the graph is bound (read-only).
                        super().replay()
                    if not _KeepGraphCUDAGraph._dmx_lazy_logged:
                        _KeepGraphCUDAGraph._dmx_lazy_logged = True
                        _mode = "lazy" if t.toggle_lazy_active else "eager"
                        print(f"[DMX] node-toggle: replay guard active (mode={_mode})", flush=True)
                    return
            super().replay()

    torch.cuda.CUDAGraph = _KeepGraphCUDAGraph
    _CUDAGRAPH_KEEP_PATCHED = True


def begin_capture_window(engine) -> None:
    """Scope node recording + keep_graph to the warmup capture that follows.
    Clears any stale registry first."""
    engine.clear_toggle_registry()
    engine.enable_toggle_capture(True)
    _patch_cudagraph_keep_graph()


def bind_captured_graphs(model, engine) -> int:
    """Find vLLM's captured CUDA graphs and bind each exec to the toggle
    registry. Returns the number bound. FULL-cudagraph path only (decode);
    the model is wrapped as CUDAGraphWrapper whose concrete_cudagraph_entries
    hold one torch.cuda.CUDAGraph per batch size."""
    try:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
    except Exception:
        return 0
    n = 0
    # Unwrap nested wrappers (UBatchWrapper etc.) to find CUDAGraphWrapper(s).
    seen = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or obj is None:
            continue
        seen.add(id(obj))
        if isinstance(obj, CUDAGraphWrapper):
            for entry in obj.concrete_cudagraph_entries.values():
                g = getattr(entry, "cudagraph", None)
                if g is None:
                    continue
                # A graph not created by the keep_graph patch means vLLM
                # resolved torch.cuda.CUDAGraph before (or without) the
                # patch -- its template is freed on instantiate, so the
                # capture-recorded node handles dangle and SetEnabled on
                # them is UB. Refuse before any handle is touched.
                if not getattr(type(g), "_dmx_keep_graph", False):
                    raise RuntimeError(
                        "[DMX] node-toggle FATAL: a captured CUDA graph was "
                        "created from the UNPATCHED torch.cuda.CUDAGraph "
                        "(keep_graph=False), so its template graph is already "
                        "freed and the recorded producer-node handles dangle. "
                        "vLLM likely binds the CUDAGraph class at import time "
                        "in this version, bypassing _patch_cudagraph_keep_graph. "
                        "Refusing to bind; fix the patch injection point before "
                        "serving with node-toggle.")
                # Don't re-instantiate if an exec already exists (that would
                # destroy it). raw_cuda_graph_exec() raises until the graph
                # is instantiated -> instantiate exactly once in that case.
                try:
                    exec_ptr = g.raw_cuda_graph_exec()
                except RuntimeError:
                    g.instantiate()
                    exec_ptr = g.raw_cuda_graph_exec()
                engine.bind_graph_exec(g.raw_cuda_graph(), exec_ptr)
                n += 1
        inner = getattr(obj, "runnable", None)
        if inner is not None:
            stack.append(inner)
    return n


def activate_after_warmup(model, engine, transport, enabled_hooks, lazy: bool) -> None:
    """Bind each captured graph's exec, close the capture window, and
    (optionally) apply the static enabled set. Call after warmup, before
    serving, so no replay is in flight."""
    n_bound = bind_captured_graphs(model, engine)
    engine.enable_toggle_capture(False)   # close the window
    print(f"[DMX] node-toggle: bound {n_bound} captured graph(s), "
          f"{engine.toggle_node_count()} nodes registered")
    if enabled_hooks is not None:
        if n_bound == 0:
            # Requested a subset but nothing bound (e.g. cudagraph_mode not
            # FULL) -> refuse to serve silently with all hooks on.
            raise RuntimeError(
                "dmx_node_toggle + dmx_enabled_hooks were set but no CUDA graph "
                "was bound. Node-toggle requires cudagraph_mode=FULL for decode "
                "(one full-model graph per batch size). PIECEWISE graphs hold only "
                "a model SUBSET, so different graphs capture different hook subsets "
                "and would fail the uniform-hook-set requirement -- use FULL. "
                "Refusing to serve with node-toggle silently inert.")
        # Arm the replay-time guard for BOTH modes: eager needs the
        # runtime-graph validation too, not just lazy.
        set_replay_guard(True)
        if lazy:
            # Lazy: defer device apply to each graph's first replay.
            transport.set_active_hooks_lazy(enabled_hooks)
            print(f"[DMX] node-toggle: LAZY active hooks set to {enabled_hooks}")
        else:
            transport.set_active_hooks(enabled_hooks)
            print(f"[DMX] node-toggle: active hooks set to {enabled_hooks}")
    elif n_bound == 0:
        print("[DMX] node-toggle: WARNING no graph bound; toggle is inert "
              "(no dmx_enabled_hooks requested, so serving continues all-on).")
