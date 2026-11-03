"""Runtime CUDA-graph node toggling for the legacy ring transport.

A monitored step that replays a captured CUDA graph runs every installed
HookPoint's producer as a kernel node of that graph.  Node toggling enables or
disables individual producer nodes between replays with
``cudaGraphNodeSetEnabled`` on the instantiated exec -- no recapture, no
re-instantiation, no model change.  A disabled node costs an empty-node
traversal, so a step pays only for the hooks it enables.

One invariant drives the design.  Every step, three lanes must agree on which
hooks fire:

1. capacity reservation (``BackendAdapter.plan_step``),
2. metadata publication (``RingTransport.pre_push_all_metas``),
3. the device node state (the engine's enabled set).

Otherwise the ring desynchronizes (reserved bytes != written bytes, or
metadata != payloads).  All three read one source of truth,
``RingTransport.specs_for_step()``, which this controller feeds.

The toggle governs only producers inside a replayed graph.  An eager or
prefill step runs producers ungated, so on such a step the integration marks
the transport ungated (``transport.set_gated_step(False)``) and all three lanes
use the full active set.

Two application modes:

* eager (``set_active_hooks``): flip every bound exec now.  Only at a
  quiescent point (no bound exec has a replay in flight), e.g. once after
  warmup.
* lazy (``set_active_hooks_lazy``): update the enabled set and the host gate
  now; each graph is brought up to date by ``ensure_graph_current`` just
  before it replays.  For per-step reconfiguration.

The second half of this module holds framework-neutral CUDA-graph helpers: a
capture window that scopes node recording, a ``torch.cuda.CUDAGraph`` subclass
that keeps the captured template alive and guards replays, and the binding of
captured graphs to the engine's registry.  A serving integration decides which
captures belong to the toggle (for example only its full-model decode graphs)
through the ``exclude_capture`` predicate, and calls ``bind_graph`` for each
graph it wants toggled.

The ``torch.cuda.CUDAGraph`` replacement is process-global, so monitoring
sessions own it: opening a capture window starts (or renews) the session of
that engine.  The session ends when the window closes with no graph bound,
when the controller is cleared, or when the transport is deactivated; once the
last session ends the original class is restored and the replay guard
disarmed.  Each capture is stamped with the session it was captured in and
with whether it captured DMI producers, so the guard validates only graphs of
the active transport's engine that are bound or carry producers: an unrelated
graph replays untouched, and a later session never adopts a graph captured
for another engine.
"""
from __future__ import annotations

import contextlib
import itertools
import weakref
from typing import (TYPE_CHECKING, Any, Callable, Iterable, Iterator, List,
                    Optional, Tuple)

if TYPE_CHECKING:  # avoid a runtime import cycle with the transport
    from .hooks.specs import HookSpec
    from .transport.ring import RingTransport


HookKey = Tuple[int, int]


class NodeToggleController:
    """Host-side toggle state for one transport.  Defaults to inactive."""

    def __init__(self, engine: Any, transport: "RingTransport") -> None:
        self._eng = engine
        self._transport = transport
        # True once set_active_hooks[_lazy] ran; the transport's
        # effective_specs gates on it.
        self.gate_active: bool = False
        # True when the active gate uses the lazy device-apply path.  The
        # replay guard reads it to choose ensure_graph_current (lazy) over
        # validation only (eager).
        self.lazy_active: bool = False
        # The enabled-and-captured subset of the transport's active specs, in
        # active order (None until the first activation).
        self.effective_enabled_specs: "Optional[List[HookSpec]]" = None
        # Bumped whenever the enabled set changes.
        self.enabled_version: int = 0
        # Reconfiguration caches, both keyed on the engine's registry version
        # (bumped by every registry mutation), so neither can serve stale:
        #   _guard_valid_version: registry version whose validation PASSED
        #     (failures are never cached);
        #   _cache: (registry version, enabled frozenset) ->
        #     (active_specs reference, effective list).  The stored reference
        #     is compared with ``is``; holding it pins the id.
        self._guard_valid_version: Optional[int] = None
        self._cache: dict = {}
        # active_specs reference -> {(hook_type, layer): [active index, ...]},
        # so a reconfiguration queries only the enabled pairs.
        self._active_index: Optional[tuple] = None

    # -- reconfiguration ---------------------------------------------------

    def set_active_hooks(self, enabled: Iterable[HookKey]) -> None:
        """Enable exactly ``enabled`` (hook_type, layer_no) pairs and disable
        the rest, on every bound exec, now.

        Eager apply does not wait for replays in flight: call it only at a
        quiescent point.  For per-step reconfiguration use
        :meth:`set_active_hooks_lazy`.  Requires producer nodes recorded at
        capture and the graphs bound (see :func:`bind_graph`).
        """
        self._validate_registry("set_active_hooks")
        pairs = [(int(ht), int(ln)) for (ht, ln) in enabled]
        self._eng.set_enabled_hooks(pairs)
        err = self._eng.apply_toggle()
        if err != 0:
            raise RuntimeError(f"apply_toggle failed with CUDA error {err}")
        self.gate_active = True
        self.lazy_active = False
        self._recompute(pairs)

    def set_active_hooks_lazy(self, enabled: Iterable[HookKey]) -> None:
        """Update the enabled set and the host gate now, and defer the device
        apply to :meth:`ensure_graph_current`, called just before each graph
        replays.  Only graphs that actually replay are touched."""
        self._validate_registry("set_active_hooks_lazy")
        pairs = [(int(ht), int(ln)) for (ht, ln) in enabled]
        self._eng.set_enabled_hooks(pairs)   # marks every graph stale
        self.gate_active = True
        self.lazy_active = True
        self._recompute(pairs)

    def clear(self) -> None:
        """Clear the engine's registry AND deactivate the host gate, so
        neither lane is left half-configured.  Call before a recapture or
        when monitoring stops.  Ends the engine's monitoring session, so the
        process-wide CUDAGraph patch is restored once no session remains."""
        self._eng.clear_toggle_registry()
        _forget_window(self._eng)
        end_session(self._eng)
        self.gate_active = False
        self.lazy_active = False
        self.effective_enabled_specs = None
        self.enabled_version += 1
        self._guard_valid_version = None
        self._cache.clear()
        self._active_index = None

    # -- replay-time hooks -------------------------------------------------

    def is_graph_ready(self, raw_graph: int) -> bool:
        """The graph has recorded producer nodes AND a bound exec."""
        return self._eng.is_graph_ready(raw_graph)

    def ensure_graph_current(self, raw_graph: int) -> int:
        """Lazy: apply the deferred toggle to the graph about to replay
        (no-op when current).  Returns a CUDA error code; nonzero is fatal."""
        return self._eng.ensure_graph_current(raw_graph)

    def record_replay_event(self, raw_graph: int) -> int:
        """Record the replay event used by the optional pre-apply wait.
        Returns a CUDA error code; nonzero is fatal."""
        return self._eng.record_replay_event(raw_graph)

    # -- internals ---------------------------------------------------------

    def _validate_registry(self, who: str) -> None:
        """Registry guards for set_active_hooks[_lazy], memoized on the
        engine's registry version.  The registry changes only at capture,
        bind and clear time, so re-running the checks on an unchanged
        registry is wasted work.  Only a pass is cached."""
        eng = self._eng
        version = eng.toggle_registry_version()   # read before validating
        if version == self._guard_valid_version:
            return
        # Gate active but the device toggle a no-op: metadata filtered while
        # every producer still fires.
        if eng.bound_graph_count() == 0 or eng.toggle_node_count() == 0:
            raise RuntimeError(
                "%s: no producer nodes registered or no graph exec bound "
                "(toggle_node_count=%d, bound_graph_count=%d). Capture inside "
                "a toggle capture window and bind the captured graphs first."
                % (who, eng.toggle_node_count(), eng.bound_graph_count()))
        # A hook in one graph but not in the replayed one would push metadata
        # with no payload.
        if not eng.toggle_registry_uniform():
            raise RuntimeError(
                "%s: captured graphs registered different hook sets; the "
                "global metadata gate cannot stay aligned. Every toggled "
                "graph must register the same hooks (exclude partial-model "
                "captures from the capture window)." % who)
        # A captured graph without a bound exec replays with every node on
        # while the gate filters.
        if not eng.toggle_registry_complete():
            raise RuntimeError(
                "%s: toggle registry incomplete -- the graphs with recorded "
                "producer nodes are not exactly the bound graphs. Every "
                "captured graph must be bound, and vice versa." % who)
        if eng.capture_anomaly_count() > 0:
            raise RuntimeError(
                "%s: capture recorded %d producer node(s) the toggle cannot "
                "manage (a non-kernel tail dependency, unreadable capture "
                "state, or a chunked producer, which the toggle does not "
                "manage). Refusing to activate."
                % (who, eng.capture_anomaly_count()))
        self._guard_valid_version = version

    def _recompute(self, pairs: List[HookKey]) -> None:
        """Recompute active AND enabled AND captured with one batched engine
        call.  Memoized per (registry version, enabled set, active_specs
        identity), so repeated presets skip the recompute."""
        active = self._transport._active_specs
        key = (self._eng.toggle_registry_version(), frozenset(pairs))
        hit = self._cache.get(key)
        if hit is not None and hit[0] is active:
            self.effective_enabled_specs = hit[1]
            self.enabled_version += 1
            return
        # The gate is active here, so the engine's mask is "enabled AND
        # registered": a spec outside `pairs` is never effective.  Query just
        # the active specs whose key is enabled, in active order.
        index = self._active_index
        if index is None or index[0] is not active:
            by_key: dict = {}
            for i, s in enumerate(active):
                by_key.setdefault((s.hook_type, s.layer_no), []).append(i)
            index = self._active_index = (active, by_key)
        by_key = index[1]
        positions = sorted(i for k in key[1] for i in by_key.get(k, ()))
        mask = self._eng.effective_enabled_mask(
            [(active[i].hook_type, active[i].layer_no) for i in positions])
        self.effective_enabled_specs = [active[i] for i, m in zip(positions, mask) if m]
        self.enabled_version += 1
        if len(self._cache) >= 64:   # ad hoc sets: keep the memo bounded
            self._cache.clear()
        self._cache[key] = (active, self.effective_enabled_specs)


def parse_enabled_hooks(value: Any) -> Optional[List[HookKey]]:
    """Parse an enabled-hooks setting into (hook_type, layer_no) pairs.

    * ``None`` or ``""`` -> ``None``: gate inactive, every hook fires.
    * ``"none"`` -> ``[]``: an explicit empty set, every hook off.
    * ``"0:1,0:2"`` -> ``[(0, 1), (0, 2)]``.  A token without a layer
      (``"14"``) is a global hook (layer ``-1``).
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() == "none":
        return []
    out: List[HookKey] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        hook_type, _, layer = token.partition(":")
        out.append((int(hook_type), int(layer) if layer != "" else -1))
    return out or None


# ---------------------------------------------------------------------------
# CUDA-graph helpers
# ---------------------------------------------------------------------------

class _CaptureWindow:
    engine: Any = None
    exclude_capture: Optional[Callable[[], bool]] = None
    excluded: int = 0


class _Session:
    """One monitoring session of one engine: an owner of the process-wide
    CUDAGraph patch while it is open.  Holds the engine weakly where it can
    (an ended session stamped on a long-lived graph must not pin the ring)."""

    _tokens = itertools.count(1)

    def __init__(self, engine: Any) -> None:
        self.token = next(self._tokens)
        self.engine_id = id(engine)
        try:
            self._engine_ref: Optional[Callable[[], Any]] = weakref.ref(engine)
        except TypeError:
            self._engine_ref = None
        # raw cudaGraph_t handles bound to the engine in this session.
        self.bound: set = set()

    def engine(self) -> Any:
        return self._engine_ref() if self._engine_ref is not None else None

    def owns_engine(self, engine: Any) -> bool:
        if engine is None:
            return False
        if self._engine_ref is not None:
            return self._engine_ref() is engine
        return self.engine_id == id(engine)


_WINDOW = _CaptureWindow()
_REPLAY_GUARD = False
# The patch: the class it replaced (restored when the last session ends) and
# the toggle-aware subclass (built once, so graphs of every session share one
# class).  _PATCHED_CLASS is set while the patch is installed.
_ORIG_CUDAGRAPH: Optional[type] = None
_TOGGLE_CLASS: Optional[type] = None
_PATCHED_CLASS: Optional[type] = None
# Open monitoring sessions, keyed by id(engine).
_SESSIONS: dict = {}


def set_replay_guard(enabled: bool) -> None:
    """Arm or disarm the replay-time guard of graphs created by the patched
    ``torch.cuda.CUDAGraph`` (see :func:`install_cudagraph_patch`)."""
    global _REPLAY_GUARD
    _REPLAY_GUARD = bool(enabled)


def excluded_capture_count() -> int:
    """Captures excluded from the registry since the window opened."""
    return _WINDOW.excluded


def _session_for(engine: Any) -> Optional[_Session]:
    session = _SESSIONS.get(id(engine)) if engine is not None else None
    if session is not None and not session.owns_engine(engine):
        return None   # a dead engine's id reused by a new object
    return session


def _start_session(engine: Any) -> _Session:
    """Begin a fresh session for `engine`.  A new capture window replaces the
    engine's previous session (the window clears its registry)."""
    session = _Session(engine)
    _SESSIONS[id(engine)] = session
    return session


def end_session(engine: Any) -> None:
    """End `engine`'s monitoring session.  When no session remains, restore
    the original ``torch.cuda.CUDAGraph`` and disarm the replay guard.
    Graphs created while patched keep their class; with the guard disarmed
    they replay exactly like unpatched graphs."""
    if engine is not None:
        _SESSIONS.pop(id(engine), None)
    if not _SESSIONS:
        restore_cudagraph_patch()


def active_session_count() -> int:
    """Open monitoring sessions (owners of the CUDAGraph patch)."""
    return len(_SESSIONS)


def _probe_engine() -> Any:
    from .transport import ring as _rt
    engine = getattr(_rt.get_active(), "_ring_engine", None)
    if engine is not None:
        return engine
    for session in list(_SESSIONS.values()):
        engine = session.engine()
        if engine is not None:
            return engine
    return None


def _launch_count(engine: Any) -> Optional[int]:
    getter = getattr(engine, "producer_launch_count", None)
    if not callable(getter):
        return None
    return int(getter())


def _node_count(engine: Any) -> Optional[int]:
    getter = getattr(engine, "toggle_node_count", None)
    if not callable(getter):
        return None
    return int(getter())


def _weak(engine: Any) -> Callable[[], Any]:
    try:
        return weakref.ref(engine)
    except TypeError:
        return lambda: engine


def _engine_of(ref: Any) -> Any:
    return ref() if ref is not None else None


def _current_exec(graph: Any) -> int:
    """The graph's exec, instantiating it first when the template is kept and
    not instantiated yet (keep_graph=True defers instantiation)."""
    try:
        return graph.raw_cuda_graph_exec()
    except RuntimeError:
        graph.instantiate()
        return graph.raw_cuda_graph_exec()


def _upload_for_device_toggle(graph: Any, engine: Any) -> None:
    """Instantiate if needed and upload the graph's exec before its first
    launch (device-updatable producers require it), once per exec.  Excluded
    captures go through here too: uploading registers nothing.  The engine
    uploads only execs whose graph captured device-updatable producers."""
    exec_ptr = _current_exec(graph)
    if exec_ptr and exec_ptr != getattr(graph, "_dmi_uploaded_exec", None):
        err = engine.upload_graph_exec(graph.raw_cuda_graph(), exec_ptr)
        if err != 0:
            raise RuntimeError(
                f"node toggle: cudaGraphUpload failed ({err})")
        graph._dmi_uploaded_exec = exec_ptr


def install_cudagraph_patch() -> type:
    """Replace ``torch.cuda.CUDAGraph`` with a toggle-aware subclass.

    The subclass forces ``keep_graph=True`` -- a framework that creates
    ``torch.cuda.CUDAGraph()`` with the default frees the captured template
    after instantiation, so the node handles recorded during capture would
    dangle.  Keeping the template costs host memory only.  It also

    * stamps each capture with the monitoring session it was captured in and
      whether it captured DMI producers;
    * excludes a capture from the registry when the capture window's
      ``exclude_capture`` predicate says so (node recording is suspended for
      that capture and the graph is tagged ``_dmi_toggle_excluded``);
    * in device-toggle mode, instantiates and uploads the exec of a capture
      that may hold producers before its first launch (required for
      device-updatable nodes), excluded captures included, and uploads a
      re-instantiated exec before it replays;
    * guards replays while :func:`set_replay_guard` is armed and a gate is
      active (see ``replay``).

    Idempotent.  Install it before the framework creates its graphs;
    :func:`bind_graph` refuses graphs of the unpatched class.  The patch is
    owned by the monitoring sessions (:func:`begin_capture_window`): when the
    last one ends, :func:`restore_cudagraph_patch` puts the original class
    back.
    """
    global _PATCHED_CLASS, _ORIG_CUDAGRAPH, _TOGGLE_CLASS
    import torch

    current = torch.cuda.CUDAGraph
    if _PATCHED_CLASS is not None and current is _PATCHED_CLASS:
        return current
    if getattr(current, "_dmi_keep_graph", False):
        # Already a keep-graph class (installed by another copy of DMI).
        _PATCHED_CLASS = current
        return current
    if _TOGGLE_CLASS is not None and current is _ORIG_CUDAGRAPH:
        torch.cuda.CUDAGraph = _TOGGLE_CLASS
        _PATCHED_CLASS = _TOGGLE_CLASS
        return _TOGGLE_CLASS
    original = current

    class ToggleCUDAGraph(original):
        # keep_graph is forced in both __new__ and __init__: CUDAGraph() with
        # no arguments would otherwise reset it to False in __init__.  A
        # subclass (not a factory) keeps isinstance checks working.
        _dmi_keep_graph = True

        def __new__(cls, *args, **kwargs):
            kwargs["keep_graph"] = True
            return original.__new__(cls, **kwargs)

        def __init__(self, *args, **kwargs):
            kwargs["keep_graph"] = True
            super().__init__(**kwargs)

        def capture_begin(self, *args, **kwargs):
            engine = _WINDOW.engine
            predicate = _WINDOW.exclude_capture
            # Owner: the session whose window is open now (None outside any
            # window).  Producer detection reads the process-wide
            # producer-launch counter (through the window's engine, else the
            # active transport's or any session's) and, inside the window,
            # the nodes recorded for this graph.
            self._dmi_owner = _session_for(engine)
            probe = engine if engine is not None else _probe_engine()
            self._dmi_probe = _weak(probe) if probe is not None else None
            self._dmi_launches0 = _launch_count(probe)
            self._dmi_nodes0 = _node_count(engine)
            self._dmi_producers = None
            self._dmi_toggle_excluded = bool(
                engine is not None and predicate is not None and predicate())
            self._dmi_toggle_suspended = False
            if self._dmi_toggle_excluded and engine.toggle_capture_enabled():
                engine.enable_toggle_capture(False)
                self._dmi_toggle_suspended = True
                _WINDOW.excluded += 1
            try:
                super().capture_begin(*args, **kwargs)
            except BaseException:
                # capture_end will not run: restore node recording now, or the
                # rest of the window would record nothing.
                if self._dmi_toggle_suspended:
                    self._dmi_toggle_suspended = False
                    engine.enable_toggle_capture(True)
                raise

        def capture_end(self, *args, **kwargs):
            engine = _WINDOW.engine
            try:
                super().capture_end(*args, **kwargs)
            finally:
                if getattr(self, "_dmi_toggle_suspended", False):
                    self._dmi_toggle_suspended = False
                    if engine is not None:
                        engine.enable_toggle_capture(True)
            self._dmi_producers = _captured_producers(self, engine)
            if (engine is not None and engine.toggle_device_mode()
                    and self._dmi_producers is not False):
                # keep_graph=True defers instantiation to the first replay,
                # which would launch device-updatable producers without the
                # required upload.  Instantiate now (what an unpatched graph
                # does at capture end) and upload; remember the engine so a
                # re-instantiated exec is uploaded before it replays.
                self._dmi_upload_engine = _weak(engine)
                _upload_for_device_toggle(self, engine)

        def replay(self):
            upload_engine = _engine_of(getattr(self, "_dmi_upload_engine", None))
            if upload_engine is not None:
                _upload_for_device_toggle(self, upload_engine)
            if _REPLAY_GUARD:
                from .transport import ring as _rt
                transport = _rt.get_active()
                if (transport is not None and transport.toggle_gate_active
                        and _guard_applies(self, transport)):
                    _guarded_replay(self, transport, super().replay)
                    return
            super().replay()

    ToggleCUDAGraph.__name__ = "CUDAGraph"
    ToggleCUDAGraph.__qualname__ = "CUDAGraph"
    torch.cuda.CUDAGraph = ToggleCUDAGraph
    _ORIG_CUDAGRAPH = original
    _TOGGLE_CLASS = ToggleCUDAGraph
    _PATCHED_CLASS = ToggleCUDAGraph
    return ToggleCUDAGraph


def restore_cudagraph_patch() -> None:
    """Put the original ``torch.cuda.CUDAGraph`` back and disarm the replay
    guard.  Called when the last monitoring session ends; graphs created
    while patched keep their class, and their guard stays inert."""
    global _PATCHED_CLASS, _REPLAY_GUARD
    _REPLAY_GUARD = False
    if _PATCHED_CLASS is None:
        return
    import torch

    if (_ORIG_CUDAGRAPH is not None and _PATCHED_CLASS is _TOGGLE_CLASS
            and torch.cuda.CUDAGraph is _PATCHED_CLASS):
        torch.cuda.CUDAGraph = _ORIG_CUDAGRAPH
    _PATCHED_CLASS = None


def _captured_producers(graph: Any, window_engine: Any) -> Optional[bool]:
    """Whether the capture that just ended recorded DMI producers.

    True or False when it can be told, None when it cannot (callers treat
    None like True).  Two independent sources of evidence, and positive
    evidence from either one wins:

    * the producer-launch counter, sampled across the capture, answers for
      any capture, excluded or outside the window;
    * the producer nodes the registry recorded for this graph (inside the
      window only; an excluded capture records none).

    An unchanged counter never overrides recorded nodes.  Without positive
    evidence, the answer is False when either source could answer (a capture
    inside the window that recorded no node has no toggle-managed producer;
    an unmanaged one is a capture anomaly, which already blocks activation),
    and None when neither could.
    """
    counted: Optional[bool] = None
    probe = _engine_of(getattr(graph, "_dmi_probe", None))
    before = getattr(graph, "_dmi_launches0", None)
    if probe is not None and before is not None:
        after = _launch_count(probe)
        if after is not None:
            if after != before:
                return True
            counted = False
    recorded = _recorded_producer_nodes(graph, window_engine)
    if recorded:
        return True
    if counted is False:
        return False
    return recorded


def _recorded_producer_nodes(graph: Any, window_engine: Any) -> Optional[bool]:
    """Whether the registry recorded producer nodes for this capture; None
    when it cannot tell (no window, or an excluded capture)."""
    if window_engine is None or getattr(graph, "_dmi_toggle_excluded", False):
        return None
    counter = getattr(window_engine, "toggle_node_count_for_graph", None)
    if counter is not None:
        return int(counter(graph.raw_cuda_graph())) > 0
    nodes0 = getattr(graph, "_dmi_nodes0", None)
    nodes1 = _node_count(window_engine)
    if nodes0 is None or nodes1 is None:
        return None
    return nodes1 != nodes0


def _guard_applies(graph: Any, transport: Any) -> bool:
    """Whether the active transport's gate governs this graph's replay.

    No for a graph stamped with another engine's session (a later session
    never adopts it) and for a graph that captured no DMI producer and is not
    bound to the transport's engine -- an unrelated CUDA-graph user in the
    process.  Yes, so the fail-closed checks run, for every bound graph and
    every graph that captured producers or may have.
    """
    engine = getattr(transport, "_ring_engine", None)
    owner = getattr(graph, "_dmi_owner", None)
    if owner is not None and engine is not None and not owner.owns_engine(engine):
        return False
    session = _session_for(engine)
    if session is not None and session.bound and graph.raw_cuda_graph() in session.bound:
        return True
    return getattr(graph, "_dmi_producers", None) is not False


def _guarded_replay(graph: Any, transport: Any, replay: Callable[[], None]) -> None:
    """Replay with the toggle's consistency checks.  Every failure here is
    fatal: this step's metadata has already been published, so replaying
    anyway would desynchronize the ring."""
    if getattr(graph, "_dmi_toggle_excluded", False):
        # Excluded from the registry by design: its producers run all on,
        # which is right exactly on an ungated step, where the full active
        # set's metadata was pushed.
        if transport.gated_step_requires_registry():
            raise RuntimeError(
                "node toggle: a graph excluded from the toggle registry is "
                "about to replay on a step classified as gated. Its producers "
                "run all on while the metadata gate published only the "
                "enabled subset, so the ring would desynchronize. The "
                "integration's step classification disagrees with the "
                "framework's graph dispatch. The worker must terminate.")
        replay()
        return
    raw = graph.raw_cuda_graph()
    rebound = _rebind_if_reinstantiated(graph, transport, raw)
    # A graph captured after the capture window closed has no recorded nodes
    # and no bound exec: its producers run default-on while the gate filters.
    if not transport.is_graph_ready(raw):
        raise RuntimeError(
            f"node toggle: graph {raw:#x} is about to replay but is not "
            "registered and bound in the toggle registry -- it was most "
            "likely captured after the capture window closed. Its producers "
            "would run all on while the metadata gate filters to the enabled "
            "subset. The worker must terminate.")
    if rebound and not transport.toggle_lazy_active:
        # Eager mode applies only at set_active_hooks time, to the execs bound
        # then.  A freshly bound exec starts with every producer node in its
        # instantiation default (on), so bring it to the enabled set now,
        # before its first replay.  Nothing of the new exec is in flight yet.
        err = transport.ensure_graph_current(raw)
        if err != 0:
            raise RuntimeError(
                f"node toggle: applying the enabled set to re-instantiated "
                f"graph {raw:#x} failed with CUDA error {err}; this step's "
                "metadata is already published. The worker must terminate.")
    if transport.toggle_lazy_active:
        err = transport.ensure_graph_current(raw)
        if err != 0:
            raise RuntimeError(
                f"node toggle: lazy apply failed with CUDA error {err} for "
                f"graph {raw:#x}; this step's metadata is already published. "
                "The worker must terminate.")
        replay()
        err = transport.record_replay_event(raw)
        if err != 0:
            raise RuntimeError(
                f"node toggle: record_replay_event failed with CUDA error "
                f"{err} for graph {raw:#x}. The worker must terminate.")
        return
    replay()


def _rebind_if_reinstantiated(graph: Any, transport: Any, raw: int) -> bool:
    """A bound graph re-instantiated since bind_graph has a new exec: bind it
    before it replays.  The registry then resets that graph's per-exec state,
    so the next ensure_graph_current sets every node of the new exec.

    Returns True when a new exec was bound.  A graph is bound to exactly one
    engine (``_dmi_bound_engine``), so at most one binding changes here."""
    bound = getattr(graph, "_dmi_bound_exec", None)
    if bound is None:
        return False
    engine = getattr(transport, "_ring_engine", None)
    if engine is None or _engine_of(getattr(graph, "_dmi_bound_engine", None)) is not engine:
        return False
    exec_ptr = _current_exec(graph)
    if exec_ptr == bound:
        return False
    engine.bind_graph_exec(raw, exec_ptr)
    graph._dmi_bound_exec = exec_ptr
    graph._dmi_uploaded_exec = exec_ptr
    return True


def begin_capture_window(
    engine: Any,
    *,
    exclude_capture: Optional[Callable[[], bool]] = None,
    device_toggle: bool = False,
) -> None:
    """Open the capture window: clear any stale registry, record producer
    nodes from now on, start the engine's monitoring session and install the
    CUDA-graph patch it owns.

    ``exclude_capture`` is called at the start of every capture; returning
    True keeps that capture out of the registry (for example a framework's
    partial-model segment graphs, which would make the registry non-uniform).
    ``device_toggle`` asks the engine for device-side toggling; it stays off
    when the device or build lacks device graph updates.
    """
    engine.clear_toggle_registry()
    # Explicit either way: the mode is a per-window choice.
    engine.enable_device_toggle(bool(device_toggle))
    engine.enable_toggle_capture(True)
    _WINDOW.engine = engine
    _WINDOW.exclude_capture = exclude_capture
    _WINDOW.excluded = 0
    _start_session(engine)
    install_cudagraph_patch()


def end_capture_window(engine: Any) -> None:
    """Close the capture window (bind graphs before or after, not during a
    capture).  A window closed with no graph bound ends the engine's session
    (restoring the CUDAGraph patch if it was the last); binding a graph
    afterwards renews it."""
    engine.enable_toggle_capture(False)
    _forget_window(engine)
    session = _session_for(engine)
    if session is not None and not session.bound:
        end_session(engine)


@contextlib.contextmanager
def capture_window(
    engine: Any,
    *,
    exclude_capture: Optional[Callable[[], bool]] = None,
    device_toggle: bool = False,
) -> Iterator[None]:
    """``begin_capture_window`` ... ``end_capture_window``, closed also when
    the body raises (so the window never keeps a reference to the engine)."""
    begin_capture_window(engine, exclude_capture=exclude_capture,
                         device_toggle=device_toggle)
    try:
        yield
    finally:
        end_capture_window(engine)


def _forget_window(engine: Any) -> None:
    """Drop the capture window's reference to `engine` (its ring buffers stay
    pinned while the window holds it)."""
    if _WINDOW.engine is engine:
        _WINDOW.engine = None
        _WINDOW.exclude_capture = None


def bind_graph(engine: Any, graph: Any) -> None:
    """Bind one captured ``torch.cuda.CUDAGraph`` to the engine's registry.

    Instantiates the graph if it has no exec yet.  Refuses a graph created
    from an unpatched ``torch.cuda.CUDAGraph``: its template was freed at
    instantiation, so the recorded node handles dangle.
    """
    if not getattr(type(graph), "_dmi_keep_graph", False):
        raise RuntimeError(
            "node toggle: the graph was created from an unpatched "
            "torch.cuda.CUDAGraph (keep_graph=False), so its template is "
            "already freed and the recorded producer-node handles dangle. "
            "Call install_cudagraph_patch() (or begin_capture_window) before "
            "the framework creates its graphs.")
    if getattr(graph, "_dmi_toggle_excluded", False):
        raise RuntimeError(
            "node toggle: refusing to bind a capture excluded from the "
            "registry")
    exec_ptr = _current_exec(graph)
    raw = graph.raw_cuda_graph()
    engine.bind_graph_exec(raw, exec_ptr)   # uploads a device-toggle exec
    graph._dmi_bound_exec = exec_ptr
    graph._dmi_bound_engine = _weak(engine)
    graph._dmi_uploaded_exec = exec_ptr
    session = _session_for(engine)
    if session is None:
        # Bound after a window that closed with nothing bound: renew the
        # session the graph was captured in (or a new one).
        owner = getattr(graph, "_dmi_owner", None)
        if owner is not None and owner.owns_engine(engine):
            _SESSIONS[id(engine)] = session = owner
        else:
            session = _start_session(engine)
        install_cudagraph_patch()
    session.bound.add(raw)


def activate(
    transport: Any,
    enabled_hooks: Optional[Iterable[HookKey]],
    *,
    lazy: bool,
) -> None:
    """Apply a static enabled set after capture and arm the replay guard.

    ``enabled_hooks=None`` leaves the gate inactive (every hook fires).  A
    requested set with no bound graph is refused rather than served with the
    toggle silently inert.
    """
    if enabled_hooks is None:
        return
    enabled = list(enabled_hooks)
    if transport._ring_engine.bound_graph_count() == 0:
        raise RuntimeError(
            "node toggle: an enabled-hook set was requested but no captured "
            "graph is bound, so the toggle would be silently inert.")
    set_replay_guard(True)
    if lazy:
        transport.set_active_hooks_lazy(enabled)
    else:
        transport.set_active_hooks(enabled)


__all__ = [
    "NodeToggleController",
    "activate",
    "active_session_count",
    "begin_capture_window",
    "bind_graph",
    "capture_window",
    "end_capture_window",
    "end_session",
    "excluded_capture_count",
    "install_cudagraph_patch",
    "parse_enabled_hooks",
    "restore_cudagraph_patch",
    "set_replay_guard",
]
