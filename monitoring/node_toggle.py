"""Host-side controller for runtime node-toggle (low-overhead hook configurability).

Owns everything Python-side that the toggle needs beyond the engine bindings:
the activation guards (with their registry-version cache), the effective-specs
recompute (with its preset memo), and the eager/lazy reconfigure entry points.
``RingTransport`` holds one ``NodeToggleController`` lazily and exposes
one-line delegates, so the transport's diff against a toggle-less tree is just
the ``effective_specs`` seam.

The engine's enabled-set remains the single source of truth; this controller
only derives host-side views from it (never the other way around). The lockstep
invariant -- capacity-reserve == meta-push == device node-enable -- is carried
by ``transport.effective_specs``, which reads this controller's precomputed
subset while the gate is active.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with ring_transport
    from monitoring.ring_transport import HookSpec, RingTransport


class NodeToggleController:
    """One per transport. All state defaults to 'toggle inactive'."""

    def __init__(self, engine, transport: "RingTransport") -> None:
        self._eng = engine
        self._transport = transport
        # True once set_active_hooks[_lazy] ran; the transport's
        # effective_specs property gates on this.
        self.gate_active: bool = False
        # True when the active gate uses the LAZY device-apply path. The
        # serving-framework replay guard reads this to decide
        # ensure_graph_current (lazy) vs validate-only (eager).
        self.lazy_active: bool = False
        # The precomputed enabled-AND-captured subset of the transport's
        # active specs (None until first activation). Single source for the
        # transport's effective_specs while the gate is active.
        self.effective_enabled_specs: "Optional[List[HookSpec]]" = None
        self.enabled_version: int = 0   # bumped whenever the enabled set changes
        # Reconfigure caches, both keyed on the engine's registry version
        # (bumped by every registry mutation) so neither can serve stale:
        #   - _guard_valid_version: registry version whose guard validation
        #     PASSED (failures are never cached).
        #   - _cache: (registry_version, enabled frozenset) ->
        #     (active_specs ref, effective list). The stored active_specs
        #     reference is compared with `is` (the strong ref pins the id).
        self._guard_valid_version: Optional[int] = None
        self._cache: dict = {}

    # -- reconfigure entry points ------------------------------------------

    def set_active_hooks(self, enabled: "Iterable[Tuple[int, int]]") -> None:
        """Eager runtime node-toggle: enable exactly `enabled` (hook_type,
        layer_no) pairs and disable the rest, NOW.

        Flips the captured producer nodes on every bound exec (apply_toggle)
        and activates the host meta gate, both driven by the engine's
        enabled-set (single source of truth) so reserve/meta/device stay in
        lockstep.

        CONTRACT: eager apply does not wait on replay events, so call only at
        a quiescent point (no bound-exec replay in flight) -- e.g. once after
        warmup. For dynamic / per-step reconfigure use set_active_hooks_lazy
        (event-guarded). Requires producer nodes registered at capture + execs
        bound.
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

    def set_active_hooks_lazy(self, enabled: "Iterable[Tuple[int, int]]") -> None:
        """Lazy reconfigure: update the enabled set + host meta gate, but
        DEFER the device apply to per-graph ensure_graph_current() (called
        just before each graph replays). set_enabled_hooks bumps the engine's
        target_version, marking every captured graph stale; the device flip
        then happens lazily, only on graphs actually replayed. Same guards +
        gate semantics as set_active_hooks, minus apply_toggle.
        """
        self._validate_registry("set_active_hooks_lazy")
        pairs = [(int(ht), int(ln)) for (ht, ln) in enabled]
        self._eng.set_enabled_hooks(pairs)   # bumps target_version; apply deferred
        self.gate_active = True
        self.lazy_active = True
        self._recompute(pairs)

    def clear(self) -> None:
        """Paired teardown: clear the engine's toggle registry AND deactivate
        the host gate, so neither lane is left half-configured. Call before a
        re-capture or when disabling monitoring."""
        self._eng.clear_toggle_registry()
        self.gate_active = False
        self.lazy_active = False
        self.effective_enabled_specs = None
        self.enabled_version += 1
        self._guard_valid_version = None
        self._cache.clear()

    # -- replay-time hooks (used by the serving-framework replay guard) -----

    def is_graph_ready(self, raw_graph: int) -> bool:
        """Read-only replay-time guard: True iff the graph has recorded
        producer nodes AND a bound exec. False for a graph the serving
        framework captured at runtime after the warmup window closed -> the
        replay hook fails loud."""
        return self._eng.is_graph_ready(raw_graph)

    def ensure_graph_current(self, raw_graph: int) -> int:
        """Lazy: apply the deferred toggle to the graph about to replay (no-op
        if already current). Call right before the graph's replay; raw_graph
        is the cudaGraph_t handle the graph was bound with."""
        return self._eng.ensure_graph_current(raw_graph)

    def record_replay_event(self, raw_graph: int) -> int:
        """Lazy event guard: record a stream event after a graph's replay so a
        later ensure_graph_current() waits for it before mutating that exec.
        Returns the CUDA error (0 = ok); nonzero is FATAL (a missing event
        would let a later ensure mutate an executing exec)."""
        return self._eng.record_replay_event(raw_graph)

    # -- internals -----------------------------------------------------------

    def _validate_registry(self, who: str) -> None:
        """Registry guards shared by set_active_hooks[_lazy], memoized on the
        engine's registry version: the registry mutates only at capture / bind
        / clear time, so re-running the O(graphs x hooks) checks on an
        unchanged registry is wasted work on every reconfigure. Only a PASS is
        cached; every registry mutation bumps the version and forces
        re-validation."""
        eng = self._eng
        version = eng.toggle_registry_version()   # read BEFORE validating
        if version == self._guard_valid_version:
            return
        # Guard: gate active but device toggle a no-op (nothing bound/captured)
        # -> metas filtered while all producers still fire -> desync. Fail loud.
        if eng.bound_graph_count() == 0 or eng.toggle_node_count() == 0:
            raise RuntimeError(
                "%s: no producer nodes registered / no graph exec bound "
                "(toggle_node_count=%d, bound_graph_count=%d). Capture with "
                "enable_toggle_capture(True) and call bind_graph_exec() first."
                % (who, eng.toggle_node_count(), eng.bound_graph_count()))
        # Guard: non-uniform hook sets across graphs -> a hook in one graph but
        # not the replayed one pushes an orphan meta.
        if not eng.toggle_registry_uniform():
            raise RuntimeError(
                "%s: captured graphs have non-uniform hook sets; the "
                "global meta gate cannot stay aligned. Ensure every captured "
                "graph registers the same hooks." % who)
        # Guard: node-registry vs exec-binding key mismatch (partial bind) -> a
        # graph replays default-ON while the meta gate filters -> desync.
        if not eng.toggle_registry_complete():
            raise RuntimeError(
                "%s: toggle registry incomplete -- the set of graphs "
                "with recorded producer nodes does not match the set of bound "
                "execs. Every captured graph must be bound (bind_graph_exec) "
                "and vice versa." % who)
        if eng.capture_anomaly_count() > 0:
            raise RuntimeError(
                "%s: capture recorded %d producer node(s) node-toggle "
                "cannot manage -- either a non-kernel tail dependency (multi-op "
                "producer / capture event-join), or a chunked producer (not "
                "supported under node-toggle; set dmx_gpu_padding_strip=False "
                "to route every hook through the basic producer). Refusing to "
                "activate (fail-closed)." % (who, eng.capture_anomaly_count()))
        self._guard_valid_version = version

    def _recompute(self, pairs) -> None:
        """Recompute the effective-enabled set (active ∩ enabled ∩ registered)
        via ONE batched pybind call. The enabled set changes only here; the
        registered set is fixed after capture. Bumps enabled_version to
        invalidate any capacity cache keyed on it.

        The result is memoized per (registry version, enabled set,
        active_specs identity), so repeated presets (graduated monitoring
        levels, profiler sweep cycles) skip the recompute. The cached
        active_specs reference is compared with `is`; holding it pins the id,
        so identity cannot be reused."""
        active = self._transport._active_specs
        key = (self._eng.toggle_registry_version(), frozenset(pairs))
        hit = self._cache.get(key)
        if hit is not None and hit[0] is active:
            self.effective_enabled_specs = hit[1]
            self.enabled_version += 1
            return
        mask = self._eng.effective_enabled_mask(
            [(s.hook_type, s.layer_no) for s in active])
        self.effective_enabled_specs = [s for s, m in zip(active, mask) if m]
        self.enabled_version += 1
        if len(self._cache) >= 64:   # ad-hoc sets: bound it
            self._cache.clear()
        self._cache[key] = (active, self.effective_enabled_specs)
