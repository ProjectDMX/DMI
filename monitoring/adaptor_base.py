"""BackendAdaptor: abstract base for framework-specific monitoring adapters.

Carved out as part of the unified-adaptor refactor (Phase 1).  Each concrete
adapter (HFAdaptor, VLLMAdaptor -- Phases 2 and 3) will live under
``integration/`` and convert framework-specific batch state into the
``RingTransport`` step protocol.

The driver in ``before_forward`` is the canonical per-step flow shared by
every concrete adapter:

    build_step_context  (subclass)
    -> plan_step        (base; walks active_specs once for
                         total_bytes + hook_count + needs_eager)
    -> commit_step      (base)
       -> prepare_step  (skipped when hook_count == 0)
       on result == 2:
         -> adapt_for_cpu_direct  (subclass; default no-op)
         -> on_capacity_exceeded  (subclass)
         -> _warn_once_capacity   (base stub; subclass overrides)
    set transport.force_eager from (result == 2) OR needs_eager.
    -> set_step_context
    -> pre_push_all_metas

Phase 1 ships the abstraction only -- no concrete subclass is wired in yet.
The unit-test gate (``tests/test_adapter_protocol.py``) exercises the driver
ordering with mocks.
"""
from __future__ import annotations

import abc
from enum import IntEnum
from typing import List, NamedTuple, Optional, TYPE_CHECKING

import torch

from .ring_transport import (
    HookSpec,
    ModelShapeConfig,
    _compute_hook_shape,
    align_up_py,
    install_ring_hooks,
)
from .selection import (
    apply_hook_selection,
    filter_by_pp_rank,
    filter_by_tp_rank,
)
from .step_context import StepContext

if TYPE_CHECKING:
    from .engine import MonitoringEngine


class StepPlan(NamedTuple):
    """Immutable, tuple-compatible reservation inputs for one model step."""

    total_bytes: int
    hook_count: int
    needs_eager: bool


class StepReservation(IntEnum):
    """Outcome of committing a :class:`StepPlan` to the ring."""

    SKIPPED = -1
    RESERVED = 0
    FLUSHED = 1
    OVERSIZED = 2


class BackendAdaptor(abc.ABC):
    """One adapter per attached model.

    Subclasses implement framework-specific shape detection, parallel-rank
    detection, and per-step context construction.  The base class owns the
    driver (``before_forward``) that all concrete adapters share, plus
    ``attach_model`` (resolves hook selection + PP/TP filters + installs
    HookPoints), ``plan_step`` (one walk over active_specs), and
    ``commit_step`` (reservation + context/metadata publication).
    """

    def __init__(self, engine: "MonitoringEngine", model_id: str) -> None:
        self.engine = engine
        self.model_id = model_id
        self.transport = engine._ring_transport
        self.ring_engine = getattr(engine, "_ring_engine", None)
        self.model_cfg: Optional[ModelShapeConfig] = None
        self.active_specs: List[HookSpec] = []
        self._warned_shapes: set = set()

    @property
    def active_hook_specs(self) -> tuple[HookSpec, ...]:
        """Read-only snapshot of selected hooks in forward firing order."""
        return tuple(self.active_specs)

    @property
    def model_shape(self) -> Optional[ModelShapeConfig]:
        """Detected model shape, or ``None`` before model attachment."""
        return self.model_cfg

    # --- subclass implements ---------------------------------------------
    @abc.abstractmethod
    def detect_model_shape(self, model) -> ModelShapeConfig:
        """Return the ModelShapeConfig for ``model`` (framework-specific)."""

    @abc.abstractmethod
    def detect_parallel_ranks(self) -> tuple:
        """Return ``(tp_rank, dp_rank, ep_rank, pp_rank)``."""

    @abc.abstractmethod
    def is_pp_first(self) -> bool: ...

    @abc.abstractmethod
    def is_pp_last(self) -> bool: ...

    @abc.abstractmethod
    def build_step_context(self, *raw) -> Optional[StepContext]:
        """Build a StepContext for one forward pass.

        Return ``None`` to skip this step entirely (e.g. when the framework
        passes a degenerate batch the adapter cannot extract metadata from).
        """

    @abc.abstractmethod
    def on_capacity_exceeded(self, ctx: StepContext) -> None:
        """Hook fired when ``prepare_step`` returns 2 (single step exceeds
        ring capacity).  Adapters use this to schedule a force-eager fallback
        or similar framework-side action."""

    # ----- optional subclass overrides ----------------------------------
    def adapt_for_cpu_direct(self, ctx: StepContext) -> StepContext:
        """Optionally rewrite ``ctx`` when cpu-direct fallback triggers.

        vLLM swaps padded -> unpadded q_len here (force_eager will disable
        CUDA-graph dispatch on the next batch, so the real tensor will not
        be padded).  HF default: no-op.
        """
        return ctx

    def _warn_once_capacity(
        self, ctx: StepContext, total_bytes: int, n_hooks: int
    ) -> None:
        """No-op stub.  Subclasses override to emit framework-flavored
        warnings when cpu-direct fallback first triggers for a given shape.
        """

    def _spec_needs_eager(self, spec: HookSpec) -> bool:
        """Return True if this spec requires eager dispatch + safety net
        for every batch it fires (independent of overflow).  Base default
        is False; subclasses override for dynamic-shape backends.
        """
        return False

    # --- shared (subclass never overrides) -------------------------------
    def attach_model(self, model, hook_selection: str = "full") -> None:
        """Resolve shape, install hooks per the selection + PP/TP filters."""
        if self.transport is None:
            raise RuntimeError(
                "BackendAdaptor.attach_model called before "
                "MonitoringEngine.enable_ring_transport()")

        cfg = self.detect_model_shape(model)
        tp, _dp, _ep, _pp = self.detect_parallel_ranks()
        cfg.tp_size = max(cfg.tp_size, 1)
        cfg.tp_rank = tp
        self.model_cfg = cfg
        self.transport.set_model_cfg(cfg)

        specs = model.get_hook_specs()
        specs = apply_hook_selection(specs, hook_selection, cfg=cfg)
        specs = filter_by_pp_rank(specs, self.is_pp_first(), self.is_pp_last())
        specs = filter_by_tp_rank(specs, tp)

        install_ring_hooks(specs, ring_payload=self.transport._ring_payload)
        self.active_specs = specs
        self.transport._active_specs = specs
        self.transport._using_forward_hooks = True

    def before_forward(self, *raw) -> None:
        """Canonical per-step driver.  See module docstring for the flow."""
        if self.transport is None or self.transport.null_offload:
            return
        ctx = self.build_step_context(*raw)
        if ctx is None:
            return

        plan = self.plan_step(ctx)
        self.commit_step(ctx, plan)

    def commit_step(
        self, ctx: StepContext, plan: Optional[StepPlan] = None
    ) -> StepReservation:
        """Reserve and publish one step, optionally using a precomputed plan.

        Supplying ``plan`` guarantees that hook shapes are not recomputed.
        This lets integrations plan before framework dispatch and commit once
        the real layout is known, without exposing the transport or native
        ring engine.
        """
        if self.transport is None or self.transport.null_offload:
            return StepReservation.SKIPPED
        if plan is None:
            plan = self.plan_step(ctx)

        # Gate prepare_step on hook_count > 0 -- matches vLLM's existing
        # behavior and is consistent with the non-zero-shape counting
        # rule for plan_step.  When the gate skips,
        # set_step_context and pre_push_all_metas still run unchanged:
        # their internal loops produce nothing to push when the
        # active-spec list is empty or every shape was empty.
        if plan.hook_count > 0:
            reservation = StepReservation(
                self.ring_engine.prepare_step(
                    plan.total_bytes, plan.hook_count
                )
            )
            self.transport.force_eager = (
                reservation is StepReservation.OVERSIZED
                or plan.needs_eager
            )
            if reservation is StepReservation.OVERSIZED:
                ctx = self.adapt_for_cpu_direct(ctx)
                self.on_capacity_exceeded(ctx)
                self._warn_once_capacity(
                    ctx, plan.total_bytes, plan.hook_count
                )
        else:
            # No hook fires this step (all shapes empty) -- no
            # safety net needed.
            self.transport.force_eager = False
            reservation = StepReservation.SKIPPED

        self.transport.set_step_context(**ctx.transport_kwargs())
        self.transport.pre_push_all_metas(
            batch=ctx.batch,
            q_len=ctx.q_len,
            kv_dim=ctx.kv_dim,
            logits_to_keep=ctx.logits_to_keep,
            token_ids_dtype=ctx.token_ids_dtype,
            actual_q_len=ctx.actual_q_len,
        )
        return reservation

    def close(self) -> None:
        self.engine.close()

    def plan_step(self, ctx: StepContext) -> StepPlan:
        """Build a side-effect-free reservation plan for one step.

        Single walk over ``active_specs``:
        - ``total_bytes`` and ``hook_count`` feed ``prepare_step``.
          ``hook_count``
          counts only specs whose ``_compute_hook_shape`` returns a
          non-empty list -- matches the count of metas
          ``pre_push_all_metas`` will push.
        - ``needs_eager`` is the OR of ``_spec_needs_eager(spec)`` over
          firing specs.  Drives the dispatch + safety-net decision in
          ``before_forward`` together with ``prepare_step``'s overflow
          result.
        """
        if self.model_cfg is None or not self.active_specs:
            return StepPlan(total_bytes=0, hook_count=0, needs_eager=False)

        total = 0
        n = 0
        needs_eager = False
        for spec in self.active_specs:
            # When the adapter has populated actual_q_len AND the spec is
            # prefix-strip-eligible, size the byte budget for the unpadded
            # data the producer will actually write.  Other specs and the
            # no-strip case use ctx.q_len.
            spec_q_len = (ctx.actual_q_len if ctx.actual_q_len is not None
                          and spec.dim0_is_actual_tokens
                          else ctx.q_len)
            shape = _compute_hook_shape(
                spec.hook_type, self.model_cfg,
                ctx.batch, spec_q_len, ctx.kv_dim,
                logits_to_keep=ctx.logits_to_keep,
            )
            if not shape:
                continue
            dtype = spec.dtype if spec.dtype is not None else self.model_cfg.dtype
            elem_size = torch._utils._element_size(dtype)
            nbytes = elem_size
            for d in shape:
                nbytes *= d
            total += align_up_py(nbytes, 16)
            n += 1
            if self._spec_needs_eager(spec):
                needs_eager = True
        return StepPlan(
            total_bytes=total,
            hook_count=n,
            needs_eager=needs_eager,
        )

    def _compute_step_plan(self, ctx: StepContext) -> StepPlan:
        """Compatibility wrapper returning a tuple-compatible plan."""
        return self.plan_step(ctx)


__all__ = ["BackendAdaptor", "StepPlan", "StepReservation"]
