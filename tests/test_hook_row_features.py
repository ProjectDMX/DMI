"""Gather tables and in-graph heads on a HookPoint: lifecycle and dispatch.

Two contracts, checked with fakes only (no native extension, no CUDA):

* Attachment lifecycle.  ``configure_hook_gather`` / ``configure_hook_heads``
  arm a HookPoint for one attachment.  ``uninstall_ring_hooks`` and
  ``install_ring_hooks`` clear that state -- the tables, the head arena and
  accumulator references, and the strip stride gather replaced -- so a
  detached HookPoint releases them and a plain re-attach runs its ordinary
  producer instead of a stale fused node.

* Task agreement.  For every combination of row mode (static, prefix,
  chunked, gather) and heads, the host plan (``BackendAdapter.plan_step``),
  the fast (capturable) path and the eager safety net agree on the tasks a
  forward publishes and on their order: the row task, then the head task.
  Adding heads to a HookPoint without a gather table keeps its row producer.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from dmi.adapters.base import BackendAdapter
from dmi.adapters.types import StepContext
from dmi.hook_rows import (
    clear_hook_row_features,
    configure_hook_gather,
    configure_hook_heads,
)
from dmi.hooks.dispatch import install_ring_hooks, uninstall_ring_hooks
from dmi.hooks.point import HookPoint
from dmi.hooks.specs import (
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_RESID_PRE_HEAD,
    HookSpec,
    ModelShapeConfig,
)

pytestmark = pytest.mark.cpu

ROW, HEAD = HOOK_TYPE_RESID_PRE, HOOK_TYPE_RESID_PRE_HEAD
ROWS, HID, SLOTS = 4, 4, 2
ROW_BYTES = HID * 4
OPS = ("producer", "producer_prefix", "producer_chunked", "producer_gather",
       "producer_head", "producer_fused")


class _FakeCudaTensor(torch.Tensor):
    @property
    def is_cuda(self) -> bool:
        return True


def _x():
    return torch.zeros(ROWS, HID).as_subclass(_FakeCudaTensor)


def _table():
    return torch.zeros(1 + 2 * 4, dtype=torch.int64)


@pytest.fixture
def calls(monkeypatch):
    """Record every producer op launch: (op, args)."""
    import dmi.transport.ring as rt

    log = []
    for name in OPS:
        monkeypatch.setattr(
            torch.ops.ring, name,
            lambda *args, _name=name: log.append((_name, args)), raising=False)
    monkeypatch.setattr(rt, "_active_transport", None)
    return log


def _published(calls):
    """Tasks a list of producer launches publishes, as (hook_type, layer).

    A producer_fused launch publishes its row part when it has a gather
    table and its head part when it has a head table (the kernel's g_on /
    h_on predicates), rows first."""
    tasks = []
    for name, args in calls:
        if name == "producer_fused":
            gather, head = args[2], args[4]
            hook_type, hook_id, head_type = args[9], args[10], args[11]
            if gather is not None:
                tasks.append((hook_type, hook_id))
            if head is not None:
                tasks.append((head_type, hook_id))
        elif name in ("producer", "producer_prefix", "producer_chunked",
                      "producer_gather"):
            tasks.append((args[-2], args[-1]))
        elif name == "cpu_direct":
            tasks.append(args)
    return tasks


def _arm_heads(hook, table=None, layer=0):
    arena = torch.zeros(80, dtype=torch.int32)
    acc = torch.zeros(ROWS, 16)
    configure_hook_heads(hook, head_type=HEAD, layer_no=layer,
                         table=_table() if table is None else table,
                         arena=arena, acc=acc, slots=SLOTS, weight_bf16=False)
    return arena, acc


def _arm_strip(hook, mode):
    if mode == "prefix":
        hook._strip_tensor = torch.tensor([ROWS], dtype=torch.int64)
        hook._strip_row_bytes = ROW_BYTES
    elif mode == "chunked":
        hook._strip_tensor = torch.tensor([ROWS * ROW_BYTES // 2] * 2,
                                          dtype=torch.int64)
        hook._strip_row_bytes = 0


def _feature_fields(hook):
    return (hook._gather_table, hook._head_hook_type, hook._head_hook_id,
            hook._head_table, hook._head_arena, hook._head_acc,
            hook._head_slots, hook._head_wbf16)


CLEARED = (None, None, None, None, None, None, 0, 1)


# ---------------------------------------------------------------------------
# Attachment lifecycle


def test_detach_clears_gather_and_heads_and_releases_the_tensors(calls):
    hook = HookPoint()
    spec = HookSpec(hook_type=ROW, layer_no=0, module=hook)
    install_ring_hooks([spec], ring_payload=object())
    table = _table()
    configure_hook_gather(hook, table, ROW_BYTES)
    arena, acc = _arm_heads(hook)
    hook.forward(_x())
    assert [c[0] for c in calls] == ["producer_fused"]

    refs = [weakref.ref(t) for t in (table, arena, acc)]
    del table, arena, acc
    calls.clear()                            # the launch log holds the args
    uninstall_ring_hooks([spec])
    gc.collect()
    assert _feature_fields(hook) == CLEARED
    assert hook._ring_hook_type is None and hook._ring_payload is None
    assert all(ref() is None for ref in refs)

    # A plain re-attach runs the ordinary producer and publishes one task.
    install_ring_hooks([spec], ring_payload=object())
    calls.clear()
    hook.forward(_x())
    assert [c[0] for c in calls] == ["producer"]
    assert _published(calls) == [(ROW, 0)]


def test_disabled_hook_point_is_cleared_and_keeps_its_selection(calls):
    hook = HookPoint()
    hook.enabled = False
    spec = HookSpec(hook_type=ROW, layer_no=0, module=hook)
    install_ring_hooks([spec], ring_payload=object())
    configure_hook_gather(hook, _table(), ROW_BYTES)
    _arm_heads(hook)
    hook.forward(_x())
    assert calls == []
    uninstall_ring_hooks([spec])
    assert _feature_fields(hook) == CLEARED
    assert hook.enabled is False            # the selection is not attachment state
    hook.enabled = True
    install_ring_hooks([spec], ring_payload=object())
    hook.forward(_x())
    assert [c[0] for c in calls] == ["producer"]


def test_detach_in_finally_after_a_failing_forward_clears_the_state(calls, monkeypatch):
    hook = HookPoint()
    spec = HookSpec(hook_type=ROW, layer_no=0, module=hook)
    install_ring_hooks([spec], ring_payload=object())
    configure_hook_gather(hook, _table(), ROW_BYTES)
    _arm_heads(hook)

    def boom(*args):
        raise RuntimeError("producer failed")

    monkeypatch.setattr(torch.ops.ring, "producer_fused", boom, raising=False)
    with pytest.raises(RuntimeError, match="producer failed"):
        try:
            hook.forward(_x())
        finally:
            uninstall_ring_hooks([spec])
    assert _feature_fields(hook) == CLEARED


def test_rejected_configuration_leaves_the_previous_state_untouched():
    hook = HookPoint()
    table = _table()
    configure_hook_gather(hook, table, ROW_BYTES)
    with pytest.raises(ValueError):
        configure_hook_gather(hook, table.to(torch.int32), ROW_BYTES)
    with pytest.raises(ValueError):
        configure_hook_heads(hook, head_type=HEAD, layer_no=0, table=table,
                             arena=torch.zeros(8, dtype=torch.int32),
                             acc=torch.zeros(1, 16), slots=17, weight_bf16=False)
    with pytest.raises(ValueError, match="head row table"):
        configure_hook_heads(hook, head_type=HEAD, layer_no=0, table=None,
                             arena=torch.zeros(8, dtype=torch.int32),
                             acc=torch.zeros(1, 16), slots=1, weight_bf16=False)
    assert hook._gather_table is table and hook._strip_row_bytes == ROW_BYTES
    assert hook._head_hook_type is None and hook._head_arena is None


def test_reinstall_without_detach_starts_a_new_attachment(calls):
    hook = HookPoint()
    spec = HookSpec(hook_type=ROW, layer_no=0, module=hook)
    install_ring_hooks([spec], ring_payload=object())
    configure_hook_gather(hook, _table(), ROW_BYTES)
    _arm_heads(hook)
    install_ring_hooks([spec], ring_payload=object())
    assert _feature_fields(hook) == CLEARED
    hook.forward(_x())
    assert [c[0] for c in calls] == ["producer"]


@pytest.mark.parametrize("mode,op", [("prefix", "producer_prefix"),
                                     ("chunked", "producer_chunked")])
def test_clearing_gather_restores_the_strip_mode(calls, mode, op):
    hook = HookPoint()
    spec = HookSpec(hook_type=ROW, layer_no=0, module=hook)
    install_ring_hooks([spec], ring_payload=object())
    _arm_strip(hook, mode)
    strip_rb = hook._strip_row_bytes
    configure_hook_gather(hook, _table(), ROW_BYTES)
    configure_hook_gather(hook, _table(), ROW_BYTES)   # re-arming keeps the original
    assert hook._strip_row_bytes == ROW_BYTES
    configure_hook_gather(hook, None, 0)
    assert hook._strip_row_bytes == strip_rb and hook._gather_table is None
    hook.forward(_x())
    configure_hook_gather(hook, _table(), ROW_BYTES)
    uninstall_ring_hooks([spec])
    install_ring_hooks([spec], ring_payload=object())
    hook.forward(_x())
    assert [c[0] for c in calls] == [op, op]
    assert hook._strip_row_bytes == strip_rb


def test_clear_is_idempotent_on_plain_modules():
    module = SimpleNamespace(_gather_table=None, _strip_row_bytes=0)
    clear_hook_row_features(module)
    clear_hook_row_features(module)
    assert module._gather_table is None and module._head_hook_type is None
    assert module._strip_row_bytes == 0


def test_a_head_spec_never_replaces_the_row_key(calls):
    hook = HookPoint()
    row = HookSpec(hook_type=ROW, layer_no=3, module=hook)
    head = HookSpec(hook_type=HEAD, layer_no=3, module=hook,
                    dtype=torch.float32)
    install_ring_hooks([row, head], ring_payload=object())
    assert (hook._ring_hook_type, hook._ring_hook_id) == (ROW, 3)
    lone = HookPoint()
    install_ring_hooks([HookSpec(hook_type=HEAD, layer_no=1, module=lone)],
                       ring_payload=object())
    assert lone._ring_hook_type is None and lone._ring_payload is not None
    # Re-installing a former row HookPoint with only its head spec (no
    # uninstall in between) leaves it head-only, not with the old row key.
    install_ring_hooks([head], ring_payload=object())
    assert hook._ring_hook_type is None and hook._ring_payload is not None


# ---------------------------------------------------------------------------
# Task agreement: plan, fast path and eager safety net


class _Engine:
    def __init__(self, enabled):
        self.enabled = enabled
        self.events = []

    def is_hook_enabled(self, hook_type, layer):
        return (hook_type, layer) in self.enabled

    def available_capacity(self):
        return 1 << 30

    def available_task_slots(self):
        return 64

    def flush_and_wait(self):
        self.events.append("flush")

    def reserve_one(self, nbytes):
        self.events.append(("reserve", nbytes))


def _eager_transport(engine, calls, *, gate):
    def cpu_direct(tensor, hook_type, hook_id):
        calls.append(("cpu_direct", (hook_type, hook_id)))

    return SimpleNamespace(
        capture_step=True, force_eager=True, _ring_engine=engine,
        toggle_gate_active=gate, _gated_step=True, effective_cap=1 << 30,
        submit_cpu_direct=cpu_direct)


def _plan(specs):
    cfg = ModelShapeConfig(hidden_dim=HID, num_heads=1, num_kv_heads=1,
                           head_dim=HID, dtype=torch.float32, head_slots=SLOTS)
    adapter = SimpleNamespace(
        model_cfg=cfg, active_specs=list(specs),
        transport=SimpleNamespace(_active_specs=list(specs)),
        _spec_needs_eager=lambda _: False)
    ctx = StepContext(model_id="m", flattened=True, req_ids=["A"],
                      token_ranges=[(0, ROWS)], dim0_offsets=[0], kv_offsets=[0],
                      q_len=ROWS, batch=0)
    return BackendAdapter.plan_step(adapter, ctx)


def _build(mode, heads, row=True):
    hook = HookPoint()
    row_spec = HookSpec(hook_type=ROW, layer_no=0, module=hook,
                        dim0_is_actual_tokens=True)
    head_spec = HookSpec(hook_type=HEAD, layer_no=0, module=hook,
                         dtype=torch.float32, dim0_is_actual_tokens=True)
    specs = ([row_spec] if row else []) + ([head_spec] if heads else [])
    install_ring_hooks(specs, ring_payload=object())
    if row:
        _arm_strip(hook, mode)
        if mode == "gather":
            configure_hook_gather(hook, _table(), ROW_BYTES)
    if heads:
        _arm_heads(hook)
    return hook, specs


FAST_OPS = {
    # (row mode, heads) -> launches of the fast (capturable) path
    ("static", True): ["producer", "producer_fused"],
    ("prefix", True): ["producer_prefix", "producer_fused"],
    ("chunked", True): ["producer_chunked", "producer_fused"],
    ("gather", True): ["producer_fused"],
    ("static", False): ["producer"],
    ("gather", False): ["producer_gather"],
}


@pytest.mark.parametrize("mode,heads", sorted(FAST_OPS))
def test_fast_path_publishes_the_planned_tasks_in_spec_order(calls, mode, heads):
    hook, specs = _build(mode, heads)
    hook.forward(_x())
    assert [c[0] for c in calls] == FAST_OPS[(mode, heads)]
    published = _published(calls)
    assert published == [(s.hook_type, s.layer_no) for s in specs]
    assert _plan(specs).hook_count == len(published)
    if heads and mode != "gather":
        # The head-only node carries no row table: it is recorded and
        # toggled under the head key alone, independent of the row node.
        fused = calls[-1][1]
        assert fused[2] is None and fused[4] is hook._head_table
        assert (fused[9], fused[10], fused[11]) == (HEAD, 0, HEAD)


def test_head_only_hook_point_publishes_the_head_task(calls):
    hook, specs = _build("static", True, row=False)
    assert hook._ring_hook_type is None
    hook.forward(_x())
    assert [c[0] for c in calls] == ["producer_fused"]
    assert _published(calls) == [(HEAD, 0)]
    assert _plan(specs).hook_count == 1


def test_reviewed_plain_plus_heads_configuration_publishes_both_tasks(calls):
    """Heads configured on an installed ordinary HookPoint, the head spec
    appended with the public replace_active_specs API: the plan has the row
    and the head task (64 + 16 bytes) and the forward publishes both."""
    hook = HookPoint()
    spec = HookSpec(hook_type=ROW, layer_no=0, module=hook)
    install_ring_hooks([spec], ring_payload=object())
    _arm_heads(hook, table=torch.tensor([1, 0, ROWS], dtype=torch.int64))
    head_spec = HookSpec(hook_type=HEAD, layer_no=0, module=hook,
                         dtype=torch.float32)
    cfg = ModelShapeConfig(hidden_dim=HID, num_heads=1, num_kv_heads=1,
                           head_dim=HID, dtype=torch.float32, head_slots=1)
    adapter = SimpleNamespace(model_cfg=cfg, active_specs=[spec],
                              transport=SimpleNamespace(_active_specs=[spec]),
                              _spec_needs_eager=lambda _: False)
    BackendAdapter.replace_active_specs(adapter, [spec, head_spec])
    ctx = StepContext(model_id="m", flattened=True, req_ids=["A"],
                      token_ranges=[(0, ROWS)], dim0_offsets=[0], kv_offsets=[0],
                      q_len=ROWS, batch=0)
    plan = BackendAdapter.plan_step(adapter, ctx)
    assert plan.hook_count == 2 and plan.total_bytes == 80
    hook.forward(_x())
    assert [c[0] for c in calls] == ["producer", "producer_fused"]
    assert _published(calls) == [(ROW, 0), (HEAD, 0)]


EAGER_ROW_OPS = {"static": "producer", "gather": "producer_gather",
                 "prefix": "cpu_direct", "chunked": "cpu_direct"}


@pytest.mark.parametrize("mode", sorted(EAGER_ROW_OPS))
@pytest.mark.parametrize("row_on,head_on", [(True, True), (True, False),
                                            (False, True), (False, False)])
def test_eager_safety_net_matches_the_fast_path_and_the_toggles(
        calls, monkeypatch, mode, row_on, head_on):
    """With independent row/head toggle states, the eager safety net
    publishes exactly the enabled subset, row first; with every key
    enabled it publishes what the fast path publishes."""
    import dmi.transport.ring as rt

    hook, specs = _build(mode, True)
    enabled = set()
    if row_on:
        enabled.add((ROW, 0))
    if head_on:
        enabled.add((HEAD, 0))
    engine = _Engine(enabled)
    monkeypatch.setattr(rt, "_active_transport",
                        _eager_transport(engine, calls, gate=True))
    hook.forward(_x())
    want = [key for key in ((ROW, 0), (HEAD, 0)) if key in enabled]
    assert _published(calls) == want
    names = [c[0] for c in calls]
    expected = ([EAGER_ROW_OPS[mode]] if row_on else []) + (
        ["producer_fused"] if head_on else [])
    assert names == expected
    if head_on:
        # The eager head is a head-only node (no row table).
        assert calls[-1][1][2] is None
    reserves = [e for e in engine.events if isinstance(e, tuple)]
    assert len(reserves) == sum(1 for n in names if n != "cpu_direct")

    if row_on and head_on:
        # Same tasks as the ungated fast path.
        calls.clear()
        monkeypatch.setattr(rt, "_active_transport", None)
        hook.forward(_x())
        assert _published(calls) == want


# ---------------------------------------------------------------------------
# Eager head sizing: every physical source row, whatever the HookPoint rank

SHAPES = {
    "batch_seq": (2, 3, HID),            # [B, S, H]
    "one_batch_many_tokens": (1, 5, HID),
    "one_token_per_batch": (3, 1, HID),
    "packed": (6, HID),                  # [N, H]
    "single_row": (HID,),                # [H]
}


def _shaped_ctx(shape):
    if len(shape) == 3:
        b, s, _ = shape
        return StepContext(model_id="m", flattened=False,
                           req_ids=[f"r{i}" for i in range(b)],
                           token_ranges=[(0, s)] * b,
                           dim0_offsets=list(range(b)), kv_offsets=[0] * b,
                           batch=b, q_len=s)
    rows = 1 if len(shape) == 1 else shape[0]
    return StepContext(model_id="m", flattened=True, req_ids=["A"],
                       token_ranges=[(0, rows)], dim0_offsets=[0],
                       kv_offsets=[0], q_len=rows, batch=0)


def _shaped_heads(shape, *, row):
    """A HookPoint with full-row heads over a ``shape`` source (the table an
    integration writes on an eager step), its specs and the step plan."""
    rows = 1
    for d in shape[:-1]:
        rows *= d
    hook = HookPoint()
    row_spec = HookSpec(hook_type=ROW, layer_no=0, module=hook)
    head_spec = HookSpec(hook_type=HEAD, layer_no=0, module=hook,
                         dtype=torch.float32)
    specs = ([row_spec] if row else []) + [head_spec]
    install_ring_hooks(specs, ring_payload=object())
    configure_hook_heads(hook, head_type=HEAD, layer_no=0,
                         table=torch.tensor([1, 0, rows], dtype=torch.int64),
                         arena=torch.zeros(80, dtype=torch.int32),
                         acc=torch.zeros(rows, 16), slots=SLOTS,
                         weight_bf16=False)
    cfg = ModelShapeConfig(hidden_dim=HID, num_heads=1, num_kv_heads=1,
                           head_dim=HID, dtype=torch.float32, head_slots=SLOTS)
    adapter = SimpleNamespace(model_cfg=cfg, active_specs=specs,
                              transport=SimpleNamespace(_active_specs=specs),
                              _spec_needs_eager=lambda _: False)
    plan = BackendAdapter.plan_step(adapter, _shaped_ctx(shape))
    return hook, rows, plan


@pytest.mark.parametrize("shape", list(SHAPES.values()), ids=list(SHAPES))
@pytest.mark.parametrize("row", [True, False], ids=["rows_and_heads", "head_only"])
def test_eager_heads_reserve_what_the_plan_budgets_for_every_rank(
        calls, monkeypatch, shape, row):
    """The eager head reserves rows x K x 4 bytes with rows the product of
    every dimension before hidden -- what the native launcher writes
    (nbytes / (hidden x element size)) and what plan_step budgets -- not
    shape[0] x K x 4."""
    import dmi.transport.ring as rt

    hook, rows, plan = _shaped_heads(shape, row=row)
    engine = _Engine({(ROW, 0), (HEAD, 0)})
    monkeypatch.setattr(rt, "_active_transport",
                        _eager_transport(engine, calls, gate=False))
    x = torch.zeros(*shape).as_subclass(_FakeCudaTensor)
    hook.forward(x)
    head_bytes = rows * SLOTS * 4
    reserves = [e[1] for e in engine.events if isinstance(e, tuple)]
    assert reserves == ([x.nbytes] if row else []) + [head_bytes]
    # The plan budgets each task 16-byte aligned, as the ring stores it.
    assert sum(-(-r // 16) * 16 for r in reserves) == plan.total_bytes
    assert [c[0] for c in calls] == (["producer"] if row else []) + [
        "producer_fused"]


@pytest.mark.parametrize("shape", [SHAPES["batch_seq"], SHAPES["packed"]],
                         ids=["batch_seq", "packed"])
def test_eager_heads_over_the_effective_capacity_are_refused(
        calls, monkeypatch, shape):
    """The capacity guard sees the full head output: 48 bytes for six rows
    of two slots is refused under a 32-byte effective capacity before any
    reservation or launch, and admitted at exactly 48."""
    import dmi.transport.ring as rt

    hook, rows, _ = _shaped_heads(shape, row=False)
    head_bytes = rows * SLOTS * 4
    assert head_bytes == 48
    engine = _Engine({(HEAD, 0)})
    transport = _eager_transport(engine, calls, gate=False)
    transport.effective_cap = 32
    monkeypatch.setattr(rt, "_active_transport", transport)
    x = torch.zeros(*shape).as_subclass(_FakeCudaTensor)
    with pytest.raises(RuntimeError, match=r"48 bytes for 6 rows x 2 slots"):
        hook.forward(x)
    assert engine.events == [] and calls == []
    transport.effective_cap = head_bytes
    hook.forward(x)
    assert engine.events == [("reserve", head_bytes)]
    assert [c[0] for c in calls] == ["producer_fused"]
