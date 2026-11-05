"""Host-only telemetry of the legacy ring (RingEngine.get_stats and
dmi.sync_stats).

The CPU tests check the readers' contracts on fake snapshots; the GPU tests
drive a real ring through ordinary, flushed and oversized steps, the
node-toggle pre-apply wait and the drain/P2P stages, and check the counters
move exactly where a wait or copy happened.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest
import torch

from dmi import sync_stats
from tests._requirements import require_cuda, require_native_backend


def _snapshot(fields, value=1):
    return SimpleNamespace(**{name: value for name in fields})


@pytest.mark.cpu
def test_readers_report_unavailable_instead_of_zeros():
    assert sync_stats.synchronization_stats(None) == {
        'available': False, 'reason': 'no ring engine'}
    partial = SimpleNamespace(prepare_ok_calls=1)
    out = sync_stats.synchronization_stats(None, snapshot=partial)
    assert out['available'] is False and 'prepare_drain_ns' in out['missing_fields']
    assert sync_stats.gather_sync_stats(object())['available'] is False


class _LegacyRingEngine:
    """The API shape of a native RingEngine built before get_stats existed."""


@pytest.mark.cpu
@pytest.mark.parametrize("reader, fields", [
    (sync_stats.synchronization_stats, sync_stats.FIELDS),
    (sync_stats.pipeline_stats, sync_stats.PIPELINE_FIELDS),
])
def test_readers_report_a_native_build_without_get_stats(reader, fields):
    out = reader(_LegacyRingEngine())
    assert out['available'] is False
    assert 'rebuild required' in out['reason'] and 'get_stats' in out['reason']
    assert out['missing_fields'] == list(fields)
    # A non-callable attribute of that name is no snapshot API either.
    engine = _LegacyRingEngine()
    engine.get_stats = None
    assert reader(engine)['available'] is False


@pytest.mark.cpu
@pytest.mark.parametrize("reader", [sync_stats.synchronization_stats,
                                    sync_stats.pipeline_stats])
def test_readers_propagate_errors_from_an_existing_get_stats(reader):
    class BrokenEngine:
        def get_stats(self):
            raise RuntimeError("snapshot failed")

    with pytest.raises(RuntimeError, match="snapshot failed"):
        reader(BrokenEngine())


@pytest.mark.cpu
def test_readers_copy_counters_and_reject_negative_ones():
    out = sync_stats.pipeline_stats(None, snapshot=_snapshot(sync_stats.PIPELINE_FIELDS, 3))
    assert out['available'] is True and out['schema_version'] == 1
    assert set(out['cumulative']) == set(sync_stats.PIPELINE_FIELDS)
    assert all(v == 3 for v in out['cumulative'].values())
    with pytest.raises(ValueError, match="negative"):
        sync_stats.synchronization_stats(
            None, snapshot=_snapshot(sync_stats.FIELDS, -1))
    counters = {f: 2 for f in sync_stats.GATHER_FIELDS}
    source = SimpleNamespace(row_table_stats=lambda: dict(counters))
    assert sync_stats.gather_sync_stats(source)['cumulative']['hook_table_uploads'] == 2
    del counters['hook_table_reuses']
    assert sync_stats.gather_sync_stats(source)['missing_fields'] == ['hook_table_reuses']
    counters['hook_table_reuses'] = -1
    with pytest.raises(ValueError, match="row-table"):
        sync_stats.gather_sync_stats(source)


QLEN, HID = 4, 8


def _ring(payload, consume, task_entries=64):
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE, HookSpec, ModelShapeConfig
    from dmi.transport import native as ne
    from dmi.transport import ring as rt

    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = payload
    cfg.pinned_staging_bytes = payload
    cfg.task_ring_entries = task_entries
    cfg.drain_poll_timeout_us = 200
    engine = ne.RingEngine(cfg, consume)
    engine.init(0)
    engine.start()
    transport = rt.RingTransport(engine)
    rt.activate(transport)
    transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2,
                                             num_kv_heads=2, head_dim=HID // 2,
                                             dtype=torch.float32))
    transport._active_specs = [HookSpec(hook_type=HOOK_TYPE_RESID_PRE,
                                        module=None, layer_no=0)]
    transport.set_step_context(model_id="m", req_ids=["r"], token_ranges=[(0, QLEN)],
                               dim0_offsets=[0], flattened=True)
    return engine, transport, rt


@pytest.mark.gpu
@pytest.mark.native_backend
@require_cuda()
@require_native_backend()
def test_get_stats_counts_prepare_outcomes_and_pipeline_stages():
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    delivered = []
    step = QLEN * HID * 4
    engine, transport, rt = _ring(4 * step, lambda *a: delivered.append(a[2]))
    try:
        before = engine.get_stats()
        assert before.prepare_ok_calls == before.pageable_copy_calls == 0
        x = torch.ones(QLEN, HID, device="cuda")
        payload = engine.payload_tensor()
        outcomes = []
        for _ in range(6):
            outcomes.append(engine.prepare_step(step, 1))
            transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
            torch.ops.ring.producer(payload, x, HOOK_TYPE_RESID_PRE, 0)
        assert engine.drain_to_sink_and_wait(0)
        # Reservations with nothing published fill the 4-step ring: the
        # fifth must synchronize and flush.
        for _ in range(5):
            outcomes.append(engine.prepare_step(step, 1))
        assert outcomes[-1] == 1
        assert engine.prepare_step(64 * step, 1) == 2   # oversized
        stats = engine.get_stats()
        assert stats.prepare_ok_calls == outcomes.count(0)
        assert stats.prepare_flushed_calls == outcomes.count(1) >= 1
        assert stats.prepare_oversized_calls == 1
        # Every flushed or oversized step synchronized and flushed once.
        waits = stats.prepare_flushed_calls + stats.prepare_oversized_calls
        assert stats.prepare_sync_calls == stats.prepare_drain_calls == waits
        assert stats.prepare_sync_ns > 0 and stats.prepare_sync_errors == 0
        assert stats.ready_commit_tasks == 6 and stats.ready_commit_invalid == 0
        assert stats.pageable_copy_calls == 6
        assert stats.pageable_copy_bytes == 6 * step
        assert stats.d2h_copy_errors == stats.d2h_sync_errors == 0
        assert stats.cpu_task_head >= stats.cpu_task_tail_committed
        out = sync_stats.synchronization_stats(engine)
        assert out['available'] and out['cumulative']['prepare_oversized_calls'] == 1
        assert sync_stats.pipeline_stats(engine)['cumulative']['pageable_copy_calls'] == 6
        assert len(delivered) == 6
    finally:
        torch.cuda.synchronize()
        engine.flush_and_wait()
        rt.deactivate()
        engine.stop()


@pytest.mark.gpu
@pytest.mark.native_backend
@require_cuda()
@require_native_backend()
def test_toggle_pre_apply_waits_are_counted_only_when_enabled():
    from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

    engine, transport, rt = _ring(1 << 20, lambda *a: None)
    try:
        x = torch.ones(QLEN, HID, device="cuda")
        payload = engine.payload_tensor()
        engine.enable_toggle_capture(True)
        engine.set_null_mode(True)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            torch.ops.ring.producer(payload, x, HOOK_TYPE_RESID_PRE, 0)
        graph.instantiate()
        engine.set_null_mode(False)
        engine.enable_toggle_capture(False)
        engine.bind_graph_exec(graph.raw_cuda_graph(), graph.raw_cuda_graph_exec())
        raw = graph.raw_cuda_graph()

        def reconfigure(enabled):
            transport.set_active_hooks_lazy([(HOOK_TYPE_RESID_PRE, 0)] if enabled else [])
            transport.pre_push_all_metas(batch=0, q_len=QLEN, kv_dim=0)
            assert transport.ensure_graph_current(raw) == 0
            graph.replay()
            assert transport.record_replay_event(raw) == 0

        reconfigure(True)
        reconfigure(False)
        assert engine.get_stats().toggle_wait_calls == 0     # off by default
        engine.set_toggle_pre_apply_sync(True)
        reconfigure(True)          # records the first replay event
        reconfigure(False)         # waits on it before mutating the exec
        assert engine.get_stats().toggle_wait_calls == 1
    finally:
        torch.cuda.synchronize()
        engine.flush_and_wait()
        rt.deactivate()
        engine.stop()


@pytest.mark.gpu
@pytest.mark.native_backend
@require_cuda()
@require_native_backend()
def test_monitoring_engine_logs_the_ring_footprint(caplog):
    from dmi.engine import MonitoringEngine
    from dmi.transport import native as ne

    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 2 << 20
    cfg.pinned_staging_bytes = 1 << 20
    engine = MonitoringEngine(model_id="m", ring_config=cfg,
                              ring_consumer=lambda *a: None)
    try:
        with caplog.at_level(logging.INFO, logger="dmi.engine"):
            engine._ensure_ring_transport()
        (record,) = [r for r in caplog.records if "legacy ring" in r.getMessage()]
        assert "2.0 MiB device payload ring" in record.getMessage()
        assert "1.0 MiB pinned host staging" in record.getMessage()
        assert "KV cache" in record.getMessage()
    finally:
        engine.close()
