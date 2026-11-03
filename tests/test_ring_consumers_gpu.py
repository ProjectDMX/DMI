"""In-process consumers of a real legacy ring.

Producers write through the ring; a SliceQueue, a BatchQueue or a Python
callable receives the result.  Checks the batch handoff's segments table,
capture leases and timestamps, the delivery barrier through the consumer,
consumer-failure accounting, the nonblocking flush hint, and the
MonitoringEngine consumer wiring.

Needs CUDA and the full native backend.  Memory use is a few MiB.
"""
from __future__ import annotations

import time
import warnings

import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]

HID = 8


def _config(ne, *, batch=False, timeout_us=200, poll_us=200):
    cfg = ne.RingConfig()
    cfg.payload_ring_bytes = 4 << 20
    cfg.pinned_staging_bytes = 4 << 20
    cfg.drain_poll_timeout_us = poll_us
    cfg.drain_flush_timeout_us = timeout_us
    cfg.batch_submit = batch
    return cfg


class Rig:
    """Ring engine + transport + one hook type over three requests."""

    def __init__(self, consumer, *, batch=False, timeout_us=200, specs=None):
        from dmi.hooks.specs import (HOOK_TYPE_RESID_PRE, HookSpec,
                                     ModelShapeConfig)
        from dmi.transport import native as ne
        from dmi.transport import ring as rt

        self.ne, self.rt = ne, rt
        self.ht = HOOK_TYPE_RESID_PRE
        self.engine = ne.RingEngine(_config(ne, batch=batch, timeout_us=timeout_us),
                                    consumer)
        self.engine.init(0)
        self.engine.start()
        self.transport = rt.RingTransport(self.engine)
        rt.activate(self.transport)
        self.transport.set_model_cfg(ModelShapeConfig(
            hidden_dim=HID, num_heads=2, num_kv_heads=2, head_dim=HID // 2,
            dtype=torch.float32, vocab_size=4))
        self.transport._active_specs = specs or [
            HookSpec(hook_type=self.ht, module=None, layer_no=layer)
            for layer in range(2)]
        self.payload = self.engine.payload_tensor()
        self.x = torch.arange(6 * HID, dtype=torch.float32,
                              device="cuda").reshape(6, HID)

    def step(self, lease=None):
        # Requests a (2 rows), b (3 rows), c (1 row) packed into 6 rows.
        self.transport.set_step_context(
            model_id="m", req_ids=["a", "b", "c"],
            token_ranges=[(5, 7), (0, 3), (9, 10)], dim0_offsets=[0, 2, 5],
            flattened=True, capture_lease=lease)
        self.transport.pre_push_all_metas(batch=0, q_len=6, kv_dim=0)
        for spec in self.transport._active_specs:
            torch.ops.ring.producer(self.payload, self.x, spec.hook_type, spec.layer_no)

    def close(self):
        torch.cuda.synchronize()
        try:
            self.engine.flush_and_wait()
        finally:
            self.rt.deactivate()
            self.engine.stop()


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def test_slice_queue_delivers_slices_with_lease_and_capture_time():
    from dmi.transport import native as ne

    queue = ne.SliceQueue()
    rig = Rig(queue)
    try:
        lease = ne.CaptureLease(11)
        before = time.monotonic_ns() // 1000
        rig.step(lease)
        torch.cuda.synchronize()
        rig.engine.flush_and_wait()
        items = []
        assert _wait_for(lambda: items.extend(queue.drain_slices(64, 10)) or len(items) >= 6)
        got = {(i[4], i[2]): i for i in items}
        assert sorted(got) == [(layer, rid) for layer in (0, 1) for rid in "abc"]
        torch.testing.assert_close(got[(0, "b")][7], rig.x[2:5].cpu())
        assert got[(1, "c")][5:7] == (9, 10)
        for item in items:
            assert item[9].same_as(lease)
            assert before <= item[10] <= time.monotonic_ns() // 1000 + 1
        assert not lease.idle
        del got, item
        items.clear()
        assert _wait_for(lambda: lease.idle)
    finally:
        rig.close()


def test_batch_queue_hands_each_hook_over_once_with_a_segments_table():
    from dmi.transport import native as ne

    queue = ne.BatchQueue()
    rig = Rig(queue, batch=True)
    try:
        rig.step(ne.CaptureLease(3))
        rig.step()
        torch.cuda.synchronize()
        rig.engine.flush_and_wait()
        batches = []
        assert _wait_for(lambda: batches.extend(queue.drain_batches(64, 10)) or len(batches) >= 4)
        assert queue.pushed == 4
        model, shard, act, layer, node, step_seq, last, tensor, rids, segments = batches[0][:10]
        assert (act, layer, node, last) == ("blocks.hook_resid_pre", 0, 0, False)
        assert batches[1][6] is True              # last hook of the step
        assert rids == ["a", "b", "c"]
        assert segments.tolist() == [[0, 5, 7, -1], [2, 0, 3, -1], [5, 9, 10, -1]]
        torch.testing.assert_close(tensor, rig.x.cpu())
        assert batches[0][10].id == 3 and batches[0][11] > 0
        assert len(batches[2]) == 10              # second step carried no lease
        assert batches[2][5] == batches[0][5] + 1  # one step_seq per step
    finally:
        rig.close()


def test_batch_handoff_indexes_final_logits_by_original_position():
    from dmi.hooks.specs import HOOK_TYPE_FINAL_LOGITS, HookSpec
    from dmi.transport import native as ne

    queue = ne.BatchQueue()
    rig = Rig(queue, batch=True, specs=[HookSpec(hook_type=HOOK_TYPE_FINAL_LOGITS,
                                                 module=None, layer_no=-1)])
    try:
        logits = torch.arange(4 * 4, dtype=torch.float32, device="cuda").reshape(4, 4)
        rig.transport.set_step_context(model_id="m", req_ids=["q1", "q3"],
                                       token_ranges=[(0, 5), (0, 2)],
                                       dim0_offsets=[3, 9], flattened=True,
                                       orig_positions=[1, 3])
        rig.transport.pre_push_all_metas(batch=0, q_len=11, kv_dim=0,
                                         logits_to_keep=4)
        torch.ops.ring.producer(rig.payload, logits, HOOK_TYPE_FINAL_LOGITS, -1)
        torch.cuda.synchronize()
        rig.engine.flush_and_wait()
        batches = []
        assert _wait_for(lambda: batches.extend(queue.drain_batches(8, 10)) or batches)
        segments = batches[0][9]
        assert segments.tolist() == [[1, 4, 5, 1], [3, 1, 2, 3]]
    finally:
        rig.close()


def test_drain_to_sink_and_wait_returns_after_the_consumer_has_everything():
    delivered = []

    def slow(model_id, shard_rank, req_id, act_name, layer_no, start, end, tensor):
        time.sleep(0.02)
        delivered.append((layer_no, req_id))

    rig = Rig(slow)
    try:
        rig.step()
        rig.step()
        assert rig.engine.drain_to_sink_and_wait(0) is True
        assert len(delivered) == 12          # 2 steps x 2 hooks x 3 requests
        rig.step()
        # A barrier that cannot finish in time reports it.
        assert rig.engine.drain_to_sink_and_wait(1) is False
        assert rig.engine.drain_to_sink_and_wait(0) is True
        assert len(delivered) == 18
    finally:
        rig.close()


def test_consumer_failures_are_counted_and_can_fail_the_next_step():
    calls = []

    def failing(*args):
        calls.append(args[2])
        raise ValueError("consumer rejected " + args[2])

    rig = Rig(failing)
    try:
        rig.step()
        assert rig.engine.drain_to_sink_and_wait(0) is True
        assert rig.engine.submit_exceptions == 6
        assert "consumer rejected" in rig.engine.last_sink_error
        assert rig.engine.sink_failed is False       # not armed
        assert rig.engine.prepare_step(256, 1) in (0, 1)
        rig.engine.flush_and_wait()
        rig.engine.set_abort_on_sink_error(True)
        assert rig.engine.sink_failed is True
        with pytest.raises(RuntimeError, match="consumer failed"):
            rig.engine.prepare_step(256, 1)
    finally:
        rig.engine.set_abort_on_sink_error(False)
        rig.close()


def test_request_flush_expedites_reserved_tasks():
    delivered = []
    rig = Rig(lambda *args: delivered.append(args[2]), timeout_us=60_000_000)
    try:
        assert rig.engine.request_flush() is False        # nothing reserved
        assert rig.engine.prepare_step(2 * 6 * HID * 4, 2) == 0
        rig.step()
        torch.cuda.synchronize()
        time.sleep(0.05)
        assert delivered == []                             # thresholds not met
        assert rig.engine.request_flush() is True
        assert _wait_for(lambda: len(delivered) == 6)
    finally:
        rig.close()


def test_monitoring_engine_delivers_to_a_ring_consumer_and_warns_without_one():
    from dmi.engine import MonitoringEngine, RingWithoutConsumerWarning
    from dmi.transport import native as ne

    delivered = []
    cfg = _config(ne)
    engine = MonitoringEngine(model_id="m", ring_config=cfg,
                              ring_consumer=lambda *args: delivered.append(args[2]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RingWithoutConsumerWarning)
        transport = engine._ensure_ring_transport()
    try:
        from dmi.hooks.specs import HOOK_TYPE_RESID_PRE, HookSpec, ModelShapeConfig
        transport.set_model_cfg(ModelShapeConfig(hidden_dim=HID, num_heads=2,
                                                 num_kv_heads=2, head_dim=HID // 2,
                                                 dtype=torch.float32))
        transport._active_specs = [HookSpec(hook_type=HOOK_TYPE_RESID_PRE,
                                            module=None, layer_no=0)]
        transport.set_step_context(model_id="m", req_ids=["a"], token_ranges=[(0, 2)],
                                   dim0_offsets=[0], flattened=True)
        transport.pre_push_all_metas(batch=0, q_len=2, kv_dim=0)
        torch.ops.ring.producer(transport._ring_payload,
                                torch.ones(2, HID, device="cuda"), HOOK_TYPE_RESID_PRE, 0)
        assert engine._ring_engine.drain_to_sink_and_wait(0)
        assert delivered == ["a"]
    finally:
        engine.close()

    with pytest.raises(ValueError, match="ring_consumer or host_engine"):
        MonitoringEngine(model_id="m", ring_config=_config(ne),
                         ring_consumer=print, host_engine=object())
    quiet = MonitoringEngine(model_id="m", ring_config=_config(ne))
    try:
        with pytest.warns(RingWithoutConsumerWarning, match="no consumer"):
            quiet._ensure_ring_transport()
    finally:
        quiet.close()


def test_monitoring_engine_switches_a_batch_queue_consumer_to_the_batch_handoff():
    from dmi.engine import MonitoringEngine
    from dmi.transport import native as ne

    queue = ne.BatchQueue()
    engine = MonitoringEngine(model_id="m", ring_config=_config(ne),
                              ring_consumer=queue)
    try:
        engine._ensure_ring_transport()
        assert engine._ring_config.batch_submit is True
    finally:
        engine.close()


def test_delivered_tensors_outlive_the_ring_engine():
    """Pooled pageable buffers stay valid after the engine that filled them
    stops, and are freed when the consumer drops them."""
    from dmi.transport import native as ne

    queue = ne.BatchQueue()
    rig = Rig(queue, batch=True)
    batches = []
    try:
        rig.step()
        torch.cuda.synchronize()
        rig.engine.flush_and_wait()
        assert _wait_for(lambda: batches.extend(queue.drain_batches(8, 10)) or len(batches) >= 2)
    finally:
        rig.close()
    del rig
    import gc
    gc.collect()
    torch.testing.assert_close(batches[0][7], torch.arange(6 * HID, dtype=torch.float32).reshape(6, HID))
    batches.clear()
    gc.collect()
