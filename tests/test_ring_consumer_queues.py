"""In-process consumer queues of the legacy ring: SliceQueue and BatchQueue.

Drives the queues directly through their push bindings (no ring, no GPU):
retained-byte backpressure, rank-ordered eviction, drop attribution through
drop notices, capture-lease identity and lifetime, and the drain/wake
protocol.  tests/test_ring_consumers_gpu.py feeds them from a real ring.
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

from tests._requirements import require_native_backend

pytestmark = [pytest.mark.native_backend, require_native_backend()]


@pytest.fixture
def ne():
    from dmi.transport import native
    return native


def _push_batch(queue, act, layer, nbytes, req_ids, step=0, lease=None):
    tensor = torch.zeros(nbytes, dtype=torch.uint8)
    segments = torch.zeros((len(req_ids), 4), dtype=torch.int32)
    kwargs = {} if lease is None else {"capture_lease": lease, "capture_ts_us": 7}
    queue.push("m", 0, act, layer, step, True, tensor, list(req_ids), segments, **kwargs)


def _notices(queue):
    drained = queue.drain_drop_notices()
    return [(n[0], n[1], n[2], tuple(n[8]), n[9]) for n in drained["notices"]], drained


def test_batch_queue_is_fifo_and_sheds_whole_batches_with_attribution(ne):
    queue = ne.BatchQueue(max_bytes=300)
    _push_batch(queue, "blocks.hook_resid_pre", 0, 100, ["a", "b"], step=1)
    _push_batch(queue, "blocks.hook_resid_pre", 1, 100, ["a"], step=1)
    _push_batch(queue, "blocks.hook_resid_pre", 2, 150, ["c"], step=1)   # over the cap
    assert queue.qsize == 2 and queue.retained_bytes == 200
    assert queue.dropped == 1 and queue.dropped_bytes == 150
    notices, drained = _notices(queue)
    assert notices == [("batch_queue", "blocks.hook_resid_pre", 2, ("c",), 2)]
    assert drained["sticky"] is False and drained["overflow"] == 0
    batches = queue.drain_batches(10, 0)
    assert [(b[3], b[5], tuple(b[8])) for b in batches] == [(0, 1, ("a", "b")), (1, 1, ("a",))]
    assert queue.retained_bytes == 0 and queue.pushed == 2
    # No notice means nothing was lost.
    assert _notices(queue)[0] == []


def test_an_empty_queue_admits_one_item_larger_than_the_cap(ne):
    queue = ne.BatchQueue(max_bytes=10)
    _push_batch(queue, "x", 0, 100, ["a"])
    assert queue.qsize == 1 and queue.dropped == 0


def test_rank_ordered_eviction_sheds_less_urgent_streams_first(ne):
    queue = ne.BatchQueue(max_bytes=300)
    queue.set_layer_class_map([("urgent", 0, 0), ("boundary", 0, 1)], 2)
    assert queue.class_map_set and queue.class_map_size == 2
    assert queue.rank_for("urgent", 0) == 0
    assert queue.rank_for("boundary", 0) == 1
    assert queue.rank_for("unnamed", 3) == 2
    _push_batch(queue, "unnamed", 3, 100, ["a"])       # rank 2
    _push_batch(queue, "boundary", 0, 100, ["b"])      # rank 1
    _push_batch(queue, "unnamed", 3, 100, ["c"])       # rank 2
    _push_batch(queue, "urgent", 0, 150, ["d"])        # rank 0: evicts two rank-2
    assert queue.evicted == 2 and queue.dropped == 2
    assert queue.dropped_by_rank == [0, 0, 2]
    assert queue.qsize_by_rank == [1, 1, 0]
    notices, _ = _notices(queue)
    assert [(n[0], n[3]) for n in notices] == [("batch_queue_evict", ("a",)),
                                               ("batch_queue_evict", ("c",))]
    # A rank-0 arrival never evicts another rank-0 item, and evicting the
    # rank-1 batch would not make room (150 + 200 > 300): the newcomer alone
    # is refused and the rank-1 batch survives.
    _push_batch(queue, "urgent", 0, 200, ["e"])
    assert queue.evicted == 2 and queue.qsize_by_rank == [1, 1, 0]
    notices, _ = _notices(queue)
    assert notices == [("batch_queue", "urgent", 0, ("e",), 0)]
    # Evicting it does make room for a smaller one.
    _push_batch(queue, "urgent", 0, 140, ["f"])
    assert queue.evicted == 3 and queue.qsize_by_rank == [2, 0, 0]
    notices, _ = _notices(queue)
    assert [(n[0], n[3]) for n in notices] == [("batch_queue_evict", ("b",))]
    assert sum(queue.dropped_by_rank) == queue.dropped


def test_eviction_that_cannot_admit_the_newcomer_evicts_nothing(ne):
    queue = ne.BatchQueue(max_bytes=100)
    queue.set_layer_class_map([("urgent", 0, 0), ("boundary", 0, 1)], 2)
    _push_batch(queue, "urgent", 0, 80, ["a"])         # rank 0
    _push_batch(queue, "other", 0, 15, ["b"])          # rank 2
    _push_batch(queue, "boundary", 0, 50, ["c"])       # rank 1: 80 + 50 > 100 anyway
    assert queue.evicted == 0 and queue.dropped == 1
    assert queue.dropped_by_rank == [0, 1, 0]
    assert queue.qsize_by_rank == [1, 0, 1] and queue.retained_bytes == 95
    notices, _ = _notices(queue)
    assert notices == [("batch_queue", "boundary", 0, ("c",), 1)]
    # Draining the queue also clears the per-rank byte accounting.
    queue.drain_batches(10, 0)
    _push_batch(queue, "other", 0, 60, ["d"])          # rank 2
    _push_batch(queue, "boundary", 0, 50, ["e"])       # rank 1 evicts d: fits
    assert queue.evicted == 1 and queue.qsize_by_rank == [0, 1, 0]


def test_class_eviction_off_restores_drop_the_newcomer(ne):
    queue = ne.BatchQueue(max_bytes=200)
    queue.set_layer_class_map([("urgent", 0, 0)], 2)
    queue.set_class_eviction(False)
    assert queue.class_eviction is False
    _push_batch(queue, "other", 0, 150, ["a"])
    _push_batch(queue, "urgent", 0, 100, ["b"])
    assert queue.evicted == 0 and queue.dropped == 1
    assert queue.dropped_by_rank == [1, 0, 0]


def test_drop_notice_buffer_overflow_is_sticky_for_one_window(ne):
    queue = ne.BatchQueue(max_bytes=1, drop_notice_cap=2)
    _push_batch(queue, "x", 0, 10, ["keep"])
    for i in range(5):
        _push_batch(queue, "x", 0, 10, [f"r{i}"])
    assert queue.drop_notices_recorded == 5
    drained = queue.drain_drop_notices()
    assert len(drained["notices"]) == 2
    assert drained["overflow"] == 3 and drained["sticky"] is True
    assert queue.drop_notices_lost == 3
    again = queue.drain_drop_notices()
    assert again["sticky"] is False and again["notices"] == []


def test_slice_queue_caps_items_and_attributes_each_dropped_slice(ne):
    queue = ne.SliceQueue(max_items=2)
    lease = ne.CaptureLease(42)
    for i in range(3):
        queue.push("m", 0, f"r{i}", "blocks.hook_resid_pre", 5, i, i + 1,
                   torch.zeros(4), lease, 1000 + i)
    assert queue.qsize == 2 and queue.dropped == 1
    notices = queue.drain_drop_notices()["notices"]
    assert len(notices) == 1
    source, hook, layer, _, start, end, nbytes, _, rids, rank, kept = notices[0]
    assert (source, hook, layer, start, end, rids, rank) == (
        "slice_queue", "blocks.hook_resid_pre", 5, 2, 3, ["r2"], 255)
    assert kept.same_as(lease) and nbytes == 16
    items = queue.drain_slices(10, 0)
    assert [i[2] for i in items] == ["r0", "r1"]
    assert items[0][9].id == 42 and items[0][10] == 1000


def test_capture_lease_identity_ranks_and_lifetime(ne):
    policy = ne.CaptureLease(1, [("a", 0, 0), ("b", 1, 7)])
    step = ne.CaptureLease(2, policy)
    assert step.id == 2 and step.present and not ne.CaptureLease().present
    assert step.rank_for("a", 0, 9) == 0
    assert step.rank_for("b", 1, 9) == 2       # clamped
    assert step.rank_for("c", 3, 9) == 2       # unnamed stream
    assert ne.CaptureLease(3).rank_for("a", 0, 1) == 1   # no map: fallback
    assert step.idle
    queue = ne.BatchQueue()
    _push_batch(queue, "a", 0, 8, ["r"], lease=step)
    assert not step.idle                       # the queued item holds a copy
    (item,) = queue.drain_batches(1, 0)
    assert item[10].same_as(step) and item[11] == 7
    del item
    assert step.idle


def test_lease_ranks_order_eviction_per_configuration(ne):
    queue = ne.BatchQueue(max_bytes=200)
    lease = ne.CaptureLease(5, [("x", 0, 0)])
    _push_batch(queue, "y", 0, 150, ["low"], lease=lease)    # rank 2 under the lease
    _push_batch(queue, "x", 0, 100, ["high"], lease=lease)   # rank 0: evicts it
    assert queue.evicted == 1
    (notice,) = queue.drain_drop_notices()["notices"]
    assert notice[8] == ["low"] and notice[10].same_as(lease)


def test_drain_waits_without_the_gil_and_wake_ends_a_wait(ne):
    queue = ne.BatchQueue()
    started = time.monotonic()
    threading.Timer(0.05, queue.wake).start()
    assert queue.drain_batches(4, 5000) == []
    assert time.monotonic() - started < 2.0
    threading.Timer(0.05, lambda: _push_batch(queue, "x", 0, 8, ["r"])).start()
    assert len(queue.drain_batches(4, 5000)) == 1
    queue.stop()
    assert queue.drain_batches(4, 5000) == []


def test_ring_config_exposes_the_handoff_settings(ne):
    cfg = ne.RingConfig()
    assert cfg.batch_submit is False
    assert cfg.p2p_pool_cap_bytes == 256 * 1024 * 1024
    with pytest.raises(ValueError, match="batch_submit"):
        ne.RingEngine(cfg, ne.BatchQueue())


def test_queues_charge_each_backing_allocation_once(ne):
    # Views of one buffer (the per-request items of one step) keep ONE
    # allocation alive: it is charged once, at its full size, until the last
    # of them leaves the queue.
    whole = torch.zeros(400, dtype=torch.uint8)
    queue = ne.BatchQueue(max_bytes=1000)
    for i in range(4):
        segments = torch.zeros((1, 4), dtype=torch.int32)
        queue.push("m", 0, "x", 0, 1, i == 3, whole[i * 100:(i + 1) * 100], ["r"], segments)
    assert queue.qsize == 4 and queue.retained_bytes == 400 and queue.dropped == 0
    queue.drain_batches(3, 0)
    assert queue.retained_bytes == 400
    queue.drain_batches(1, 0)
    assert queue.retained_bytes == 0
    # A small view of a large allocation is charged the allocation.
    big = torch.zeros(800, dtype=torch.uint8)
    queue.push("m", 0, "x", 0, 2, True, big[:8], ["r"], torch.zeros((1, 4), dtype=torch.int32))
    assert queue.retained_bytes == 800
    _push_batch(queue, "x", 1, 300, ["s"], step=2)          # 800 + 300 > 1000
    assert queue.dropped == 1 and queue.dropped_bytes == 300


def test_slice_queue_reports_retention_and_honours_an_optional_byte_cap(ne):
    unbounded = ne.SliceQueue()
    assert unbounded.max_bytes == 0
    whole = torch.zeros(256, dtype=torch.uint8)
    for i in range(4):
        unbounded.push("m", 0, f"r{i}", "x", 0, 0, 1, whole[i * 64:(i + 1) * 64])
    assert unbounded.qsize == 4 and unbounded.retained_bytes == 256

    capped = ne.SliceQueue(max_items=100, drop_notice_cap=16, max_bytes=300)
    capped.push("m", 0, "a", "x", 0, 0, 1, torch.zeros(200, dtype=torch.uint8))
    capped.push("m", 0, "b", "x", 0, 0, 1, torch.zeros(200, dtype=torch.uint8))  # over
    assert capped.qsize == 1 and capped.dropped == 1 and capped.retained_bytes == 200
    (notice,) = capped.drain_drop_notices()["notices"]
    assert notice[8] == ["b"]
    capped.drain_slices(10, 0)
    assert capped.retained_bytes == 0
