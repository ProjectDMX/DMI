from __future__ import annotations

import time

import pytest


def test_capped_drain_continues_after_initial_timeout_and_staging_is_bounded():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native ring drain regression test")

    from monitoring import _native_engine as native
    from monitoring.ring_transport import RingTransport, activate, deactivate

    mib = 1024 * 1024
    chunk_bytes = 4 * mib
    num_chunks = 16

    cfg = native.RingConfig()
    cfg.payload_ring_bytes = 128 * mib
    cfg.pinned_staging_bytes = 64 * mib
    cfg.task_ring_entries = 128
    cfg.drain_poll_timeout_us = 100
    cfg.drain_flush_timeout_us = 100_000

    engine = native.RingEngine(cfg, None)
    engine.init()
    engine.start()
    transport = RingTransport(engine)
    activate(transport)

    try:
        engine.set_drain_control(0, chunk_bytes, 0)
        assert engine.prepare_step(chunk_bytes * num_chunks, num_chunks) == 0

        source = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda")
        for hook_id in range(num_chunks):
            torch.ops.ring.producer(
                transport._ring_payload, source, 0, hook_id
            )
        torch.cuda.synchronize()

        deadline = time.monotonic() + 5.0
        first_batch_at = None
        last_batch_at = None
        stats = {}
        while time.monotonic() < deadline:
            stats = engine.link_stats()
            now = time.monotonic()
            if int(stats["d2h_batches"]) > 0 and first_batch_at is None:
                first_batch_at = now
            assert 0 <= int(stats["staging_used_bytes"]) <= int(stats["staging_cap"])
            if int(stats["d2h_batches"]) >= num_chunks:
                last_batch_at = now
                break
            time.sleep(0.001)

        assert int(stats["d2h_batches"]) == num_chunks
        assert int(stats["d2h_bytes"]) == chunk_bytes * num_chunks
        assert first_batch_at is not None and last_batch_at is not None
        # The 100 ms timeout starts a drain episode; it must not be paid again
        # for every capped chunk left in the same backlog.
        assert last_batch_at - first_batch_at < 0.75
    finally:
        deactivate()
        engine.stop()
