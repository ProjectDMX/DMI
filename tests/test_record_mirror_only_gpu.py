"""Mirror-only record rings: GPU ring retention without any host copy.

A ``RingConfig.mirror_only`` record ring keeps what producers write in the
GPU payload ring and never copies it to the host: no pinned staging, no drain
or record threads, no sink admission. A reservation retires at once, so the
ring never fills or blocks; bytes stay until a later reservation wraps onto
them. Used on TP ranks whose captured data another rank copies to the host.

Build: make -C native SM_ARCH=... PYTHON=...
"""

from __future__ import annotations

import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]

RING_BYTES = 4096
RECORD_ELEMENTS = 256  # float32: 1 KiB per record


class _Target:
    """Capture target that must never be reached on a mirror-only ring."""

    def __init__(self):
        self.submitted = 0

    def _attach(self):
        pass

    def _detach(self):
        pass

    def _submit_capture(self, metadata_json, payload):
        self.submitted += 1

    def _flush_capture(self, timeout_s):
        return True

    def _rethrow_capture(self):
        pass


class _HookRuntime:
    def __init__(self, runtime):
        self._runtime = runtime
        self.metadata = None

    def should_emit(self, hook):
        return True

    def prepare_output(self, *, hook, output_index, output_id, output_spec,
                       output):
        from dmi.api.v1 import ProducerPlanBuilder

        entry = ProducerPlanBuilder().record_output(
            output_id=output_id, output_spec=output_spec, output=output)
        return self._runtime.emit_output(entry, self.metadata, output)


def _metadata(step: int, tensor: torch.Tensor):
    from dmi.storage.capture import CaptureMetadata

    return CaptureMetadata(
        capture_id=f"mirror-{step:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{step}",
        sequence_id="n", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v",
        hook_name="capture_tensor", layer_number=0, producer_rank=0,
        step_number=step, token_start=step, token_end=step + 1,
        batch_position=0, dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


class _MirrorRing:
    def __init__(self, mirror_only: bool = True):
        from dmi.api.v1 import (
            HookPointV1, HookSpecV1, MonitoringEngine, RingConfig,
            TransportSpec,
        )
        from dmi.storage.capture import CaptureRecordFormat
        from dmi.transport import native

        config = RingConfig()
        config.task_ring_entries = 64
        config.payload_ring_bytes = RING_BYTES
        config.pinned_staging_bytes = RING_BYTES
        config.mirror_only = mirror_only
        self.target = _Target()
        self.engine = MonitoringEngine(model_id="mirror-gpu",
                                       ring_config=config)
        sink = native.ReferencePythonCaptureSink(
            self.target, CaptureRecordFormat.LAYOUT_NAME)
        self.runtime = self.engine.create_record_runtime(
            CaptureRecordFormat(), record_sink=sink)
        self.hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        self.hook_runtime = _HookRuntime(self.runtime)
        self.runtime.bind_hook(self.hook, hook_runtime=self.hook_runtime)
        self.ring = self.engine._ring_engine
        self.cap = int(self.ring.payload_cap())
        self.payload = self.ring.payload_tensor()

    def emit(self, step: int, tensor: torch.Tensor) -> int:
        """Emit one record; return its logical ring offset."""
        self.hook_runtime.metadata = _metadata(step, tensor)
        self.runtime.begin_step()
        head = int(self.ring.payload_reserved_head())
        self.hook(tensor)
        return head

    def bytes_at(self, offset: int, nbytes: int) -> bytes:
        torch.cuda.current_stream().synchronize()
        start = offset % self.cap
        return bytes(self.payload[start:start + nbytes].cpu().numpy())


def _record(step: int) -> torch.Tensor:
    return torch.arange(RECORD_ELEMENTS, dtype=torch.float32,
                        device="cuda") + 1000 * step


def _as_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.contiguous().view(torch.uint8).cpu().numpy())


def test_mirror_only_keeps_records_in_the_ring_without_a_host_copy():
    ring = _MirrorRing()
    try:
        assert int(ring.ring.staging_cap()) == 0  # no pinned staging
        offsets = []
        for step in range(3):
            tensor = _record(step)
            offset = ring.emit(step, tensor)
            head = int(ring.ring.payload_reserved_head())
            assert head - offset == tensor.numel() * tensor.element_size()
            # Retired at once: nothing is pending for a drain.
            assert int(ring.ring.payload_committed_tail()) == head
            offsets.append(offset)
        for step, offset in enumerate(offsets):
            assert ring.bytes_at(offset, 1024) == _as_bytes(_record(step))
        ring.engine.flush_and_wait(5.0)  # nothing to flush; returns at once
        assert ring.target.submitted == 0
    finally:
        ring.engine.close()


def test_mirror_only_never_blocks_and_wraps_onto_the_oldest_records():
    ring = _MirrorRing()
    try:
        steps = 6  # 6 KiB through a 4 KiB ring
        offsets = [ring.emit(step, _record(step)) for step in range(steps)]
        head = int(ring.ring.payload_reserved_head())
        for step, offset in enumerate(offsets):
            intact = head - offset <= ring.cap
            data = ring.bytes_at(offset, 1024)
            if intact:
                assert data == _as_bytes(_record(step)), step
            else:
                # Overwritten by the record that wrapped onto its position.
                newer = step + ring.cap // 1024
                assert data == _as_bytes(_record(newer)), step
        assert ring.target.submitted == 0
    finally:
        ring.engine.close()


def test_mirror_only_skips_an_item_larger_than_the_ring():
    ring = _MirrorRing()
    try:
        before = int(ring.ring.payload_reserved_head())
        big = torch.zeros(2 * RING_BYTES // 4, dtype=torch.float32,
                          device="cuda")
        ring.emit(0, big)
        assert int(ring.ring.payload_reserved_head()) == before
        assert ring.target.submitted == 0
    finally:
        ring.engine.close()


def test_default_ring_still_copies_to_the_host():
    ring = _MirrorRing(mirror_only=False)
    try:
        assert int(ring.ring.staging_cap()) > 0
        ring.emit(0, _record(0))
        ring.engine.flush_and_wait(30.0)
        assert ring.target.submitted == 1
    finally:
        ring.engine.close()
