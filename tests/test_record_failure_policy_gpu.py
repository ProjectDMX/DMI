"""Sink refusals and stalls on a real record ring: what the forward sees.

The native suites pin the mechanism (tests/native/ring). This drives it the
way an integration does -- ``create_record_runtime``, a bound
``HookPointV1`` firing on a CUDA tensor, one ``begin_step`` per step -- and
measures what matters to serving: whether a hook call raises, and how long
the slowest step took.

The stalling sink is a Python target behind the native reference bridge:
its admission sleeps, like a block-mode sink waiting for queue room, or
refuses, like one that dropped a record. The burst case uses the real
NativePackSink with its default config.

Build: make -C native SM_ARCH=... PYTHON=... (and cpu-goals for the sink)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
import torch

from tests._requirements import require_cuda, require_native_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
SINK_BUILT = bool(
    sorted((REPO_ROOT / "native" / "build").glob("_dmi_native_sink*.so")))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.native_backend,
    require_cuda(),
    require_native_backend(),
]

BUDGET_MS = 50
ADMISSION_S = 0.4
RECORD_ELEMENTS = 256  # float32: 1 KiB per record
STEPS = 16


class _Target:
    """A capture target whose every admission takes ADMISSION_S, and which
    refuses from record `refuse_from` on when set."""

    def __init__(self, *, admission_s=0.0, refuse_from=None):
        self.admission_s = admission_s
        self.refuse_from = refuse_from
        self.submitted = 0

    def _attach(self):
        pass

    def _detach(self):
        pass

    def _submit_capture(self, metadata_json, payload):
        if self.refuse_from is not None and self.submitted >= self.refuse_from:
            raise RuntimeError("sink refused durable admission: dropped")
        time.sleep(self.admission_s)
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
        capture_id=f"policy-{step:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{step}",
        sequence_id="n", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v",
        hook_name="capture_tensor", layer_number=0, producer_rank=0,
        step_number=step, token_start=step, token_end=step + 1,
        batch_position=0, dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        captured_at_ns=1_700_000_000_000_000_000 + step,
    )


def _ring_config(payload_bytes: int):
    from dmi.api.v1 import RingConfig

    config = RingConfig()
    config.task_ring_entries = 64
    config.payload_ring_bytes = payload_bytes
    config.pinned_staging_bytes = payload_bytes
    # Drain every record as soon as it is published, so a refusal reaches
    # the worker while the steps are still running, as under serving load.
    config.drain_flush_entry_threshold = 1
    return config


def _run_steps(record_sink, *, policy, budget_ms, payload_bytes=4096,
               steps=STEPS, elements=RECORD_ELEMENTS, pace_s=0.0):
    """Fire one hook per step; return per-step wall times, the first error
    a hook call raised (if any), the status, and the flush outcome."""
    from dmi.api.v1 import (
        HookPointV1, HookSpecV1, MonitoringEngine, TransportSpec,
    )
    from dmi.storage.capture import CaptureRecordFormat

    engine = MonitoringEngine(model_id="policy-gpu",
                              ring_config=_ring_config(payload_bytes))
    outcome = {"step_s": [], "errors": [], "flush_error": None}
    try:
        runtime = engine.create_record_runtime(
            CaptureRecordFormat(), record_sink=record_sink,
            failure_policy=policy, step_stall_budget_ms=budget_ms)
        hook = HookPointV1(
            HookSpecV1("capture_tensor", (TransportSpec("payload"),)))
        hook_runtime = _HookRuntime(runtime)
        runtime.bind_hook(hook, hook_runtime=hook_runtime)
        tensor = torch.arange(elements, dtype=torch.float32, device="cuda")
        for step in range(steps):
            hook_runtime.metadata = _metadata(step, tensor)
            runtime.begin_step()
            started = time.monotonic()
            try:
                hook(tensor + step)
                torch.cuda.current_stream().synchronize()
            except RuntimeError as exc:
                outcome["errors"].append((step, str(exc)))
            outcome["step_s"].append(time.monotonic() - started)
            time.sleep(pace_s)
        outcome["status"] = engine.capture_status()
        try:
            engine.flush_and_wait(30.0)
        except RuntimeError as exc:
            outcome["flush_error"] = str(exc)
        outcome["flushed_status"] = engine.capture_status()
    finally:
        engine.close()
    return outcome


def _slow_sink(target):
    from dmi.storage.capture import CaptureRecordFormat
    from dmi.transport import native

    return native.ReferencePythonCaptureSink(
        target, CaptureRecordFormat.LAYOUT_NAME)


def test_disable_capture_keeps_the_forward_running_through_a_stalled_sink(
        caplog):
    """A 4 KiB ring against a sink that takes 400 ms per record: the ninth
    record's reservation waits for the sink. With a 50 ms budget every step
    returns, none waits longer than the budget plus one admission, and
    capture reports why it stopped."""
    target = _Target(admission_s=ADMISSION_S)
    with caplog.at_level(logging.WARNING, logger="dmi.engine"):
        outcome = _run_steps(_slow_sink(target), policy="disable_capture",
                             budget_ms=BUDGET_MS)

    assert outcome["errors"] == []
    worst = max(outcome["step_s"])
    assert worst < BUDGET_MS / 1000 + ADMISSION_S + 0.3, outcome["step_s"]
    status = outcome["status"]
    assert status["capture_active"] is False
    assert "stall budget" in status["failure"]
    assert status["failure_policy"] == "disable_capture"
    assert status["stall_budget_exhaustions"] == 1
    assert status["step_stall_budget_ms"] == BUDGET_MS
    assert status["max_step_wait_s"] >= BUDGET_MS / 1000
    # Only the record the sink was already admitting reached it; everything
    # after the latch was dropped on the worker, not submitted.
    assert target.submitted == 1
    assert outcome["flushed_status"]["discarded_payloads"] == STEPS - 1
    assert "stall budget" in outcome["flush_error"]
    assert any("stall budget" in r.getMessage() for r in caplog.records)


def test_raise_fails_fast_when_the_stall_budget_is_spent():
    target = _Target(admission_s=ADMISSION_S)
    outcome = _run_steps(_slow_sink(target), policy="raise",
                         budget_ms=BUDGET_MS)

    assert outcome["errors"], "the stalled step must raise under 'raise'"
    first_step, message = outcome["errors"][0]
    assert "stall budget" in message
    # Raised at the budget, not after the sink's 400 ms admission.
    assert outcome["step_s"][first_step] < 0.3, outcome["step_s"]
    # Latched: every later step raises at once.
    assert [step for step, _ in outcome["errors"]] == list(
        range(first_step, STEPS))
    assert max(outcome["step_s"][first_step + 1:]) < 0.1
    assert outcome["status"]["capture_active"] is False
    assert "stall budget" in outcome["flush_error"]


def test_disable_capture_turns_a_sink_refusal_into_stopped_capture():
    target = _Target(refuse_from=3)
    outcome = _run_steps(_slow_sink(target), policy="disable_capture",
                         budget_ms=None, payload_bytes=64 * 1024,
                         pace_s=0.02)

    assert outcome["errors"] == []
    status = outcome["status"]
    assert status["capture_active"] is False
    assert "sink refused durable admission" in status["failure"]
    assert status["stall_budget_exhaustions"] == 0
    assert target.submitted == 3
    # The refused fourth record and the twelve after it.
    assert outcome["flushed_status"]["discarded_payloads"] == STEPS - 3
    assert "sink refused durable admission" in outcome["flush_error"]


def test_raise_surfaces_a_sink_refusal_in_a_later_forward():
    target = _Target(refuse_from=3)
    outcome = _run_steps(_slow_sink(target), policy="raise", budget_ms=None,
                         payload_bytes=64 * 1024, pace_s=0.02)

    assert outcome["errors"]
    # The refusal happens on the worker after its record's forward returned,
    # so it raises from a LATER step, and from every step after that.
    first_step = outcome["errors"][0][0]
    assert first_step > 3
    assert [step for step, _ in outcome["errors"]] == list(
        range(first_step, STEPS))
    assert all("sink refused durable admission" in message
               for _, message in outcome["errors"])
    assert outcome["status"]["capture_active"] is False


@pytest.mark.skipif(
    not SINK_BUILT,
    reason="native/build/_dmi_native_sink*.so is not built; run "
    "`make -C native cpu-goals PYTHON=<venv>/bin/python`")
def test_a_burst_four_times_the_sink_queue_is_stored_on_a_record_ring(
        tmp_path):
    """64 x 1 MiB records back to back through the real NativePackSink and
    its default config (16 MiB queue, block with 2 s): the record worker
    waits for room, nothing is refused, capture stays active. Under the
    binding's old drop_newest default this burst dropped a record after
    18-25 and latched the ring."""
    from dmi.storage.capture.native_sink import create_native_pack_sink
    from dmi.storage.native_capture import NativeSinkConfig

    handle = create_native_pack_sink(
        NativeSinkConfig(spool_root=str(tmp_path / "spool")))
    sink = handle.native_sink
    outcome = _run_steps(sink, policy="disable_capture", budget_ms=2000,
                         payload_bytes=128 << 20, steps=64,
                         elements=(1 << 20) // 4)

    assert outcome["errors"] == []
    assert outcome["flush_error"] is None
    status = outcome["flushed_status"]
    assert status["capture_active"] is True, status
    assert status["sink"]["persisted_records"] == 64, status["sink"]
    assert status["sink"]["dropped_records"] == 0
    assert status["sink"]["timed_out_records"] == 0
