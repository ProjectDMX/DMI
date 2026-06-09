"""Unit tests for monitoring/config.py.

Covers ``CaptureSchedule`` (step / request scheduling with warmup, offset,
stride, and phase filtering) and ``MonitoringConfig`` (the bundle and its
default values).  Pure Python with no torch dependency.

The pre-refactor ``HookSelection`` dataclass was removed in Phase 2a of
the unified-adaptor refactor (the path was a silent no-op due to a dead
``_filter_specs`` call).  Selection now flows through
``monitoring.selection.apply_hook_selection`` driven by the ``hook_selection=``
arg on ``generate_with_monitoring`` / ``DMXGPUWorker.additional_config``.
"""

import pytest

from monitoring.config import CaptureSchedule, MonitoringConfig

pytestmark = pytest.mark.cpu


# =============================================================================
# CaptureSchedule
# =============================================================================


class TestCaptureScheduleDefaults:
    def test_defaults_are_capture_everything(self):
        schedule = CaptureSchedule()
        assert schedule.step_stride == 1
        assert schedule.step_offset == 0
        assert schedule.warmup_steps == 0
        assert schedule.capture_prefill is True
        assert schedule.capture_decode is True
        assert schedule.request_stride == 1
        assert schedule.request_offset == 0
        assert schedule.warmup_requests == 0

    def test_defaults_capture_all_steps(self):
        schedule = CaptureSchedule()
        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is True
            assert schedule.should_capture_step(step_id, "decode") is True

    def test_defaults_capture_all_requests(self):
        schedule = CaptureSchedule()
        for request_id in range(10):
            assert schedule.should_capture_request(request_id) is True


class TestCaptureScheduleValidation:
    def test_step_stride_must_be_positive(self):
        with pytest.raises(ValueError, match="step_stride must be >= 1"):
            CaptureSchedule(step_stride=0)
        with pytest.raises(ValueError, match="step_stride must be >= 1"):
            CaptureSchedule(step_stride=-1)

    def test_request_stride_must_be_positive(self):
        with pytest.raises(ValueError, match="request_stride must be >= 1"):
            CaptureSchedule(request_stride=0)

    def test_negative_step_offset_raises(self):
        with pytest.raises(ValueError, match="offsets must be >= 0"):
            CaptureSchedule(step_offset=-1)

    def test_negative_request_offset_raises(self):
        with pytest.raises(ValueError, match="offsets must be >= 0"):
            CaptureSchedule(request_offset=-1)

    def test_negative_warmup_steps_raises(self):
        with pytest.raises(ValueError, match="warmup values must be >= 0"):
            CaptureSchedule(warmup_steps=-1)

    def test_negative_warmup_requests_raises(self):
        with pytest.raises(ValueError, match="warmup values must be >= 0"):
            CaptureSchedule(warmup_requests=-1)


class TestCaptureScheduleRequestCapture:
    def test_warmup_requests_are_skipped(self):
        schedule = CaptureSchedule(warmup_requests=5)
        for request_id in range(5):
            assert schedule.should_capture_request(request_id) is False
        assert schedule.should_capture_request(5) is True

    def test_request_offset_boundary(self):
        schedule = CaptureSchedule(warmup_requests=2, request_offset=3)
        assert schedule.should_capture_request(0) is False
        assert schedule.should_capture_request(1) is False
        assert schedule.should_capture_request(2) is False
        assert schedule.should_capture_request(3) is False
        assert schedule.should_capture_request(4) is False
        assert schedule.should_capture_request(5) is True

    def test_request_stride_pattern(self):
        schedule = CaptureSchedule(request_stride=3, request_offset=1, warmup_requests=2)
        assert schedule.should_capture_request(0) is False
        assert schedule.should_capture_request(1) is False
        assert schedule.should_capture_request(2) is False
        assert schedule.should_capture_request(3) is True
        assert schedule.should_capture_request(4) is False
        assert schedule.should_capture_request(5) is False
        assert schedule.should_capture_request(6) is True


class TestCaptureScheduleStepCapture:
    def test_warmup_steps_are_skipped(self):
        schedule = CaptureSchedule(warmup_steps=5)
        for step_id in range(5):
            assert schedule.should_capture_step(step_id, "decode") is False
        assert schedule.should_capture_step(5, "decode") is True

    def test_step_offset_boundary(self):
        schedule = CaptureSchedule(warmup_steps=2, step_offset=3)
        assert schedule.should_capture_step(0, "decode") is False
        assert schedule.should_capture_step(1, "decode") is False
        assert schedule.should_capture_step(2, "decode") is False
        assert schedule.should_capture_step(3, "decode") is False
        assert schedule.should_capture_step(4, "decode") is False
        assert schedule.should_capture_step(5, "decode") is True

    def test_step_stride_pattern(self):
        schedule = CaptureSchedule(step_stride=4, step_offset=2, warmup_steps=5)
        for step_id in range(5):
            assert schedule.should_capture_step(step_id, "decode") is False
        assert schedule.should_capture_step(5, "decode") is False
        assert schedule.should_capture_step(6, "decode") is False
        assert schedule.should_capture_step(7, "decode") is True
        assert schedule.should_capture_step(8, "decode") is False
        assert schedule.should_capture_step(9, "decode") is False
        assert schedule.should_capture_step(10, "decode") is False
        assert schedule.should_capture_step(11, "decode") is True


class TestCaptureSchedulePhaseFiltering:
    def test_capture_prefill_false_skips_prefill(self):
        schedule = CaptureSchedule(capture_prefill=False, capture_decode=True)
        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is False
            assert schedule.should_capture_step(step_id, "decode") is True

    def test_capture_decode_false_skips_decode(self):
        schedule = CaptureSchedule(capture_prefill=True, capture_decode=False)
        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is True
            assert schedule.should_capture_step(step_id, "decode") is False

    def test_both_phases_disabled(self):
        schedule = CaptureSchedule(capture_prefill=False, capture_decode=False)
        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is False
            assert schedule.should_capture_step(step_id, "decode") is False

    def test_invalid_phase_raises_valueerror(self):
        schedule = CaptureSchedule()
        with pytest.raises(ValueError, match="Unsupported phase"):
            schedule.should_capture_step(0, "invalid")  # type: ignore


# =============================================================================
# MonitoringConfig
# =============================================================================


class TestMonitoringConfigDefaults:
    def test_defaults_create_valid_config(self):
        config = MonitoringConfig()
        assert isinstance(config.schedule, CaptureSchedule)
        assert config.schedule.step_stride == 1

    def test_custom_schedule(self):
        schedule = CaptureSchedule(step_stride=4)
        config = MonitoringConfig(schedule=schedule)
        assert config.schedule.step_stride == 4
