"""
Unit tests for monitoring/config.py

This module tests the configuration classes for the monitoring engine:
- HookSelection: Tests hook filtering by mode (full/attention/mlp/custom) and include/exclude lists
- CaptureSchedule: Tests step-level and request-level capture scheduling with warmup, offset, and stride
- MonitoringConfig: Tests the bundle class and its serialization to dict format

All tests are pure Python with no torch dependency.
"""

import pytest

from monitoring.config import (
    CaptureSchedule,
    HookSelection,
    MonitoringConfig,
    _ATTENTION_SUFFIXES,
    _MLP_SUFFIXES,
)


# =============================================================================
# Test fixtures and helper data
# =============================================================================

SAMPLE_HOOK_NAMES = [
    "blocks.0.hook_resid_pre",
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_k",
    "blocks.0.attn.hook_v",
    "blocks.0.attn.hook_z",
    "blocks.0.attn.hook_attn_scores",
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_result",
    "blocks.0.hook_resid_mid",
    "blocks.0.mlp.hook_mlp_in",
    "blocks.0.mlp.hook_mlp_post",
    "blocks.0.mlp.hook_mlp_out",
    "blocks.0.hook_resid_post",
    "blocks.1.hook_resid_pre",
    "blocks.1.attn.hook_q",
    "blocks.1.attn.hook_k",
]


# =============================================================================
# HookSelection Tests
# =============================================================================


class TestHookSelectionFullMode:
    """Tests for HookSelection with mode='full' (default)."""

    def test_full_mode_returns_all_hooks(self):
        """Full mode should return all input hooks without filtering."""
        selector = HookSelection(mode="full")
        result = selector.compile(SAMPLE_HOOK_NAMES)
        assert result == SAMPLE_HOOK_NAMES

    def test_full_mode_is_default(self):
        """Default mode should be 'full'."""
        selector = HookSelection()
        assert selector.mode == "full"
        result = selector.compile(SAMPLE_HOOK_NAMES)
        assert result == SAMPLE_HOOK_NAMES

    def test_full_mode_with_empty_input(self):
        """Full mode with empty input should return empty list."""
        selector = HookSelection(mode="full")
        result = selector.compile([])
        assert result == []


class TestHookSelectionAttentionMode:
    """Tests for HookSelection with mode='attention'."""

    def test_attention_mode_filters_correctly(self):
        """Attention mode should only return hooks ending with attention suffixes."""
        selector = HookSelection(mode="attention")
        result = selector.compile(SAMPLE_HOOK_NAMES)

        # All results should end with an attention suffix
        for name in result:
            assert any(name.endswith(suffix) for suffix in _ATTENTION_SUFFIXES), f"{name} is not an attention hook"

        # Should include these attention hooks
        assert "blocks.0.attn.hook_q" in result
        assert "blocks.0.attn.hook_k" in result
        assert "blocks.0.attn.hook_v" in result
        assert "blocks.0.attn.hook_z" in result

        # Should NOT include MLP or resid hooks
        assert "blocks.0.mlp.hook_mlp_in" not in result
        assert "blocks.0.hook_resid_pre" not in result

    def test_attention_mode_with_no_attention_hooks(self):
        """Attention mode with no matching hooks should return empty list."""
        non_attention_hooks = ["blocks.0.hook_resid_pre", "blocks.0.mlp.hook_mlp_in"]
        selector = HookSelection(mode="attention")
        result = selector.compile(non_attention_hooks)
        assert result == []


class TestHookSelectionMlpMode:
    """Tests for HookSelection with mode='mlp'."""

    def test_mlp_mode_filters_correctly(self):
        """MLP mode should only return hooks ending with MLP suffixes."""
        selector = HookSelection(mode="mlp")
        result = selector.compile(SAMPLE_HOOK_NAMES)

        # All results should end with an MLP suffix
        for name in result:
            assert any(name.endswith(suffix) for suffix in _MLP_SUFFIXES), f"{name} is not an MLP hook"

        # Should include MLP hooks
        assert "blocks.0.mlp.hook_mlp_in" in result
        assert "blocks.0.mlp.hook_mlp_out" in result

        # Should NOT include attention hooks
        assert "blocks.0.attn.hook_q" not in result

    def test_mlp_mode_with_no_mlp_hooks(self):
        """MLP mode with no matching hooks should return empty list."""
        non_mlp_hooks = ["blocks.0.hook_resid_pre", "blocks.0.attn.hook_q"]
        selector = HookSelection(mode="mlp")
        result = selector.compile(non_mlp_hooks)
        assert result == []


class TestHookSelectionCustomMode:
    """Tests for HookSelection with mode='custom'."""

    def test_custom_mode_with_include(self):
        """Custom mode should return only hooks in the include list."""
        include_list = ["blocks.0.hook_resid_pre", "blocks.0.attn.hook_q"]
        selector = HookSelection(mode="custom", include=include_list)
        result = selector.compile(SAMPLE_HOOK_NAMES)

        assert set(result) == set(include_list)

    def test_custom_mode_requires_include(self):
        """Custom mode without include should raise ValueError."""
        selector = HookSelection(mode="custom", include=None)

        with pytest.raises(ValueError, match="requires include to be provided"):
            selector.compile(SAMPLE_HOOK_NAMES)

    def test_custom_mode_with_nonexistent_hooks(self):
        """Custom mode with include list containing nonexistent hooks should return only existing ones."""
        include_list = ["blocks.0.hook_resid_pre", "nonexistent_hook"]
        selector = HookSelection(mode="custom", include=include_list)
        result = selector.compile(SAMPLE_HOOK_NAMES)

        assert result == ["blocks.0.hook_resid_pre"]


class TestHookSelectionExclude:
    """Tests for HookSelection exclude functionality."""

    def test_exclude_removes_specified_hooks(self):
        """Exclude should remove specified hooks from the result."""
        exclude_list = ["blocks.0.hook_resid_pre", "blocks.0.attn.hook_q"]
        selector = HookSelection(mode="full", exclude=exclude_list)
        result = selector.compile(SAMPLE_HOOK_NAMES)

        assert "blocks.0.hook_resid_pre" not in result
        assert "blocks.0.attn.hook_q" not in result
        assert "blocks.0.attn.hook_k" in result  # Other hooks should remain

    def test_exclude_with_attention_mode(self):
        """Exclude should work with preset modes."""
        exclude_list = ["blocks.0.attn.hook_q"]
        selector = HookSelection(mode="attention", exclude=exclude_list)
        result = selector.compile(SAMPLE_HOOK_NAMES)

        assert "blocks.0.attn.hook_q" not in result
        assert "blocks.0.attn.hook_k" in result

    def test_exclude_nonexistent_hooks_is_harmless(self):
        """Excluding nonexistent hooks should not cause errors."""
        exclude_list = ["nonexistent_hook"]
        selector = HookSelection(mode="full", exclude=exclude_list)
        result = selector.compile(SAMPLE_HOOK_NAMES)

        assert result == SAMPLE_HOOK_NAMES


class TestHookSelectionIncludeWithPreset:
    """Tests for HookSelection include with preset modes (non-custom)."""

    def test_include_narrows_preset_results(self):
        """Include with preset mode should intersect with preset filter."""
        # attention mode + include should return only hooks that are both attention AND in include
        include_list = ["blocks.0.attn.hook_q", "blocks.0.hook_resid_pre"]
        selector = HookSelection(mode="attention", include=include_list)
        result = selector.compile(SAMPLE_HOOK_NAMES)

        # Only hook_q should be included (resid_pre is not an attention hook)
        assert result == ["blocks.0.attn.hook_q"]


class TestHookSelectionInvalidMode:
    """Tests for HookSelection with invalid mode."""

    def test_invalid_mode_raises_valueerror(self):
        """Invalid mode should raise ValueError."""
        selector = HookSelection(mode="invalid")  # type: ignore

        with pytest.raises(ValueError, match="Unsupported hook selection mode"):
            selector.compile(SAMPLE_HOOK_NAMES)


# =============================================================================
# CaptureSchedule Tests
# =============================================================================


class TestCaptureScheduleDefaults:
    """Tests for CaptureSchedule default values."""

    def test_defaults_are_capture_everything(self):
        """Default values should capture every step and request."""
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
        """Default schedule should capture all steps."""
        schedule = CaptureSchedule()

        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is True
            assert schedule.should_capture_step(step_id, "decode") is True

    def test_defaults_capture_all_requests(self):
        """Default schedule should capture all requests."""
        schedule = CaptureSchedule()

        for request_id in range(10):
            assert schedule.should_capture_request(request_id) is True


class TestCaptureScheduleValidation:
    """Tests for CaptureSchedule __post_init__ validation."""

    def test_step_stride_must_be_positive(self):
        """step_stride < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="step_stride must be >= 1"):
            CaptureSchedule(step_stride=0)

        with pytest.raises(ValueError, match="step_stride must be >= 1"):
            CaptureSchedule(step_stride=-1)

    def test_request_stride_must_be_positive(self):
        """request_stride < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="request_stride must be >= 1"):
            CaptureSchedule(request_stride=0)

    def test_negative_step_offset_raises(self):
        """Negative step_offset should raise ValueError."""
        with pytest.raises(ValueError, match="offsets must be >= 0"):
            CaptureSchedule(step_offset=-1)

    def test_negative_request_offset_raises(self):
        """Negative request_offset should raise ValueError."""
        with pytest.raises(ValueError, match="offsets must be >= 0"):
            CaptureSchedule(request_offset=-1)

    def test_negative_warmup_steps_raises(self):
        """Negative warmup_steps should raise ValueError."""
        with pytest.raises(ValueError, match="warmup values must be >= 0"):
            CaptureSchedule(warmup_steps=-1)

    def test_negative_warmup_requests_raises(self):
        """Negative warmup_requests should raise ValueError."""
        with pytest.raises(ValueError, match="warmup values must be >= 0"):
            CaptureSchedule(warmup_requests=-1)


class TestCaptureScheduleRequestCapture:
    """Tests for CaptureSchedule.should_capture_request()."""

    def test_warmup_requests_are_skipped(self):
        """Requests during warmup period should not be captured."""
        schedule = CaptureSchedule(warmup_requests=5)

        for request_id in range(5):
            assert schedule.should_capture_request(request_id) is False

        assert schedule.should_capture_request(5) is True

    def test_request_offset_boundary(self):
        """Requests before offset (after warmup) should not be captured."""
        schedule = CaptureSchedule(warmup_requests=2, request_offset=3)

        # request 0, 1: warmup -> False
        assert schedule.should_capture_request(0) is False
        assert schedule.should_capture_request(1) is False

        # request 2, 3, 4: past warmup but effective < offset -> False
        # effective = request_id - warmup = 0, 1, 2 (all < 3)
        assert schedule.should_capture_request(2) is False
        assert schedule.should_capture_request(3) is False
        assert schedule.should_capture_request(4) is False

        # request 5: effective = 5 - 2 = 3, (3 - 3) % 1 == 0 -> True
        assert schedule.should_capture_request(5) is True

    def test_request_stride_pattern(self):
        """Request stride should capture every Nth request after warmup+offset."""
        schedule = CaptureSchedule(request_stride=3, request_offset=1, warmup_requests=2)

        # warmup: 0, 1 -> False
        assert schedule.should_capture_request(0) is False
        assert schedule.should_capture_request(1) is False

        # effective = request_id - 2
        # request 2: effective = 0 < offset(1) -> False
        assert schedule.should_capture_request(2) is False

        # request 3: effective = 1, (1-1) % 3 == 0 -> True (first capture)
        assert schedule.should_capture_request(3) is True

        # request 4: effective = 2, (2-1) % 3 == 1 -> False
        assert schedule.should_capture_request(4) is False

        # request 5: effective = 3, (3-1) % 3 == 2 -> False
        assert schedule.should_capture_request(5) is False

        # request 6: effective = 4, (4-1) % 3 == 0 -> True (second capture)
        assert schedule.should_capture_request(6) is True


class TestCaptureScheduleStepCapture:
    """Tests for CaptureSchedule.should_capture_step()."""

    def test_warmup_steps_are_skipped(self):
        """Steps during warmup period should not be captured."""
        schedule = CaptureSchedule(warmup_steps=5)

        for step_id in range(5):
            assert schedule.should_capture_step(step_id, "decode") is False

        assert schedule.should_capture_step(5, "decode") is True

    def test_step_offset_boundary(self):
        """Steps before offset (after warmup) should not be captured."""
        schedule = CaptureSchedule(warmup_steps=2, step_offset=3)

        # step 0, 1: warmup -> False
        assert schedule.should_capture_step(0, "decode") is False
        assert schedule.should_capture_step(1, "decode") is False

        # step 2, 3, 4: past warmup but effective < offset -> False
        assert schedule.should_capture_step(2, "decode") is False
        assert schedule.should_capture_step(3, "decode") is False
        assert schedule.should_capture_step(4, "decode") is False

        # step 5: effective = 5 - 2 = 3, (3 - 3) % 1 == 0 -> True
        assert schedule.should_capture_step(5, "decode") is True

    def test_step_stride_pattern(self):
        """Step stride should capture every Nth step after warmup+offset."""
        schedule = CaptureSchedule(step_stride=4, step_offset=2, warmup_steps=5)

        # warmup: 0-4 -> False
        for step_id in range(5):
            assert schedule.should_capture_step(step_id, "decode") is False

        # effective = step_id - 5
        # step 5: effective = 0 < offset(2) -> False
        assert schedule.should_capture_step(5, "decode") is False

        # step 6: effective = 1 < offset(2) -> False
        assert schedule.should_capture_step(6, "decode") is False

        # step 7: effective = 2, (2-2) % 4 == 0 -> True (first capture)
        assert schedule.should_capture_step(7, "decode") is True

        # step 8, 9, 10: (3-2)%4=1, (4-2)%4=2, (5-2)%4=3 -> False
        assert schedule.should_capture_step(8, "decode") is False
        assert schedule.should_capture_step(9, "decode") is False
        assert schedule.should_capture_step(10, "decode") is False

        # step 11: effective = 6, (6-2) % 4 == 0 -> True (second capture)
        assert schedule.should_capture_step(11, "decode") is True


class TestCaptureSchedulePhaseFiltering:
    """Tests for CaptureSchedule phase filtering (prefill/decode)."""

    def test_capture_prefill_false_skips_prefill(self):
        """capture_prefill=False should skip all prefill phases."""
        schedule = CaptureSchedule(capture_prefill=False, capture_decode=True)

        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is False
            assert schedule.should_capture_step(step_id, "decode") is True

    def test_capture_decode_false_skips_decode(self):
        """capture_decode=False should skip all decode phases."""
        schedule = CaptureSchedule(capture_prefill=True, capture_decode=False)

        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is True
            assert schedule.should_capture_step(step_id, "decode") is False

    def test_both_phases_disabled(self):
        """Both phases disabled should skip all steps."""
        schedule = CaptureSchedule(capture_prefill=False, capture_decode=False)

        for step_id in range(10):
            assert schedule.should_capture_step(step_id, "prefill") is False
            assert schedule.should_capture_step(step_id, "decode") is False

    def test_invalid_phase_raises_valueerror(self):
        """Invalid phase should raise ValueError."""
        schedule = CaptureSchedule()

        with pytest.raises(ValueError, match="Unsupported phase"):
            schedule.should_capture_step(0, "invalid")  # type: ignore


# =============================================================================
# MonitoringConfig Tests
# =============================================================================


class TestMonitoringConfigDefaults:
    """Tests for MonitoringConfig default values."""

    def test_defaults_create_valid_config(self):
        """Default MonitoringConfig should have valid HookSelection and CaptureSchedule."""
        config = MonitoringConfig()

        assert isinstance(config.hooks, HookSelection)
        assert isinstance(config.schedule, CaptureSchedule)
        assert config.hooks.mode == "full"
        assert config.schedule.step_stride == 1

    def test_custom_hooks_and_schedule(self):
        """MonitoringConfig should accept custom hooks and schedule."""
        hooks = HookSelection(mode="attention")
        schedule = CaptureSchedule(step_stride=4)
        config = MonitoringConfig(hooks=hooks, schedule=schedule)

        assert config.hooks.mode == "attention"
        assert config.schedule.step_stride == 4


# =============================================================================
# Integration-style tests
# =============================================================================


class TestConfigIntegration:
    """Integration tests combining multiple config features."""

    def test_full_workflow_example(self):
        """Test a realistic configuration workflow."""
        # Create config for: attention hooks only, every 4th decode step, skip first 10 steps
        config = MonitoringConfig(
            hooks=HookSelection(mode="attention"),
            schedule=CaptureSchedule(
                step_stride=4,
                warmup_steps=10,
                capture_prefill=False,
            ),
        )

        # Compile hooks
        hooks = config.hooks.compile(SAMPLE_HOOK_NAMES)
        assert all(any(h.endswith(s) for s in _ATTENTION_SUFFIXES) for h in hooks)

        # Check schedule
        assert config.schedule.should_capture_step(5, "prefill") is False  # prefill disabled
        assert config.schedule.should_capture_step(5, "decode") is False   # warmup
        assert config.schedule.should_capture_step(10, "decode") is True   # first capture
        assert config.schedule.should_capture_step(11, "decode") is False  # not on stride
        assert config.schedule.should_capture_step(14, "decode") is True   # second capture
