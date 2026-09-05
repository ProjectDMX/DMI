from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dmi.transport.ring import RingTransport

pytestmark = pytest.mark.cpu


def test_window_configuration_types_are_available_on_the_lazy_native_surface():
    from dmi.transport.native import (
        D2HWindowGrantPolicyKind,
        D2HWindowMode,
        D2HWindowProgressKind,
        D2HWindowRuntimeSnapshot,
        RecurringD2HWindowConfig,
        RingConfig,
    )

    windows = RecurringD2HWindowConfig()
    windows.enabled = True
    windows.progress = D2HWindowProgressKind.PACKED_VERSION_COUNTER
    windows.grant_policy = D2HWindowGrantPolicyKind.LAST_K_ADAPTIVE
    windows.history_size = 3
    windows.minimum_record_probe_retry_interval_occurrences = 2
    windows.capacity_flush_fallback_threshold = 4
    config = RingConfig()
    config.recurring_d2h_windows = windows

    assert config.recurring_d2h_windows.enabled is True
    assert config.recurring_d2h_windows.history_size == 3
    assert D2HWindowMode.ENABLED_NO_PATTERN is not None
    assert D2HWindowRuntimeSnapshot is not None


def _transport_for_window_methods():
    transport = RingTransport.__new__(RingTransport)
    calls = []
    transport._ring_engine = SimpleNamespace(
        define_d2h_window_pattern=lambda *args: calls.append(("define", args))
    )
    transport._ring_payload = torch.empty(0, dtype=torch.uint8)
    transport._d2h_window_device_progress = torch.empty(1, dtype=torch.int64)
    transport._d2h_window_cpu_visible_progress = torch.empty(1, dtype=torch.int64)
    transport._d2h_window_marker = lambda *args: calls.append(("advance", args))
    transport._d2h_window_pattern_defined = False
    return transport, calls


def test_boundary_is_rejected_before_a_pattern_is_accepted():
    transport, calls = _transport_for_window_methods()

    with pytest.raises(RuntimeError, match="has not been defined"):
        transport.advance_boundary()

    assert calls == []


def test_pattern_acceptance_enables_the_stored_marker_callable():
    transport, calls = _transport_for_window_methods()

    transport.define_d2h_window_pattern(
        period=10,
        windows=((1, 4),),
        initial_counter=2,
    )
    transport.advance_boundary()

    assert calls[0] == ("define", (10, ((1, 4),), 2))
    assert calls[1][0] == "advance"
    assert calls[1][1] == (
        transport._ring_payload,
        transport._d2h_window_device_progress,
        transport._d2h_window_cpu_visible_progress,
    )


def test_failed_pattern_definition_does_not_enable_boundary_calls():
    transport, calls = _transport_for_window_methods()

    def reject(*_args):
        raise ValueError("invalid pattern")

    transport._ring_engine.define_d2h_window_pattern = reject
    with pytest.raises(ValueError, match="invalid pattern"):
        transport.define_d2h_window_pattern(period=0, windows=())
    with pytest.raises(RuntimeError, match="has not been defined"):
        transport.advance_boundary()

    assert calls == []
