"""Checks that the Python hook catalog matches the compiled native ABI."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.native_backend


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
    windows.grant_policy = D2HWindowGrantPolicyKind.BINARY_ADAPTIVE
    windows.minimum_record_probe_retry_interval_occurrences = 2
    windows.timing_revalidation_retry_interval_occurrences = 3
    windows.capacity_flush_fallback_threshold = 4
    assert windows.capacity_flush_count_reset_interval_periods == 32
    windows.capacity_flush_count_reset_interval_periods = 16
    config = RingConfig()
    config.recurring_d2h_windows = windows

    assert config.recurring_d2h_windows.enabled is True
    assert config.recurring_d2h_windows.timing_revalidation_retry_interval_occurrences == 3
    assert config.recurring_d2h_windows.capacity_flush_count_reset_interval_periods == 16
    assert D2HWindowMode.ENABLED_NO_PATTERN is not None
    assert D2HWindowRuntimeSnapshot is not None


def test_python_hook_catalog_matches_native_extension():
    from dmi.hooks.catalog import HOOK_DEFS
    from dmi.transport.native import _load_extension

    try:
        native = _load_extension()
    except ImportError as exc:
        pytest.skip(str(exc))

    assert tuple(tuple(row) for row in native.HOOK_DEFS) == HOOK_DEFS
