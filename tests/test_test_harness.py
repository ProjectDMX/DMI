from __future__ import annotations

import pytest


pytestmark = pytest.mark.cpu


def test_cpu_items_do_not_require_optional_resources(request):
    resource_markers = {
        "clickhouse",
        "e2e",
        "framework_fork",
        "gpu",
        "hf",
        "multi_gpu",
        "native_backend",
        "ring_native",
    }
    violations = []
    for item in request.session.items:
        if item.get_closest_marker("cpu") is None:
            continue
        overlap = sorted(
            name for name in resource_markers if item.get_closest_marker(name)
        )
        if overlap:
            violations.append(f"{item.nodeid}: {', '.join(overlap)}")

    assert violations == []


def test_framework_source_checks_are_not_cpu_tests(request):
    cpu_nodeids = {
        item.nodeid
        for item in request.session.items
        if item.get_closest_marker("cpu") is not None
    }

    assert not any("test_round_trip_byte_identical" in nodeid for nodeid in cpu_nodeids)
    assert not any(
        "test_smoke_hooks_present_in_compare_source" in nodeid
        for nodeid in cpu_nodeids
    )
