"""CPU tests for precise external-resource guards."""

from __future__ import annotations

import pytest

from tests import _requirements


pytestmark = pytest.mark.cpu


def test_clickhouse_guard_reports_missing_python_driver(monkeypatch):
    monkeypatch.setattr(
        _requirements.importlib.util,
        "find_spec",
        lambda name: None if name == "clickhouse_driver" else object(),
    )

    marker = _requirements.require_clickhouse("127.0.0.1", 9000)

    assert marker.name == "skipif"
    assert marker.args == (True,)
    assert marker.kwargs["reason"] == "clickhouse-driver is not installed"
