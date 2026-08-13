"""CPU contracts for capture-scoped ClickHouse test cleanup."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tests._clickhouse_test_utils import delete_capture


pytestmark = pytest.mark.cpu


def test_delete_capture_is_scoped_to_model_id(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, host, port):
            calls.append(("connect", host, port))

        def execute(self, query, params, settings):
            calls.append(("execute", query, params, settings))

    monkeypatch.setitem(
        sys.modules,
        "clickhouse_driver",
        SimpleNamespace(Client=FakeClient),
    )

    delete_capture("db", 9000, "capture-123", database="test_db", table="rows")

    assert calls == [
        ("connect", "db", 9000),
        (
            "execute",
            "ALTER TABLE test_db.rows DELETE WHERE model_id = %(model_id)s",
            {"model_id": "capture-123"},
            {"mutations_sync": 1},
        ),
    ]


def test_delete_capture_rejects_untrusted_identifier():
    with pytest.raises(ValueError, match="simple identifiers"):
        delete_capture("db", 9000, "capture", table="offload; DROP TABLE other")
