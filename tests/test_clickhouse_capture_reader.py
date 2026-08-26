from __future__ import annotations

import re

import pytest

from benchmarks.bench_capture_catalog import synthetic_descriptors
from dmi.storage.capture import (
    CaptureDescriptor,
    CaptureQuery,
    ClickHouseCaptureCatalog,
    ClickHouseCatalogConfig,
    ClickHouseReaderConfig,
    InvalidCursorError,
    PackFormatError,
)
from dmi.storage.capture.clickhouse_catalog import _CAPTURE_COLUMNS
from dmi.storage.capture.clickhouse_reader import _PROJECTION


pytestmark = pytest.mark.cpu


_WATERMARK = 1_756_142_093_000_000_000


def _row(descriptor: CaptureDescriptor) -> tuple:
    """The row a catalog read returns, in projection order."""
    metadata, locator = descriptor.metadata, descriptor.locator
    source = {
        **metadata.to_mapping(),
        "pack_id": locator.pack_id,
        "store_id": locator.store_id,
        "object_key": locator.object_key,
        "object_bytes": locator.object_bytes,
        "pack_checksum": locator.pack_checksum,
        "pack_record_count": locator.pack_record_count,
        "payload_offset": locator.offset,
        "stored_length": locator.stored_length,
        "decoded_length": locator.decoded_length,
        "codec": locator.codec,
        "payload_checksum": locator.checksum,
    }
    return tuple(source[name] for name in _PROJECTION)


class _Client:
    """Records every statement and replays canned result pages."""

    def __init__(self, *, descriptors=(), watermark=_WATERMARK, pages=None):
        self.calls: list[tuple[str, dict | None, dict]] = []
        self._watermark = watermark
        self._pages = list(pages) if pages is not None else [list(descriptors)]

    def execute(self, query, params=None, **kwargs):
        self.calls.append((" ".join(query.split()), params, kwargs))
        if "max(index_version)" in query:
            return [(self._watermark,)]
        page = self._pages.pop(0) if self._pages else []
        return [_row(item) for item in page]

    @property
    def selects(self) -> list[str]:
        return [call[0] for call in self.calls if "max(index_version)" not in call[0]]


def _catalog(**kwargs) -> tuple[ClickHouseCaptureCatalog, _Client]:
    client = _Client(**kwargs)
    return ClickHouseCaptureCatalog(client, ClickHouseReaderConfig()), client


# --- projection and reconstruction ------------------------------------------


def test_projection_covers_every_written_column_but_the_version():
    assert _PROJECTION == _CAPTURE_COLUMNS[:-1]
    assert "index_version" not in _PROJECTION


def test_search_reconstructs_descriptors_exactly():
    expected = synthetic_descriptors(3)
    catalog, _ = _catalog(descriptors=expected)

    page = catalog.search(CaptureQuery(limit=10))

    assert page.items == expected
    assert page.watermark == str(_WATERMARK)
    assert page.next_cursor is None


def test_search_rejects_a_row_of_the_wrong_width():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))
    client.execute = lambda *a, **k: [(1, 2, 3)]  # type: ignore[method-assign]

    with pytest.raises(PackFormatError, match="columns"):
        catalog.search(CaptureQuery(limit=10))


# --- snapshot semantics -----------------------------------------------------


def test_search_reads_the_raw_table_not_the_final_view():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=10))

    sql = client.selects[0]
    assert "_capture_raw" in sql
    assert "FINAL" not in sql


def test_search_resolves_columns_with_argmax_at_the_watermark():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=10))

    sql, params, _ = [call for call in client.calls if "argMax" in call[0]][0]
    assert "index_version <= %(watermark)s" in sql
    assert params["watermark"] == _WATERMARK
    # Every non-sort-key column resolves through argMax rather than an
    # arbitrary row from the group, and is never aliased back to its own name
    # -- that would shadow the raw column and break filters on it.
    for name in ("pack_id", "payload_offset", "stored_length", "dtype"):
        assert f"argMax(`{name}`, index_version)" in sql
        assert f"AS `{name}`" not in sql


def test_search_groups_and_orders_by_the_sort_key():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=10))

    sql = client.selects[0]
    key = "`tenant_id`, `experiment_id`, `run_id`, `captured_at_ns`, `capture_id`"
    assert f"GROUP BY {key}" in sql
    assert f"ORDER BY {key}" in sql


# --- pagination -------------------------------------------------------------


def test_search_requests_one_row_beyond_the_page():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=25))

    _, params, _ = client.calls[-1]
    assert params["row_limit"] == 26


def test_a_full_page_issues_a_cursor_and_truncates():
    descriptors = synthetic_descriptors(4)
    catalog, _ = _catalog(descriptors=descriptors)

    page = catalog.search(CaptureQuery(limit=3))

    assert page.items == descriptors[:3]
    assert page.next_cursor is not None


def test_a_partial_page_issues_no_cursor():
    catalog, _ = _catalog(descriptors=synthetic_descriptors(2))

    page = catalog.search(CaptureQuery(limit=3))

    assert page.next_cursor is None


def test_a_walk_advances_past_the_last_row_of_the_previous_page():
    descriptors = synthetic_descriptors(4)
    catalog, client = _catalog(pages=[descriptors, descriptors[3:]])

    first = catalog.search(CaptureQuery(limit=3))
    second = catalog.search(CaptureQuery(limit=3, cursor=first.next_cursor))

    _, params, _ = client.calls[-1]
    last_of_first = first.items[-1]
    assert params["after_capture_id"] == last_of_first.capture_id
    assert params["after_captured_at_ns"] == last_of_first.metadata.captured_at_ns
    assert second.items == descriptors[3:]
    assert second.next_cursor is None


def test_a_walk_stays_pinned_to_the_first_watermark():
    descriptors = synthetic_descriptors(4)
    catalog, client = _catalog(pages=[descriptors, descriptors[3:]])

    first = catalog.search(CaptureQuery(limit=3))
    client._watermark = _WATERMARK + 5_000  # a later indexing run lands
    second = catalog.search(CaptureQuery(limit=3, cursor=first.next_cursor))

    assert second.watermark == first.watermark
    _, params, _ = client.calls[-1]
    assert params["watermark"] == _WATERMARK


def test_a_cursor_cannot_be_replayed_against_different_filters():
    catalog, _ = _catalog(descriptors=synthetic_descriptors(4))
    page = catalog.search(CaptureQuery(limit=3, run_id="run-a"))

    with pytest.raises(InvalidCursorError, match="filter"):
        catalog.search(CaptureQuery(limit=3, run_id="run-z", cursor=page.next_cursor))


# --- filters ----------------------------------------------------------------


def test_every_filter_is_parameterised_not_interpolated():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(
        CaptureQuery(
            tenant_id="tenant-a",
            experiment_id="exp-a",
            run_id="run-a",
            session_id="session-a",
            model_id="model-a",
            hook_names=("resid_pre",),
            layer_numbers=(3,),
            captured_after_ns=1,
            captured_before_ns=2,
            limit=10,
        )
    )

    sql, params, _ = client.calls[-1]
    for name in (
        "tenant_id", "experiment_id", "run_id", "session_id", "model_id",
        "hook_names", "layer_numbers", "captured_after_ns", "captured_before_ns",
    ):
        assert f"%({name})s" in sql
        assert name in params
    # No filter value reaches the statement text.
    for value in ("tenant-a", "exp-a", "run-a", "session-a", "model-a", "resid_pre"):
        assert value not in sql


def test_absent_filters_add_no_clauses():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=10))

    sql, params, _ = client.calls[-1]
    assert sql.count("AND") == 0
    assert set(params) == {"watermark", "row_limit"}


# --- limits and injection ---------------------------------------------------


def test_every_statement_carries_the_query_limits():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))
    config = ClickHouseReaderConfig()

    catalog.search(CaptureQuery(limit=10))

    assert client.calls, "expected at least one statement"
    for _, _, kwargs in client.calls:
        assert kwargs["settings"] == config.settings
    assert config.settings["read_overflow_mode"] == "throw"


def test_get_by_ids_refuses_an_oversized_lookup():
    catalog, _ = _catalog()
    config = ClickHouseReaderConfig()

    with pytest.raises(ValueError, match="max_capture_ids"):
        catalog.get_by_ids(
            [f"capture-{index}" for index in range(config.max_capture_ids + 1)],
            watermark=str(_WATERMARK),
        )


def test_get_by_ids_pins_the_watermark_and_parameterises_ids():
    expected = synthetic_descriptors(2)
    catalog, client = _catalog(descriptors=expected)

    resolved = catalog.get_by_ids(
        [item.capture_id for item in expected], watermark=str(_WATERMARK)
    )

    sql, params, _ = client.calls[-1]
    assert "index_version <= %(watermark)s" in sql
    assert "capture_id IN %(capture_ids)s" in sql
    assert params["watermark"] == _WATERMARK
    assert resolved == expected


def test_get_by_ids_short_circuits_on_an_empty_request():
    catalog, client = _catalog()

    assert catalog.get_by_ids([], watermark=str(_WATERMARK)) == ()
    assert client.calls == []


@pytest.mark.parametrize("watermark", ("", "abc", "-1", "1.5", str(2**64)))
def test_get_by_ids_rejects_a_malformed_watermark(watermark: str):
    catalog, _ = _catalog()

    with pytest.raises(ValueError, match="watermark"):
        catalog.get_by_ids(["capture-0"], watermark=watermark)


@pytest.mark.parametrize(
    "field,value",
    (
        ("database", "default; DROP TABLE users"),
        ("database", "`injected`"),
        ("table_prefix", "dmi; DROP TABLE users"),
        ("table_prefix", "1_leading_digit"),
    ),
)
def test_config_refuses_hostile_identifiers(field: str, value: str):
    with pytest.raises(ValueError, match="identifier"):
        ClickHouseReaderConfig(**{field: value})


@pytest.mark.parametrize(
    "field", ("max_capture_ids", "max_rows_to_read", "max_bytes_to_read", "max_execution_time")
)
def test_config_refuses_non_positive_limits(field: str):
    with pytest.raises(ValueError, match=field):
        ClickHouseReaderConfig(**{field: 0})


def test_config_can_follow_a_writer_configuration():
    writer = ClickHouseCatalogConfig(database="analytics", table_prefix="dmi_test")

    config = ClickHouseReaderConfig.from_catalog(writer)

    assert (config.database, config.table_prefix) == ("analytics", "dmi_test")


def test_identifiers_are_quoted_in_the_statement():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=10))

    assert re.search(r"FROM `default`\.`dmi_capture_raw`", client.selects[0])


# --- wiring into CaptureReader ----------------------------------------------


class _StubStore:
    store_id = "garage"

    def put(self, pack, object_key):  # pragma: no cover - unused by select()
        raise NotImplementedError

    def stat(self, ref):  # pragma: no cover - unused by select()
        raise NotImplementedError

    def read_range(self, ref, offset, length):  # pragma: no cover - unused
        raise NotImplementedError


def test_catalog_satisfies_the_capture_catalog_protocol():
    from dmi.storage.capture import CaptureCatalog

    catalog, _ = _catalog()

    assert isinstance(catalog, CaptureCatalog)


def test_capture_reader_selects_through_the_clickhouse_catalog():
    from dmi.storage.capture import CaptureReader

    expected = synthetic_descriptors(3)
    client = _Client(descriptors=expected)
    reader = CaptureReader(
        ClickHouseCaptureCatalog(client, ClickHouseReaderConfig()),
        {"garage": _StubStore()},
    )

    selection = reader.select(CaptureQuery(limit=10))

    assert selection.capture_ids == tuple(item.capture_id for item in expected)
    assert selection.catalog_watermark == str(_WATERMARK)


def test_capture_reader_refuses_a_selection_that_spans_pages():
    from dmi.storage.capture import CaptureReader

    client = _Client(descriptors=synthetic_descriptors(4))
    reader = CaptureReader(
        ClickHouseCaptureCatalog(client, ClickHouseReaderConfig()),
        {"garage": _StubStore()},
    )

    with pytest.raises(ValueError, match="one bounded page"):
        reader.select(CaptureQuery(limit=3))
