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
from dmi.storage.capture.clickhouse_catalog import (
    _CAPTURE_COLUMNS,
    _CAPTURE_TABLE_ORDER,
)
from dmi.storage.capture.clickhouse_reader import (
    _EQUALITY_FILTERS,
    _PROJECTION,
    _RESOLUTION_ORDER,
    _RESOLVED,
    _SORT_KEY,
)
from dmi.storage.capture.model import PayloadLocator


pytestmark = pytest.mark.cpu


_WATERMARK = 1_756_142_093_000_000_000

# The ordering argument the projection's argMax must carry. Spelled out here
# rather than imported so that a change to it fails these tests instead of
# silently travelling through them.
_ORDER = "(index_version, store_id, pack_id)"


def _source(descriptor: CaptureDescriptor) -> dict:
    """Every catalog column of one descriptor, keyed by column name."""
    metadata, locator = descriptor.metadata, descriptor.locator
    return {
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


def _shaped(source: dict) -> tuple:
    """The row a catalog read returns: identity columns, then one tuple.

    Everything that is not grouped on comes back inside a single aggregate, so
    the row is six wide however many columns the catalog gains.
    """
    return tuple(source[name] for name in _SORT_KEY) + (
        tuple(source[name] for name in _RESOLVED),
    )


def _row(descriptor: CaptureDescriptor) -> tuple:
    return _shaped(_source(descriptor))


def _resolved_tuple(sql: str) -> str:
    """The column list inside the projection's single argMax tuple."""
    assert sql.count("argMax(") == 1, f"expected exactly one aggregate: {sql}"
    return sql.split("argMax(tuple(")[1].split("), ")[0]


class _Client:
    """Records every statement and replays canned result pages."""

    def __init__(self, *, descriptors=(), watermark=_WATERMARK, pages=None):
        self.calls: list[tuple[str, dict | None, dict]] = []
        self._watermark = watermark
        self._pages = list(pages) if pages is not None else [list(descriptors)]

    def execute(self, query, params=None, **kwargs):
        self.calls.append((" ".join(query.split()), params, kwargs))
        if "max(index_version)" in query:
            # The watermark now comes from the published log, not the
            # descriptor table.
            assert "_index_watermark" in query, query
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
    # Grouped on, or resolved: every column is exactly one of the two, so a
    # column added to the writer cannot fall out of the read.
    assert set(_SORT_KEY) | set(_RESOLVED) == set(_PROJECTION)
    assert not set(_SORT_KEY) & set(_RESOLVED)
    assert _RESOLVED == tuple(n for n in _PROJECTION if n not in _SORT_KEY)


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


@pytest.mark.parametrize(
    "resolved",
    (
        pytest.param("not-a-tuple", id="scalar"),
        pytest.param((1, 2, 3), id="too-few-columns"),
    ),
)
def test_search_rejects_a_malformed_resolved_tuple(resolved):
    """Everything not grouped on arrives as one aggregate, so its width matters.

    A row of the right *outer* width can still carry a tuple of the wrong
    length -- a catalog written by a build with different columns, say -- and
    zipping it against the column names would then silently slide every field
    onto the wrong one.
    """
    descriptor = synthetic_descriptors(1)[0]
    row = tuple(_source(descriptor)[name] for name in _SORT_KEY) + (resolved,)
    catalog = _raw_row_catalog(row)

    with pytest.raises(PackFormatError, match="resolved-column tuple"):
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
    # Every non-sort-key column is resolved inside the one aggregate rather than
    # taken from an arbitrary row of the group, and nothing is aliased back to
    # a source column name -- that would shadow the raw column and break the
    # filters on it.
    resolved = _resolved_tuple(sql)
    for name in ("pack_id", "payload_offset", "stored_length", "dtype"):
        assert f"`{name}`" in resolved
        assert f"AS `{name}`" not in sql


def test_the_logical_sort_key_is_a_prefix_of_the_table_order():
    """The two keys are decoupled, but only in one direction.

    The reader groups, orders and paginates on capture identity; the table is
    physically ordered on capture identity *plus* pack identity, so a merge
    cannot collapse two packs' rows for one capture. Those must stay in a
    prefix relationship: if the logical key ever stopped being a prefix, the
    keyset comparison that advances a page would stop pruning on the primary
    index and every page would scan the table.
    """
    from dmi.storage.capture.clickhouse_catalog import _CAPTURE_TABLE_ORDER
    from dmi.storage.capture.clickhouse_reader import _SORT_KEY

    assert _CAPTURE_TABLE_ORDER[: len(_SORT_KEY)] == _SORT_KEY
    # And the extra columns really are pack identity, resolved by argMax rather
    # than grouped on -- grouping on them would emit one row per pack for a
    # re-described capture instead of one newest-wins row.
    assert _CAPTURE_TABLE_ORDER[len(_SORT_KEY) :] == ("store_id", "pack_id")


def test_pack_identity_is_resolved_by_argmax_not_grouped_on():
    """Supersession lives here: newest pack wins, one row per capture."""
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=10))

    sql = client.selects[0]
    group_by = sql.split("GROUP BY")[1].split("ORDER BY")[0]
    resolved = _resolved_tuple(sql)
    for name in ("store_id", "pack_id"):
        assert f"`{name}`" in resolved
        assert name not in group_by


def test_one_aggregate_on_a_total_order_resolves_both_query_sites():
    """The two properties the projection's shape buys, at both query sites.

    ONE aggregate over a tuple of every resolved column, rather than one
    aggregate per column. Separate aggregates each pick their own row out of a
    tie, and ClickHouse does not say which -- nothing in that shape forbids
    ``store_id`` coming from one row and ``object_key`` from another, a
    descriptor describing no pack that exists. A single aggregate keeps a single
    row, so a mixed descriptor is impossible rather than merely unobserved.

    A TOTAL ordering argument, rather than ``index_version``. One
    ``CatalogIndexer.index`` call writes every pack of a batch at one version,
    so two packs describing the same capture in one batch tie on it, and the row
    the engine then keeps moves with the physical layout -- the live suite pins
    a merge changing it. Appending pack identity, which is what a capture's rows
    differ by, leaves exactly one maximum.

    Asserted at both sites, because a page and a lookup that resolved
    differently would make a selection stop round-tripping.
    """
    expected = synthetic_descriptors(1)
    catalog, client = _catalog(pages=[expected, expected])

    catalog.search(CaptureQuery(limit=10))
    catalog.get_by_ids(
        [expected[0].capture_id], tenant_id="tenant-a", watermark=str(_WATERMARK)
    )

    assert len(client.selects) == 2
    for sql in client.selects:
        # Exactly one aggregate (asserted inside _resolved_tuple), carrying
        # every resolved column, in _RESOLVED order so the row maps positionally.
        assert _resolved_tuple(sql) == ", ".join(f"`{n}`" for n in _RESOLVED)
        assert f"argMax(tuple({_resolved_tuple(sql)}), {_ORDER})" in sql
        # And nothing is left resolving on the version alone.
        assert ", index_version)" not in sql
        # The grouping columns still project directly, not through the tuple.
        for name in _SORT_KEY:
            assert f"`{name}`" in sql.split("argMax(")[0]
    # The ordering key is exactly (index_version, store_id, pack_id): version
    # first, so a later pack still supersedes an earlier one, then the columns
    # the table is physically ordered on beyond capture identity -- the only
    # ones a capture's rows can differ in, and therefore the only ones that can
    # break the tie a shared version leaves.
    assert _RESOLUTION_ORDER == _ORDER
    tail = _CAPTURE_TABLE_ORDER[len(_SORT_KEY) :]
    assert _ORDER == "(" + ", ".join(("index_version",) + tail) + ")"


def test_only_the_locator_may_differ_between_a_captures_rows():
    """The identity rule the query shape rests on, pinned column by column.

    ``(tenant_id, capture_id)`` identifies a capture. Every descriptor field
    except the locator is immutable for that identity, so two rows describing
    one capture differ only in where its bytes are. That is what reconciles a
    five-column ``GROUP BY`` with a ``CaptureSelection`` that dedups on
    ``capture_id`` alone, and -- the load-bearing half -- what makes the
    reader's pre-aggregation ``WHERE`` filters safe: a filter on an immutable
    column cannot match only the row that loses the argMax, because every row
    for the capture carries the same value.

    Asserted as an exact set rather than a subset. A new column that a later
    description could change has to land in one of these two lists, and landing
    in the immutable one is a claim someone has to make on purpose.
    """
    locator = frozenset(
        {
            "store_id", "pack_id", "object_key", "object_bytes", "pack_checksum",
            "pack_record_count", "payload_offset", "stored_length",
            "decoded_length", "codec", "payload_checksum",
        }
    )
    immutable = frozenset(
        {
            "session_id", "request_id", "sequence_id", "model_id",
            "model_revision", "adapter_revision", "capture_policy_version",
            "hook_name", "layer_number", "producer_rank", "step_number",
            "token_start", "token_end", "batch_position", "dtype", "shape",
        }
    )
    assert frozenset(_RESOLVED) - locator == immutable
    # The locator list is the PayloadLocator's own fields, so growing the
    # dataclass cannot quietly reclassify a mutable field as immutable. Two
    # fields are stored under a prefixed column name.
    renamed = {"offset": "payload_offset", "checksum": "payload_checksum"}
    assert locator == {
        renamed.get(name, name) for name in PayloadLocator.__dataclass_fields__
    }
    # And every filter the reader applies before grouping reads an immutable
    # column or a grouping column -- never a locator column.
    filtered = frozenset(_EQUALITY_FILTERS) | {
        "hook_name", "layer_number", "captured_at_ns"
    }
    assert not filtered & locator


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


def test_the_keyset_cursor_advances_with_a_strict_comparison():
    """`>` not `>=`, asserted on the operator rather than the parameters.

    A non-strict comparison re-reads the row the cursor was issued for, so
    every page boundary duplicates one capture and the walk returns more rows
    than the corpus holds. The parameter assertions above cannot see that --
    they check which values are bound, not how they are compared -- so only the
    live pagination walks caught it, and the PR gate (`-m cpu`) shipped green.
    """
    descriptors = synthetic_descriptors(4)
    catalog, client = _catalog(pages=[descriptors, descriptors[3:]])

    first = catalog.search(CaptureQuery(limit=3))
    catalog.search(CaptureQuery(limit=3, cursor=first.next_cursor))

    sql = client.selects[-1]
    key = "(`tenant_id`, `experiment_id`, `run_id`, `captured_at_ns`, `capture_id`)"
    bound = (
        "(%(after_tenant_id)s, %(after_experiment_id)s, %(after_run_id)s, "
        "%(after_captured_at_ns)s, %(after_capture_id)s)"
    )
    assert f"{key} > {bound}" in sql
    # Spelled out, because a blanket ">=" search would also match the
    # captured_after_ns range filter, which is legitimately non-strict.
    assert f"{key} >= {bound}" not in sql


def test_a_walk_stays_pinned_to_the_first_watermark():
    descriptors = synthetic_descriptors(4)
    catalog, client = _catalog(pages=[descriptors, descriptors[3:]])

    first = catalog.search(CaptureQuery(limit=3))
    client._watermark = _WATERMARK + 5_000  # a later indexing run lands
    second = catalog.search(CaptureQuery(limit=3, cursor=first.next_cursor))

    assert second.watermark == first.watermark
    _, params, _ = client.calls[-1]
    assert params["watermark"] == _WATERMARK


def test_only_a_cursor_bearing_search_reads_the_head_as_deciding():
    """The head read that bounds a cursor refuses the call; the other pins.

    A cursorless search that reads a stale head merely pins a slightly older
    snapshot, indistinguishable from having run a moment earlier, so it stays
    cheap. A cursor-bearing search hands that head to ``decode_cursor``, whose
    answer is a hard rejection -- so that read carries the same
    ``select_sequential_consistency`` the writer's deciding reads do.
    """
    from dmi.storage.capture.clickhouse_catalog import _DECIDING_READ

    descriptors = synthetic_descriptors(4)
    catalog, client = _catalog(pages=[descriptors, descriptors[3:]])
    config = ClickHouseReaderConfig()

    first = catalog.search(CaptureQuery(limit=3))
    catalog.search(CaptureQuery(limit=3, cursor=first.next_cursor))

    heads = [
        kwargs["settings"]
        for sql, _, kwargs in client.calls
        if "max(index_version)" in sql
    ]
    assert heads == [config.settings, {**config.settings, **_DECIDING_READ}]


def test_a_cursor_at_a_published_watermark_survives_replica_lag():
    """A cursor stamped with a watermark the catalog issued must stay valid.

    On a lagging replica the plain ``max(index_version)`` read answers below
    the published head, so a cursor the catalog itself handed out would read
    as "ahead of the catalog" and be rejected -- a false refusal of valid
    caller data, where a stale cursorless search merely pins an older snapshot.
    """
    from dmi.storage.capture.cursor import CursorKey, encode_cursor

    descriptors = synthetic_descriptors(1)

    class _LaggingClient(_Client):
        def execute(self, query, params=None, **kwargs):
            if "max(index_version)" in query:
                self.calls.append((" ".join(query.split()), params, kwargs))
                settings = kwargs.get("settings") or {}
                if settings.get("select_sequential_consistency"):
                    return [(self._watermark,)]
                return [(self._watermark - 1,)]
            return super().execute(query, params, **kwargs)

    client = _LaggingClient(descriptors=descriptors)
    catalog = ClickHouseCaptureCatalog(client, ClickHouseReaderConfig())
    cursor = encode_cursor(
        CursorKey(
            tenant_id="tenant-a",
            experiment_id="exp-a",
            run_id="run-a",
            captured_at_ns=1,
            capture_id="capture-0",
        ),
        watermark=_WATERMARK,
        filter_hash=CaptureQuery(limit=3).filter_hash,
    )

    page = catalog.search(CaptureQuery(limit=3, cursor=cursor))

    assert page.watermark == str(_WATERMARK)
    assert page.items == tuple(descriptors)


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
    assert set(params) == {"watermark", "row_limit"}
    # The membership subquery carries its own AND (the manifest version must
    # also appear in the watermark log), so counting the token no longer says
    # anything. Assert on the filters themselves instead.
    for fragment in (
        "tenant_id = ", "experiment_id = ", "run_id = ", "session_id = ",
        "model_id = ", "hook_name IN", "layer_number IN", "captured_at_ns >=",
        "captured_at_ns <=",
    ):
        assert fragment not in sql


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
            tenant_id="tenant-a",
            watermark=str(_WATERMARK),
        )


def test_get_by_ids_pins_the_watermark_and_parameterises_ids():
    expected = synthetic_descriptors(2)
    catalog, client = _catalog(descriptors=expected)

    resolved = catalog.get_by_ids(
        [item.capture_id for item in expected],
        tenant_id="tenant-a",
        watermark=str(_WATERMARK),
    )

    sql, params, _ = client.calls[-1]
    assert "index_version <= %(watermark)s" in sql
    assert "capture_id IN %(capture_ids)s" in sql
    assert params["watermark"] == _WATERMARK
    assert resolved == expected


def test_get_by_ids_chunks_the_inlined_id_list():
    """The ids ride in the statement TEXT, so a full lookup has to be split.

    The driver substitutes non-VALUES parameters client-side, and the server
    parses at most ``max_query_size`` bytes (256 KiB by default) -- an
    unchunked ``max_capture_ids`` lookup breaches it and dies with Code 62.
    Each chunk reads the same pinned snapshot, so the union of the chunk
    results is the unchunked result.
    """
    ids = [f"capture-{index:05d}" for index in range(10_000)]
    catalog, client = _catalog(pages=[[] for _ in range(50)])

    catalog.get_by_ids(ids, tenant_id="tenant-a", watermark=str(_WATERMARK))

    batches = [
        params["capture_ids"]
        for _, params, _ in client.calls
        if params is not None and "capture_ids" in params
    ]
    assert max(map(len, batches)) > 200
    # Every id is sent, in order, exactly once.
    assert [item for batch in batches for item in batch] == ids
    # Every chunk runs the same statement -- same tenant lead, same watermark
    # bound, same membership subquery.
    assert len(set(client.selects)) == 1
    # And the published-head check runs once, not once per chunk.
    heads = [sql for sql, _, _ in client.calls if "max(index_version)" in sql]
    assert len(heads) == 1


def test_every_id_chunk_stays_inside_the_rendered_byte_budget():
    from clickhouse_driver.util.escape import escape_param

    from dmi.storage.capture.clickhouse_sql import MAX_INLINE_PARAMETER_BYTES

    ids = [f"{index:04d}" + "'" * 508 for index in range(401)]
    catalog, client = _catalog(pages=[[], [], []])

    catalog.get_by_ids(ids, tenant_id="tenant-a", watermark=str(_WATERMARK))

    batches = [
        params["capture_ids"]
        for _, params, _ in client.calls
        if params is not None and "capture_ids" in params
    ]
    rendered = [
        escape_param(batch, {"strings_as_bytes": False}).encode("utf-8")
        for batch in batches
    ]
    assert all(len(value) <= MAX_INLINE_PARAMETER_BYTES for value in rendered)


def test_get_by_ids_sends_each_id_once():
    """A repeated id must not surface a capture twice.

    Within one statement the GROUP BY collapses a repeated id; two chunks
    naming the same id would each return its row. Deduplicating before
    chunking keeps the two shapes equivalent.
    """
    expected = synthetic_descriptors(1)
    catalog, client = _catalog(descriptors=expected)

    resolved = catalog.get_by_ids(
        [expected[0].capture_id, expected[0].capture_id],
        tenant_id="tenant-a",
        watermark=str(_WATERMARK),
    )

    _, params, _ = client.calls[-1]
    assert params["capture_ids"] == [expected[0].capture_id]
    assert resolved == tuple(expected)


def test_get_by_ids_filters_on_the_tenant_before_anything_else():
    expected = synthetic_descriptors(1)
    catalog, client = _catalog(descriptors=expected)

    catalog.get_by_ids(
        [expected[0].capture_id], tenant_id="tenant-a", watermark=str(_WATERMARK)
    )

    sql, params, _ = client.calls[-1]
    # tenant_id is the first ORDER BY column, so leading with it is what lets
    # the primary index prune the read to one tenant's range instead of
    # scanning the whole table for a capture_id match.
    assert "WHERE tenant_id = %(tenant_id)s AND capture_id IN %(capture_ids)s" in sql
    assert params["tenant_id"] == "tenant-a"
    # The tenant value is parameterised, never interpolated.
    assert "tenant-a" not in sql


@pytest.mark.parametrize("tenant_id", ("", None, 7))
def test_get_by_ids_rejects_a_blank_or_non_string_tenant(tenant_id):
    catalog, client = _catalog()

    with pytest.raises(ValueError, match="tenant_id"):
        catalog.get_by_ids(
            ["capture-0"], tenant_id=tenant_id, watermark=str(_WATERMARK)
        )
    assert client.calls == []


def test_get_by_ids_short_circuits_on_an_empty_request():
    catalog, client = _catalog()

    assert (
        catalog.get_by_ids([], tenant_id="tenant-a", watermark=str(_WATERMARK)) == ()
    )
    assert client.calls == []


@pytest.mark.parametrize("watermark", ("", "abc", "-1", "1.5", str(2**64)))
def test_get_by_ids_rejects_a_malformed_watermark(watermark: str):
    catalog, _ = _catalog()

    with pytest.raises(ValueError, match="watermark"):
        catalog.get_by_ids(["capture-0"], tenant_id="tenant-a", watermark=watermark)


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
    assert selection.tenant_id == "tenant-a"


def test_capture_reader_refuses_a_selection_that_spans_pages():
    from dmi.storage.capture import CaptureReader

    client = _Client(descriptors=synthetic_descriptors(4))
    reader = CaptureReader(
        ClickHouseCaptureCatalog(client, ClickHouseReaderConfig()),
        {"garage": _StubStore()},
    )

    with pytest.raises(ValueError, match="one bounded page"):
        reader.select(CaptureQuery(limit=3))


def test_get_by_ids_matches_commit_membership_on_store_and_pack():
    expected = synthetic_descriptors(1)
    catalog, client = _catalog(descriptors=expected)

    catalog.get_by_ids(
        [expected[0].capture_id], tenant_id="tenant-a", watermark=str(_WATERMARK)
    )

    sql = client.selects[0]
    # Pack identity is (store_id, pack_id); matching pack_id alone would let
    # the same UUID committed by a second store slip inside a pinned snapshot.
    assert "(store_id, pack_id) IN (SELECT store_id, pack_id FROM" in sql


def test_search_matches_commit_membership_on_store_and_pack():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    catalog.search(CaptureQuery(limit=10))

    sql = client.selects[0]
    assert "(store_id, pack_id) IN (SELECT store_id, pack_id FROM" in sql


def test_get_by_ids_rejects_an_unpublished_watermark():
    catalog, client = _catalog(descriptors=synthetic_descriptors(1))

    with pytest.raises(ValueError, match="published watermark"):
        catalog.get_by_ids(
            ["capture-0"], tenant_id="tenant-a", watermark=str(_WATERMARK + 1)
        )

    # Nothing was read beyond the watermark log itself.
    assert client.selects == []


# --- row value normalisation --------------------------------------------------


def _patched_row(descriptor: CaptureDescriptor, **overrides) -> tuple:
    return _shaped({**_source(descriptor), **overrides})


def _raw_row_catalog(row: tuple) -> ClickHouseCaptureCatalog:
    """A catalog whose page query returns one raw row, bypassing _row()."""
    client = _Client()
    original = client.execute

    def execute(query, params=None, **kwargs):
        if "max(index_version)" in query:
            return original(query, params, **kwargs)
        client.calls.append((" ".join(query.split()), params, kwargs))
        return [row]

    client.execute = execute  # type: ignore[method-assign]
    return ClickHouseCaptureCatalog(client, ClickHouseReaderConfig())


def test_row_mapping_strips_nul_padding_from_fixedstring_columns():
    descriptor = synthetic_descriptors(1)[0]
    padded = descriptor.locator.pack_checksum.encode() + b"\x00" * 4
    catalog = _raw_row_catalog(_patched_row(descriptor, pack_checksum=padded))

    page = catalog.search(CaptureQuery(limit=10))

    assert page.items == (descriptor,)


def test_row_mapping_converts_uuid_columns_to_text():
    from uuid import UUID

    descriptor = synthetic_descriptors(1)[0]
    catalog = _raw_row_catalog(
        _patched_row(descriptor, pack_id=UUID(descriptor.locator.pack_id))
    )

    page = catalog.search(CaptureQuery(limit=10))

    assert page.items[0].locator.pack_id == descriptor.locator.pack_id


def test_row_mapping_rejects_a_non_text_column():
    descriptor = synthetic_descriptors(1)[0]
    catalog = _raw_row_catalog(_patched_row(descriptor, capture_id=5))

    with pytest.raises(PackFormatError, match="non-text capture_id"):
        catalog.search(CaptureQuery(limit=10))


def test_row_mapping_rejects_a_non_integer_column():
    descriptor = synthetic_descriptors(1)[0]
    catalog = _raw_row_catalog(_patched_row(descriptor, layer_number=True))

    with pytest.raises(PackFormatError, match="non-integer layer_number"):
        catalog.search(CaptureQuery(limit=10))


def test_row_mapping_rejects_a_non_array_shape():
    descriptor = synthetic_descriptors(1)[0]
    catalog = _raw_row_catalog(_patched_row(descriptor, shape=5))

    with pytest.raises(PackFormatError, match="non-array shape"):
        catalog.search(CaptureQuery(limit=10))


@pytest.mark.parametrize("value", (True, "5"))
def test_current_watermark_rejects_a_non_integer_version(value):
    catalog, _ = _catalog(watermark=value)

    with pytest.raises(PackFormatError, match="non-integer watermark"):
        catalog.current_watermark()


def test_current_watermark_reports_zero_before_any_publish():
    catalog, _ = _catalog(watermark=None)

    assert catalog.current_watermark() == "0"


def test_reader_catalog_exposes_its_config():
    config = ClickHouseReaderConfig(max_capture_ids=7)
    catalog = ClickHouseCaptureCatalog(_Client(), config)

    assert catalog.config is config
