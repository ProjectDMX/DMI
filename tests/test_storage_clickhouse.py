"""Unit tests for dmi.storage.clickhouse -- no live ClickHouse server needed.

`clickhouse_driver.Client` is lazy: constructing it opens no connection, so the
`_ensure_client` construction arm is exercised for real. Query paths run
against a module-local fake client that records `execute()` calls and replays
canned rows.
"""
import pytest
import torch

from dmi.storage.clickhouse import CHClickhouseDriverReadOnly

pytestmark = pytest.mark.cpu


class _FakeClient:
    """Records every execute() call and replays canned rows."""

    def __init__(self, rows=()):
        self.calls = []
        self._rows = list(rows)
        self.disconnected = False

    def execute(self, query, params=None, **kwargs):
        self.calls.append((query, params))
        return list(self._rows)

    def disconnect(self):
        self.disconnected = True


def _payload(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().numpy().tobytes()


def _reader(rows=(), **kwargs) -> tuple[CHClickhouseDriverReadOnly, _FakeClient]:
    """Reader with a small 2-column primary key and an injected fake client."""
    reader = CHClickhouseDriverReadOnly(
        primary_key_column_names=("model_id", "request_id"),
        **kwargs,
    )
    fake = _FakeClient(rows)
    reader._client = fake
    return reader, fake


# --- construction -------------------------------------------------------------


def test_default_construction_is_lazy_and_forces_strings_as_bytes():
    reader = CHClickhouseDriverReadOnly()

    assert reader._client is None
    assert reader._client_settings["strings_as_bytes"] == 1
    assert reader._pk_count == 7
    assert len(reader._prefix_select_sql_with_key) == reader._pk_count + 1
    assert len(reader._prefix_select_sql_values_only) == reader._pk_count + 1


def test_construction_preserves_caller_client_settings():
    reader = CHClickhouseDriverReadOnly(client_settings={"max_threads": 2})

    assert reader._client_settings == {"max_threads": 2, "strings_as_bytes": 1}


def test_order_by_defaults_to_primary_key_and_can_be_overridden():
    reader = CHClickhouseDriverReadOnly(
        primary_key_column_names=("model_id", "request_id"),
    )
    assert reader._order_by_column_names == ("model_id", "request_id")

    reader = CHClickhouseDriverReadOnly(
        primary_key_column_names=("model_id", "request_id"),
        order_by_column_names=("request_id",),
    )
    assert reader._order_by_column_names == ("request_id",)


@pytest.mark.parametrize("port", [8123, 8443])
def test_http_ports_rejected(port):
    with pytest.raises(ValueError, match="HTTP port"):
        CHClickhouseDriverReadOnly(port=port)


def test_value_column_names_must_be_exactly_three():
    with pytest.raises(ValueError, match="exactly 3 columns"):
        CHClickhouseDriverReadOnly(value_column_names=("dtype", "bytes"))


def test_bad_table_identifier_rejected():
    with pytest.raises(ValueError, match="Invalid identifier"):
        CHClickhouseDriverReadOnly(table="off; DROP TABLE x")


def test_bad_primary_key_identifier_rejected():
    with pytest.raises(ValueError, match="Invalid identifier"):
        CHClickhouseDriverReadOnly(primary_key_column_names=("model_id", "bad-col"))


# --- helpers ------------------------------------------------------------------


def test_validate_ident_accepts_legal_names():
    assert CHClickhouseDriverReadOnly._validate_ident("model_id") == "model_id"
    assert CHClickhouseDriverReadOnly._validate_ident("_x9") == "_x9"


@pytest.mark.parametrize("bad", ["9lives", "a b", "a`b", "", "a;b"])
def test_validate_ident_rejects_illegal_names(bad):
    with pytest.raises(ValueError, match="Invalid identifier"):
        CHClickhouseDriverReadOnly._validate_ident(bad)


def test_backtick():
    assert CHClickhouseDriverReadOnly._backtick("col") == "`col`"


def test_build_select_sql_no_prefix_no_order():
    sql = CHClickhouseDriverReadOnly._build_select_sql(
        db="d", table="t", pk_names=(), select_col_names=("a", "b"), order_by=None,
    )
    assert sql == "SELECT `a`, `b` FROM `d`.`t`"


def test_build_select_sql_two_key_prefix_with_order():
    sql = CHClickhouseDriverReadOnly._build_select_sql(
        db="d",
        table="t",
        pk_names=("k1", "k2"),
        select_col_names=("a",),
        order_by=("k1", "k2"),
    )
    assert sql == (
        "SELECT `a` FROM `d`.`t` "
        "WHERE `k1` = %(k1)s AND `k2` = %(k2)s "
        "ORDER BY `k1`, `k2`"
    )


@pytest.mark.parametrize(
    "cell, expected",
    [
        (memoryview(b"mv"), "mv"),
        (bytearray(b"ba"), "ba"),
        (b"raw", "raw"),
        (7, 7),
        ("already-str", "already-str"),
    ],
)
def test_decode_key_cell(cell, expected):
    assert CHClickhouseDriverReadOnly._decode_key_cell(cell) == expected


def test_ensure_client_early_returns_when_client_present():
    reader, fake = _reader()

    reader._ensure_client()

    assert reader._client is fake


def test_ensure_client_builds_a_real_lazy_client():
    from clickhouse_driver import Client

    reader = CHClickhouseDriverReadOnly()
    reader._ensure_client()

    assert isinstance(reader._client, Client)
    reader.close()
    assert reader._client is None


# --- static decoders ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (b"torch.float32", torch.float32),
        (bytearray(b"torch.float16"), torch.float16),
        (memoryview(b"torch.int64"), torch.int64),
        ("torch.bfloat16", torch.bfloat16),
    ],
)
def test_bytes_to_torch_dtype(raw, expected):
    assert CHClickhouseDriverReadOnly.bytes_to_torch_dtype(raw) is expected


def test_bytes_to_torch_dtype_requires_torch_prefix():
    with pytest.raises(ValueError, match="starting with 'torch.'"):
        CHClickhouseDriverReadOnly.bytes_to_torch_dtype(b"float32")


def test_torch_decode_round_trip():
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    decoded = CHClickhouseDriverReadOnly.torch_decode(
        b"torch.float32", (2, 3), _payload(original),
    )

    assert decoded.dtype is torch.float32
    assert torch.equal(decoded, original)


def test_torch_decode_accepts_memoryview_and_bytearray_payloads():
    original = torch.tensor([1, 2, 3, 4], dtype=torch.int64).reshape(2, 2)
    raw = _payload(original)

    assert torch.equal(
        CHClickhouseDriverReadOnly.torch_decode("torch.int64", (2, 2), memoryview(raw)),
        original,
    )
    assert torch.equal(
        CHClickhouseDriverReadOnly.torch_decode("torch.int64", (2, 2), bytearray(raw)),
        original,
    )


# --- prefix_get ---------------------------------------------------------------


def _full_row(model=b"m", request=b"r0", tensor=None):
    tensor = tensor if tensor is not None else torch.ones(2, dtype=torch.float32)
    return (model, request, b"torch.float32", tuple(tensor.shape), _payload(tensor))


def test_prefix_get_full_key_tuple_decodes_strings():
    tensor = torch.tensor([1.5, 2.5], dtype=torch.float32)
    reader, fake = _reader([_full_row(tensor=tensor)])

    out = reader.prefix_get((b"m",))

    assert len(out) == 1
    key, value = out[0]
    assert key == ("m", "r0")
    assert torch.equal(value, tensor)
    assert fake.calls == [
        (reader._prefix_select_sql_with_key[1], {"model_id": b"m"}),
    ]


def test_prefix_get_full_key_tuple_keeps_raw_bytes_when_decode_disabled():
    reader, _ = _reader([_full_row()], decode_strings=False)

    (key, _), = reader.prefix_get((b"m",))

    assert key == (b"m", b"r0")


def test_prefix_get_values_only():
    tensor = torch.tensor([3.0, 4.0], dtype=torch.float32)
    reader, fake = _reader([(b"torch.float32", (2,), _payload(tensor))])

    out = reader.prefix_get((b"m", b"r0"), return_full_key_tuple=False)

    assert len(out) == 1
    assert torch.equal(out[0], tensor)
    assert fake.calls == [
        (
            reader._prefix_select_sql_values_only[2],
            {"model_id": b"m", "request_id": b"r0"},
        ),
    ]


def test_prefix_get_empty_rows_both_branches():
    reader, _ = _reader([])

    assert reader.prefix_get((b"m",)) == []
    assert reader.prefix_get((b"m",), return_full_key_tuple=False) == []


def test_prefix_get_rejects_non_tuple():
    reader, _ = _reader()
    with pytest.raises(TypeError, match="must be a tuple"):
        reader.prefix_get([b"m"])


def test_prefix_get_rejects_empty_prefix():
    reader, _ = _reader()
    with pytest.raises(ValueError, match="non-empty"):
        reader.prefix_get(())


def test_prefix_get_rejects_prefix_longer_than_primary_key():
    reader, _ = _reader()
    with pytest.raises(ValueError, match="too long"):
        reader.prefix_get((b"m", b"r0", b"extra"))


# --- custom_select ------------------------------------------------------------


@pytest.mark.parametrize("query", ["DROP TABLE offload", "INSERT INTO t VALUES (1)"])
def test_custom_select_rejects_non_select(query):
    reader, fake = _reader()

    with pytest.raises(ValueError, match="requires a SELECT"):
        reader.custom_select(query)
    assert fake.calls == []


def test_custom_select_executes_through_client():
    reader, fake = _reader([(b"x", 1)])

    rows = reader.custom_select("  select 1 WHERE a = %(a)s", {"a": 1})

    assert rows == [(b"x", 1)]
    assert fake.calls == [("  select 1 WHERE a = %(a)s", {"a": 1})]


def test_custom_select_empty_result_returns_empty_list():
    reader, _ = _reader([])

    assert reader.custom_select("SELECT 1") == []


# --- close and properties -----------------------------------------------------


def test_close_disconnects_and_resets_client():
    reader, fake = _reader()

    reader.close()

    assert fake.disconnected is True
    assert reader._client is None

    reader.close()  # no client: a no-op
    assert reader._client is None


def test_close_resets_client_even_when_disconnect_raises():
    reader, fake = _reader()

    def _boom():
        raise RuntimeError("socket gone")

    fake.disconnect = _boom
    with pytest.raises(RuntimeError, match="socket gone"):
        reader.close()
    assert reader._client is None


def test_property_getters():
    reader = CHClickhouseDriverReadOnly(
        database="db1",
        table="tbl",
        primary_key_column_names=("model_id", "request_id"),
    )

    assert reader.database == "db1"
    assert reader.table == "tbl"
    assert reader.primary_keys_columns == ("model_id", "request_id")
    assert reader.value_columns == ("dtype", "shape", "bytes")
    assert reader.columns == ("model_id", "request_id", "dtype", "shape", "bytes")
