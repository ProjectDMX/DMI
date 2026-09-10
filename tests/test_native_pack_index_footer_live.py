"""B3: the native pack-footer reader against the Python reader oracle.

The pack footer is EXTERNAL data. `pack_index.h` says so, and
`hydration.cpp` proves it: the ref is rebuilt from ClickHouse descriptor
rows and the footer is re-read from the object store at QUERY time, in a
different process, long after whatever wrote it. Both pack builders
validate metadata before sealing a footer, so a hostile footer needs a
CRC-consistent object from a foreign producer with bucket write access
rather than bit rot -- but the reader-side layer is the defence the header
comment promises, and `model.py` states its purpose outright: "Rejecting
out-of-range values here keeps one poison record from failing the
persistence thread or wedging catalog indexing."

Every footer here is re-sealed with a valid CRC and body hash through the
ORACLE's own `_reseal` helper, so each case fails on the field it corrupts
and not on a checksum. Each case is then driven through BOTH readers --
native `op: "index"` and `PackIndex.from_store` -- over the same bytes, and
the two refusals are compared.

Which tests are bug proofs and which are regression guards:

* `test_metadata_bounds_are_refused_at_the_footer_boundary`,
  `test_a_poison_shape_that_clickhouse_stores_is_refused_by_the_reader`
  and `test_footer_arithmetic_is_overflow_checked` are BUG PROOFS: they
  failed against the build that admitted footers the oracle refuses, and
  admitted catalog rows the Python reader then refused on read-back.
* `test_the_trailer_and_footer_refusal_matrix_matches_the_oracle` is
  mostly a REGRESSION GUARD -- twelve of its sixteen cases passed on
  first write and are here to keep passing, because the sentence a
  refusal comes out with is what an operator reads. Four did not:
  `records-not-list`, `duplicate-capture-id`, `offset-0` and
  `footer-not-json` each named the wrong refusal, and those four are bug
  proofs for the message and ordering half of the parity.
* `test_the_admitted_dtype_set_agrees_with_the_oracle` is a DRIFT GUARD
  in both directions -- it asserts the two readers agree on each dtype
  name, not that any particular name is refused, because the contract's
  dtype table grows.
* `test_a_well_formed_pack_still_indexes_through_the_same_path` is the
  control: without it a reader that refused everything would pass every
  refusal case above.

These need a live ClickHouse and the driver binary; they run in the
clickhouse-live job beside the rest of the native catalog suite.
"""

from __future__ import annotations

import hashlib
import struct

import pytest

# Module-level so the fake-S3 fixture re-exports into this module's
# namespace (function-local imports do not register fixtures).
from tests.test_native_s3_client import (  # noqa: E402,F401
    ACCESS, BUCKET, REGION, SECRET, STATE, fake_s3,
)
from tests.test_native_catalog_lease_live import (  # noqa: E402
    DRIVER, CatalogDriver, _catalog, _open,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.clickhouse,
    pytest.mark.skipif(
        not DRIVER.exists(),
        reason="native/build/conformance_catalog is not built; run "
        "`make -C native build/conformance_catalog`",
    ),
]

OBJECT_KEY = "packs/footer-parity.dmi-pack"
STORE_ID = "local"


# --- the oracle's own fixtures, reused rather than re-derived ---------------


def _oracle():
    from tests.test_capture_pack_format import (
        PACK_ID, _MemStore, _record, _reseal, _sealed_pack,
    )

    return PACK_ID, _MemStore, _record, _reseal, _sealed_pack


def _two_record_pack():
    """A sealed two-record pack, plus the oracle helpers bound to it."""
    _, _, record, reseal, sealed_pack = _oracle()
    sealed = sealed_pack(
        record("capture-a", b"\x00" * 8),
        record("capture-b", b"\x01" * 8, step=1),
    )
    return sealed, reseal


def _one_record_pack():
    """A sealed one-record pack, for the cases a second record would mask.

    A footer whose record ranges wrap is caught by the range ORDERING check
    when a second record follows it -- by accident, and with the wrong
    sentence. One record shows the acceptance itself.
    """
    _, _, record, reseal, sealed_pack = _oracle()
    return sealed_pack(record("capture-a", b"\x00" * 8)), reseal


def _put(data: bytes) -> None:
    STATE.objects[OBJECT_KEY] = {
        "body": data,
        "meta": {},
        "content_type": "",
        "etag": f'"{hashlib.md5(data).hexdigest()}"',
    }


def _native_ref(data: bytes, checksum: str, *, record_count: int,
                object_bytes: int | None = None) -> dict:
    PACK_ID = _oracle()[0]
    return {
        "pack_id": str(PACK_ID),
        "store_id": STORE_ID,
        "object_key": OBJECT_KEY,
        "object_bytes": len(data) if object_bytes is None else object_bytes,
        "checksum": checksum,
        "record_count": record_count,
    }


def _oracle_refusal(data: bytes, ref: dict) -> str:
    """What `PackIndex.from_store` says about the same bytes, or "".

    An empty answer means the oracle ACCEPTED the footer, which is itself
    a divergence when native refused it.
    """
    from dmi.storage.capture import PackRef
    from dmi.storage.capture.model import CaptureStorageError

    _, mem_store, _, _, _ = _oracle()
    from dmi.storage.capture.pack import PackIndex

    store = mem_store(data)
    py_ref = PackRef(
        pack_id=ref["pack_id"],
        store_id=ref["store_id"],
        object_key=ref["object_key"],
        object_bytes=ref["object_bytes"],
        checksum=ref["checksum"],
        record_count=ref["record_count"],
    )
    try:
        PackIndex.from_store(store, py_ref).descriptors()
    except (CaptureStorageError, ValueError) as exc:
        return str(exc)
    return ""


def _native_refusal(driver, endpoint: str, ref: dict) -> str:
    """What the native reader says about the same bytes, or "".

    A pack whose read fails is a per-pack failure, so the batch itself
    still succeeds; an empty answer means native ACCEPTED the footer.
    """
    result = driver.call(
        op="index", refs=[ref], endpoint=endpoint, bucket=BUCKET,
        region=REGION, access=ACCESS, secret=SECRET, insecure=True,
    )
    assert result["ok"], result
    failures = result["result"]["failures"]
    if not failures:
        return ""
    assert len(failures) == 1, failures
    return failures[0]["message"]


def _assert_same_refusal(case: str, native: str, oracle: str) -> None:
    """Both readers refused, and named the same thing.

    Exact equality, or the oracle's sentence being native's followed by
    ``: <the offending value>`` -- which is the only shape in which the
    two are allowed to differ, because the oracle sometimes quotes the
    value it refused ("unsupported pack version: 99.0") where this port
    names only the field.
    """
    assert native, f"{case}: native ACCEPTED a footer the oracle refuses " \
                   f"with {oracle!r}"
    assert oracle, f"{case}: the oracle accepted a footer native refuses " \
                   f"with {native!r}"
    assert oracle == native or oracle.startswith(native + ": "), (
        f"{case}: native says {native!r}, oracle says {oracle!r}"
    )


# --- the metadata bounds ---------------------------------------------------


def _metadata_set(key, value):
    def mutate(decoded):
        decoded["records"][0]["metadata"][key] = value

    return mutate


def _shape_and_length(dims, decoded_length):
    """Set the first record's shape, keeping its lengths self-consistent."""

    def mutate(decoded):
        record = decoded["records"][0]
        record["metadata"]["shape"] = list(dims)
        record["stored_length"] = decoded_length
        record["decoded_length"] = decoded_length

    return mutate


METADATA_CASES = (
    # (case name, footer mutation)
    ("shape-dim-above-int32", _shape_and_length([0, 2**32 - 1], 0)),
    ("producer_rank-negative", _metadata_set("producer_rank", -1)),
    ("producer_rank-above-uint32", _metadata_set("producer_rank", 2**32 + 9)),
    ("layer_number-below-minus-one", _metadata_set("layer_number", -5)),
    ("captured_at_ns-negative", _metadata_set("captured_at_ns", -1)),
    # token_end is 1 on this record, so token_start=5 is the violation;
    # token_end=-1 would be refused as a negative counter first.
    ("token_end-before-token_start", _metadata_set("token_start", 5)),
    ("shape-rank-40", _shape_and_length([1] * 40, 4)),
    ("capture_id-empty", _metadata_set("capture_id", "")),
    # adapter_revision is the one optional identifier, and "optional" on
    # the oracle's side means None -- a JSON null -- and nothing else. The
    # empty STRING is refused like any other identifier.
    ("adapter_revision-empty", _metadata_set("adapter_revision", "")),
)


@pytest.mark.parametrize("case,mutate", METADATA_CASES,
                         ids=[name for name, _ in METADATA_CASES])
def test_metadata_bounds_are_refused_at_the_footer_boundary(
        fake_s3, case, mutate):
    """BUG PROOF: native copied these straight into SQL text.

    `render_record_row` validated the dtype x shape == decoded_length
    product and nothing else, while the oracle runs the whole of
    `CaptureMetadata.__post_init__` at the same boundary
    (`pack.py` `_parse_record` -> `CaptureMetadata.from_mapping`). Every
    case here is a footer the oracle refuses and native admitted.
    """
    sealed, reseal = _two_record_pack()
    data = reseal(sealed.data, mutate=mutate)
    _put(data)
    ref = _native_ref(data, sealed.checksum, record_count=2)

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            # A lease, so that a build which ACCEPTS the footer gets as far
            # as writing the row -- the failure then reads as "native
            # accepted it", not as a missing lease.
            driver.call(op="acquire", holder="indexer")
            native = _native_refusal(driver, fake_s3, ref)
        finally:
            driver.close()
        _assert_same_refusal(case, native, _oracle_refusal(data, ref))
        # Refused means nothing was written, for any of these.
        assert client.execute(
            f"SELECT count() FROM `{config.database}`."
            f"`{prefix}_capture_raw`") == [(0,)]


def test_a_poison_shape_that_clickhouse_stores_is_refused_by_the_reader(
        fake_s3):
    """BUG PROOF, worst case: the row LANDS, and poisons the page.

    `stored_length=0, decoded_length=0, shape=[0, 4294967295]` is
    ClickHouse-representable -- it fits `Array(UInt32)` and stores cleanly
    -- so the insert does not error. Native wrote the row, and then
    `clickhouse_reader._descriptor_of` refuses it on read-back, taking the
    whole search page down with it.

    Two halves, so the harm is on the record independently of the fix:
    first that such a row really is unreadable (written straight through
    the driver's descriptor path, which is not what this change touches),
    and then that the footer carrying it is refused before it can become
    one.
    """
    from dmi.storage.capture import CaptureQuery
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog,
        ClickHouseReaderConfig,
    )
    from tests.test_native_catalog_lease_live import (
        _descriptor_dicts, _refs_of,
    )

    poison_shape = [0, 2**32 - 1]
    sealed, reseal = _two_record_pack()
    data = reseal(sealed.data, mutate=_shape_and_length(poison_shape, 0))
    _put(data)
    ref = _native_ref(data, sealed.checksum, record_count=2)

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")

            # Half one: a row with this shape is unreadable. ClickHouse
            # accepts it without complaint -- it wraps out-of-range
            # integers silently rather than erroring -- and the Python
            # reader is where it detonates.
            descriptors = _descriptor_dicts(1)
            descriptors[0]["shape"] = poison_shape
            version = driver.call(op="allocate_version")["version"]
            assert driver.call(op="write_descriptors",
                               descriptors=descriptors,
                               index_version=version)["ok"]
            assert driver.call(op="publish_snapshot", index_version=version,
                               refs=_refs_of(1), published_at_ns=version,
                               indexed_rows=1, indexed_packs=1)["ok"]
            (stored_shape,), = client.execute(
                f"SELECT shape FROM `{config.database}`."
                f"`{prefix}_capture_raw`")
            assert list(stored_shape) == poison_shape, stored_shape
            reader = ClickHouseCaptureCatalog(
                client, ClickHouseReaderConfig.from_catalog(config))
            with pytest.raises(ValueError,
                               match=r"shape dimensions must be integers"):
                reader.search(CaptureQuery(limit=10))

            # Half two: the footer that would have produced it is refused
            # at the boundary, so the row never exists.
            native = _native_refusal(driver, fake_s3, ref)
        finally:
            driver.close()
        _assert_same_refusal("poison-shape", native,
                             _oracle_refusal(data, ref))


# --- the admitted dtype set ------------------------------------------------


def _dtype_and_length(dtype, decoded_length):
    def mutate(decoded):
        record = decoded["records"][0]
        record["metadata"]["dtype"] = dtype
        record["metadata"]["shape"] = [2]
        record["stored_length"] = decoded_length
        record["decoded_length"] = decoded_length

    return mutate


# Names spanning three groups: admitted by every version of the contract,
# admitted by some (fp8 and the wide integers become first-class further up
# this stack), and admitted by none.
DTYPE_PROBES = ("float32", "bool", "int64", "uint16", "uint32",
                "float8_e4m3fn", "float8_e5m2", "uint64", "float128",
                "not-a-dtype")


@pytest.mark.parametrize("dtype", DTYPE_PROBES)
def test_the_admitted_dtype_set_agrees_with_the_oracle(fake_s3, dtype):
    """Native admits a footer's dtype exactly when the oracle admits it.

    Deliberately NOT a list of names this test expects to be refused.
    `_DTYPE_BYTES` in model.py is the contract and it GROWS: a test that
    pinned today's membership would fail the moment the oracle widened,
    and a reader that pinned it would refuse dtypes the pipeline had
    declared first-class -- which is why `dtype_bytes` asks
    `dmi_pack::DtypeSupported` for membership rather than keeping a list.
    What is asserted here is AGREEMENT, in whichever direction the
    contract moves: for each name, both readers admit the footer or both
    refuse it with the same sentence.

    `uint64` earns its place: `dtype_bytes` knows a width for it, and no
    version of the contract admits it. It is the case that would catch a
    width table quietly widening what is admitted.
    """
    from dmi.storage.capture.model import _DTYPE_BYTES

    # A length consistent with the oracle's OWN width where it has one.
    # Where it has none the dtype is refused before any length is looked
    # at, so the value cannot matter.
    length = 2 * _DTYPE_BYTES.get(dtype, 4)
    sealed, reseal = _one_record_pack()
    data = reseal(sealed.data, mutate=_dtype_and_length(dtype, length))
    _put(data)
    ref = _native_ref(data, sealed.checksum, record_count=1)
    oracle = _oracle_refusal(data, ref)

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            native = _native_refusal(driver, fake_s3, ref)
        finally:
            driver.close()
        rows = client.execute(
            f"SELECT count() FROM `{config.database}`."
            f"`{prefix}_capture_raw`")
        if oracle:
            _assert_same_refusal(dtype, native, oracle)
            assert rows == [(0,)], (dtype, rows)
        else:
            # The oracle admits it, so native must, and must have written
            # the row rather than merely not complaining.
            assert native == "", (
                f"{dtype}: native refuses a footer the oracle admits: "
                f"{native!r}")
            assert rows == [(1,)], (dtype, rows)


def _offset_beyond_int64(decoded):
    record = decoded["records"][0]
    record["offset"] = 9223372036854775800
    record["stored_length"] = 16
    record["decoded_length"] = 16
    record["metadata"]["shape"] = [4]


OVERFLOW_CASES = (
    # A dimension above 2^64 wraps to 2 in a naive digit loop, which is
    # inside every bound the model has.
    ("shape-dim-2p64", _shape_and_length([2**64 + 2], 8)),
    # offset + stored_length overflows int64, and signed overflow is
    # undefined: the wrapped sum compared BELOW footer_offset.
    ("offset-int64-overflow", _offset_beyond_int64),
)


@pytest.mark.parametrize("case,mutate", OVERFLOW_CASES,
                         ids=[name for name, _ in OVERFLOW_CASES])
def test_footer_arithmetic_is_overflow_checked(fake_s3, case, mutate):
    """BUG PROOF: `shape_product` wrapped and `offset + stored` was UB.

    Both are computed from footer numbers, so both are attacker-chosen.
    A wrapping product can land back on `decoded_length`; a wrapped
    `offset + stored_length` compares below `footer_offset` and admits a
    record that reaches into the footer. Executed at -O2, the level the
    driver ships at, so neither is a -O0 artefact.
    """
    sealed, reseal = _one_record_pack()
    data = reseal(sealed.data, mutate=mutate)
    _put(data)
    ref = _native_ref(data, sealed.checksum, record_count=1)

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            native = _native_refusal(driver, fake_s3, ref)
        finally:
            driver.close()
        _assert_same_refusal(case, native, _oracle_refusal(data, ref))
        assert client.execute(
            f"SELECT count() FROM `{config.database}`."
            f"`{prefix}_capture_raw`") == [(0,)]


# --- the trailer and footer refusal matrix ---------------------------------


def _patch(data: bytes, offset: int, value: bytes) -> bytes:
    changed = bytearray(data)
    changed[offset:offset + len(value)] = value
    return bytes(changed)


def _footer_set(key, value):
    def mutate(decoded):
        decoded[key] = value

    return mutate


def _duplicated(decoded):
    decoded["records"] = [decoded["records"][0], decoded["records"][0]]


def _reordered(decoded):
    decoded["records"] = decoded["records"][::-1]


def _offset_zero(decoded):
    decoded["records"][0]["offset"] = 0


def _trailer_at(data: bytes) -> int:
    from dmi.storage.capture.pack import _TRAILER

    return len(data) - _TRAILER.size


def _flip_footer_crc(data: bytes) -> bytes:
    """Invert the trailer's footer CRC, whatever it happens to be.

    Writing a constant would be a no-op on the one pack whose footer CRC
    already equalled it.
    """
    at = _trailer_at(data) + 28
    (crc,) = struct.unpack_from("<I", data, at)
    return _patch(data, at, struct.pack("<I", crc ^ 0xFFFFFFFF))


# Each case yields the object bytes and the ref overrides, from a sealed
# two-record pack. `reseal` re-signs the footer CRC and the body hash, so
# a case that mutates the footer JSON is refused on its own merits.
FOOTER_CASES = (
    ("truncated", lambda d, r: (d, {"object_bytes": 16})),
    ("invalid-trailer",
     lambda d, r: (_patch(d, _trailer_at(d), b"NOTAFTRX"), {})),
    ("unsupported-version",
     lambda d, r: (_patch(d, _trailer_at(d) + 8, struct.pack("<H", 99)), {})),
    ("footer-too-large",
     lambda d, r: (_patch(d, _trailer_at(d) + 20,
                          struct.pack("<Q", 64 * 1024 * 1024 + 1)), {})),
    ("footer-range-invalid",
     lambda d, r: (_patch(d, _trailer_at(d) + 12, struct.pack("<Q", 1)), {})),
    ("footer-crc-mismatch", lambda d, r: (_flip_footer_crc(d), {})),
    ("footer-not-json", lambda d, r: (r(d, footer=b"{broken"), {})),
    ("footer-not-an-object", lambda d, r: (r(d, footer=b"[]"), {})),
    ("format-marker",
     lambda d, r: (r(d, mutate=_footer_set("format", "zip")), {})),
    ("footer-version-mismatch",
     lambda d, r: (r(d, mutate=_footer_set("major_version", 2)), {})),
    ("foreign-pack-id",
     lambda d, r: (r(d, mutate=_footer_set(
         "pack_id", "018f0000-0000-7000-8000-00000000ffff")), {})),
    ("records-not-list",
     lambda d, r: (r(d, mutate=_footer_set("records", 5)), {})),
    ("record-count-mismatch", lambda d, r: (d, {"record_count": 5})),
    ("records-out-of-order", lambda d, r: (r(d, mutate=_reordered), {})),
    ("duplicate-capture-id",
     lambda d, r: (r(d, mutate=_duplicated), {})),
    ("offset-0", lambda d, r: (r(d, mutate=_offset_zero), {})),
)


@pytest.mark.parametrize("case,build", FOOTER_CASES,
                         ids=[name for name, _ in FOOTER_CASES])
def test_the_trailer_and_footer_refusal_matrix_matches_the_oracle(
        fake_s3, case, build):
    """Mostly a REGRESSION GUARD; four cases are bug proofs.

    Twelve of these sixteen passed on first write and are here to keep
    passing -- they pin the sentence each refusal comes out with, which is
    what an operator reads. The four that did not pass named the wrong
    refusal: `records-not-list` blamed the record's metadata instead of
    the record list, `duplicate-capture-id` and `offset-0` both came out
    as "ranges overlap or are out of order" because the range ordering was
    checked before the record was, and `footer-not-json` came out as an
    invalid format marker.
    """
    sealed, reseal = _two_record_pack()
    data, overrides = build(sealed.data, reseal)
    _put(data)
    ref = _native_ref(data, sealed.checksum, record_count=2)
    ref.update(overrides)

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            native = _native_refusal(driver, fake_s3, ref)
        finally:
            driver.close()
        _assert_same_refusal(case, native, _oracle_refusal(data, ref))


def test_a_well_formed_pack_still_indexes_through_the_same_path(fake_s3):
    """The matrix above is a guard only if the unmutated pack still passes.

    Same helpers, same injection, no mutation: two records read out of the
    footer and rendered into two descriptor rows, which the Python reader
    then resolves. Without this, every refusal case above would also pass
    against a reader that refused everything.
    """
    from dmi.storage.capture import CaptureQuery
    from dmi.storage.capture.clickhouse_reader import (
        ClickHouseCaptureCatalog,
        ClickHouseReaderConfig,
    )

    sealed, _ = _two_record_pack()
    _put(sealed.data)
    ref = _native_ref(sealed.data, sealed.checksum, record_count=2)

    with _catalog() as (client, config, prefix):
        driver = CatalogDriver()
        try:
            _open(driver, prefix)
            driver.call(op="acquire", holder="indexer")
            result = driver.call(
                op="index", refs=[ref], endpoint=fake_s3, bucket=BUCKET,
                region=REGION, access=ACCESS, secret=SECRET, insecure=True,
            )
            assert result["ok"], result
            assert result["result"]["failures"] == [], result
            assert result["result"]["indexed_rows"] == 2, result
        finally:
            driver.close()

        reader = ClickHouseCaptureCatalog(
            client, ClickHouseReaderConfig.from_catalog(config))
        page = reader.search(CaptureQuery(limit=10))
        assert {item.capture_id for item in page.items} == {
            "capture-a", "capture-b"}, page
        # And the oracle read the same two out of the same bytes.
        assert _oracle_refusal(sealed.data, ref) == ""
