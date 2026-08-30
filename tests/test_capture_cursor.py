from __future__ import annotations

from base64 import urlsafe_b64encode
import json

import pytest

from dmi.storage.capture import (
    CaptureQuery,
    Cursor,
    CursorKey,
    InvalidCursorError,
    decode_cursor,
    encode_cursor,
)


pytestmark = pytest.mark.cpu


_FILTER_HASH = "a" * 64
_WATERMARK = 1_756_142_093_000_000_000
_MAX_WATERMARK = 1_756_142_099_000_000_000


def _key(**overrides) -> CursorKey:
    base = {
        "tenant_id": "tenant-a",
        "experiment_id": "experiment-a",
        "run_id": "run-a",
        "captured_at_ns": 1_756_142_090_000_000_000,
        "capture_id": "capture-0f31",
    }
    return CursorKey(**{**base, **overrides})


def _encoded(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _valid_payload(**overrides) -> dict:
    payload = {
        "v": 1,
        "w": _WATERMARK,
        "fh": _FILTER_HASH,
        "k": [
            "tenant-a",
            "experiment-a",
            "run-a",
            1_756_142_090_000_000_000,
            "capture-0f31",
        ],
    }
    payload.update(overrides)
    return payload


def _decode(cursor: str) -> Cursor:
    return decode_cursor(
        cursor, filter_hash=_FILTER_HASH, max_watermark=_MAX_WATERMARK
    )


# --- round trip -------------------------------------------------------------


def test_round_trip_preserves_every_field():
    encoded = encode_cursor(_key(), watermark=_WATERMARK, filter_hash=_FILTER_HASH)

    decoded = _decode(encoded)

    assert decoded == Cursor(
        version=1, watermark=_WATERMARK, filter_hash=_FILTER_HASH, key=_key()
    )


def test_encoding_is_deterministic():
    first = encode_cursor(_key(), watermark=_WATERMARK, filter_hash=_FILTER_HASH)
    second = encode_cursor(_key(), watermark=_WATERMARK, filter_hash=_FILTER_HASH)

    assert first == second


def test_encoded_cursor_is_accepted_by_capture_query():
    encoded = encode_cursor(_key(), watermark=_WATERMARK, filter_hash=_FILTER_HASH)

    # Unpadded url-safe base64 keeps the cursor inside CaptureQuery's validator
    # and free of characters that would need escaping in a URL.
    assert "=" not in encoded
    assert CaptureQuery(cursor=encoded).cursor == encoded


def test_encode_refuses_a_cursor_beyond_the_query_limit():
    with pytest.raises(InvalidCursorError, match="cursor"):
        encode_cursor(
            _key(tenant_id="t" * 512, experiment_id="e" * 512, run_id="r" * 512,
                 capture_id="c" * 512),
            watermark=_WATERMARK,
            filter_hash=_FILTER_HASH,
        )


# --- binding ----------------------------------------------------------------


def test_decode_rejects_a_cursor_from_different_filters():
    encoded = encode_cursor(_key(), watermark=_WATERMARK, filter_hash="b" * 64)

    with pytest.raises(InvalidCursorError, match="filter"):
        _decode(encoded)


def test_decode_rejects_a_watermark_above_the_catalog():
    encoded = encode_cursor(
        _key(), watermark=_MAX_WATERMARK + 1, filter_hash=_FILTER_HASH
    )

    with pytest.raises(InvalidCursorError, match="watermark"):
        _decode(encoded)


def test_decode_accepts_a_watermark_equal_to_the_catalog():
    encoded = encode_cursor(
        _key(), watermark=_MAX_WATERMARK, filter_hash=_FILTER_HASH
    )

    assert _decode(encoded).watermark == _MAX_WATERMARK


# --- malformed input --------------------------------------------------------


@pytest.mark.parametrize(
    "cursor",
    (
        "",
        "!!!not base64!!!",
        "z",
        urlsafe_b64encode(b"not json").rstrip(b"=").decode("ascii"),
        _encoded([1, 2, 3]),
        _encoded("a string"),
        _encoded(None),
    ),
    ids=(
        "empty", "non-base64", "truncated", "not-json",
        "json-array", "json-string", "json-null",
    ),
)
def test_decode_rejects_unparseable_cursors(cursor: str):
    with pytest.raises(InvalidCursorError):
        _decode(cursor)


@pytest.mark.parametrize(
    "payload,reason",
    (
        (_valid_payload(v=2), "unknown-version"),
        (_valid_payload(v="1"), "version-not-int"),
        (_valid_payload(w=-1), "negative-watermark"),
        (_valid_payload(w=2**64), "watermark-overflow"),
        (_valid_payload(w="1"), "watermark-not-int"),
        (_valid_payload(fh=123), "filter-hash-not-str"),
        (_valid_payload(k=["a", "b", "c", 1]), "key-too-short"),
        (_valid_payload(k=["a", "b", "c", 1, "d", "e"]), "key-too-long"),
        (_valid_payload(k="not-a-list"), "key-not-list"),
        (_valid_payload(k=["a", "b", "c", "not-int", "d"]), "timestamp-not-int"),
        (_valid_payload(k=["a", "b", "c", -1, "d"]), "negative-timestamp"),
        (_valid_payload(k=[1, "b", "c", 1, "d"]), "tenant-not-str"),
        (_valid_payload(k=["", "b", "c", 1, "d"]), "empty-tenant"),
        (_valid_payload(k=["a", "b", "c", 1, ""]), "empty-capture-id"),
    ),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_decode_rejects_malformed_payloads(payload: dict, reason: str):
    with pytest.raises(InvalidCursorError):
        _decode(_encoded(payload))


@pytest.mark.parametrize("missing", ("v", "w", "fh", "k"))
def test_decode_rejects_missing_fields(missing: str):
    payload = _valid_payload()
    del payload[missing]

    with pytest.raises(InvalidCursorError):
        _decode(_encoded(payload))


@pytest.mark.parametrize("junk", ("!!!!", "....", "    ", "\n\n\n\n"))
def test_decode_rejects_characters_outside_the_base64_alphabet(junk: str):
    # Python's default base64 decoder discards these silently. Injecting a
    # multiple of four keeps the padding arithmetic intact, so a lax decoder
    # accepts the tampered cursor and returns the original payload.
    encoded = encode_cursor(_key(), watermark=_WATERMARK, filter_hash=_FILTER_HASH)
    tampered = encoded[:10] + junk + encoded[10:]

    with pytest.raises(InvalidCursorError, match="base64"):
        _decode(tampered)


def test_decode_rejects_unexpected_fields():
    # A strict envelope keeps a future field from being silently ignored by an
    # older reader.
    with pytest.raises(InvalidCursorError):
        _decode(_encoded(_valid_payload(extra="surprise")))


def test_invalid_cursor_error_is_a_capture_storage_error():
    from dmi.storage.capture import CaptureStorageError

    assert issubclass(InvalidCursorError, CaptureStorageError)


def test_decode_rejects_a_non_canonical_base64_spelling():
    from base64 import b64decode, b64encode

    encoded = encode_cursor(_key(), watermark=_WATERMARK, filter_hash=_FILTER_HASH)
    padding = "=" * (-len(encoded) % 4)
    raw = b64decode(encoded + padding, altchars=b"-_")
    standard = b64encode(raw).rstrip(b"=").decode("ascii")

    if standard == encoded:
        # No '+' or '/' arose from this payload; force a non-canonical
        # character to prove the alphabet check itself.
        standard = encoded[:-1] + "+"

    # CPython's b64decode(altchars=..., validate=True) still accepts literal
    # '+' and '/', so without an explicit alphabet check two spellings would
    # alias the same page position.
    with pytest.raises(InvalidCursorError, match="canonical"):
        decode_cursor(
            standard, filter_hash=_FILTER_HASH, max_watermark=_MAX_WATERMARK
        )


def test_decode_rejects_a_boolean_cursor_version():
    encoded = encode_cursor(_key(), watermark=_WATERMARK, filter_hash=_FILTER_HASH)
    from base64 import b64decode as _b64decode

    padding = "=" * (-len(encoded) % 4)
    payload = json.loads(_b64decode(encoded + padding, altchars=b"-_"))
    payload["v"] = True  # True == 1 in Python; the version check must reject it

    with pytest.raises(InvalidCursorError, match="version"):
        decode_cursor(
            _encoded(payload), filter_hash=_FILTER_HASH, max_watermark=_MAX_WATERMARK
        )
