"""Keyset pagination cursors for bounded catalog search.

A cursor addresses a position in the catalog's own sort order --
``(tenant_id, experiment_id, run_id, captured_at_ns, capture_id)`` -- so page
latency does not grow with depth and a concurrent insert cannot shift a page
boundary.

The encoding is deliberately strict. A cursor is data handed back by a caller,
so every field is validated on the way in and any deviation raises
:class:`InvalidCursorError` rather than silently yielding a different page.
"""

from __future__ import annotations

from base64 import b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from dataclasses import dataclass
import json

from .model import _CURSOR_LIMIT, InvalidCursorError


_CURSOR_VERSION = 1
_ENVELOPE_FIELDS = frozenset({"v", "w", "fh", "k"})
_UINT64_MAX = 2**64 - 1


@dataclass(frozen=True, slots=True)
class CursorKey:
    """A position in the catalog sort order."""

    tenant_id: str
    experiment_id: str
    run_id: str
    captured_at_ns: int
    capture_id: str


@dataclass(frozen=True, slots=True)
class Cursor:
    """A decoded cursor, bound to the query and snapshot that issued it."""

    version: int
    watermark: int
    filter_hash: str
    key: CursorKey


def _require(condition: object, message: str) -> None:
    if not condition:
        raise InvalidCursorError(message)


def _uint64(value: object, name: str) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    _require(0 <= value <= _UINT64_MAX, f"{name} must fit UInt64")
    return value  # type: ignore[return-value]


def _text(value: object, name: str) -> str:
    _require(isinstance(value, str) and value, f"{name} must be a non-empty string")
    return value  # type: ignore[return-value]


def encode_cursor(key: CursorKey, *, watermark: int, filter_hash: str) -> str:
    """Encode a position as an opaque, url-safe cursor."""
    payload = {
        "v": _CURSOR_VERSION,
        "w": _uint64(watermark, "watermark"),
        "fh": _text(filter_hash, "filter_hash"),
        "k": [
            _text(key.tenant_id, "tenant_id"),
            _text(key.experiment_id, "experiment_id"),
            _text(key.run_id, "run_id"),
            _uint64(key.captured_at_ns, "captured_at_ns"),
            _text(key.capture_id, "capture_id"),
        ],
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if len(encoded.encode("utf-8")) > _CURSOR_LIMIT:
        raise InvalidCursorError(
            f"encoded cursor exceeds the cursor limit of {_CURSOR_LIMIT} bytes"
        )
    return encoded


def decode_cursor(cursor: str, *, filter_hash: str, max_watermark: int) -> Cursor:
    """Decode and validate a cursor against the query and catalog presenting it.

    ``filter_hash`` binds the cursor to one set of filters, so a cursor cannot
    be replayed against a different query. ``max_watermark`` is the catalog's
    current high-water version; a cursor above it does not describe a snapshot
    this catalog can serve.
    """
    _require(isinstance(cursor, str) and cursor, "cursor must be a non-empty string")
    _require(
        len(cursor.encode("utf-8")) <= _CURSOR_LIMIT,
        f"cursor exceeds the cursor limit of {_CURSOR_LIMIT} bytes",
    )

    # ``validate=True`` matters here: the default decoder silently discards
    # characters outside the base64 alphabet, so a cursor with injected junk
    # would decode to the same payload and pass every check below.
    padding = "=" * (-len(cursor) % 4)
    try:
        raw = b64decode(cursor + padding, altchars=b"-_", validate=True)
    except (BinasciiError, ValueError) as exc:
        raise InvalidCursorError("cursor is not valid url-safe base64") from exc

    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor does not contain valid JSON") from exc

    _require(isinstance(payload, dict), "cursor payload must be a JSON object")
    unexpected = set(payload) - _ENVELOPE_FIELDS
    _require(not unexpected, f"cursor has unexpected fields: {sorted(unexpected)}")
    missing = _ENVELOPE_FIELDS - set(payload)
    _require(not missing, f"cursor is missing fields: {sorted(missing)}")

    _require(payload["v"] == _CURSOR_VERSION, f"unsupported cursor version: {payload['v']!r}")

    watermark = _uint64(payload["w"], "cursor watermark")
    _require(
        watermark <= max_watermark,
        "cursor watermark is ahead of the catalog: "
        f"{watermark} > {max_watermark}",
    )

    encoded_hash = _text(payload["fh"], "cursor filter hash")
    _require(
        encoded_hash == filter_hash,
        "cursor was issued for different filters",
    )

    key = payload["k"]
    _require(isinstance(key, list), "cursor key must be a JSON array")
    _require(len(key) == 5, f"cursor key must have five elements, got {len(key)}")

    return Cursor(
        version=_CURSOR_VERSION,
        watermark=watermark,
        filter_hash=encoded_hash,
        key=CursorKey(
            tenant_id=_text(key[0], "cursor tenant_id"),
            experiment_id=_text(key[1], "cursor experiment_id"),
            run_id=_text(key[2], "cursor run_id"),
            captured_at_ns=_uint64(key[3], "cursor captured_at_ns"),
            capture_id=_text(key[4], "cursor capture_id"),
        ),
    )
