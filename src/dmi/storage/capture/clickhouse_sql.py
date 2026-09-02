from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

MAX_INLINE_PARAMETER_BYTES = 192 * 1024

T = TypeVar("T")


def inline_text_bytes(value: str) -> int:
    if not isinstance(value, str):
        raise TypeError("inline text parameters must be strings")
    return 2 * len(value.encode("utf-8")) + 2


def inline_tuple_bytes(item: tuple[str, str]) -> int:
    if len(item) != 2:
        raise ValueError("pack identities must contain two text values")
    return sum(inline_text_bytes(value) for value in item) + 4


def inline_chunks(
    items: Iterable[T], *, item_bytes: Callable[[T], int]
) -> Iterator[list[T]]:
    chunk: list[T] = []
    size = 2
    for item in items:
        encoded = item_bytes(item)
        if encoded + 2 > MAX_INLINE_PARAMETER_BYTES:
            raise ValueError("item exceeds inline query byte budget")
        separator = 2 if chunk else 0
        if chunk and size + separator + encoded > MAX_INLINE_PARAMETER_BYTES:
            yield chunk
            chunk = []
            size = 2
            separator = 0
        chunk.append(item)
        size += separator + encoded
    if chunk:
        yield chunk


def membership_predicate(manifest: str, watermark: str, *, bounded: bool) -> str:
    manifest_bound = "index_version <= %(watermark)s AND " if bounded else ""
    watermark_bound = " WHERE index_version <= %(watermark)s" if bounded else ""
    return (
        "(store_id, pack_id) IN ("
        f"SELECT store_id, pack_id FROM {manifest} "
        f"WHERE {manifest_bound}(index_version, publish_id) IN "
        f"(SELECT index_version, publish_id FROM {watermark}{watermark_bound}))"
    )
