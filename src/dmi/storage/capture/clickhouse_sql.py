from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol, TypeVar

MAX_INLINE_PARAMETER_BYTES = 192 * 1024

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# Settings for the statements whose answers DECIDE something: which claimant
# owns a version, whether a publish landed, what the published head is, whether
# ensure_schema refuses. Each is a read-back of the reader's own write, and the
# sole-claimant protocols are only sound while a later write always observes an
# earlier one. A single node gives that for free; a ReplicatedMergeTree serves
# reads from whatever log entries a replica has fetched, so a read-back can miss
# a row another writer has already committed.
#
# It lives here, in the module both the schema and the lease coordinator sit on
# top of, so that "this read decides something" is one definition rather than an
# import from whichever module happened to declare it first.
DECIDING_READ = {"select_sequential_consistency": 1}

T = TypeVar("T")


class QuorumConfig(Protocol):
    publish_timeout_ns: int
    insert_quorum: int | None


def quorum_write(config: QuorumConfig) -> dict[str, object]:
    """The settings that make a DECIDING write durable before it is read.

    Empty unless an operator has opted in, so a single-node deployment pays
    nothing and nothing changes for anyone who has not asked for it.
    ``insert_quorum_parallel`` is turned OFF alongside, because
    ``select_sequential_consistency`` does not work with it -- setting the
    quorum without clearing the parallel flag would buy latency and no
    guarantee at all.

    The timeout is BOUNDED, and this is not a detail: ClickHouse waits
    ``insert_quorum_timeout`` for the replicas to acknowledge, and its default
    is 600 seconds. The fenced publish statements are already capped by
    ``max_execution_time``, but the version and lease claims are not -- so an
    unreachable replica would park a claim for ten minutes, twenty times the
    publish cap and twenty times the lease TTL, while the writer believes it is
    mid-publish. Capped at the publish timeout, so a quorum that cannot be met
    fails on the same clock everything else here does.

    One definition, beside DECIDING_READ and for the same reason: the lease
    coordinator and the catalog writer both make deciding writes, and the
    replicated verifier requires every one of them to carry identical settings
    -- a check that needs a Keeper-backed server and so cannot run in CI.
    """
    quorum = config.insert_quorum
    if quorum is None:
        return {}
    return {
        "insert_quorum": quorum,
        "insert_quorum_parallel": 0,
        "insert_quorum_timeout": config.publish_timeout_ns // 1_000_000,
    }


class ClickHouseClient(Protocol):
    def execute(self, query: str, params=None, **kwargs): ...


def identifier(value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid ClickHouse identifier: {value!r}")
    return value


def quoted(value: str) -> str:
    return f"`{value}`"


def text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("ClickHouse returned a non-text identifier")
    return value


def inline_text_bytes(value: str) -> int:
    if not isinstance(value, str):
        raise TypeError("inline text parameters must be strings")
    return 2 * len(value.encode("utf-8")) + 2


def inline_tuple_bytes(item: tuple[str, str]) -> int:
    if len(item) != 2:
        raise ValueError("pack identities must contain two text values")
    return sum(inline_text_bytes(value) for value in item) + 4


def inline_version_identity_bytes(item: tuple[int, str]) -> int:
    """Rendered size of a ``(index_version, publish_id)`` pair.

    The version renders as decimal digits and the identity as a quoted string
    with room for escaping, plus the tuple's own punctuation."""
    if len(item) != 2 or type(item[0]) is not int:
        raise ValueError("publish identities must be (version, publish_id) pairs")
    return len(str(item[0])) + inline_text_bytes(item[1]) + 4


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
