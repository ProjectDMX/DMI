"""Deterministic fault injection for the capture storage path.

Phase 6 requires fault injection before the default sink can be switched, and
the components that must survive faults are spread across three boundaries: the
object store, the ClickHouse client, and the pack sink. This module provides one
wrapper per boundary so a test states *which* fault it wants rather than
hand-rolling another one-off fake.

Every fault is **scripted, not random**. A schedule says which call numbers fail
and how, so a failing test reproduces exactly. Nothing here uses a random source.

    store = FaultyPackStore(inner, read_range=fail_on(2, OSError("reset")))
    store.read_range(ref, 0, 16)   # call 1 -- fine
    store.read_range(ref, 0, 16)   # call 2 -- raises OSError

These wrappers also serve the conformance role the Python implementation now
has: they describe the failure behaviour a native writer has to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class FaultInjected(Exception):
    """Raised by a scripted fault, so tests can tell it from a real bug."""


# --- schedules ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Schedule:
    """Which calls misbehave, and how.

    ``calls`` are 1-based call numbers. A call not listed behaves normally, so
    the default schedule is a pass-through.
    """

    calls: frozenset[int] = frozenset()
    error: BaseException | None = None
    truncate_by: int = 0
    repeat_result: bool = False

    def applies_to(self, call_number: int) -> bool:
        return call_number in self.calls


def never() -> Schedule:
    """No faults. Useful as an explicit default."""
    return Schedule()


def fail_on(*calls: int, error: BaseException | None = None) -> Schedule:
    """Raise on the given 1-based call numbers."""
    if not calls:
        raise ValueError("fail_on needs at least one call number")
    return Schedule(
        calls=frozenset(calls),
        error=error or FaultInjected("scripted failure"),
    )


def truncate_on(*calls: int, by: int = 1) -> Schedule:
    """Return fewer bytes than asked for -- a short read.

    Object stores are allowed to return short reads, and code that assumes
    otherwise corrupts payloads silently rather than failing.
    """
    if by <= 0:
        raise ValueError("truncate_on needs a positive byte count")
    return Schedule(calls=frozenset(calls), truncate_by=by)


def duplicate_on(*calls: int) -> Schedule:
    """Apply the call twice -- an ambiguous write that actually landed twice."""
    return Schedule(calls=frozenset(calls), repeat_result=True)


def fail_then_succeed(count: int, error: BaseException | None = None) -> Schedule:
    """Fail the first ``count`` calls, then behave. Models a transient outage."""
    if count <= 0:
        raise ValueError("fail_then_succeed needs a positive count")
    return Schedule(
        calls=frozenset(range(1, count + 1)),
        error=error or FaultInjected("transient failure"),
    )


# --- object store ------------------------------------------------------------


@dataclass
class _Counter:
    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, name: str) -> int:
        self.counts[name] = self.counts.get(name, 0) + 1
        return self.counts[name]


class FaultyPackStore:
    """Wraps any PackStore, injecting scripted faults per method.

    Delegates everything it does not intercept, so it stays usable as a drop-in
    even as the store protocol grows.
    """

    def __init__(
        self,
        inner: Any,
        *,
        put: Schedule | None = None,
        stat: Schedule | None = None,
        read_range: Schedule | None = None,
    ) -> None:
        self._inner = inner
        self._schedules = {
            "put": put or never(),
            "stat": stat or never(),
            "read_range": read_range or never(),
        }
        self._calls = _Counter()

    @property
    def store_id(self) -> str:
        return self._inner.store_id

    @property
    def call_counts(self) -> Mapping[str, int]:
        return dict(self._calls.counts)

    def _guard(self, name: str) -> Schedule | None:
        schedule = self._schedules[name]
        number = self._calls.bump(name)
        if not schedule.applies_to(number):
            return None
        if schedule.error is not None:
            raise schedule.error
        return schedule

    def put(self, pack: Any, object_key: str) -> Any:
        schedule = self._guard("put")
        result = self._inner.put(pack, object_key)
        if schedule is not None and schedule.repeat_result:
            # The same object written twice: the second write must be a no-op
            # for an immutable key, not a conflict.
            self._inner.put(pack, object_key)
        return result

    def stat(self, ref: Any) -> Any:
        self._guard("stat")
        return self._inner.stat(ref)

    def read_range(self, ref: Any, offset: int, length: int) -> bytes:
        schedule = self._guard("read_range")
        data = self._inner.read_range(ref, offset, length)
        if schedule is not None and schedule.truncate_by:
            return data[: max(0, len(data) - schedule.truncate_by)]
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# --- ClickHouse client -------------------------------------------------------


class FaultyClickHouseClient:
    """Wraps a ClickHouse client, injecting faults by statement kind.

    Statements are classified by their leading keyword, so a test can fail only
    inserts while leaving schema and reads alone -- which is the interesting
    case for an indexer that must not lose or double-count rows.
    """

    def __init__(
        self,
        inner: Any,
        *,
        insert: Schedule | None = None,
        select: Schedule | None = None,
        ddl: Schedule | None = None,
    ) -> None:
        self._inner = inner
        self._schedules = {
            "insert": insert or never(),
            "select": select or never(),
            "ddl": ddl or never(),
        }
        self._calls = _Counter()
        self.statements: list[str] = []

    @property
    def call_counts(self) -> Mapping[str, int]:
        return dict(self._calls.counts)

    @staticmethod
    def _kind(query: str) -> str:
        head = query.lstrip().split(None, 1)[0].upper() if query.strip() else ""
        if head == "INSERT":
            return "insert"
        if head in {"SELECT", "WITH"}:
            return "select"
        return "ddl"

    def execute(self, query: str, params: Any = None, **kwargs: Any) -> Any:
        kind = self._kind(query)
        self.statements.append(query)
        number = self._calls.bump(kind)
        schedule = self._schedules[kind]
        if schedule.applies_to(number):
            if schedule.error is not None:
                raise schedule.error
            if schedule.repeat_result:
                # An ambiguous insert: the client never learned it succeeded, so
                # the row lands twice and dedup has to absorb it.
                self._inner.execute(query, params, **kwargs)
        return self._inner.execute(query, params, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# --- pack sink ---------------------------------------------------------------


class FaultyPackSink:
    """Wraps a pack sink so persistence can fail on chosen packs."""

    def __init__(self, inner: Any, *, persist: Schedule | None = None) -> None:
        self._inner = inner
        self._schedule = persist or never()
        self._calls = _Counter()

    @property
    def call_counts(self) -> Mapping[str, int]:
        return dict(self._calls.counts)

    def persist(self, ready: Any) -> Any:
        number = self._calls.bump("persist")
        if self._schedule.applies_to(number) and self._schedule.error is not None:
            raise self._schedule.error
        return self._inner.persist(ready)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def call_sequence(store: FaultyPackStore) -> Sequence[tuple[str, int]]:
    """The recorded call counts, sorted -- handy in assertion messages."""
    return sorted(store.call_counts.items())


__all__ = [
    "FaultInjected",
    "FaultyClickHouseClient",
    "FaultyPackSink",
    "FaultyPackStore",
    "Schedule",
    "call_sequence",
    "duplicate_on",
    "fail_on",
    "fail_then_succeed",
    "never",
    "truncate_on",
]
