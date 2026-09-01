from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from time import monotonic_ns, time_ns
from typing import Callable, Protocol, Sequence

from .model import CaptureDescriptor, CaptureStorageError, ObjectPage, PackRef, PackStore
from .pack import PackIndex


PackIdentity = tuple[str, str]


class SnapshotPublishRaceError(CaptureStorageError):
    """A publish lost the race to a higher version: no snapshot became visible.

    Raised when the watermark table holds no row carrying this attempt's
    ``publish_id`` -- the conditional INSERT was refused, either by the version
    barrier or by the lease fence. Nothing this publish wrote can enter a
    snapshot: membership pairs a manifest row with the watermark row of the
    same publish, and that watermark row does not exist.

    Retryable by construction, and re-allocating a higher version and
    republishing is the whole recovery. What it does NOT promise is that the
    catalog is byte for byte as the publish found it: ``publish_snapshot``
    issues the manifest INSERT and the watermark INSERT as two separately
    fenced statements, so a takeover landing between them leaves the manifest
    rows of the first behind. They are inert -- no snapshot can ever admit them
    -- but they are durable, and collecting them is the GC obligation
    docs/catalog-descriptor-key.md records.
    """


class SnapshotPublishConflictError(CaptureStorageError):
    """This publish landed, and so did somebody else's, at the same version.

    The read-back found this attempt's ``publish_id`` on a watermark row for
    the version AND at least one row it did not write. That is not a lost race
    and must not be retried as one: this publish IS visible -- its watermark
    row admits its manifest rows and its packs are in the snapshot -- so
    republishing at a higher version would add a second visible copy rather
    than repair anything.

    It is still an anomaly, because the *contents* of that version are now the
    union of two publishes rather than this one's batch. Every protocol that
    should have prevented it -- the sole-claimant allocator, the version
    barrier, the publisher lease -- has been bypassed or shares its prefix with
    a second writer, so the catalog needs an operator rather than a retry.
    """


class SnapshotPublishExhaustedError(CaptureStorageError):
    """Every bounded publish attempt lost its version race.

    Raised by the indexer after ``max_publish_attempts`` allocations were each
    refused by the version barrier, chained (``__cause__``) to the last
    :class:`SnapshotPublishRaceError`. Each individual race is retryable, but
    this many in a row means something is publishing continuously against this
    catalog -- or the barrier itself is refusing everything, e.g. a server
    profile where an aggregate over the empty watermark table answers NULL --
    so it is surfaced through the module's own error taxonomy for a supervisor
    to back off on, rather than as a bare RuntimeError that no
    ``except CaptureStorageError`` handler sees.

    The batch's descriptor rows are durable at this point and no snapshot
    admits them; the next indexing pass re-reads the packs and tries again.
    """


class PublisherLeaseError(CaptureStorageError):
    """This publisher does not hold the lease a publish requires.

    Raised when no lease was ever acquired, when acquiring one was contested
    past its attempt bound, and -- the case that matters -- when a fenced
    publish made NO SNAPSHOT VISIBLE because the lease had been taken over or
    had expired. That last one is NOT a lost version race: re-allocating a
    version and trying again would fail the same fence every time, so this is
    deliberately not a ``SnapshotPublishRaceError`` and the indexer's publish
    retry does not absorb it. The recovery is to acquire a lease again and
    re-index.

    It does NOT promise the catalog is byte for byte as the publish found it.
    ``publish_snapshot`` issues the manifest rows and the watermark row as two
    separately fenced statements, so a takeover before the first leaves nothing
    behind, while one landing in the gap between them leaves the manifest rows
    of the first. Those rows are inert -- membership pairs a manifest row with
    the watermark row of the SAME publish, and that row will never exist -- so
    there is nothing to undo, but they are durable and they accumulate. See
    docs/catalog-descriptor-key.md, "What this does not close".
    """


class PublisherLeaseHeldError(PublisherLeaseError):
    """Another publisher holds a live lease on this catalog.

    Not retryable until that lease expires or its holder gives it back. The
    message names the holder and the expiry so an operator can tell a healthy
    handover from a wedged one.
    """


class CatalogSchemaVersionError(CaptureStorageError):
    """The catalog on the server is not a schema this build can read or write.

    Not retryable: the state it reports is durable and an indexer that carried
    on would make it worse. The message names the version found, the version
    required, and the rebuild that resolves it -- the catalog is a derived
    projection over immutable packs, so dropping it and reconciling the object
    store again loses nothing.
    """


class PackInventory(PackStore, Protocol):
    def inspect(self, object_key: str) -> PackRef: ...
    def list_objects(
        self, *, prefix: str = "", cursor: str | None = None, limit: int = 1000
    ) -> ObjectPage: ...


class CatalogWriter(Protocol):
    def committed_pack_ids(
        self, identities: Sequence[PackIdentity]
    ) -> set[PackIdentity]: ...
    def write_descriptors(
        self, descriptors: Sequence[CaptureDescriptor], *, index_version: int
    ) -> None: ...
    def commit_packs(
        self, refs: Sequence[PackRef], *, index_version: int
    ) -> None: ...

    def publish_snapshot(
        self,
        *,
        index_version: int,
        refs: Sequence[PackRef],
        published_at_ns: int,
        indexed_rows: int,
        indexed_packs: int,
    ) -> None: ...

    def last_published_version(self) -> int: ...

    def allocate_version(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CatalogIndexerConfig:
    max_packs: int = 64
    max_rows_per_insert: int = 10_000
    max_estimated_bytes: int = 128 * 1024**2
    max_failure_details: int = 128
    max_publish_attempts: int = 8

    def __post_init__(self) -> None:
        for name in (
            "max_packs",
            "max_rows_per_insert",
            "max_estimated_bytes",
            "max_failure_details",
            "max_publish_attempts",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class IndexFailure:
    pack_id: str
    object_key: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class IndexResult:
    requested_packs: int = 0
    skipped_packs: int = 0
    indexed_packs: int = 0
    indexed_rows: int = 0
    failed_packs: int = 0
    descriptor_inserts: int = 0
    estimated_bytes: int = 0
    elapsed_ns: int = 0
    failures: tuple[IndexFailure, ...] = ()

    def merge(
        self, other: IndexResult, *, failure_limit: int | None = None
    ) -> IndexResult:
        failures = self.failures + other.failures
        if failure_limit is not None:
            failures = failures[:failure_limit]
        return IndexResult(
            requested_packs=self.requested_packs + other.requested_packs,
            skipped_packs=self.skipped_packs + other.skipped_packs,
            indexed_packs=self.indexed_packs + other.indexed_packs,
            indexed_rows=self.indexed_rows + other.indexed_rows,
            failed_packs=self.failed_packs + other.failed_packs,
            descriptor_inserts=self.descriptor_inserts + other.descriptor_inserts,
            estimated_bytes=self.estimated_bytes + other.estimated_bytes,
            elapsed_ns=self.elapsed_ns + other.elapsed_ns,
            failures=failures,
        )


@dataclass(frozen=True, slots=True)
class IndexEvent:
    event: str
    requested_packs: int
    skipped_packs: int
    indexed_packs: int
    indexed_rows: int
    failed_packs: int
    descriptor_inserts: int
    estimated_bytes: int
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    index: IndexResult
    next_cursor: str | None
    pages: int


def _identity(ref: PackRef) -> PackIdentity:
    return ref.store_id, ref.pack_id


def _partition_refs(
    refs: Sequence[PackRef],
) -> tuple[tuple[PackRef, ...], tuple[IndexFailure, ...]]:
    """Deduplicate refs, containing identity conflicts as per-pack failures.

    Two different refs claiming one (store_id, pack_id) means at least one of
    them is wrong, with no way to tell which, so every claimant is failed and
    none indexed. Raising here instead would let one duplicated bucket object
    abort each reconcile pass that lists it, forever.
    """

    by_identity: dict[PackIdentity, PackRef] = {}
    conflicted: dict[PackIdentity, list[PackRef]] = {}
    for ref in refs:
        identity = _identity(ref)
        current = by_identity.get(identity)
        if current is None:
            by_identity[identity] = ref
            continue
        if current != ref:
            claimants = conflicted.setdefault(identity, [current])
            if ref not in claimants:
                claimants.append(ref)
    unique = tuple(
        ref
        for identity, ref in by_identity.items()
        if identity not in conflicted
    )
    failures = tuple(
        IndexFailure(
            pack_id=ref.pack_id,
            object_key=ref.object_key,
            error_type="PackConflictError",
            message=f"conflicting pack identity: {identity!r}",
        )
        for identity, claimants in conflicted.items()
        for ref in claimants
    )
    return unique, failures


def _estimated_bytes(descriptor: CaptureDescriptor) -> int:
    value = {
        "metadata": descriptor.metadata.to_mapping(),
        "locator": asdict(descriptor.locator),
    }
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


class CatalogIndexer:
    def __init__(
        self,
        store: PackStore,
        writer: CatalogWriter,
        *,
        config: CatalogIndexerConfig | None = None,
        clock_ns: Callable[[], int] = time_ns,
        timer_ns: Callable[[], int] = monotonic_ns,
        on_event: Callable[[IndexEvent], None] | None = None,
    ) -> None:
        self._store = store
        self._writer = writer
        self._config = config or CatalogIndexerConfig()
        self._clock_ns = clock_ns
        self._timer_ns = timer_ns
        self._on_event = on_event
        self._callback_failures = 0
        self._published_version: int | None = None

    @property
    def store_id(self) -> str:
        return self._store.store_id

    @property
    def max_packs(self) -> int:
        return self._config.max_packs

    @property
    def callback_failures(self) -> int:
        return self._callback_failures

    @property
    def max_failure_details(self) -> int:
        return self._config.max_failure_details

    def index(self, refs: Sequence[PackRef]) -> IndexResult:
        started_ns = self._timer_ns()
        if len(refs) > self._config.max_packs:
            raise ValueError(
                f"pack batch exceeds max_packs: {len(refs)} > {self._config.max_packs}"
            )
        unique, conflict_failures = _partition_refs(refs)
        identities = tuple(_identity(ref) for ref in unique)
        committed = self._writer.committed_pack_ids(identities) if identities else set()
        pending = tuple(ref for ref in unique if _identity(ref) not in committed)
        descriptors: list[CaptureDescriptor] = []
        valid_refs: list[PackRef] = []
        failures: list[IndexFailure] = list(conflict_failures)
        estimated_bytes = 0
        for ref in pending:
            try:
                pack_descriptors = PackIndex.from_store(self._store, ref).descriptors()
                pack_bytes = sum(_estimated_bytes(item) for item in pack_descriptors)
            except Exception as exc:
                failures.append(
                    IndexFailure(
                        pack_id=ref.pack_id,
                        object_key=ref.object_key,
                        error_type=type(exc).__name__,
                        message=str(exc)[:512],
                    )
                )
                continue
            # A batch that is too large is a caller error, not a property of the
            # pack being read. Reporting it as a per-pack failure would blame an
            # innocent pack and, because the loop continues, silently skip every
            # remaining pack while index() still returned normally.
            if estimated_bytes + pack_bytes > self._config.max_estimated_bytes:
                raise ValueError(
                    "catalog batch exceeds max_estimated_bytes: "
                    f"{estimated_bytes + pack_bytes} > "
                    f"{self._config.max_estimated_bytes}"
                )
            estimated_bytes += pack_bytes
            descriptors.extend(pack_descriptors)
            valid_refs.append(ref)

        if descriptors or valid_refs:
            version = self._allocate_version()
            descriptor_inserts = self._write_descriptor_batches(
                descriptors, version
            )
            # Publish before the inventory, never after. committed_pack_ids
            # reads the inventory to skip replays, so a pack recorded there but
            # never made visible is skipped forever *and* invisible -- silent,
            # permanent loss. This way round the same crash costs only
            # redundant work on the next pass. Publishing is also what makes
            # the descriptor rows above readable at all: it writes the
            # membership rows for `valid_refs` and the watermark that admits
            # them.
            published, rewrite_inserts = self._publish(
                version, valid_refs, descriptors
            )
            descriptor_inserts += rewrite_inserts
            if valid_refs:
                self._writer.commit_packs(valid_refs, index_version=published)
        else:
            # A complete no-op: every pack was already committed (or failed to
            # read), nothing was written, and nothing new could become visible
            # -- so nothing is published. Publishing here anyway would burn a
            # version and, since only the lease holder may publish, make a
            # sweep that merely CONFIRMS a catalog contend for the publisher
            # lease: a periodic CatalogReconciler.rebuild running beside the
            # live indexer would hard-fail on pages it has nothing to say
            # about. A no-op pass performs no catalog writes at all.
            descriptor_inserts = 0
        result = IndexResult(
            requested_packs=len(unique) + len(conflict_failures),
            skipped_packs=len(unique) - len(pending),
            indexed_packs=len(valid_refs),
            indexed_rows=len(descriptors),
            failed_packs=len(failures),
            descriptor_inserts=descriptor_inserts,
            estimated_bytes=estimated_bytes,
            elapsed_ns=self._timer_ns() - started_ns,
            failures=tuple(failures),
        )
        self._emit(result)
        return result

    def _allocate_version(self) -> int:
        # index_version has to increase strictly across every live indexer, or
        # a batch lands underneath a watermark a reader already pinned and that
        # snapshot grows after the fact. The version is a DB-allocated
        # generation owned by the catalog itself -- no wall-clock value
        # participates in the ordering, so clock skew between indexers (or an
        # NTP step on one of them) cannot reorder publications. The wall clock
        # only stamps published_at_ns.
        version = self._writer.allocate_version()
        if type(version) is not int or version < 1:
            # Positive, not merely non-negative: last_published_version() is at
            # least 0, so an allocator that can ever return 0 has already
            # broken the strictly-above-the-head contract.
            raise ValueError("allocate_version must return a positive integer")
        if self._published_version is None:
            # Cross-check against durable state: a broken allocator must fail
            # loudly here rather than publish under a pinned watermark.
            self._published_version = self._writer.last_published_version()
        if version <= self._published_version:
            raise RuntimeError(
                "catalog version allocator returned a non-monotonic version"
            )
        return version

    def _write_descriptor_batches(
        self, descriptors: Sequence[CaptureDescriptor], version: int
    ) -> int:
        """Write the descriptors at ``version`` in bounded chunks; count them."""
        step = self._config.max_rows_per_insert
        inserts = 0
        for start in range(0, len(descriptors), step):
            self._writer.write_descriptors(
                descriptors[start : start + step], index_version=version
            )
            inserts += 1
        return inserts

    def _publish(
        self,
        version: int,
        refs: Sequence[PackRef],
        descriptors: Sequence[CaptureDescriptor],
    ) -> tuple[int, int]:
        """Publish the batch, re-allocating around a lost publish race.

        Returns ``(published_version, rewrite_inserts)`` -- the version that
        became visible and how many extra descriptor INSERTs the retries cost.

        Publishing is a required part of the CatalogWriter contract: skipping
        it would leave every row this call wrote durably stored but permanently
        invisible to readers, with no error anywhere.

        Whatever authority a writer needs in order to publish is the writer's
        own affair and is acquired before an indexer is handed one -- the
        ClickHouse writer takes a publisher lease. Only a lost VERSION race is
        retried here: it is repaired by allocating a higher version, which is
        exactly what this loop does. A writer that has lost its authority to
        publish raises something else, and it propagates, because every retry
        would fail the same way.

        A losing publish made nothing visible, so recovery is a fresh version
        and another attempt -- and the DESCRIPTORS are rewritten at that
        version, not only the manifest rows. VISIBILITY does not need the
        rewrite (membership decides it, and membership is rewritten at the
        version that wins), but SUPERSESSION does: ``index_version`` leads
        ``clickhouse_reader._RESOLUTION_ORDER``, the primary ordering between
        two rows describing one capture in two DIFFERENT packs. Rows left at
        the lost version would rank below another pack's rows written between
        the lost and the winning version, so the reader would resolve a capture
        to the OLDER publish's pack -- and to its locator, which is exactly
        what may differ. The rewrite is byte-identical rows at the new version;
        the superseded rows share their full sort key with them (pack identity
        included), so the ReplacingMergeTree collapses each pair to the new
        version and ``argMax`` resolves the same rows in the meantime.

        A publish that CONFLICTED is the opposite of a loss: it is visible, by
        that error's own contract, so its packs enter the replay inventory
        before the error propagates. Skipping that would hand the next pass a
        batch it re-indexes and re-publishes automatically -- the retry the
        error exists to forbid -- while burying the anomaly under a success.
        """
        attempts = self._config.max_publish_attempts
        rewrite_inserts = 0
        last_race: SnapshotPublishRaceError | None = None
        for attempt in range(attempts):
            # The wall clock stamps published_at_ns only -- a human-readable
            # record of when the version was published, never part of its
            # ordering.
            published_at_ns = self._clock_ns()
            if type(published_at_ns) is not int or published_at_ns < 0:
                raise ValueError("clock_ns must return a non-negative integer")
            try:
                self._writer.publish_snapshot(
                    index_version=version,
                    refs=refs,
                    published_at_ns=published_at_ns,
                    indexed_rows=len(descriptors),
                    indexed_packs=len(refs),
                )
            except SnapshotPublishConflictError:
                # Visible, so skippable: record the packs before propagating.
                if refs:
                    self._writer.commit_packs(refs, index_version=version)
                raise
            except SnapshotPublishRaceError as race:
                last_race = race
                if attempt + 1 == attempts:
                    break
                version = self._allocate_version()
                rewrite_inserts += self._write_descriptor_batches(
                    descriptors, version
                )
                continue
            self._published_version = version
            return version, rewrite_inserts
        raise SnapshotPublishExhaustedError(
            f"could not publish a catalog snapshot after {attempts} attempts"
        ) from last_race

    def _emit(self, result: IndexResult) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(
                IndexEvent(
                    event="catalog_index_completed",
                    requested_packs=result.requested_packs,
                    skipped_packs=result.skipped_packs,
                    indexed_packs=result.indexed_packs,
                    indexed_rows=result.indexed_rows,
                    failed_packs=result.failed_packs,
                    descriptor_inserts=result.descriptor_inserts,
                    estimated_bytes=result.estimated_bytes,
                    elapsed_ns=result.elapsed_ns,
                )
            )
        except Exception:
            self._callback_failures += 1


class CatalogReconciler:
    def __init__(self, inventory: PackInventory, indexer: CatalogIndexer) -> None:
        if inventory.store_id != indexer.store_id:
            raise ValueError("inventory and indexer store IDs differ")
        self._inventory = inventory
        self._indexer = indexer

    def index_object_keys(self, object_keys: Sequence[str]) -> IndexResult:
        if len(object_keys) > self._indexer.max_packs:
            raise ValueError("object notification batch exceeds max_packs")
        keys = tuple(dict.fromkeys(object_keys))
        # A bucket holds whatever anyone put in it. Inspection runs outside
        # CatalogIndexer.index's per-pack handling, so without this one foreign
        # object would abort an entire rebuild instead of being one failure.
        refs: list[PackRef] = []
        failures: list[IndexFailure] = []
        for key in keys:
            try:
                refs.append(self._inventory.inspect(key))
            except Exception as exc:
                failures.append(
                    IndexFailure(
                        pack_id="",
                        object_key=key,
                        error_type=type(exc).__name__,
                        message=str(exc)[:512],
                    )
                )
        result = self._indexer.index(refs)
        if not failures:
            return result
        rejected = IndexResult(
            requested_packs=len(failures),
            skipped_packs=0,
            indexed_packs=0,
            indexed_rows=0,
            failed_packs=len(failures),
            descriptor_inserts=0,
            estimated_bytes=0,
            elapsed_ns=0,
            failures=tuple(failures),
        )
        return result.merge(
            rejected, failure_limit=self._indexer.max_failure_details
        )

    def reconcile_page(
        self, *, prefix: str = "", cursor: str | None = None, limit: int = 64
    ) -> ReconcileResult:
        if limit > self._indexer.max_packs:
            raise ValueError("listing limit exceeds max_packs")
        page = self._inventory.list_objects(prefix=prefix, cursor=cursor, limit=limit)
        result = self.index_object_keys([item.object_key for item in page.items])
        return ReconcileResult(index=result, next_cursor=page.next_cursor, pages=1)

    def rebuild(
        self, *, prefix: str = "", page_size: int = 64, max_pages: int = 10_000
    ) -> IndexResult:
        if type(max_pages) is not int or max_pages <= 0:
            raise ValueError("max_pages must be positive")
        cursor = None
        total = IndexResult()
        for _ in range(max_pages):
            page = self.reconcile_page(prefix=prefix, cursor=cursor, limit=page_size)
            total = total.merge(
                page.index, failure_limit=self._indexer.max_failure_details
            )
            cursor = page.next_cursor
            if cursor is None:
                return total
        raise RuntimeError("rebuild exceeded max_pages")
