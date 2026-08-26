from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from time import monotonic_ns, time_ns
from typing import Callable, Protocol, Sequence

from .model import CaptureDescriptor, ObjectPage, PackRef, PackStore
from .pack import PackIndex


PackIdentity = tuple[str, str]


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

    def publish_watermark(
        self,
        *,
        index_version: int,
        published_at_ns: int,
        indexed_rows: int,
        indexed_packs: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CatalogIndexerConfig:
    max_packs: int = 64
    max_rows_per_insert: int = 10_000
    max_estimated_bytes: int = 128 * 1024**2
    max_failure_details: int = 128

    def __post_init__(self) -> None:
        for name in (
            "max_packs",
            "max_rows_per_insert",
            "max_estimated_bytes",
            "max_failure_details",
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


def _deduplicate_refs(refs: Sequence[PackRef]) -> tuple[PackRef, ...]:
    by_identity: dict[PackIdentity, PackRef] = {}
    for ref in refs:
        identity = _identity(ref)
        current = by_identity.get(identity)
        if current is not None and current != ref:
            raise ValueError(f"conflicting pack identity: {identity!r}")
        by_identity[identity] = ref
    return tuple(by_identity.values())


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
        unique = _deduplicate_refs(refs)
        identities = tuple(_identity(ref) for ref in unique)
        committed = self._writer.committed_pack_ids(identities) if identities else set()
        pending = tuple(ref for ref in unique if _identity(ref) not in committed)
        descriptors: list[CaptureDescriptor] = []
        valid_refs: list[PackRef] = []
        failures: list[IndexFailure] = []
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

        version = self._clock_ns()
        if type(version) is not int or version < 0:
            raise ValueError("clock_ns must return a non-negative integer")
        # index_version has to increase strictly, or a batch lands underneath a
        # watermark a reader already pinned and that snapshot grows after the
        # fact. A wall clock does not guarantee that: NTP steps backwards, and
        # a coarse clock can return the same value twice in a row. Advance past
        # the last published version rather than failing -- the version is an
        # ordering token, not a timestamp, and published_at_ns still records the
        # real time. Cross-process skew is a separate problem, documented under
        # the Phase 5 limitations.
        if self._published_version is not None and version <= self._published_version:
            version = self._published_version + 1
        step = self._config.max_rows_per_insert
        descriptor_inserts = 0
        for start in range(0, len(descriptors), step):
            self._writer.write_descriptors(
                descriptors[start : start + step], index_version=version
            )
            descriptor_inserts += 1
        if valid_refs:
            self._writer.commit_packs(valid_refs, index_version=version)
        # Publish last. Descriptors go out across several INSERTs and the pack
        # markers after them, so the version is only a truthful snapshot once
        # all of that is durable. A reader that derived the watermark from the
        # descriptor table itself would see this version mid-batch.
        self._published_version = version
        publish = getattr(self._writer, "publish_watermark", None)
        if publish is not None:
            publish(
                index_version=version,
                published_at_ns=self._clock_ns(),
                indexed_rows=len(descriptors),
                indexed_packs=len(valid_refs),
            )
        result = IndexResult(
            requested_packs=len(unique),
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
