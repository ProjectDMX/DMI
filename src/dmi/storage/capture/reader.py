from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
from typing import Mapping, Sequence

from .model import (
    CaptureCatalog,
    CaptureDescriptor,
    CapturePage,
    CaptureQuery,
    CaptureSelection,
    DuplicateCaptureError,
    HydratedCapture,
    HydrationEstimate,
    HydrationLimitError,
    PackFormatError,
    PackIntegrityError,
    PackRef,
    PackStore,
)
from .extensions import (
    ArtifactSink,
    ExtensionFailure,
    ExtensionRegistry,
)
from .pack import PackIndex, verify_payload
from .summary import ArtifactRef, CoreTensorSummaryV1, decode_tensor, summarize_tensor


# Footer indexes verified during hydration are cached per pack, bounded so a
# reader that touches many packs cannot hold every footer in memory at once.
_FOOTER_CACHE_PACKS = 128
_FOOTER_CACHE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    """A versioned core summary plus whatever extensions contributed."""

    capture_id: str
    core: CoreTensorSummaryV1
    scalars: Mapping[str, float]
    artifacts: tuple[ArtifactRef, ...]
    failures: tuple[ExtensionFailure, ...]


def _record_locator(descriptor: CaptureDescriptor) -> tuple[int, int, int, str, str]:
    """The per-record placement fields the footer is authoritative for.

    The locator's pack-level fields (store id, object key, size, pack
    checksum) came from the same catalog row under test, so comparing them
    against a footer index built from that row would prove nothing.
    """
    locator = descriptor.locator
    return (
        locator.offset,
        locator.stored_length,
        locator.decoded_length,
        locator.codec,
        locator.checksum,
    )


@dataclass(frozen=True, slots=True)
class _ReadRange:
    descriptor_indexes: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class _ObjectPlan:
    store_id: str
    pack_id: str
    object_key: str
    descriptor_indexes: tuple[int, ...]
    ranges: tuple[_ReadRange, ...]


@dataclass(frozen=True, slots=True)
class _FooterCacheEntry:
    descriptors: Mapping[str, CaptureDescriptor]
    wire_bytes: int


class _ReadBudget:
    """Budget metadata reads after reserving every planned payload read."""

    def __init__(self, *, requests: int, bytes_: int) -> None:
        self._requests = requests
        self._bytes = bytes_

    def consume(self, length: int) -> None:
        if self._requests < 1:
            raise HydrationLimitError("hydration request limit exceeded")
        if length > self._bytes:
            raise HydrationLimitError("hydration byte limit exceeded")
        self._requests -= 1
        self._bytes -= length


class _BudgetedPackStore:
    """The footer-only PackStore surface with pre-read budget checks."""

    def __init__(self, store: PackStore, budget: _ReadBudget) -> None:
        self._store = store
        self._budget = budget
        self.store_id = store.store_id

    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes:
        self._budget.consume(length)
        return self._store.read_range(ref, offset, length)


class CaptureReader:
    def __init__(
        self,
        catalog: CaptureCatalog,
        stores: Mapping[str, PackStore],
        *,
        max_coalesce_gap_bytes: int = 4096,
    ) -> None:
        if max_coalesce_gap_bytes < 0:
            raise ValueError("max_coalesce_gap_bytes must be non-negative")
        if not stores:
            raise ValueError("at least one pack store is required")
        for store_id, store in stores.items():
            if store_id != store.store_id:
                raise ValueError(f"store mapping key does not match {store.store_id!r}")
        self._catalog = catalog
        self._stores = dict(stores)
        self._max_coalesce_gap_bytes = max_coalesce_gap_bytes
        # LRU of {capture_id: descriptor} footer indexes keyed by pack
        # identity, so repeated hydrations of the same pack verify against
        # the footer without re-reading it.
        self._footer_cache: OrderedDict[
            tuple[str, str, str], _FooterCacheEntry
        ] = OrderedDict()
        self._footer_cache_limit = _FOOTER_CACHE_PACKS
        self._footer_cache_bytes_limit = _FOOTER_CACHE_BYTES
        self._footer_cache_bytes = 0
        self._footer_cache_lock = threading.Lock()

    def search(self, query: CaptureQuery) -> CapturePage:
        return self._catalog.search(query)

    def select(self, query: CaptureQuery) -> CaptureSelection:
        page = self.search(query)
        if page.next_cursor is not None:
            raise ValueError(
                "selection exceeds one bounded page; narrow the query or paginate explicitly"
            )
        return CaptureSelection.create(
            page.items,
            catalog_watermark=page.watermark,
            filter_hash=query.filter_hash,
        )

    def estimate(self, selection: CaptureSelection) -> HydrationEstimate:
        descriptors = self._resolve(selection)
        plans = self._plan(descriptors)
        payload_requests = sum(len(plan.ranges) for plan in plans)
        payload_bytes = sum(item.length for plan in plans for item in plan.ranges)
        footer_bytes = sum(
            PackIndex.max_footer_read_bytes(
                descriptors[plan.descriptor_indexes[0]].locator.pack_ref
            )
            for plan in plans
        )
        return HydrationEstimate(
            capture_count=len(descriptors),
            object_count=len(plans),
            request_count=payload_requests + 2 * len(plans),
            logical_bytes=sum(item.locator.decoded_length for item in descriptors),
            stored_bytes=sum(item.locator.stored_length for item in descriptors),
            request_bytes=payload_bytes + footer_bytes,
        )

    def hydrate(
        self,
        selection: CaptureSelection,
        *,
        byte_limit: int,
        request_limit: int = 1024,
    ) -> tuple[HydratedCapture, ...]:
        if byte_limit < 0:
            raise ValueError("byte_limit must be non-negative")
        if request_limit <= 0:
            raise ValueError("request_limit must be positive")
        descriptors = self._resolve(selection)
        plans = self._plan(descriptors)
        payload_requests = sum(len(plan.ranges) for plan in plans)
        if payload_requests > request_limit:
            raise HydrationLimitError(
                f"hydration request limit exceeded: "
                f"{payload_requests} > {request_limit}"
            )
        payload_bytes = sum(item.length for plan in plans for item in plan.ranges)
        if payload_bytes > byte_limit:
            raise HydrationLimitError(
                f"hydration byte limit exceeded: {payload_bytes} > {byte_limit}"
            )

        footer_budget = _ReadBudget(
            requests=request_limit - payload_requests,
            bytes_=byte_limit - payload_bytes,
        )

        payloads: list[bytes | None] = [None] * len(descriptors)
        verified_plans: list[tuple[_ObjectPlan, PackStore, PackRef]] = []
        for plan in plans:
            store = self._stores.get(plan.store_id)
            if store is None:
                raise PackFormatError(f"unknown pack store: {plan.store_id}")
            ref = descriptors[plan.descriptor_indexes[0]].locator.pack_ref
            # The catalog is the query index, but the pack footer is the
            # authority for what each payload *is*: verify_payload checks
            # length and CRC32 of the bytes only, so a re-described catalog
            # row (same bytes, different dtype/shape) would otherwise decode
            # garbage. Binding every catalog descriptor to the footer costs
            # two small range reads (trailer + footer) per uncached pack per
            # hydration -- the price of the footer being authoritative -- and
            # happens before any payload range is fetched.
            footer = self._footer_descriptors(store, ref, footer_budget)
            for index in plan.descriptor_indexes:
                self._require_footer_match(descriptors[index], footer)
            verified_plans.append((plan, store, ref))

        # Footer verification is a separate first phase. A later pack cannot
        # fail its metadata budget or descriptor check after earlier payloads
        # have already been fetched.
        for plan, store, ref in verified_plans:
            for read_range in plan.ranges:
                block = store.read_range(ref, read_range.offset, read_range.length)
                if len(block) != read_range.length:
                    raise PackIntegrityError(
                        f"object store returned a short range: "
                        f"{len(block)} != {read_range.length}"
                    )
                for index in read_range.descriptor_indexes:
                    descriptor = descriptors[index]
                    start = descriptor.locator.offset - read_range.offset
                    end = start + descriptor.locator.stored_length
                    payload = block[start:end]
                    verify_payload(descriptor, payload)
                    payloads[index] = bytes(payload)

        if any(payload is None for payload in payloads):
            raise PackFormatError(  # pragma: no cover - defensive invariant
                "hydration plan did not resolve every capture"
            )
        return tuple(
            HydratedCapture(descriptor=descriptor, payload=payload)
            for descriptor, payload in zip(descriptors, payloads)
            if payload is not None
        )

    def summarize(
        self,
        selection: CaptureSelection,
        *,
        byte_limit: int,
        request_limit: int = 1024,
        registry: ExtensionRegistry | None = None,
        artifact_sink: ArtifactSink | None = None,
        max_summary_captures: int = 1000,
        max_summary_elements: int = 64_000_000,
    ) -> tuple[CaptureSummary, ...]:
        """Summarise a selection from payloads fetched under the same limits.

        Hydration is the only thing that reads bytes, so summarising cannot
        widen the read set: whatever the byte and request limits allowed is
        exactly what gets decoded.
        """
        if max_summary_captures <= 0:
            raise ValueError("max_summary_captures must be positive")
        if max_summary_elements <= 0:
            raise ValueError("max_summary_elements must be positive")
        if len(selection.capture_ids) > max_summary_captures:
            raise HydrationLimitError(
                f"summary capture limit exceeded: "
                f"{len(selection.capture_ids)} > {max_summary_captures}"
            )

        hydrated = self.hydrate(
            selection, byte_limit=byte_limit, request_limit=request_limit
        )
        total_elements = sum(
            math.prod(item.descriptor.metadata.shape) for item in hydrated
        )
        if total_elements > max_summary_elements:
            raise HydrationLimitError(
                f"summary element limit exceeded: "
                f"{total_elements} > {max_summary_elements}"
            )

        summaries: list[CaptureSummary] = []
        for item in hydrated:
            core = summarize_tensor(item.descriptor, item.payload)
            scalars: dict[str, float] = {}
            artifacts: tuple[ArtifactRef, ...] = ()
            failures: tuple[ExtensionFailure, ...] = ()
            if registry is not None:
                scalars, artifacts, failures = registry.evaluate(
                    decode_tensor(item.descriptor, item.payload),
                    capture_id=item.capture_id,
                    sink=artifact_sink,
                )
            summaries.append(
                CaptureSummary(
                    capture_id=item.capture_id,
                    core=core,
                    scalars=dict(scalars),
                    artifacts=artifacts,
                    failures=failures,
                )
            )
        return tuple(summaries)

    def _footer_descriptors(
        self,
        store: PackStore,
        ref: PackRef,
        budget: _ReadBudget,
    ) -> Mapping[str, CaptureDescriptor]:
        key = (ref.store_id, ref.pack_id, ref.checksum)
        with self._footer_cache_lock:
            cached = self._footer_cache.get(key)
            if cached is not None:
                self._footer_cache.move_to_end(key)
                return cached.descriptors

        index = PackIndex.from_store(_BudgetedPackStore(store, budget), ref)
        footer = {item.capture_id: item for item in index.descriptors()}
        entry = _FooterCacheEntry(footer, index.footer_read_bytes)
        with self._footer_cache_lock:
            cached = self._footer_cache.get(key)
            if cached is not None:
                self._footer_cache.move_to_end(key)
                return cached.descriptors
            if (
                self._footer_cache_limit > 0
                and entry.wire_bytes <= self._footer_cache_bytes_limit
            ):
                while self._footer_cache and (
                    len(self._footer_cache) >= self._footer_cache_limit
                    or self._footer_cache_bytes + entry.wire_bytes
                    > self._footer_cache_bytes_limit
                ):
                    _, evicted = self._footer_cache.popitem(last=False)
                    self._footer_cache_bytes -= evicted.wire_bytes
                self._footer_cache[key] = entry
                self._footer_cache_bytes += entry.wire_bytes
        return footer

    @staticmethod
    def _require_footer_match(
        descriptor: CaptureDescriptor,
        footer: Mapping[str, CaptureDescriptor],
    ) -> None:
        authoritative = footer.get(descriptor.capture_id)
        if authoritative is None:
            raise PackFormatError(
                "catalog descriptor does not match the pack footer: "
                f"{descriptor.capture_id} is not in pack {descriptor.locator.pack_id}"
            )
        for domain, catalog_value, footer_value in (
            ("metadata", descriptor.metadata, authoritative.metadata),
            (
                "locator",
                _record_locator(descriptor),
                _record_locator(authoritative),
            ),
        ):
            if catalog_value != footer_value:
                raise PackFormatError(
                    "catalog descriptor does not match the pack footer: "
                    f"{domain} for {descriptor.capture_id}"
                )

    def _resolve(self, selection: CaptureSelection) -> tuple[CaptureDescriptor, ...]:
        resolved = self._catalog.get_by_ids(
            selection.capture_ids,
            tenant_id=selection.tenant_id,
            watermark=selection.catalog_watermark,
        )
        by_id: dict[str, CaptureDescriptor] = {}
        for descriptor in resolved:
            if descriptor.capture_id in by_id:
                raise DuplicateCaptureError(
                    f"catalog returned duplicate capture: {descriptor.capture_id}"
                )
            by_id[descriptor.capture_id] = descriptor
        if set(by_id) != set(selection.capture_ids):
            raise PackFormatError("selection no longer resolves at its catalog watermark")
        return tuple(by_id[capture_id] for capture_id in selection.capture_ids)

    def _plan(self, descriptors: Sequence[CaptureDescriptor]) -> tuple[_ObjectPlan, ...]:
        grouped: dict[tuple[str, str, str], list[int]] = {}
        for index, descriptor in enumerate(descriptors):
            locator = descriptor.locator
            grouped.setdefault(
                (locator.store_id, locator.pack_id, locator.object_key), []
            ).append(index)

        plans: list[_ObjectPlan] = []
        for (store_id, pack_id, object_key), indexes in grouped.items():
            indexes.sort(key=lambda index: descriptors[index].locator.offset)
            ranges: list[_ReadRange] = []
            current: list[int] = []
            start = end = 0
            for index in indexes:
                locator = descriptors[index].locator
                if not current:
                    current = [index]
                    start = locator.offset
                    end = locator.offset + locator.stored_length
                    continue
                if locator.offset <= end + self._max_coalesce_gap_bytes:
                    current.append(index)
                    end = max(end, locator.offset + locator.stored_length)
                    continue
                ranges.append(
                    _ReadRange(tuple(current), offset=start, length=end - start)
                )
                current = [index]
                start = locator.offset
                end = locator.offset + locator.stored_length
            if current:
                ranges.append(
                    _ReadRange(tuple(current), offset=start, length=end - start)
                )
            plans.append(
                _ObjectPlan(
                    store_id=store_id,
                    pack_id=pack_id,
                    object_key=object_key,
                    descriptor_indexes=tuple(indexes),
                    ranges=tuple(ranges),
                )
            )
        return tuple(plans)
