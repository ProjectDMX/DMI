from __future__ import annotations

from dataclasses import dataclass
import math
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
    PackStore,
)
from .extensions import (
    ArtifactSink,
    ExtensionFailure,
    ExtensionRegistry,
)
from .pack import verify_payload
from .summary import ArtifactRef, CoreTensorSummaryV1, decode_tensor, summarize_tensor


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    """A versioned core summary plus whatever extensions contributed."""

    capture_id: str
    core: CoreTensorSummaryV1
    scalars: Mapping[str, float]
    artifacts: tuple[ArtifactRef, ...]
    failures: tuple[ExtensionFailure, ...]


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
        return HydrationEstimate(
            capture_count=len(descriptors),
            object_count=len(plans),
            request_count=sum(len(plan.ranges) for plan in plans),
            logical_bytes=sum(item.locator.decoded_length for item in descriptors),
            stored_bytes=sum(item.locator.stored_length for item in descriptors),
            request_bytes=sum(item.length for plan in plans for item in plan.ranges),
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
        request_count = sum(len(plan.ranges) for plan in plans)
        if request_count > request_limit:
            raise HydrationLimitError(
                f"hydration request limit exceeded: {request_count} > {request_limit}"
            )
        request_bytes = sum(item.length for plan in plans for item in plan.ranges)
        if request_bytes > byte_limit:
            raise HydrationLimitError(
                f"hydration byte limit exceeded: {request_bytes} > {byte_limit}"
            )

        payloads: list[bytes | None] = [None] * len(descriptors)
        for plan in plans:
            store = self._stores.get(plan.store_id)
            if store is None:
                raise PackFormatError(f"unknown pack store: {plan.store_id}")
            ref = descriptors[plan.descriptor_indexes[0]].locator.pack_ref
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
            raise PackFormatError("hydration plan did not resolve every capture")
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

    def _resolve(self, selection: CaptureSelection) -> tuple[CaptureDescriptor, ...]:
        resolved = self._catalog.get_by_ids(
            selection.capture_ids,
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
