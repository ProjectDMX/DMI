"""Immutable capture packs and bounded analysis hydration."""

from .filesystem import FilesystemPackStore
from .model import (
    CaptureCatalog,
    CaptureDescriptor,
    CaptureMetadata,
    CapturePage,
    CaptureQuery,
    CaptureRecord,
    CaptureSelection,
    CaptureStorageError,
    DuplicateCaptureError,
    HydratedCapture,
    HydrationEstimate,
    HydrationLimitError,
    InvalidCursorError,
    ObjectInfo,
    ObjectPage,
    PackConflictError,
    PackFormatError,
    PackIntegrityError,
    PackRef,
    PackSource,
    PackStore,
    PayloadLocator,
    StoredObject,
    VerifiedPackSource,
)
from .pack import PackIndex, PackReader, PackWriter, SealedPack
from .pipeline import (
    AdmissionResult,
    BoundedRecordQueue,
    DirectPackSink,
    FlushReason,
    HistogramSnapshot,
    HostCapturePipeline,
    OverloadPolicy,
    OversizedRecordError,
    PackAssembler,
    PipelineConfig,
    PipelineEvent,
    PipelineFailedError,
    PipelineSnapshot,
    QueueSnapshot,
    ReadyPack,
    object_key_for,
)
from .extensions import (
    ArtifactProducer,
    ArtifactSink,
    ExtensionError,
    ExtensionFailure,
    ExtensionRegistry,
    ScalarMetric,
)
from .reader import CaptureReader, CaptureSummary
from .summary import (
    CORE_SUMMARY_VERSION,
    ArtifactRef,
    CoreTensorSummaryV1,
    decode_tensor,
    summarize_tensor,
)
from .catalog import (
    CatalogIndexer,
    CatalogIndexerConfig,
    CatalogReconciler,
    CatalogWriter,
    IndexEvent,
    IndexFailure,
    IndexResult,
    PackIdentity,
    PackInventory,
    ReconcileResult,
)
from .clickhouse_catalog import ClickHouseCatalogConfig, ClickHouseCatalogWriter
from .clickhouse_reader import ClickHouseCaptureCatalog, ClickHouseReaderConfig
from .cursor import Cursor, CursorKey, decode_cursor, encode_cursor
from .s3 import S3PackStore, S3StoreConfig
from .spool import (
    DurablePackSink,
    DurablePackSpool,
    SpoolFullError,
    SpoolSnapshot,
    SpoolUploader,
    StagedPack,
    ParallelSpoolUploader,
    ParallelUploadConfig,
    UploadBatchResult,
    UploadEvent,
    UploadFailure,
    UploadSnapshot,
)


_REFERENCE_ADAPTER_EXPORTS = frozenset(
    {"CapturePackReferenceSink", "CapturePayloadSlice", "CaptureRecordFormat"}
)


def __getattr__(name: str):
    if name in _REFERENCE_ADAPTER_EXPORTS:
        from . import record_adapter

        return getattr(record_adapter, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _REFERENCE_ADAPTER_EXPORTS)

__all__ = [
    "ArtifactProducer",
    "ArtifactRef",
    "ArtifactSink",
    "AdmissionResult",
    "BoundedRecordQueue",
    "CaptureCatalog",
    "CaptureDescriptor",
    "CaptureMetadata",
    "CapturePage",
    "CapturePackReferenceSink",
    "CapturePayloadSlice",
    "CaptureQuery",
    "CaptureReader",
    "CaptureRecord",
    "CaptureRecordFormat",
    "CaptureSelection",
    "CaptureSummary",
    "CaptureStorageError",
    "CatalogIndexer",
    "CatalogIndexerConfig",
    "CatalogReconciler",
    "CatalogWriter",
    "ClickHouseCatalogConfig",
    "ClickHouseCatalogWriter",
    "ClickHouseCaptureCatalog",
    "ClickHouseReaderConfig",
    "Cursor",
    "CursorKey",
    "CoreTensorSummaryV1",
    "CORE_SUMMARY_VERSION",
    "DuplicateCaptureError",
    "DirectPackSink",
    "DurablePackSink",
    "DurablePackSpool",
    "ExtensionError",
    "ExtensionFailure",
    "ExtensionRegistry",
    "FilesystemPackStore",
    "FlushReason",
    "HistogramSnapshot",
    "HostCapturePipeline",
    "HydratedCapture",
    "HydrationEstimate",
    "HydrationLimitError",
    "IndexEvent",
    "IndexFailure",
    "IndexResult",
    "InvalidCursorError",
    "ObjectInfo",
    "ObjectPage",
    "OverloadPolicy",
    "OversizedRecordError",
    "PackAssembler",
    "PackConflictError",
    "PackFormatError",
    "PackIntegrityError",
    "PackIndex",
    "PackIdentity",
    "PackInventory",
    "PackReader",
    "PackRef",
    "PackSource",
    "PackStore",
    "PackWriter",
    "ParallelSpoolUploader",
    "ParallelUploadConfig",
    "PayloadLocator",
    "PipelineConfig",
    "PipelineEvent",
    "PipelineFailedError",
    "PipelineSnapshot",
    "QueueSnapshot",
    "ReadyPack",
    "ReconcileResult",
    "ScalarMetric",
    "S3PackStore",
    "S3StoreConfig",
    "SealedPack",
    "SpoolFullError",
    "SpoolSnapshot",
    "SpoolUploader",
    "StagedPack",
    "StoredObject",
    "UploadBatchResult",
    "UploadEvent",
    "UploadFailure",
    "UploadSnapshot",
    "VerifiedPackSource",
    "decode_cursor",
    "encode_cursor",
    "decode_tensor",
    "summarize_tensor",
    "object_key_for",
]
