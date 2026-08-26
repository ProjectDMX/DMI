from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, Sequence

from .catalog import PackIdentity
from .model import CaptureDescriptor, PackRef


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ClickHouseClient(Protocol):
    def execute(self, query: str, params=None, **kwargs): ...


def _identifier(value: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid ClickHouse identifier: {value!r}")
    return value


def _quoted(value: str) -> str:
    return f"`{value}`"


@dataclass(frozen=True, slots=True)
class ClickHouseCatalogConfig:
    database: str = "default"
    table_prefix: str = "dmi"
    query_pack_limit: int = 10_000

    def __post_init__(self) -> None:
        _identifier(self.database)
        _identifier(self.table_prefix)
        if type(self.query_pack_limit) is not int or self.query_pack_limit <= 0:
            raise ValueError("query_pack_limit must be positive")


_CAPTURE_COLUMNS = (
    "capture_id", "tenant_id", "experiment_id", "run_id", "session_id",
    "request_id", "sequence_id", "model_id", "model_revision",
    "adapter_revision", "capture_policy_version", "hook_name", "layer_number",
    "producer_rank", "step_number", "token_start", "token_end",
    "batch_position", "dtype", "shape", "captured_at_ns", "pack_id",
    "store_id", "object_key", "object_bytes", "pack_checksum",
    "pack_record_count", "payload_offset", "stored_length", "decoded_length",
    "codec", "payload_checksum", "index_version",
)

# Catalog facets: descriptor-derived columns that make the catalog filterable
# and sortable server-side. They are pure functions of columns the writer
# already stores, so MATERIALIZED computes them at insert with no indexer
# change and no extra object reads.
#
# The casts are not decoration. On ClickHouse 26.9 ``arrayProduct`` returns
# Float64, ``UInt64 - UInt64`` returns Int64, and ``nullIf`` makes an
# expression Nullable -- none of which fit these column types.
_FACET_COLUMNS = (
    ("facet_version", "UInt16", "1"),
    ("element_count", "UInt64", "toUInt64(arrayProduct(shape))"),
    ("tensor_rank", "UInt8", "toUInt8(length(shape))"),
    ("token_span", "UInt64", "toUInt64(token_end - token_start)"),
    (
        "compression_ratio",
        "Float32",
        "toFloat32(if(stored_length = 0, 0, decoded_length / stored_length))",
    ),
)


def _facet_ddl() -> str:
    return ",\n".join(
        f"{name} {kind} MATERIALIZED {expression}"
        for name, kind, expression in _FACET_COLUMNS
    )


_PACK_COLUMNS = (
    "pack_id", "store_id", "object_key", "object_bytes", "pack_checksum",
    "record_count", "index_version",
)


class ClickHouseCatalogWriter:
    def __init__(
        self, client: ClickHouseClient, config: ClickHouseCatalogConfig | None = None
    ) -> None:
        self._client = client
        self._config = config or ClickHouseCatalogConfig()
        prefix = self._config.table_prefix
        self._capture_raw = f"{prefix}_capture_raw"
        self._capture_view = f"{prefix}_capture"
        self._pack_raw = f"{prefix}_pack_inventory_raw"
        self._pack_view = f"{prefix}_pack_inventory"
        self._watermark = f"{prefix}_index_watermark"

    def ensure_schema(self) -> None:
        database = _quoted(self._config.database)
        capture_raw = f"{database}.{_quoted(self._capture_raw)}"
        capture_view = f"{database}.{_quoted(self._capture_view)}"
        pack_raw = f"{database}.{_quoted(self._pack_raw)}"
        pack_view = f"{database}.{_quoted(self._pack_view)}"
        self._client.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {capture_raw} (
capture_id String, tenant_id String, experiment_id String, run_id String,
session_id String, request_id String, sequence_id String, model_id String,
model_revision String, adapter_revision Nullable(String),
capture_policy_version String, hook_name LowCardinality(String), layer_number Int32,
producer_rank UInt32, step_number UInt64, token_start UInt64, token_end UInt64,
batch_position UInt32, dtype LowCardinality(String), shape Array(UInt32),
captured_at_ns UInt64, pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), pack_record_count UInt32,
payload_offset UInt64, stored_length UInt64, decoded_length UInt64,
codec LowCardinality(String), payload_checksum FixedString(8), index_version UInt64,
{_facet_ddl()}
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY (tenant_id, experiment_id, run_id, captured_at_ns, capture_id)"""
        )
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {pack_raw} (
pack_id UUID, store_id LowCardinality(String), object_key String,
object_bytes UInt64, pack_checksum FixedString(64), record_count UInt32,
index_version UInt64
) ENGINE = ReplacingMergeTree(index_version)
ORDER BY (store_id, pack_id)"""
        )
        # Tables created by an earlier build predate the facet columns, and
        # CREATE TABLE IF NOT EXISTS will not add them. ADD COLUMN IF NOT
        # EXISTS is idempotent, so this is safe on every start.
        for name, kind, expression in _FACET_COLUMNS:
            self._client.execute(
                f"ALTER TABLE {capture_raw} ADD COLUMN IF NOT EXISTS "
                f"{name} {kind} MATERIALIZED {expression}"
            )

        # A version becomes readable only once its whole batch is durable, so
        # the watermark cannot be derived from the descriptor table: a reader
        # sampling max(index_version) there sees a version mid-batch, between
        # the INSERTs that make it up. This table is written as the last step of
        # an indexing call instead. Plain MergeTree, because it is a log --
        # ReplacingMergeTree would eventually collapse the history a pinned
        # snapshot reads.
        self._client.execute(
            f"""CREATE TABLE IF NOT EXISTS {database}.{_quoted(self._watermark)} (
index_version UInt64, published_at_ns UInt64, indexed_rows UInt64, indexed_packs UInt32
) ENGINE = MergeTree ORDER BY index_version"""
        )

        capture_public = ", ".join(_CAPTURE_COLUMNS[:-1])
        pack_public = ", ".join(_PACK_COLUMNS[:-1])
        self._client.execute(
            f"CREATE VIEW IF NOT EXISTS {capture_view} AS "
            f"SELECT {capture_public} FROM {capture_raw} FINAL"
        )
        self._client.execute(
            f"CREATE VIEW IF NOT EXISTS {pack_view} AS "
            f"SELECT {pack_public} FROM {pack_raw} FINAL"
        )

    def committed_pack_ids(
        self, identities: Sequence[PackIdentity]
    ) -> set[PackIdentity]:
        if not identities:
            return set()
        if len(identities) > self._config.query_pack_limit:
            raise ValueError("pack identity query exceeds query_pack_limit")
        table = self._qualified(self._pack_view)
        rows = self._client.execute(
            f"SELECT store_id, toString(pack_id) FROM {table} "
            "WHERE (store_id, pack_id) IN %(identities)s",
            {"identities": list(identities)},
        )
        return {(self._text(row[0]), self._text(row[1])) for row in rows}

    def write_descriptors(
        self, descriptors: Sequence[CaptureDescriptor], *, index_version: int
    ) -> None:
        if not descriptors:
            return
        self._validate_version(index_version)
        rows = [self._descriptor_row(item, index_version) for item in descriptors]
        self._client.execute(
            f"INSERT INTO {self._qualified(self._capture_raw)} "
            f"({', '.join(_CAPTURE_COLUMNS)}) VALUES",
            rows,
        )

    def publish_watermark(
        self,
        *,
        index_version: int,
        published_at_ns: int,
        indexed_rows: int,
        indexed_packs: int,
    ) -> None:
        """Make a version readable, after everything it covers is durable."""
        self._validate_version(index_version)
        self._validate_version(published_at_ns)
        self._client.execute(
            f"INSERT INTO {self._qualified(self._watermark)} "
            "(index_version, published_at_ns, indexed_rows, indexed_packs) VALUES",
            [(index_version, published_at_ns, indexed_rows, indexed_packs)],
        )

    def commit_packs(
        self, refs: Sequence[PackRef], *, index_version: int
    ) -> None:
        if not refs:
            return
        self._validate_version(index_version)
        rows = [
            (
                ref.pack_id, ref.store_id, ref.object_key, ref.object_bytes,
                ref.checksum, ref.record_count, index_version,
            )
            for ref in refs
        ]
        self._client.execute(
            f"INSERT INTO {self._qualified(self._pack_raw)} "
            f"({', '.join(_PACK_COLUMNS)}) VALUES",
            rows,
        )

    def _qualified(self, table: str) -> str:
        return f"{_quoted(self._config.database)}.{_quoted(table)}"

    @staticmethod
    def _validate_version(index_version: int) -> None:
        if type(index_version) is not int or not 0 <= index_version < 2**64:
            raise ValueError("index_version must fit UInt64")

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if not isinstance(value, str):
            raise ValueError("ClickHouse returned an invalid pack identity")
        return value

    @staticmethod
    def _descriptor_row(item: CaptureDescriptor, index_version: int) -> tuple:
        metadata = item.metadata
        locator = item.locator
        return (
            metadata.capture_id, metadata.tenant_id, metadata.experiment_id,
            metadata.run_id, metadata.session_id, metadata.request_id,
            metadata.sequence_id, metadata.model_id, metadata.model_revision,
            metadata.adapter_revision, metadata.capture_policy_version,
            metadata.hook_name, metadata.layer_number, metadata.producer_rank,
            metadata.step_number, metadata.token_start, metadata.token_end,
            metadata.batch_position, metadata.dtype, list(metadata.shape),
            metadata.captured_at_ns, locator.pack_id, locator.store_id,
            locator.object_key, locator.object_bytes, locator.pack_checksum,
            locator.pack_record_count, locator.offset, locator.stored_length,
            locator.decoded_length, locator.codec, locator.checksum, index_version,
        )
