from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import PurePosixPath
import re
from typing import Mapping, Protocol
from urllib.parse import urlsplit
from uuid import UUID

from .filesystem import validate_object_key, validate_pack_source
from .model import (
    ObjectInfo,
    ObjectPage,
    PackConflictError,
    PackFormatError,
    PackIntegrityError,
    PackRef,
    PackSource,
    StoredObject,
)


_MIN_MULTIPART_BYTES = 5 * 1024**2
_MAX_CURSOR_BYTES = 2048
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONTENT_VERIFY_CHUNK_BYTES = 1024**2


def _validate_bounded_text(name: str, value: object, limit: int = 255) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode()) > limit
    ):
        raise ValueError(f"{name} must be non-empty and at most {limit} bytes")


def _validate_credentials(config: S3StoreConfig) -> None:
    if (config.access_key_id is None) != (config.secret_access_key is None):
        raise ValueError("access_key_id and secret_access_key must be set together")
    for value in (
        config.access_key_id,
        config.secret_access_key,
        config.session_token,
    ):
        if value is not None:
            _validate_bounded_text("credentials", value, 4096)


def _validate_endpoint(config: S3StoreConfig) -> None:
    if config.endpoint_url is None:
        return
    parsed = urlsplit(config.endpoint_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint_url must be an HTTP(S) origin")
    if parsed.scheme == "http" and not config.allow_insecure_http:
        raise ValueError("HTTP endpoints require allow_insecure_http=True")


class S3Client(Protocol):
    def head_object(self, **kwargs): ...
    def upload_fileobj(self, *args, **kwargs) -> None: ...
    def get_object(self, **kwargs): ...
    def list_objects_v2(self, **kwargs): ...
    def delete_object(self, **kwargs): ...


class _HashingReader:
    """Tee a pack stream through SHA-256 while the transfer manager reads it.

    ``seekable()`` is False on purpose: a non-seekable stream forces boto's
    transfer manager into sequential reads, so this tee sees every byte exactly
    once and in order.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._digest = sha256()
        self._bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._inner.read(size)
        self._digest.update(chunk)
        self._bytes_read += len(chunk)
        return chunk

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class S3StoreConfig:
    endpoint_url: str | None
    bucket: str
    region: str
    access_key_id: str | None = field(default=None, repr=False)
    secret_access_key: str | None = field(default=None, repr=False)
    session_token: str | None = field(default=None, repr=False)
    store_id: str = "s3"
    allow_insecure_http: bool = False
    multipart_threshold_bytes: int = 64 * 1024**2
    multipart_chunk_bytes: int = 16 * 1024**2
    multipart_concurrency: int = 4
    max_pool_connections: int = 32
    connect_timeout_seconds: float = 5
    read_timeout_seconds: float = 120
    max_attempts: int = 4

    def __post_init__(self) -> None:
        for name in ("bucket", "region", "store_id"):
            _validate_bounded_text(name, getattr(self, name))
        _validate_credentials(self)
        _validate_endpoint(self)
        integers = (
            "multipart_threshold_bytes",
            "multipart_chunk_bytes",
            "multipart_concurrency",
            "max_pool_connections",
            "max_attempts",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.multipart_chunk_bytes < _MIN_MULTIPART_BYTES:
            raise ValueError(
                f"multipart_chunk_bytes must be at least {_MIN_MULTIPART_BYTES}"
            )
        if self.max_pool_connections < self.multipart_concurrency:
            raise ValueError(
                "max_pool_connections must cover multipart_concurrency"
            )
        for name in ("connect_timeout_seconds", "read_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")


class S3PackStore:
    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        store_id: str = "s3",
        transfer_config: object = None,
    ) -> None:
        _validate_bounded_text("bucket", bucket)
        _validate_bounded_text("store_id", store_id)
        self._client = client
        self._bucket = bucket
        self._transfer_config = transfer_config
        self.store_id = store_id

    @classmethod
    def from_config(cls, config: S3StoreConfig) -> S3PackStore:
        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "S3 support requires the optional 's3' dependencies"
            ) from exc

        client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            aws_session_token=config.session_token,
            config=Config(
                signature_version="s3v4",
                connect_timeout=config.connect_timeout_seconds,
                read_timeout=config.read_timeout_seconds,
                max_pool_connections=config.max_pool_connections,
                retries={"mode": "standard", "max_attempts": config.max_attempts},
                s3={"addressing_style": "path"},
            ),
        )
        transfer = TransferConfig(
            multipart_threshold=config.multipart_threshold_bytes,
            multipart_chunksize=config.multipart_chunk_bytes,
            max_concurrency=config.multipart_concurrency,
            use_threads=config.multipart_concurrency > 1,
        )
        return cls(
            client,
            bucket=config.bucket,
            store_id=config.store_id,
            transfer_config=transfer,
        )

    def put(self, pack: PackSource, object_key: str) -> PackRef:
        validate_pack_source(pack)
        key = str(validate_object_key(object_key))
        existing = self._head_or_none(key)
        if existing is not None:
            # The metadata comparison is a cheap pre-filter; the object's
            # content is then hashed before it is blessed. Metadata alone is
            # only an assertion recorded at upload time: a failed upload whose
            # cleanup delete also failed leaves corrupt bytes wearing the
            # declared checksum, and a retry finding them here must not turn
            # that into silent corruption. The full read-back makes retries
            # and idempotent replays more expensive, which is acceptable
            # because they are rare.
            ref = self._existing_ref(pack, key, existing)
            self._verify_existing_content(pack, key)
            return ref

        # upload_fileobj hands the stream straight to the transfer manager, so
        # unlike FilesystemPackStore.put nothing here would otherwise see the
        # bytes. PackRef.sha256 == SHA256(bytes actually uploaded) is enforced
        # by hashing the single upload stream itself: a separate verification
        # pass would read the source twice, and PackSource does not promise
        # identical repeated reads. The dmi-sha256 object metadata below is a
        # stored assertion for later identity checks, not proof of server-side
        # verification.
        metadata = {
            "dmi-format": "dmi-pack-v1",
            "dmi-pack-id": pack.pack_id,
            "dmi-sha256": pack.checksum,
            "dmi-record-count": str(pack.record_count),
            "dmi-created-at-ns": str(pack.created_at_ns),
        }
        with pack.open() as source:
            reader = _HashingReader(source)
            self._client.upload_fileobj(
                reader,
                self._bucket,
                key,
                ExtraArgs={
                    "ContentType": "application/vnd.dmi.pack",
                    "Metadata": metadata,
                    # Defence in depth, on a different failure than the tee
                    # above: the client-side digest catches a source whose
                    # bytes contradict its declaration, this catches
                    # corruption between the bytes boto read and the bytes the
                    # server stored (each multipart part is checked on
                    # arrival). Verified against Garage 2.3, which rejects a
                    # mismatching digest with InvalidDigest, on both the
                    # single-part and multipart paths and with the
                    # non-seekable stream this reader presents. Note the
                    # stored value for a multipart object is S3's composite
                    # checksum-of-checksums, not the whole-object digest, so
                    # it is never compared against PackRef.checksum.
                    "ChecksumAlgorithm": "SHA256",
                },
                Config=self._transfer_config,
            )
        if (
            reader.bytes_read != pack.object_bytes
            or reader.hexdigest != pack.checksum
        ):
            self._delete_uploaded(key)
            raise PackIntegrityError(
                "uploaded bytes do not match the pack source declaration"
            )
        uploaded = self._head_or_none(key)
        if uploaded is None:
            raise PackIntegrityError("uploaded object is not visible to HeadObject")
        return self._existing_ref(pack, key, uploaded)

    def _verify_existing_content(self, pack: PackSource, key: str) -> None:
        """Hash an existing object's bytes before blessing it as ``pack``.

        This process cannot prove it wrote the object, and the object's
        metadata carries the checksum the uploader *declared*, not one the
        server computed -- so only the content itself can prove the object is
        the pack. The read is streamed in bounded chunks: packs run to
        hundreds of MiB and must not be buffered whole.
        """
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=key,
            Range=f"bytes=0-{pack.object_bytes - 1}",
        )
        if not isinstance(response, Mapping) or "Body" not in response:
            raise PackIntegrityError("S3 returned an invalid range response")
        digest = sha256()
        total = 0
        body = response["Body"]
        try:
            while True:
                chunk = body.read(_CONTENT_VERIFY_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
        finally:
            body.close()
        # HeadObject already matched the size, so a short or long body is the
        # transport lying rather than a conflicting object.
        if total != pack.object_bytes:
            raise PackIntegrityError("S3 returned a short or oversized range")
        if digest.hexdigest() != pack.checksum:
            raise PackConflictError(f"object key contains different content: {key}")

    def _delete_uploaded(self, key: str) -> None:
        # Best effort: removal is cleanup, not correctness -- a raised delete
        # error must not mask the integrity error. Note the leftover object's
        # metadata still asserts the DECLARED checksum (only the bytes lie),
        # so inspect() and stat() would accept it from metadata alone; what
        # keeps a later retry from blessing it is put() re-hashing an existing
        # object's content in _verify_existing_content.
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except Exception:
            pass

    def stat(self, ref: PackRef) -> ObjectInfo:
        self._validate_ref(ref)
        response = self._client.head_object(
            Bucket=self._bucket, Key=str(validate_object_key(ref.object_key))
        )
        if not isinstance(response, Mapping):
            raise PackIntegrityError("S3 returned an invalid HeadObject response")
        size, metadata = self._parse_head(response)
        checksum = metadata.get("dmi-sha256")
        if checksum is None:
            raise PackIntegrityError("S3 object is missing DMI checksum metadata")
        if size != ref.object_bytes or checksum != ref.checksum:
            raise PackIntegrityError("S3 object does not match its pack reference")
        # NOTE: this checksum is object metadata written at upload time, not a
        # digest S3 recomputed from stored bytes. It proves identity and size,
        # not that the stored content is intact -- detecting silent corruption
        # would take a server-side checksum or a read-back.
        return ObjectInfo(size=size, checksum=checksum)

    def inspect(self, object_key: str) -> PackRef:
        key = str(validate_object_key(object_key))
        response = self._client.head_object(Bucket=self._bucket, Key=key)
        if not isinstance(response, Mapping):
            raise PackIntegrityError("S3 returned an invalid HeadObject response")
        size, metadata = self._parse_head(response)
        try:
            pack_id = str(UUID(metadata["dmi-pack-id"]))
            checksum = metadata["dmi-sha256"]
            record_count = int(metadata["dmi-record-count"])
            created_at_ns = int(metadata["dmi-created-at-ns"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PackIntegrityError("S3 object has invalid DMI metadata") from exc
        if (
            metadata.get("dmi-format") != "dmi-pack-v1"
            or _SHA256.fullmatch(checksum) is None
            or not 1 <= record_count <= 1_000_000
            or created_at_ns < 0
            or size <= 0
        ):
            raise PackIntegrityError("S3 object has invalid DMI metadata")
        return PackRef(
            pack_id=pack_id,
            store_id=self.store_id,
            object_key=key,
            object_bytes=size,
            checksum=checksum,
            record_count=record_count,
        )

    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes:
        self._validate_ref(ref)
        if (
            type(offset) is not int
            or type(length) is not int
            or offset < 0
            or length < 0
        ):
            raise ValueError("range offset and length must be non-negative integers")
        if offset + length > ref.object_bytes:
            raise PackFormatError("requested range exceeds object size")
        if length == 0:
            return b""
        response = self._client.get_object(
            Bucket=self._bucket,
            Key=str(validate_object_key(ref.object_key)),
            Range=f"bytes={offset}-{offset + length - 1}",
        )
        if not isinstance(response, Mapping) or "Body" not in response:
            raise PackIntegrityError("S3 returned an invalid range response")
        body = response["Body"]
        try:
            data = body.read(length + 1)
        finally:
            body.close()
        if not isinstance(data, bytes) or len(data) != length:
            raise PackIntegrityError("S3 returned a short or oversized range")
        return data

    def list_objects(
        self,
        *,
        prefix: str = "",
        cursor: str | None = None,
        limit: int = 1000,
    ) -> ObjectPage:
        if prefix:
            self._validate_prefix(prefix)
        if cursor is not None and (
            not isinstance(cursor, str)
            or not cursor
            or len(cursor.encode()) > _MAX_CURSOR_BYTES
        ):
            raise ValueError("cursor must be non-empty and bounded")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        request: dict[str, object] = {
            "Bucket": self._bucket,
            "Prefix": prefix,
            "MaxKeys": limit,
        }
        if cursor is not None:
            request["ContinuationToken"] = cursor
        response = self._client.list_objects_v2(**request)
        if not isinstance(response, Mapping):
            raise PackIntegrityError("S3 returned an invalid listing response")
        items = self._parse_listing_items(response)
        next_cursor = self._parse_listing_cursor(response)
        return ObjectPage(items=items, next_cursor=next_cursor)

    @staticmethod
    def _parse_listing_items(
        response: Mapping[str, object],
    ) -> tuple[StoredObject, ...]:
        raw_items = response.get("Contents", ())
        if not isinstance(raw_items, (list, tuple)):
            raise PackIntegrityError("S3 listing contents are invalid")
        items = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise PackIntegrityError("S3 listing item is invalid")
            key = raw.get("Key")
            size = raw.get("Size")
            if not isinstance(key, str) or type(size) is not int or size < 0:
                raise PackIntegrityError("S3 listing item is invalid")
            validate_object_key(key)
            items.append(StoredObject(object_key=key, object_bytes=size))
        return tuple(items)

    @staticmethod
    def _parse_listing_cursor(response: Mapping[str, object]) -> str | None:
        is_truncated = response.get("IsTruncated", False)
        if type(is_truncated) is not bool:
            raise PackIntegrityError("S3 listing truncation state is invalid")
        next_cursor = response.get("NextContinuationToken") if is_truncated else None
        if is_truncated and next_cursor is None:
            raise PackIntegrityError("S3 truncated listing is missing its cursor")
        if next_cursor is not None and (
            not isinstance(next_cursor, str)
            or not next_cursor
            or len(next_cursor.encode()) > _MAX_CURSOR_BYTES
        ):
            raise PackIntegrityError("S3 listing cursor is invalid")
        return next_cursor

    def _head_or_none(self, key: str) -> Mapping[str, object] | None:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise
        if not isinstance(response, Mapping):
            raise PackIntegrityError("S3 returned an invalid HeadObject response")
        return response

    def _existing_ref(
        self, pack: PackSource, key: str, response: Mapping[str, object]
    ) -> PackRef:
        size, metadata = self._parse_head(response)
        expected = {
            # The format marker must be part of this check: an object put()
            # accepts here is one inspect() must also accept later, or a
            # retry deletes the local copy of a pack reconciliation can
            # never index.
            "dmi-format": "dmi-pack-v1",
            "dmi-pack-id": pack.pack_id,
            "dmi-sha256": pack.checksum,
            "dmi-record-count": str(pack.record_count),
            "dmi-created-at-ns": str(pack.created_at_ns),
        }
        if size != pack.object_bytes or any(
            metadata.get(name) != value for name, value in expected.items()
        ):
            raise PackConflictError(f"object key contains different content: {key}")
        return PackRef(
            pack_id=pack.pack_id,
            store_id=self.store_id,
            object_key=key,
            object_bytes=pack.object_bytes,
            checksum=pack.checksum,
            record_count=pack.record_count,
        )

    @staticmethod
    def _parse_head(response: Mapping[str, object]) -> tuple[int, Mapping[str, str]]:
        size = response.get("ContentLength")
        metadata = response.get("Metadata")
        if type(size) is not int or size < 0 or not isinstance(metadata, Mapping):
            raise PackIntegrityError("S3 returned invalid object metadata")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise PackIntegrityError("S3 returned invalid object metadata")
        return size, metadata

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if not isinstance(response, Mapping):
            return False
        error = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})
        code = error.get("Code") if isinstance(error, Mapping) else None
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
        return status == 404 or code in {"404", "NoSuchKey", "NotFound"}

    @staticmethod
    def _validate_prefix(prefix: str) -> None:
        if (
            not isinstance(prefix, str)
            or prefix.startswith("/")
            or "\\" in prefix
            or any(part in {"", ".", ".."} for part in prefix.rstrip("/").split("/"))
        ):
            raise ValueError("prefix is invalid")
        PurePosixPath(prefix)

    def _validate_ref(self, ref: PackRef) -> None:
        if ref.store_id != self.store_id:
            raise ValueError(
                f"pack store mismatch: {ref.store_id!r} != {self.store_id!r}"
            )
