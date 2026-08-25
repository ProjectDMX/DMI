from __future__ import annotations

from hashlib import sha256
import errno
import os
from pathlib import Path, PurePosixPath
import re
import stat as stat_mode
import tempfile
from typing import BinaryIO
from uuid import UUID

from .model import (
    ObjectInfo,
    PackConflictError,
    PackFormatError,
    PackIntegrityError,
    PackRef,
    PackSource,
)


_KEY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=%-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COPY_BYTES = 1024 * 1024


def validate_object_key(object_key: str) -> PurePosixPath:
    if not isinstance(object_key, str) or not object_key or "\\" in object_key:
        raise ValueError("object key is invalid")
    key = PurePosixPath(object_key)
    if key.is_absolute() or any(
        part in ("", ".", "..") or _KEY_COMPONENT.fullmatch(part) is None
        for part in key.parts
    ):
        raise ValueError("object key is invalid")
    return key


def validate_pack_source(pack: PackSource) -> None:
    if not isinstance(pack, PackSource):
        raise TypeError("pack must implement PackSource")
    try:
        canonical_id = str(UUID(pack.pack_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("pack source has an invalid pack ID") from exc
    if canonical_id != pack.pack_id:
        raise ValueError("pack source must use a canonical pack ID")
    integers = (pack.created_at_ns, pack.record_count, pack.object_bytes)
    if any(
        type(value) is not int or value < 0 or value > 2**63 - 1
        for value in integers
    ):
        raise ValueError("pack source has invalid numeric metadata")
    if pack.record_count == 0 or pack.object_bytes == 0:
        raise ValueError("pack source must contain records and bytes")
    if not isinstance(pack.checksum, str) or _SHA256.fullmatch(pack.checksum) is None:
        raise ValueError("pack source has an invalid checksum")


def copy_pack_source(pack: PackSource, destination: BinaryIO) -> None:
    validate_pack_source(pack)
    digest = sha256()
    remaining = pack.object_bytes
    with pack.open() as source:
        while remaining:
            requested = min(_COPY_BYTES, remaining)
            chunk = source.read(requested)
            if not isinstance(chunk, bytes) or not chunk or len(chunk) > requested:
                raise PackIntegrityError("pack source returned an invalid byte stream")
            destination.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise PackIntegrityError("pack source exceeds its declared size")
    if digest.hexdigest() != pack.checksum:
        raise PackIntegrityError("pack source checksum does not match its metadata")


class FilesystemPackStore:
    def __init__(self, root: str | Path, *, store_id: str = "filesystem") -> None:
        if not store_id or len(store_id.encode("utf-8")) > 128:
            raise ValueError("store_id must be non-empty and at most 128 bytes")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store_id = store_id

    @staticmethod
    def _validate_key(object_key: str) -> PurePosixPath:
        return validate_object_key(object_key)

    def _path(self, object_key: str) -> Path:
        key = self._validate_key(object_key)
        path = self.root.joinpath(*key.parts)
        parent = path.parent.resolve(strict=False)
        if not parent.is_relative_to(self.root):
            raise ValueError("object key escapes the store root")
        return path

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = sha256()
        with FilesystemPackStore._open_regular(path) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _open_regular(path: Path) -> BinaryIO:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise PackFormatError("pack object must be a regular file") from exc
            raise
        if not stat_mode.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise PackFormatError("pack object must be a regular file")
        return os.fdopen(descriptor, "rb")

    @staticmethod
    def _size(path: Path) -> int:
        status = os.stat(path, follow_symlinks=False)
        if not stat_mode.S_ISREG(status.st_mode):
            raise PackFormatError("pack object must be a regular file")
        return status.st_size

    def _existing_ref(self, pack: PackSource, object_key: str, path: Path) -> PackRef:
        if self._size(path) != pack.object_bytes or self._checksum(path) != pack.checksum:
            raise PackConflictError(f"object key contains different content: {object_key}")
        return PackRef(
            pack_id=pack.pack_id,
            store_id=self.store_id,
            object_key=object_key,
            object_bytes=pack.object_bytes,
            checksum=pack.checksum,
            record_count=pack.record_count,
        )

    def put(self, pack: PackSource, object_key: str) -> PackRef:
        validate_pack_source(pack)
        path = self._path(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return self._existing_ref(pack, object_key, path)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".open",
                dir=path.parent,
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                copy_pack_source(pack, handle)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                return self._existing_ref(pack, object_key, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        return PackRef(
            pack_id=pack.pack_id,
            store_id=self.store_id,
            object_key=object_key,
            object_bytes=pack.object_bytes,
            checksum=pack.checksum,
            record_count=pack.record_count,
        )

    def stat(self, ref: PackRef) -> ObjectInfo:
        self._validate_ref(ref)
        path = self._path(ref.object_key)
        return ObjectInfo(size=self._size(path), checksum=self._checksum(path))

    def read_range(self, ref: PackRef, offset: int, length: int) -> bytes:
        self._validate_ref(ref)
        if (
            type(offset) is not int
            or type(length) is not int
            or offset < 0
            or length < 0
        ):
            raise ValueError("range offset and length must be non-negative integers")
        path = self._path(ref.object_key)
        size = self._size(path)
        if offset + length > size:
            raise PackFormatError("requested range exceeds object size")
        with self._open_regular(path) as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise PackIntegrityError("filesystem returned a short range")
        return data

    def _validate_ref(self, ref: PackRef) -> None:
        if ref.store_id != self.store_id:
            raise ValueError(
                f"pack store mismatch: {ref.store_id!r} != {self.store_id!r}"
            )
