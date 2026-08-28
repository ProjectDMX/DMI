from __future__ import annotations

from io import BytesIO
from uuid import UUID

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    PackConflictError,
    PackFormatError,
    PackIntegrityError,
    PackRef,
    PackWriter,
    S3PackStore,
    S3StoreConfig,
)


pytestmark = pytest.mark.cpu


class _ClientError(Exception):
    def __init__(self, status: int, code: str):
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }
        super().__init__(code)


class _Body(BytesIO):
    def close(self) -> None:
        super().close()


class _S3Client:
    def __init__(self):
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.upload_configs: list[object] = []

    def head_object(self, *, Bucket: str, Key: str):
        try:
            data, metadata = self.objects[Key]
        except KeyError as exc:
            raise _ClientError(404, "NoSuchKey") from exc
        return {"ContentLength": len(data), "Metadata": metadata}

    def upload_fileobj(
        self,
        Fileobj,
        Bucket: str,
        Key: str,
        ExtraArgs: dict[str, object],
        Config: object,
    ) -> None:
        self.upload_configs.append(Config)
        self.objects[Key] = (Fileobj.read(), dict(ExtraArgs["Metadata"]))

    def get_object(self, *, Bucket: str, Key: str, Range: str):
        start, end = (int(value) for value in Range.removeprefix("bytes=").split("-"))
        data = self.objects[Key][0][start : end + 1]
        return {"Body": _Body(data), "ContentLength": len(data)}

    def list_objects_v2(self, **request):
        keys = sorted(
            key for key in self.objects if key.startswith(request.get("Prefix", ""))
        )
        start = int(request.get("ContinuationToken", "0"))
        limit = request["MaxKeys"]
        selected = keys[start : start + limit]
        next_index = start + len(selected)
        return {
            "Contents": [
                {"Key": key, "Size": len(self.objects[key][0])}
                for key in selected
            ],
            "IsTruncated": next_index < len(keys),
            **(
                {"NextContinuationToken": str(next_index)}
                if next_index < len(keys)
                else {}
            ),
        }


def _pack(pack_id: str = "018f0000-0000-7000-8000-000000000001"):
    metadata = CaptureMetadata(
        capture_id="capture-a",
        tenant_id="tenant-a",
        experiment_id="exp-a",
        run_id="run-a",
        session_id="session-a",
        request_id="request-a",
        sequence_id="sequence-a",
        model_id="model-a",
        model_revision="revision-a",
        adapter_revision=None,
        capture_policy_version="policy-v1",
        hook_name="resid_pre",
        layer_number=3,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="float32",
        shape=(2,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    writer = PackWriter(
        pack_id=UUID(pack_id),
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=1024 * 1024,
    )
    writer.append(CaptureRecord(metadata=metadata, payload=b"abcdefgh"))
    return writer.seal()


def _store(client: _S3Client) -> S3PackStore:
    return S3PackStore(
        client,
        bucket="captures",
        store_id="garage-local",
        transfer_config="bounded-transfer",
    )


def test_s3_config_rejects_insecure_remote_endpoint_and_hides_secrets():
    with pytest.raises(ValueError, match="allow_insecure_http"):
        S3StoreConfig(
            endpoint_url="http://garage.example.com",
            bucket="captures",
            region="garage",
            access_key_id="access",
            secret_access_key="secret",
        )

    config = S3StoreConfig(
        endpoint_url="http://127.0.0.1:3900",
        bucket="captures",
        region="garage",
        access_key_id="access",
        secret_access_key="secret",
        allow_insecure_http=True,
    )

    assert "secret" not in repr(config)
    assert config.multipart_chunk_bytes >= 5 * 1024**2

    with pytest.raises(ValueError, match="origin"):
        S3StoreConfig(
            endpoint_url="https://garage.example.com/api",
            bucket="captures",
            region="garage",
        )
    with pytest.raises(ValueError, match="credentials"):
        S3StoreConfig(
            endpoint_url="https://garage.example.com",
            bucket="captures",
            region="garage",
            access_key_id="",
            secret_access_key="secret",
        )


def test_s3_put_is_retry_safe_and_uses_checksum_metadata():
    client = _S3Client()
    store = _store(client)
    pack = _pack()
    key = "v1/tenant=tenant-a/pack.dmi-pack"

    first = store.put(pack, key)
    second = store.put(pack, key)

    assert first == second
    assert len(client.upload_configs) == 1
    assert client.objects[key][1]["dmi-sha256"] == pack.checksum
    assert store.stat(first).checksum == pack.checksum


def test_s3_discovers_pack_reference_from_object_metadata():
    client = _S3Client()
    store = _store(client)
    pack = _pack()
    expected = store.put(pack, "packs/a.dmi-pack")

    assert store.inspect("packs/a.dmi-pack") == expected

    data, metadata = client.objects["packs/a.dmi-pack"]
    client.objects["packs/a.dmi-pack"] = (data, {**metadata, "dmi-sha256": "bad"})
    with pytest.raises(PackIntegrityError, match="metadata"):
        store.inspect("packs/a.dmi-pack")


def test_s3_put_rejects_an_existing_key_with_different_metadata():
    client = _S3Client()
    store = _store(client)
    pack = _pack()
    key = "v1/tenant=tenant-a/pack.dmi-pack"
    client.objects[key] = (b"different", {"dmi-sha256": "0" * 64})

    with pytest.raises(PackConflictError, match="different content"):
        store.put(pack, key)


def test_s3_stat_rejects_missing_integrity_metadata():
    client = _S3Client()
    store = _store(client)
    pack = _pack()
    key = "v1/tenant=tenant-a/pack.dmi-pack"
    client.objects[key] = (pack.data, {})
    ref = PackRef(
        pack_id=pack.pack_id,
        store_id=store.store_id,
        object_key=key,
        object_bytes=len(pack.data),
        checksum=pack.checksum,
        record_count=pack.record_count,
    )

    with pytest.raises(PackIntegrityError, match="metadata"):
        store.stat(ref)


def test_s3_stat_rejects_a_malformed_head_response():
    client = _S3Client()
    client.head_object = lambda **_: []
    pack = _pack()
    ref = PackRef(
        pack_id=pack.pack_id,
        store_id="garage-local",
        object_key="v1/pack.dmi-pack",
        object_bytes=len(pack.data),
        checksum=pack.checksum,
        record_count=pack.record_count,
    )

    with pytest.raises(PackIntegrityError, match="HeadObject"):
        _store(client).stat(ref)


def test_s3_range_reads_are_exact_and_bounded():
    client = _S3Client()
    store = _store(client)
    pack = _pack()
    ref = store.put(pack, "v1/tenant=tenant-a/pack.dmi-pack")

    assert store.read_range(ref, 10, 17) == pack.data[10:27]
    assert store.read_range(ref, 0, 0) == b""
    with pytest.raises(PackFormatError, match="exceeds object size"):
        store.read_range(ref, len(pack.data), 1)


def test_s3_listing_is_prefix_scoped_and_cursor_bounded():
    client = _S3Client()
    store = _store(client)
    for index in range(3):
        client.objects[f"v1/day=2026-08-25/{index}.dmi-pack"] = (
            bytes(index + 1),
            {"dmi-sha256": str(index) * 64},
        )
    client.objects["other/ignored.dmi-pack"] = (b"x", {"dmi-sha256": "f" * 64})

    first = store.list_objects(prefix="v1/day=2026-08-25/", limit=2)
    second = store.list_objects(
        prefix="v1/day=2026-08-25/", cursor=first.next_cursor, limit=2
    )

    assert [item.object_key for item in first.items] == [
        "v1/day=2026-08-25/0.dmi-pack",
        "v1/day=2026-08-25/1.dmi-pack",
    ]
    assert first.next_cursor == "2"
    assert [item.object_key for item in second.items] == [
        "v1/day=2026-08-25/2.dmi-pack"
    ]
    assert second.next_cursor is None


def test_s3_listing_rejects_a_truncated_page_without_a_cursor():
    client = _S3Client()
    client.list_objects_v2 = lambda **_: {
        "Contents": [],
        "IsTruncated": True,
    }

    with pytest.raises(PackIntegrityError, match="cursor"):
        _store(client).list_objects(limit=1)


class _LyingPack:
    """A pack source whose bytes do not match the checksum it declares."""

    def __init__(self, sealed):
        self.pack_id = sealed.pack_id
        self.created_at_ns = sealed.created_at_ns
        self.record_count = sealed.record_count
        self.checksum = sealed.checksum
        self._data = b"\x00" * len(sealed.data)

    @property
    def object_bytes(self) -> int:
        return len(self._data)

    def open(self):
        from io import BytesIO

        return BytesIO(self._data)


def test_s3_put_verifies_the_bytes_it_uploads():
    client = _S3Client()
    store = _store(client)

    # upload_fileobj hands the stream to the transfer manager, so put never sees
    # the bytes. Without hashing the source it would happily store content that
    # contradicts the checksum it writes into object metadata -- and stat()
    # compares against that same metadata, so nothing downstream could tell.
    with pytest.raises(PackIntegrityError, match="checksum"):
        store.put(_LyingPack(_pack()), "packs/lying.dmi-pack")

    assert client.objects == {}


def test_s3_put_still_accepts_a_faithful_source():
    client = _S3Client()
    store = _store(client)

    ref = store.put(_pack(), "packs/honest.dmi-pack")

    assert ref.checksum == _pack().checksum
    assert len(client.objects) == 1


def test_s3_put_retry_rejects_an_object_missing_the_format_marker():
    client = _S3Client()
    store = _store(client)
    pack = _pack()
    key = "v1/tenant=tenant-a/pack.dmi-pack"
    store.put(pack, key)
    data, metadata = client.objects[key]
    client.objects[key] = (
        data,
        {name: value for name, value in metadata.items() if name != "dmi-format"},
    )

    # An object accepted by a put() retry must be one inspect() will accept
    # later; a format-less object would otherwise have its local copy deleted
    # while reconciliation can never index the remote one.
    with pytest.raises(PackConflictError, match="different content"):
        store.put(pack, key)
