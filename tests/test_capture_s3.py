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


# --- configuration validation ---------------------------------------------------


def test_s3_config_requires_both_credential_halves():
    with pytest.raises(ValueError, match="set together"):
        S3StoreConfig(
            endpoint_url="https://garage.example.com",
            bucket="captures",
            region="garage",
            access_key_id="access",
        )


def test_s3_config_accepts_a_default_endpoint():
    config = S3StoreConfig(endpoint_url=None, bucket="captures", region="garage")

    assert config.endpoint_url is None


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"multipart_concurrency": 0}, "must be positive"),
        ({"max_attempts": True}, "must be positive"),
        ({"multipart_chunk_bytes": 1024 * 1024}, "at least"),
        (
            {"max_pool_connections": 2, "multipart_concurrency": 4},
            "cover multipart_concurrency",
        ),
        ({"connect_timeout_seconds": 0}, "must be positive"),
        ({"read_timeout_seconds": True}, "must be positive"),
    ),
)
def test_s3_config_validates_transfer_and_timeout_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        S3StoreConfig(
            endpoint_url=None, bucket="captures", region="garage", **kwargs
        )


# --- from_config -----------------------------------------------------------------


def test_s3_from_config_requires_the_optional_dependency(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(RuntimeError, match="optional 's3' dependencies"):
        S3PackStore.from_config(
            S3StoreConfig(endpoint_url=None, bucket="captures", region="garage")
        )


def test_s3_from_config_builds_a_bounded_client(monkeypatch):
    import sys
    import types

    recorded: dict[str, object] = {}

    class _TransferConfig:
        def __init__(self, **kwargs):
            recorded["transfer"] = kwargs

    class _Config:
        def __init__(self, **kwargs):
            recorded["config"] = kwargs

    def client(service, **kwargs):
        recorded["service"] = service
        recorded["client"] = kwargs
        return _S3Client()

    boto3_module = types.ModuleType("boto3")
    boto3_module.client = client
    boto3_s3 = types.ModuleType("boto3.s3")
    boto3_transfer = types.ModuleType("boto3.s3.transfer")
    boto3_transfer.TransferConfig = _TransferConfig
    boto3_module.s3 = boto3_s3
    boto3_s3.transfer = boto3_transfer
    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = _Config

    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "boto3.s3", boto3_s3)
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", boto3_transfer)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)

    config = S3StoreConfig(
        endpoint_url="http://127.0.0.1:3900",
        bucket="captures",
        region="garage",
        access_key_id="access",
        secret_access_key="secret",
        allow_insecure_http=True,
        multipart_concurrency=2,
    )
    store = S3PackStore.from_config(config)

    assert store.store_id == "s3"
    assert recorded["service"] == "s3"
    client_kwargs = recorded["client"]
    assert client_kwargs["endpoint_url"] == "http://127.0.0.1:3900"
    assert client_kwargs["region_name"] == "garage"
    assert client_kwargs["aws_access_key_id"] == "access"
    assert client_kwargs["aws_secret_access_key"] == "secret"
    config_kwargs = recorded["config"]
    assert config_kwargs["signature_version"] == "s3v4"
    assert config_kwargs["retries"] == {"mode": "standard", "max_attempts": 4}
    assert config_kwargs["s3"] == {"addressing_style": "path"}
    assert config_kwargs["max_pool_connections"] == config.max_pool_connections
    transfer_kwargs = recorded["transfer"]
    assert transfer_kwargs["multipart_threshold"] == config.multipart_threshold_bytes
    assert transfer_kwargs["multipart_chunksize"] == config.multipart_chunk_bytes
    assert transfer_kwargs["max_concurrency"] == 2
    assert transfer_kwargs["use_threads"] is True


# --- response validation ----------------------------------------------------------


def _ref_for(pack, key: str) -> PackRef:
    return PackRef(
        pack_id=pack.pack_id,
        store_id="garage-local",
        object_key=key,
        object_bytes=len(pack.data),
        checksum=pack.checksum,
        record_count=pack.record_count,
    )


def test_s3_put_detects_an_upload_that_never_became_visible():
    client = _S3Client()
    original = client.upload_fileobj

    def vanish(Fileobj, Bucket, Key, ExtraArgs, Config):
        original(Fileobj, Bucket, Key, ExtraArgs=ExtraArgs, Config=Config)
        del client.objects[Key]

    client.upload_fileobj = vanish  # type: ignore[method-assign]

    with pytest.raises(PackIntegrityError, match="not visible"):
        _store(client).put(_pack(), "packs/a.dmi-pack")


def test_s3_put_reraises_a_non_404_head_failure():
    client = _S3Client()

    def unavailable(**_):
        raise _ClientError(500, "InternalError")

    client.head_object = unavailable  # type: ignore[method-assign]

    with pytest.raises(_ClientError, match="InternalError"):
        _store(client).put(_pack(), "packs/a.dmi-pack")


def test_s3_put_reraises_a_failure_without_an_http_response():
    client = _S3Client()

    def broken(**_):
        raise RuntimeError("client exploded")

    client.head_object = broken  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="client exploded"):
        _store(client).put(_pack(), "packs/a.dmi-pack")


def test_s3_put_rejects_a_malformed_head_response():
    client = _S3Client()
    client.head_object = lambda **_: []  # type: ignore[method-assign]

    with pytest.raises(PackIntegrityError, match="HeadObject"):
        _store(client).put(_pack(), "packs/a.dmi-pack")


def test_s3_stat_rejects_a_reference_that_no_longer_matches():
    from dataclasses import replace

    client = _S3Client()
    store = _store(client)
    pack = _pack()
    ref = store.put(pack, "packs/a.dmi-pack")

    with pytest.raises(PackIntegrityError, match="does not match its pack reference"):
        store.stat(replace(ref, object_bytes=ref.object_bytes + 1))


def test_s3_stat_rejects_a_foreign_store_reference():
    from dataclasses import replace

    client = _S3Client()
    store = _store(client)
    ref = store.put(_pack(), "packs/a.dmi-pack")

    with pytest.raises(ValueError, match="pack store mismatch"):
        store.stat(replace(ref, store_id="other"))


def test_s3_inspect_rejects_a_malformed_head_response():
    client = _S3Client()
    client.head_object = lambda **_: []  # type: ignore[method-assign]

    with pytest.raises(PackIntegrityError, match="HeadObject"):
        _store(client).inspect("packs/a.dmi-pack")


def test_s3_inspect_rejects_missing_or_unparseable_dmi_metadata():
    client = _S3Client()
    store = _store(client)
    client.objects["packs/bare.dmi-pack"] = (b"x" * 8, {})
    client.objects["packs/mangled.dmi-pack"] = (
        b"x" * 8,
        {
            "dmi-format": "dmi-pack-v1",
            "dmi-pack-id": "not-a-uuid",
            "dmi-sha256": "f" * 64,
            "dmi-record-count": "1",
            "dmi-created-at-ns": "1",
        },
    )

    with pytest.raises(PackIntegrityError, match="invalid DMI metadata"):
        store.inspect("packs/bare.dmi-pack")
    with pytest.raises(PackIntegrityError, match="invalid DMI metadata"):
        store.inspect("packs/mangled.dmi-pack")


def test_s3_head_metadata_must_be_well_typed():
    client = _S3Client()
    store = _store(client)
    client.head_object = lambda **_: {"ContentLength": "8", "Metadata": {}}  # type: ignore[method-assign]
    with pytest.raises(PackIntegrityError, match="invalid object metadata"):
        store.inspect("packs/a.dmi-pack")

    client.head_object = lambda **_: {"ContentLength": 8, "Metadata": {"a": 1}}  # type: ignore[method-assign]
    with pytest.raises(PackIntegrityError, match="invalid object metadata"):
        store.inspect("packs/a.dmi-pack")


def test_s3_read_range_validates_its_arguments():
    client = _S3Client()
    store = _store(client)
    ref = store.put(_pack(), "packs/a.dmi-pack")

    with pytest.raises(ValueError, match="non-negative integers"):
        store.read_range(ref, -1, 4)
    with pytest.raises(ValueError, match="non-negative integers"):
        store.read_range(ref, 0, 4.0)


def test_s3_read_range_rejects_a_malformed_or_lying_body():
    client = _S3Client()
    store = _store(client)
    ref = store.put(_pack(), "packs/a.dmi-pack")

    client.get_object = lambda **_: {}  # type: ignore[method-assign]
    with pytest.raises(PackIntegrityError, match="invalid range response"):
        store.read_range(ref, 0, 8)

    client.get_object = lambda **_: {"Body": _Body(b"xy")}  # type: ignore[method-assign]
    with pytest.raises(PackIntegrityError, match="short or oversized"):
        store.read_range(ref, 0, 8)

    client.get_object = lambda **_: {"Body": _Body(b"x" * 16)}  # type: ignore[method-assign]
    with pytest.raises(PackIntegrityError, match="short or oversized"):
        store.read_range(ref, 0, 8)


def test_s3_listing_validates_its_arguments():
    store = _store(_S3Client())

    with pytest.raises(ValueError, match="cursor"):
        store.list_objects(cursor="")
    with pytest.raises(ValueError, match="cursor"):
        store.list_objects(cursor="x" * 4096)
    with pytest.raises(ValueError, match="limit"):
        store.list_objects(limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.list_objects(limit=1001)


@pytest.mark.parametrize("prefix", ("/absolute", "a\\b", "a/../b", "a//b"))
def test_s3_listing_rejects_hostile_prefixes(prefix: str):
    with pytest.raises(ValueError, match="prefix"):
        _store(_S3Client()).list_objects(prefix=prefix)


@pytest.mark.parametrize(
    "response,match",
    (
        ([], "invalid listing response"),
        ({"Contents": "bogus", "IsTruncated": False}, "listing contents"),
        ({"Contents": [5], "IsTruncated": False}, "listing item"),
        (
            {"Contents": [{"Key": 5, "Size": 1}], "IsTruncated": False},
            "listing item",
        ),
        (
            {"Contents": [{"Key": "a.dmi-pack", "Size": "1"}], "IsTruncated": False},
            "listing item",
        ),
        ({"Contents": [], "IsTruncated": "yes"}, "truncation state"),
        (
            {"Contents": [], "IsTruncated": True, "NextContinuationToken": ""},
            "listing cursor",
        ),
        (
            {
                "Contents": [],
                "IsTruncated": True,
                "NextContinuationToken": "x" * 4096,
            },
            "listing cursor",
        ),
    ),
)
def test_s3_listing_rejects_malformed_responses(response, match):
    client = _S3Client()
    client.list_objects_v2 = lambda **_: response  # type: ignore[method-assign]

    with pytest.raises(PackIntegrityError, match=match):
        _store(client).list_objects(limit=1)
