from __future__ import annotations

import os
from uuid import uuid4

import pytest

from dmi.storage.capture import (
    CaptureMetadata,
    CaptureRecord,
    PackIndex,
    PackWriter,
    S3PackStore,
    S3StoreConfig,
)


pytestmark = [pytest.mark.manual, pytest.mark.garage]


def _store() -> S3PackStore:
    names = (
        "DMI_S3_ENDPOINT",
        "DMI_S3_BUCKET",
        "DMI_S3_ACCESS_KEY_ID",
        "DMI_S3_SECRET_ACCESS_KEY",
    )
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip("missing Garage environment: " + ", ".join(missing))
    config = S3StoreConfig(
        endpoint_url=values["DMI_S3_ENDPOINT"],
        bucket=values["DMI_S3_BUCKET"],
        region=os.environ.get("DMI_S3_REGION", "garage"),
        access_key_id=values["DMI_S3_ACCESS_KEY_ID"],
        secret_access_key=values["DMI_S3_SECRET_ACCESS_KEY"],
        store_id="garage-live",
        allow_insecure_http=os.environ.get("DMI_S3_ALLOW_HTTP") == "1",
        multipart_threshold_bytes=5 * 1024**2,
        multipart_chunk_bytes=5 * 1024**2,
        multipart_concurrency=2,
    )
    return S3PackStore.from_config(config)


def test_garage_multipart_retry_listing_and_two_range_footer_read():
    store = _store()
    pack_id = uuid4()
    metadata = CaptureMetadata(
        capture_id=f"garage-live-{pack_id}",
        tenant_id="test",
        experiment_id="garage-live",
        run_id=str(pack_id),
        session_id="session-0",
        request_id="request-0",
        sequence_id="sequence-0",
        model_id="synthetic",
        model_revision="test-v1",
        adapter_revision=None,
        capture_policy_version="all-v1",
        hook_name="resid_pre",
        layer_number=0,
        producer_rank=0,
        step_number=0,
        token_start=0,
        token_end=1,
        batch_position=0,
        dtype="uint8",
        shape=(6 * 1024**2,),
        captured_at_ns=1_700_000_000_000_000_000,
    )
    writer = PackWriter(
        pack_id=pack_id,
        created_at_ns=metadata.captured_at_ns,
        max_pack_bytes=8 * 1024**2,
    )
    writer.append(
        CaptureRecord(metadata=metadata, payload=b"x" * (6 * 1024**2))
    )
    pack = writer.seal()
    prefix = f"tests/dmi/{pack_id}/"
    key = f"{prefix}{pack.pack_id}.dmi-pack"

    first = store.put(pack, key)
    second = store.put(pack, key)
    info = store.stat(first)
    index = PackIndex.from_store(store, first)
    page = store.list_objects(prefix=prefix, limit=10)

    assert first == second
    assert info.size == len(pack.data)
    assert info.checksum == pack.checksum
    assert index.descriptors()[0].capture_id == metadata.capture_id
    assert [item.object_key for item in page.items] == [key]
