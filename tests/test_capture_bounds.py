"""Capture bounds are checked at attach, never discovered in the forward.

A record the sink can never admit -- larger than its queue, or than an empty
pack -- is refused on the record worker, which latches the record runtime:
under the raise policy the next forward raises. Likewise a pack larger than
the uploader's in-flight budget is staged and then never uploaded. All three
depend only on configuration and the largest record an adapter will emit,
so ``validate_capture_bounds`` checks them before any forward runs.
"""

from __future__ import annotations

import pytest

from dmi.configuration.errors import ConfigurationError

pytestmark = pytest.mark.cpu

MiB = 1 << 20


def _sink(tmp_path, **fields):
    from dmi.storage.native_capture import NativeSinkConfig

    return NativeSinkConfig(spool_root=str(tmp_path), **fields)


def _storage(**fields):
    from dmi.storage.native_capture import NativeCaptureStorageConfig

    base = dict(s3_endpoint="https://s3.example.test", s3_bucket="bucket",
                s3_access_key="AKIA-test", s3_secret_key="secret-test")
    base.update(fields)
    return NativeCaptureStorageConfig(**base)


# --- uploader_max_in_flight_bytes ---------------------------------------------


def test_in_flight_budget_defaults_to_the_native_uploader_default():
    storage = _storage()
    # native/csrc/store/uploader.h UploaderConfig::max_in_flight_bytes
    assert storage.uploader_max_in_flight_bytes == 256 * MiB
    assert storage._native_dict()["uploader_max_in_flight_bytes"] == 256 * MiB


def test_in_flight_budget_reaches_the_native_service_config():
    storage = _storage(uploader_max_in_flight_bytes=512 * MiB)
    assert storage._native_dict()["uploader_max_in_flight_bytes"] == 512 * MiB


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_in_flight_budget_must_be_a_positive_int(value):
    with pytest.raises(ValueError, match="uploader_max_in_flight_bytes"):
        _storage(uploader_max_in_flight_bytes=value)


# --- validate_capture_bounds ----------------------------------------------------


def test_bounds_that_fit_pass(tmp_path):
    from dmi.storage.native_capture import validate_capture_bounds

    validate_capture_bounds(_sink(tmp_path), 8 * MiB, storage_config=_storage())


def test_a_record_larger_than_the_queue_is_refused(tmp_path):
    from dmi.storage.native_capture import validate_capture_bounds

    # The default 16 MiB queue, and a 17 MiB prefill row.
    with pytest.raises(ConfigurationError, match="max_queue_bytes"):
        validate_capture_bounds(_sink(tmp_path), 17 * MiB)


def test_a_record_that_fills_an_empty_pack_is_refused(tmp_path):
    from dmi.storage.native_capture import (
        PACK_FRAMING_RESERVE_BYTES, validate_capture_bounds,
    )

    # Admitted by the queue, but an empty pack still needs its header, row
    # footer and trailer: the sink would admit the record and then count it
    # oversized on the pack worker, storing nothing.
    sink = _sink(tmp_path, max_queue_bytes=64 * MiB, max_pack_bytes=32 * MiB)
    with pytest.raises(ConfigurationError, match="max_pack_bytes"):
        validate_capture_bounds(sink, 32 * MiB)
    validate_capture_bounds(sink, 32 * MiB - PACK_FRAMING_RESERVE_BYTES)


def test_a_pack_over_the_in_flight_budget_is_refused(tmp_path):
    from dmi.storage.native_capture import validate_capture_bounds

    sink = _sink(tmp_path, max_pack_bytes=512 * MiB)
    with pytest.raises(ConfigurationError,
                       match="uploader_max_in_flight_bytes"):
        validate_capture_bounds(sink, MiB, storage_config=_storage())
    validate_capture_bounds(
        sink, MiB,
        storage_config=_storage(uploader_max_in_flight_bytes=512 * MiB))


@pytest.mark.parametrize("value", [0, -1, 1.0, True])
def test_the_record_size_must_be_a_positive_int(tmp_path, value):
    from dmi.storage.native_capture import validate_capture_bounds

    with pytest.raises(ValueError, match="max_record_bytes"):
        validate_capture_bounds(_sink(tmp_path), value)


# --- the engine's entry point ----------------------------------------------------


def _engine(tmp_path, **sink_fields):
    from dmi.config import MonitoringConfig
    from dmi.engine import MonitoringEngine

    config = MonitoringConfig(
        storage_backend="persistent",
        capture_sink_config=_sink(tmp_path, **sink_fields),
        capture_storage_config=_storage(),
    )
    return MonitoringEngine(config=config, model_id="bounds",
                            enable_ring_transport=False)


def test_the_engine_checks_its_own_capture_configs(tmp_path):
    engine = _engine(tmp_path)
    engine.validate_capture_bounds(MiB)
    with pytest.raises(ConfigurationError, match="max_queue_bytes"):
        engine.validate_capture_bounds(17 * MiB)
    engine = _engine(tmp_path, max_pack_bytes=512 * MiB,
                     max_queue_bytes=64 * MiB)
    with pytest.raises(ConfigurationError,
                       match="uploader_max_in_flight_bytes"):
        engine.validate_capture_bounds(MiB)


def test_the_engine_has_nothing_to_check_without_a_sink_config():
    from dmi.engine import MonitoringEngine

    engine = MonitoringEngine(config=None, model_id="bounds",
                              enable_ring_transport=False)
    engine.validate_capture_bounds(1 << 40)
