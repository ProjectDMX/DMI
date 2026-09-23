"""The storage choice a user makes: ``in-memory``, ``persistent`` or ``none``.

``persistent`` is the native capture storage path (object store + catalog +
ClickHouse). ``in-memory`` delivers records in memory; until its consumer
interface exists it runs today's ``ClickHouseRecordSink`` path, which needs
a host engine. ``none`` turns capture off entirely: no ring is allocated, and
nothing that would capture can be attached.

The earlier names ``native`` and ``capture`` still work, with a
DeprecationWarning, and mean ``in-memory`` and ``persistent``. ``auto`` is the
unset default -- the inference every caller made before the field existed --
and is not one of the user's choices.
"""

from __future__ import annotations

import warnings

import pytest

from dmi.config import USER_STORAGE_CHOICES, MonitoringConfig, StorageBackend
from dmi.engine import MonitoringEngine

pytestmark = pytest.mark.cpu


def test_the_user_choices_are_in_memory_persistent_and_none():
    assert USER_STORAGE_CHOICES == ("in-memory", "persistent", "none")
    for choice in USER_STORAGE_CHOICES:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # none of them is deprecated
            assert MonitoringConfig(storage_backend=choice).storage_backend == choice


def test_auto_is_the_unset_default_not_a_user_choice():
    assert MonitoringConfig().storage_backend == "auto"
    assert "auto" not in USER_STORAGE_CHOICES


@pytest.mark.parametrize(
    "legacy, canonical", [("native", "in-memory"), ("capture", "persistent")])
def test_the_old_names_still_work_and_say_what_replaced_them(legacy, canonical):
    with pytest.warns(DeprecationWarning, match=f"'{canonical}'"):
        config = MonitoringConfig(storage_backend=legacy)
    # The config keeps what the caller wrote; the engine acts on the new name.
    assert config.storage_backend == legacy
    assert config.canonical_storage_backend == canonical


def test_an_unknown_choice_names_the_three_real_ones():
    with pytest.raises(ValueError) as refused:
        MonitoringConfig(storage_backend="object-store")
    message = str(refused.value)
    for choice in USER_STORAGE_CHOICES:
        assert repr(choice) in message


def test_every_accepted_value_is_in_the_type():
    for value in ("in-memory", "persistent", "none", "auto", "native", "capture"):
        assert value in StorageBackend.__args__


def test_persistent_takes_the_capture_sink_config(tmp_path):
    from dmi.storage.capture.native_sink import NativeSinkConfig

    sink = NativeSinkConfig(spool_root=str(tmp_path))
    config = MonitoringConfig(storage_backend="persistent", capture_sink_config=sink)
    assert config.capture_sink_config is sink
    with pytest.raises(ValueError, match="storage_backend='persistent'"):
        MonitoringConfig(storage_backend="in-memory", capture_sink_config=sink)


# --- none: capture off entirely ---------------------------------------------


def test_none_allocates_no_ring(monkeypatch):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("storage_backend='none' allocated a ring")

    monkeypatch.setattr(MonitoringEngine, "enable_ring_transport", _must_not_run)
    monkeypatch.setattr(MonitoringEngine, "_make_default_ring_config",
                        staticmethod(_must_not_run))

    engine = MonitoringEngine(config=MonitoringConfig(storage_backend="none"))

    assert engine._ring_transport is None
    assert engine.capture_enabled is False


def test_none_refuses_an_explicit_ring_config():
    with pytest.raises(ValueError, match="capture off"):
        MonitoringEngine(config=MonitoringConfig(storage_backend="none"),
                         ring_config=object())


def test_none_refuses_a_record_runtime():
    engine = MonitoringEngine(config=MonitoringConfig(storage_backend="none"))

    with pytest.raises(RuntimeError, match="storage_backend='none' turns capture off"):
        engine.create_record_runtime(object())


def test_none_refuses_a_host_engine():
    with pytest.raises(ValueError, match="'none'"):
        MonitoringEngine(config=MonitoringConfig(storage_backend="none"),
                         model_id="m", host_engine=object())


def test_an_adapter_refuses_to_attach_when_capture_is_off():
    from dmi.adapters.base import _refuse_unwired_capture_storage
    from dmi.configuration.errors import ConfigurationError

    engine = MonitoringEngine(config=MonitoringConfig(storage_backend="none"))

    with pytest.raises(ConfigurationError, match="capture is off"):
        _refuse_unwired_capture_storage(engine, "attach_model()", "SomeAdapter")


def test_an_adapter_refuses_persistent_by_its_new_name():
    from dmi.adapters.base import _refuse_unwired_capture_storage
    from dmi.configuration.errors import ConfigurationError

    engine = MonitoringEngine(enable_ring_transport=False)
    engine._storage_backend = "persistent"

    with pytest.raises(ConfigurationError, match="storage_backend='persistent'"):
        _refuse_unwired_capture_storage(engine, "attach_model()", "SomeAdapter")


def test_none_refuses_a_ring_enabled_after_construction():
    """The constructor skipping the ring is not enough: the public
    enable_ring_transport() would still allocate one and make it active."""
    engine = MonitoringEngine(config=MonitoringConfig(storage_backend="none"))

    with pytest.raises(RuntimeError, match="storage_backend='none' turns capture off"):
        engine.enable_ring_transport(object())
    assert engine._ring_transport is None


# --- the old names reach the engine, not only the config ---------------------


def test_the_engine_acts_on_native_as_in_memory():
    with pytest.warns(DeprecationWarning):
        config = MonitoringConfig(storage_backend="native")
    with pytest.raises(ValueError, match="needs a host engine"):
        MonitoringEngine(config=config, enable_ring_transport=False)


def test_the_engine_acts_on_capture_as_persistent(tmp_path):
    from dmi.storage.capture.native_sink import NativeSinkConfig

    with pytest.warns(DeprecationWarning):
        refused = MonitoringConfig(storage_backend="capture")
    with pytest.raises(ValueError, match="does not use the C\\+\\+ ClickHouse host"):
        MonitoringEngine(config=refused, model_id="m", host_engine=object(),
                         enable_ring_transport=False)

    with pytest.warns(DeprecationWarning):
        config = MonitoringConfig(
            storage_backend="capture",
            capture_sink_config=NativeSinkConfig(spool_root=str(tmp_path)))
    engine = MonitoringEngine(config=config, enable_ring_transport=False)
    assert engine._storage_backend == "persistent"


def test_a_duck_typed_config_with_an_old_name_is_still_checked():
    from types import SimpleNamespace

    config = SimpleNamespace(storage_backend="native", capture_sink_config=None,
                             capture_storage_config=None)
    with pytest.raises(ValueError, match="needs a host engine"):
        MonitoringEngine(config=config, enable_ring_transport=False)
