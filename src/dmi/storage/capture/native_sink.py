"""Native capture sink selection: the production writer for capture packs.

The reference path (:class:`CapturePackReferenceSink` in
:mod:`dmi.storage.capture.record_adapter`) crosses the Python GIL once per
envelope and exists to exercise the format end to end. This module selects
the native writer instead: envelopes travel the ring's record worker
straight into pack assembly with no Python on the capture path.

Selection is explicit at the call site — pass ``handle.native_sink`` to
``MonitoringEngine.create_record_runtime(record_sink=...)`` exactly as with
the reference sink. Rolling back is the same call with the other sink; packs
staged by either writer are readable by the same Python reader, which the
rollback test pins.

The extension (``native/build/_dmi_native_sink*.so``) loads lazily so that
importing :mod:`dmi.storage.capture` never requires torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LAYOUT_NAME = "capture_pack_reference_v1"


@dataclass(frozen=True, slots=True)
class NativeSinkConfig:
    """Pack-sink bounds for the native writer. Field-by-field the same
    contract as the pipeline config the reference sink takes."""

    spool_root: str
    spool_max_bytes: int = 1 << 40
    num_workers: int = 1
    max_queue_records: int = 256
    max_queue_bytes: int = 16 * 1024 * 1024
    max_pack_bytes: int = 128 * 1024 * 1024
    max_pack_records: int = 10_000
    max_linger_ns: int = 1_000_000_000

    def __post_init__(self) -> None:
        if not self.spool_root:
            raise ValueError("spool_root is required")
        for name in (
            "spool_max_bytes",
            "num_workers",
            "max_queue_records",
            "max_queue_bytes",
            "max_pack_bytes",
            "max_pack_records",
            "max_linger_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")


def _load_native_sink_extension() -> Any:
    from ...transport import native as _native_transport

    try:
        return _native_transport._load_named_extension("_dmi_native_sink")
    except ImportError as exc:
        raise ImportError(
            "The native capture sink is unavailable. Build it with "
            "`make -C native build/_dmi_native_sink "
            "PYTHON=<venv>/bin/python`."
        ) from exc


class NativePackSinkHandle:
    """Owns a native pack sink; mirrors CapturePackReferenceSink's surface
    (``record_format`` + ``native_sink``) so call sites switch by factory."""

    def __init__(self, config: NativeSinkConfig) -> None:
        module = _load_native_sink_extension()
        self._native_sink = module.NativePackSink(
            spool_root=config.spool_root,
            layout=LAYOUT_NAME,
            num_workers=config.num_workers,
            max_queue_records=config.max_queue_records,
            max_queue_bytes=config.max_queue_bytes,
            max_pack_bytes=config.max_pack_bytes,
            max_pack_records=config.max_pack_records,
            max_linger_ns=config.max_linger_ns,
        )
        # Engine ownership is taken by create_record_runtime; holding no
        # lease here keeps the handle closable without an engine.
        self._record_format_layout = LAYOUT_NAME

    @property
    def record_format_layout(self) -> str:
        """The wire layout this sink consumes (shared with the reference)."""

        return self._record_format_layout

    @property
    def native_sink(self) -> Any:
        """Native ``RecordSink`` passed explicitly to ``create_record_runtime``."""

        return self._native_sink


def create_native_pack_sink(config: NativeSinkConfig) -> NativePackSinkHandle:
    """Select the native capture writer for one record runtime."""

    if not isinstance(config, NativeSinkConfig):
        raise TypeError("config must be a NativeSinkConfig")
    return NativePackSinkHandle(config)


__all__ = [
    "LAYOUT_NAME",
    "NativePackSinkHandle",
    "NativeSinkConfig",
    "create_native_pack_sink",
]
