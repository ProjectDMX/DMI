"""Compatibility namespace for DMI 1.x.

New code should import from :mod:`dmi`. Existing ``monitoring`` imports remain
supported and resolve to the same implementation objects.
"""

from dmi import CaptureSchedule, HostEngineConfig, MonitoringConfig, MonitoringEngine

_NATIVE_EXPORTS = (
    "StageConfig",
    "DMXHostEngine",
    "ClickHouseClientConfig",
    "QueueConfig",
    "EnqueuePolicy",
    "OnFullPolicy",
    "OnClosedPolicy",
)

def __getattr__(name: str):
    if name in _NATIVE_EXPORTS:
        from dmi.transport import native

        return getattr(native, name)
    raise AttributeError(name)


__all__ = [
    "MonitoringEngine",
    "HostEngineConfig",
    "CaptureSchedule",
    "MonitoringConfig",
    *_NATIVE_EXPORTS,
]
