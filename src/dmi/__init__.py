"""DMI: decoupled monitoring infrastructure for LLM inference.

The top-level package deliberately avoids loading the CUDA extension. Native
resources are imported lazily when an engine or transport is constructed.
"""

from .config import CaptureSchedule, MonitoringConfig
from .engine import HostEngineConfig, MonitoringEngine, RingCapacities

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
        from .transport import native

        return getattr(native, name)
    raise AttributeError(name)


__all__ = [
    "MonitoringEngine",
    "HostEngineConfig",
    "RingCapacities",
    "CaptureSchedule",
    "MonitoringConfig",
    *_NATIVE_EXPORTS,
]
