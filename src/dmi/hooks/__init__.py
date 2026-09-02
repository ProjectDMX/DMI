"""Hook definitions, dispatch, selection policy, and model hook points.

Re-exports resolve lazily (PEP 562): :mod:`dmi.hooks.catalog` is deliberately
torch-free so descriptor and validation paths can import it, and an eager
``from .dispatch import ...`` here would pull torch into every one of them.
"""

from typing import TYPE_CHECKING

_EXPORTS = {
    "HookPoint": "point",
    "HookedRootModule": "point",
    "set_monitoring_debug": "point",
    "HookRowBasis": "specs",
    "HookSpec": "specs",
    "ModelShapeConfig": "specs",
    "compute_hook_shape": "specs",
    "install_ring_hooks": "dispatch",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    from .dispatch import install_ring_hooks
    from .point import HookPoint, HookedRootModule, set_monitoring_debug
    from .specs import (
        HookRowBasis,
        HookSpec,
        ModelShapeConfig,
        compute_hook_shape,
    )


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value
