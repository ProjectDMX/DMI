"""DMI-configurator: a local web UI for authoring DMI capture configurations.

Launch it with ``dmi ui MODEL_DESCRIPTOR``. The web dependencies are optional::

    pip install "DMI[ui]"
"""

from __future__ import annotations

__all__ = ["create_app", "serve"]


def __getattr__(name: str):
    # Deferred so importing dmi.ui does not require FastAPI to be installed.
    if name == "create_app":
        from .app import create_app

        return create_app
    if name == "serve":
        from .server import serve

        return serve
    raise AttributeError(name)
