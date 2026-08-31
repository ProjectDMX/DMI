"""Uvicorn entry point for DMI-configurator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def serve(
    source: str | Path,
    config_path: Optional[str | Path] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Run the configurator until interrupted.

    Binds to loopback by default. The server reads one descriptor and writes at
    most one configuration file, both named by the caller.
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised by install state
        raise RuntimeError(
            "DMI-configurator needs the optional UI dependencies. Install them "
            'with:\n    pip install "DMI[ui]"'
        ) from exc

    app = create_app(source, config_path)
    ui = app.state.ui
    print(f"DMI-configurator  ·  {ui.descriptor.model.name}")
    print(f"  model from : {ui.source}")
    print(f"  saves to   : {ui.save_path}")
    print(f"  serving    : http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["serve", "DEFAULT_HOST", "DEFAULT_PORT"]
