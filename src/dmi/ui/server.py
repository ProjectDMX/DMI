"""Uvicorn entry point for DMI-configurator."""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

from .app import create_app

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# How many ports past the default to try before giving up. Only used when the
# caller did not name a port: an explicit --port that is busy should fail
# loudly rather than quietly serving somewhere else.
PORT_SEARCH_RANGE = 20

# Descriptor filename patterns, in the order they are offered.
DESCRIPTOR_GLOBS = ("*.model.yaml", "*.model.yml", "*.dmi-model.yaml")


def find_descriptors(directory: str | Path = ".") -> list[Path]:
    """Return descriptor files in ``directory``, sorted, without duplicates.

    Used when no model is named on the command line. Deliberately shallow and
    pattern-based: guessing which of several models someone meant is worse
    than listing them.
    """
    base = Path(directory)
    found: list[Path] = []
    for pattern in DESCRIPTOR_GLOBS:
        for path in sorted(base.glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def resolve_port(host: str, port: Optional[int]) -> int:
    """Pick a port to bind.

    An explicit ``port`` is returned unchanged -- if it is busy, uvicorn should
    say so. ``None`` means "the usual one, or the next free one after it", so
    that a second configurator does not fail with an address-in-use trace.
    """
    if port is not None:
        return port
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SEARCH_RANGE):
        if port_is_free(host, candidate):
            return candidate
    return DEFAULT_PORT  # let uvicorn report the real failure


def _open_when_ready(url: str, host: str, port: int, timeout: float = 20.0) -> None:
    """Open a browser once the server accepts connections.

    Polls rather than sleeping a fixed interval, so a slow first import does
    not open a browser on a page that is not being served yet.
    """

    def wait_and_open() -> None:
        step = 0.1
        # Wall-clock deadline: each iteration spends real time in both the
        # connect probe and the sleep, so counting iterations would stretch
        # the stated timeout to roughly double.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(step)
                if probe.connect_ex((host, port)) == 0:
                    webbrowser.open(url)
                    return
            time.sleep(step)

    thread = threading.Thread(target=wait_and_open, name="dmi-ui-open", daemon=True)
    thread.start()


def serve(
    source: str | Path,
    config_path: Optional[str | Path] = None,
    host: str = DEFAULT_HOST,
    port: Optional[int] = None,
    open_browser: bool = True,
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

    app = create_app(source, config_path, bind_host=host)
    ui = app.state.ui
    bound_port = resolve_port(host, port)
    url = f"http://{host}:{bound_port}"

    topology = ui.descriptor.topology
    lines = [
        f"DMI-configurator  ·  {ui.descriptor.model.name}",
        f"  model      : {topology.num_layers} layers, "
        f"hidden {topology.hidden_size}"
        + (f", {topology.num_experts} experts" if topology.is_moe else ""),
        f"  read from  : {ui.source}",
        f"  saves to   : {ui.save_path}",
        f"  serving    : {url}",
    ]
    if port is None and bound_port != DEFAULT_PORT:
        lines.append(f"  (port {DEFAULT_PORT} was busy)")
    lines.append("  Ctrl-C to stop.")
    # Flushed explicitly: this banner carries the URL, and a piped or
    # make-wrapped stdout would otherwise hold it until the server exits.
    print("\n".join(lines), flush=True)

    if open_browser:
        _open_when_ready(url, host, bound_port)

    uvicorn.run(app, host=host, port=bound_port, log_level="warning")


__all__ = [
    "serve",
    "find_descriptors",
    "port_is_free",
    "resolve_port",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DESCRIPTOR_GLOBS",
]
