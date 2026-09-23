"""``dmi ui`` argument handling, descriptor discovery, and port selection.

None of this starts a server. The pieces that decide *what* to serve and
*where* are separated from ``serve()`` precisely so they can be tested without
binding a socket or importing uvicorn.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

from dmi.cli import _resolve_ui_model, build_parser
from dmi.configuration.errors import ConfigurationError
from dmi.ui.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    find_descriptors,
    port_is_free,
    resolve_port,
)

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "model_descriptors" / "llama3-8b.yaml"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_model_is_optional():
    args = build_parser().parse_args(["ui"])

    assert args.model is None


def test_model_is_taken_positionally_when_given():
    args = build_parser().parse_args(["ui", "./some-model"])

    assert args.model == "./some-model"


def test_port_defaults_to_none_so_it_can_fall_back():
    """An explicit port must be distinguishable from no port at all."""
    args = build_parser().parse_args(["ui"])

    assert args.port is None


def test_explicit_port_is_preserved():
    args = build_parser().parse_args(["ui", "--port", "9001"])

    assert args.port == 9001


def test_browser_opens_by_default():
    args = build_parser().parse_args(["ui"])

    assert args.no_browser is False


def test_no_browser_can_be_disabled():
    args = build_parser().parse_args(["ui", "--no-browser"])

    assert args.no_browser is True


# ---------------------------------------------------------------------------
# Descriptor discovery
# ---------------------------------------------------------------------------


def test_no_descriptors_found_in_an_empty_directory(tmp_path):
    assert find_descriptors(tmp_path) == []


def test_descriptor_is_found_by_its_suffix(tmp_path):
    target = tmp_path / "qwen3.model.yaml"
    target.write_text("schema_version: 1\n")

    assert find_descriptors(tmp_path) == [target]


@pytest.mark.parametrize(
    "name",
    ["a.model.yaml", "a.model.yml", "a.dmi-model.yaml"],
)
def test_every_supported_descriptor_suffix_is_discovered(tmp_path, name):
    (tmp_path / name).write_text("schema_version: 1\n")

    assert len(find_descriptors(tmp_path)) == 1


def test_unrelated_yaml_is_not_treated_as_a_descriptor(tmp_path):
    (tmp_path / "config.yaml").write_text("nope: true\n")
    (tmp_path / "attention-debug.dmi.yaml").write_text("version: 1\n")

    assert find_descriptors(tmp_path) == []


def test_discovery_is_sorted_and_deduplicated(tmp_path):
    for name in ["b.model.yaml", "a.model.yaml"]:
        (tmp_path / name).write_text("schema_version: 1\n")

    found = find_descriptors(tmp_path)

    assert [path.name for path in found] == ["a.model.yaml", "b.model.yaml"]


def test_a_directory_is_not_mistaken_for_a_descriptor(tmp_path):
    (tmp_path / "weird.model.yaml").mkdir()

    assert find_descriptors(tmp_path) == []


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def test_an_explicit_model_is_returned_unchanged():
    assert _resolve_ui_model("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"


def test_a_lone_descriptor_is_used_without_being_named(tmp_path, monkeypatch):
    target = tmp_path / "solo.model.yaml"
    target.write_text("schema_version: 1\n")
    monkeypatch.chdir(tmp_path)

    assert _resolve_ui_model(None) == str(Path("solo.model.yaml"))


def test_no_descriptor_explains_how_to_make_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError) as excinfo:
        _resolve_ui_model(None)

    message = str(excinfo.value)
    assert "No model given" in message
    assert "describe-model" in message


def test_several_descriptors_ask_rather_than_guess(tmp_path, monkeypatch):
    for name in ["one.model.yaml", "two.model.yaml"]:
        (tmp_path / name).write_text("schema_version: 1\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError) as excinfo:
        _resolve_ui_model(None)

    message = str(excinfo.value)
    assert "2 descriptors" in message
    # Both candidates offered as runnable commands.
    assert "one.model.yaml" in message
    assert "two.model.yaml" in message


# ---------------------------------------------------------------------------
# Port selection
# ---------------------------------------------------------------------------


def _fake_availability(monkeypatch, busy: set[int]):
    """Drive resolve_port's branching without binding a real socket.

    Binding is the wrong dependency for a unit test: a sandboxed or otherwise
    restricted CI environment refuses it, and the logic worth pinning here is
    which port gets chosen, not whether the kernel hands out sockets.
    """
    monkeypatch.setattr(
        "dmi.ui.server.port_is_free",
        lambda host, port: port not in busy,
    )


def test_an_explicit_port_is_never_moved(monkeypatch):
    """A named port that is busy must fail loudly, not serve elsewhere."""
    _fake_availability(monkeypatch, busy={9001})

    assert resolve_port(DEFAULT_HOST, 9001) == 9001


def test_an_explicit_port_is_not_probed_at_all(monkeypatch):
    """Nothing should touch the network when the caller named a port."""
    calls = []

    def tripwire(host, port):
        calls.append(port)
        return True

    monkeypatch.setattr("dmi.ui.server.port_is_free", tripwire)

    assert resolve_port(DEFAULT_HOST, 9001) == 9001
    assert calls == []


def test_automatic_port_prefers_the_default_when_it_is_free(monkeypatch):
    _fake_availability(monkeypatch, busy=set())

    assert resolve_port(DEFAULT_HOST, None) == DEFAULT_PORT


def test_automatic_port_skips_the_default_when_it_is_taken(monkeypatch):
    _fake_availability(monkeypatch, busy={DEFAULT_PORT})

    assert resolve_port(DEFAULT_HOST, None) == DEFAULT_PORT + 1


def test_automatic_port_walks_past_a_run_of_busy_ports(monkeypatch):
    _fake_availability(
        monkeypatch, busy=set(range(DEFAULT_PORT, DEFAULT_PORT + 5))
    )

    assert resolve_port(DEFAULT_HOST, None) == DEFAULT_PORT + 5


def test_automatic_port_falls_back_to_the_default_when_none_are_free(monkeypatch):
    """Exhausting the search hands the real failure to uvicorn to report."""
    _fake_availability(
        monkeypatch, busy=set(range(DEFAULT_PORT, DEFAULT_PORT + 10_000))
    )

    assert resolve_port(DEFAULT_HOST, None) == DEFAULT_PORT


def test_port_is_free_reports_a_boolean(monkeypatch):
    """The probe answers a question; it must not leak an OSError upward."""

    class RefusingSocket:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            raise OSError("address in use")

    monkeypatch.setattr(socket, "socket", lambda *a, **k: RefusingSocket())

    assert port_is_free(DEFAULT_HOST, 8000) is False


def test_the_documented_default_port_is_unchanged():
    """Docs and the launch config name 8000; keep them honest."""
    assert DEFAULT_PORT == 8000


# ---------------------------------------------------------------------------
# Browser opening
# ---------------------------------------------------------------------------


def _patch_connect(monkeypatch, results):
    """Fake connect_ex, returning each queued result in turn (0 == connected)."""
    queue = list(results)

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def settimeout(self, _seconds):
            pass

        def connect_ex(self, _address):
            return queue.pop(0) if queue else 111

    monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSocket())


def test_browser_opens_once_the_port_accepts(monkeypatch):
    from dmi.ui import server

    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    _patch_connect(monkeypatch, [0])

    server._open_when_ready("http://127.0.0.1:8000", DEFAULT_HOST, 8000, timeout=2.0)
    _join_opener()

    assert opened == ["http://127.0.0.1:8000"]


def test_browser_waits_for_the_server_before_opening(monkeypatch):
    """A slow first import must not open a browser on a dead port."""
    from dmi.ui import server

    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    # Refused twice, then accepted.
    _patch_connect(monkeypatch, [111, 111, 0])

    server._open_when_ready("http://127.0.0.1:8000", DEFAULT_HOST, 8000, timeout=5.0)
    _join_opener()

    assert opened == ["http://127.0.0.1:8000"]


def test_browser_gives_up_instead_of_hanging(monkeypatch):
    from dmi.ui import server

    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    _patch_connect(monkeypatch, [])  # always refused

    server._open_when_ready("http://127.0.0.1:8000", DEFAULT_HOST, 8000, timeout=0.3)
    _join_opener()

    assert opened == []


def test_the_opener_thread_does_not_outlive_the_process():
    from dmi.ui import server

    import threading as _threading

    before = {t.name for t in _threading.enumerate()}
    server._open_when_ready("http://127.0.0.1:1", DEFAULT_HOST, 1, timeout=0.2)
    opener = [t for t in _threading.enumerate() if t.name == "dmi-ui-open"]

    assert opener, "opener thread was not started"
    assert opener[0].daemon is True, "a non-daemon thread would block exit"
    _join_opener()
    assert "dmi-ui-open" not in before


def _join_opener(timeout: float = 5.0) -> None:
    import threading as _threading

    for thread in _threading.enumerate():
        if thread.name == "dmi-ui-open":
            thread.join(timeout)
