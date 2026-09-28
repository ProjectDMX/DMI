"""B6: one owner per spool directory, across processes.

A spool's crash cleanup (Recover) deletes every ``.open`` file its own object
is not writing, so it is only safe while no other process writes there. The
C++ spool therefore takes an owner lock -- flock on ``<dir>/.owner.lock`` --
when it opens a directory with ``owner_lock="take"`` (the default), and a
second process that tries is refused, told who holds it. What that lock
must also refuse, and what it must leave alone:

* a directory nested under, or containing, an owned directory: Scan walks
  recursively, so the outer spool's cleanup would reach into the inner one;
* ``owner_lock="held_by_caller"`` with nothing holding the lock: that mode
  opens without a lock of its own, on the caller's word that one is held;
* ``<dir>/_refs/``, where the upload handoff's ref files will live: no scan
  may sweep, quarantine or list anything under it.

The drivers are separate processes, so these are real cross-process locks.
The in-process cases -- two Spool objects in one process, the node-local
check, the directory layout -- are pinned by tests/native/
test_spool_owner_lock.cpp (test_native_spool_owner_lock_unit.py).

The Python spool (dmi.storage.capture.spool) takes no lock; the C++ spool is
deliberately stricter, and that is not ported to the reference.

Build: make -C native build/conformance_spool build/conformance_sink
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD = REPO_ROOT / "native" / "build"
SPOOL_DRIVER = BUILD / "conformance_spool"
SINK_DRIVER = BUILD / "conformance_sink"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not (SPOOL_DRIVER.exists() and SINK_DRIVER.exists()),
        reason="native spool/sink drivers are not built; run "
        "`make -C native build/conformance_spool build/conformance_sink`",
    ),
]

READY_NAME = (
    "018f0000-0000-7000-8000-000000000001.1700000000000000000.1."
    + "0" * 64 + ".dmi-pack.ready"
)


class _Holder:
    """A conformance_sink process whose open PackSink holds a spool."""

    def __init__(self, root: Path, **fields):
        self.proc = subprocess.Popen(
            [str(SINK_DRIVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        self.opened = self.call(
            op="open", root=str(root), max_bytes=1 << 30,
            max_queue_records=16, max_queue_bytes=1 << 20,
            max_pack_bytes=1 << 20, max_pack_records=16,
            max_linger_ns=1_000_000_000, overload="drop_newest",
            admission_timeout=-1, **fields)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def call(self, **fields) -> dict:
        self.proc.stdin.write(json.dumps(fields) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        if self.opened.get("ok"):
            self.call(op="close", timeout=10)
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=30)


def _spool(**fields) -> dict:
    """One conformance_spool op in a fresh process (it opens per op)."""
    fields.setdefault("max_bytes", 1 << 30)
    proc = subprocess.run(
        [str(SPOOL_DRIVER)], input=json.dumps(fields) + "\n",
        capture_output=True, text=True, timeout=60)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, proc.stderr
    return json.loads(lines[0])


def test_a_second_process_is_refused_and_told_who_holds_the_spool(tmp_path):
    root = tmp_path / "spool"
    holder = _Holder(root)
    try:
        assert holder.opened["ok"], holder.opened
        response = _spool(op="recover", root=str(root))
        assert not response["ok"], response
        assert response["status"] == "open", response
        what = response["what"]
        assert f"pid {holder.pid}" in what, what
        assert socket.gethostname() in what, what
        assert str(root) in what, what
    finally:
        holder.close()
    # The lock goes with its holder: the next process opens the directory.
    assert _spool(op="recover", root=str(root))["ok"]


def test_the_lock_file_records_the_holder(tmp_path):
    root = tmp_path / "spool"
    holder = _Holder(root)
    try:
        assert holder.opened["ok"], holder.opened
        record = (root / ".owner.lock").read_text()
        assert record.split() == [socket.gethostname(), str(holder.pid)]
    finally:
        holder.close()


def test_a_directory_nested_under_an_owned_one_is_refused(tmp_path):
    outer = tmp_path / "spool"
    holder = _Holder(outer)
    try:
        assert holder.opened["ok"], holder.opened
        response = _spool(op="recover", root=str(outer / "inner"))
        assert not response["ok"], response
        assert "nested" in response["what"], response
        assert str(outer) in response["what"], response
    finally:
        holder.close()


def test_a_directory_containing_an_owned_one_is_refused(tmp_path):
    outer = tmp_path / "spool"
    holder = _Holder(outer / "a" / "inner")
    try:
        assert holder.opened["ok"], holder.opened
        response = _spool(op="recover", root=str(outer))
        assert not response["ok"], response
        assert "contains" in response["what"], response
        assert str(outer / "a" / "inner") in response["what"], response
    finally:
        holder.close()


def test_a_stale_lock_file_still_marks_an_owned_directory(tmp_path):
    """Nesting is judged by the lock FILE, not by a live holder: an outer
    directory some spool once owned is still a spool directory, and the
    next process to open it would sweep the inner one."""
    outer = tmp_path / "spool"
    assert _spool(op="recover", root=str(outer))["ok"]
    assert (outer / ".owner.lock").exists()
    response = _spool(op="recover", root=str(outer / "inner"))
    assert not response["ok"], response
    assert "nested" in response["what"], response


def test_held_by_caller_opens_beside_the_holder(tmp_path):
    root = tmp_path / "spool"
    holder = _Holder(root)
    try:
        assert holder.opened["ok"], holder.opened
        response = _spool(op="recover", root=str(root),
                          owner_lock="held_by_caller")
        assert response["ok"], response
    finally:
        holder.close()


def test_held_by_caller_is_refused_when_nothing_holds_the_lock(tmp_path):
    root = tmp_path / "spool"
    response = _spool(op="recover", root=str(root),
                      owner_lock="held_by_caller")
    assert not response["ok"], response
    assert "held_by_caller" in response["what"], response
    # Once some spool has taken and let go of the lock, the file exists and
    # is unlocked: still refused.
    assert _spool(op="recover", root=str(root))["ok"]
    response = _spool(op="recover", root=str(root),
                      owner_lock="held_by_caller")
    assert not response["ok"], response


def test_an_unknown_owner_lock_mode_is_refused(tmp_path):
    response = _spool(op="recover", root=str(tmp_path / "spool"),
                      owner_lock="share")
    assert not response["ok"], response
    assert "owner_lock" in response["what"], response


def test_recovery_leaves_the_refs_directory_alone(tmp_path):
    """<spool>/_refs/ will hold the upload handoff's ref files (E2a). A
    sweep that deleted, quarantined or listed them would break it."""
    root = tmp_path / "spool"
    refs = root / "_refs"
    refs.mkdir(parents=True)
    stale = refs / ".018f0000-0000-7000-8000-000000000001.abcd1234.open"
    stale.write_bytes(b"in progress")
    bogus = refs / READY_NAME  # the wrong checksum: would be quarantined
    bogus.write_bytes(b"not a pack")
    ref = refs / "018f0000-0000-7000-8000-000000000001.uploaded"
    ref.write_text("{}")

    response = _spool(op="recover", root=str(root))

    assert response["ok"], response
    assert response["staged"] == []
    assert response["snapshot"] == {"entries": 0, "bytes": 0}
    assert stale.exists() and bogus.exists() and ref.exists()
    assert sorted(p.name for p in refs.iterdir()) == sorted(
        [stale.name, bogus.name, ref.name])


def test_the_open_accounting_skips_the_refs_directory(tmp_path):
    root = tmp_path / "spool"
    refs = root / "_refs"
    refs.mkdir(parents=True)
    (refs / READY_NAME).write_bytes(b"x" * 100)
    (refs / ".018f0000-0000-7000-8000-000000000001.abcd1234.open"
     ).write_bytes(b"y" * 50)
    assert _spool(op="snapshot", root=str(root))["snapshot"]["bytes"] == 0
