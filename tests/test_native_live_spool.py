"""A live pack writer owns its spool: no other process may sweep it.

Crash cleanup (Recover) deletes every ``.open`` file its Spool is not
writing itself, so running it against a live writer in another process
deletes the writer's in-flight temp file and fails its stage with "cannot
link ready file". The spool's owner lock makes that impossible rather than
merely avoided: while the writer holds its directory, a second process's
open is refused, naming the writer. Once the writer dies -- even by
SIGKILL, mid-stage -- the lock goes with it, and the next process recovers
the directory: the stale temp file is swept and the sealed pack uploads.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.test_native_s3_client import STATE, fake_s3
from tests.test_native_uploader import DriverSession, STORE_DRIVER, _upload_pending

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(sys.platform != "linux", reason="uses GNU linker wrapping"),
    pytest.mark.skipif(not STORE_DRIVER.exists(), reason="native store driver not built"),
]

SPOOL_DRIVER = STORE_DRIVER.parent / "conformance_spool"


@pytest.fixture
def writer_binary(tmp_path) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[1]
    csrc = root / "native" / "csrc"
    executable = tmp_path / "live_spool_stage"
    subprocess.run(
        [compiler, "-std=c++17", "-pthread", f"-I{csrc}",
         str(root / "tests/native/live_spool_stage.cpp"),
         str(csrc / "store/spool.cpp"), "-lcrypto", "-Wl,--wrap=open",
         "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    return executable


def _start_writer(binary: Path, spool: Path, packs: int = 1):
    """A writer paused with its last pack's temp file open."""
    process = subprocess.Popen(
        [str(binary), str(spool), str(packs)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # Blocking reads under a watchdog: select() on the pipe cannot see a line
    # the text wrapper has already buffered behind an earlier one.
    watchdog = threading.Timer(20, process.kill)
    watchdog.start()
    seen = []
    try:
        for line in process.stdout:
            if line.strip() == "OPEN":
                return process
            seen.append(line.strip())  # an earlier pack's stage status
    finally:
        watchdog.cancel()
    raise AssertionError(
        f"writer never opened its temp file: {seen} {process.stderr.read()}")


def _recover(spool: Path) -> dict:
    import json

    proc = subprocess.run(
        [str(SPOOL_DRIVER)],
        input=json.dumps({"op": "recover", "root": str(spool),
                          "max_bytes": 1 << 30}) + "\n",
        capture_output=True, text=True, timeout=30)
    return json.loads(proc.stdout.strip())


def test_an_upload_scan_is_refused_while_a_writer_holds_the_spool(
        fake_s3, tmp_path, writer_binary):
    spool = tmp_path / "spool"
    process = _start_writer(writer_binary, spool)
    uploader = DriverSession(STORE_DRIVER)
    try:
        result = _upload_pending(uploader, fake_s3, spool)
        # Refused before it touched anything, naming the holder.
        assert not result["ok"], result
        assert f"pid {process.pid}" in result["what"], result
        assert socket.gethostname() in result["what"], result
        assert len(list(spool.rglob("*.open"))) == 1

        output, error = process.communicate("continue\n", timeout=10)
        assert process.returncode == 0, output + error
        assert len(list(spool.rglob("*.dmi-pack.ready"))) == 1
        assert list(spool.rglob("*.open")) == []

        # The writer is gone, and its lock with it.
        result = _upload_pending(uploader, fake_s3, spool)
        assert result["ok"], result
        assert len(result["refs"]) == 1, result
        assert list(spool.rglob("*.dmi-pack.ready")) == []
    finally:
        uploader.close()
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_a_sigkilled_writers_spool_is_recovered_by_the_next_process(
        fake_s3, tmp_path, writer_binary):
    spool = tmp_path / "spool"
    # The first pack is sealed; the second is mid-stage when the writer dies.
    process = _start_writer(writer_binary, spool, packs=2)
    assert len(list(spool.rglob("*.dmi-pack.ready"))) == 1
    (stale,) = spool.rglob("*.open")
    os.kill(process.pid, signal.SIGKILL)
    process.communicate(timeout=10)
    assert process.returncode == -signal.SIGKILL

    recovered = _recover(spool)
    assert recovered["ok"], recovered
    assert len(recovered["staged"]) == 1, recovered
    assert not stale.exists()

    uploader = DriverSession(STORE_DRIVER)
    try:
        result = _upload_pending(uploader, fake_s3, spool)
    finally:
        uploader.close()
    assert result["ok"], result
    assert [ref["pack_id"] for ref in result["refs"]] == [
        recovered["staged"][0]["pack_id"]]
    assert list(spool.rglob("*.dmi-pack.ready")) == []
    with STATE.lock:
        assert result["refs"][0]["object_key"] in STATE.objects
