"""An upload scan must not run crash cleanup against a live pack writer."""
import select
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_native_s3_client import fake_s3
from tests.test_native_uploader import DriverSession, STORE_DRIVER, _upload_pending

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(sys.platform != "linux", reason="uses GNU linker wrapping"),
    pytest.mark.skipif(not STORE_DRIVER.exists(), reason="native store driver not built"),
]


def test_upload_scan_preserves_an_inflight_stage(fake_s3, tmp_path):
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
    spool = tmp_path / "spool"
    process = subprocess.Popen(
        [str(executable), str(spool)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    uploader = DriverSession(STORE_DRIVER)
    try:
        assert select.select([process.stdout], [], [], 10)[0], "writer did not open its temp file"
        assert process.stdout.readline().strip() == "OPEN"
        result = _upload_pending(uploader, fake_s3, spool)
        output, error = process.communicate("continue\n", timeout=10)
        assert result["ok"], result
        assert process.returncode == 0, output + error
        assert len(list(spool.rglob("*.dmi-pack.ready"))) == 1
    finally:
        uploader.close()
        if process.poll() is None:
            process.kill()
            process.communicate()
