"""CPU contracts for bounded process-tree cleanup in release tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.process_group import run_process_group


pytestmark = pytest.mark.cpu


def _is_running(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    # A zombie has exited and owns no GPU or other runtime resources.
    try:
        return stat.read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def test_process_group_returns_completed_output(tmp_path):
    result = run_process_group(
        [sys.executable, "-c", "print('complete')"],
        cwd=tmp_path,
        env=os.environ,
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout == "complete\n"
    assert result.stderr == ""


def test_process_group_timeout_kills_term_ignoring_descendant(tmp_path):
    script = """
import subprocess
import sys
import time

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
print(child.pid, flush=True)
time.sleep(30)
"""
    with pytest.raises(subprocess.TimeoutExpired) as captured:
        run_process_group(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ,
            timeout=0.2,
            termination_grace=0.2,
        )

    descendant_pid = int(captured.value.output.strip())
    deadline = time.monotonic() + 3
    while _is_running(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _is_running(descendant_pid)


@pytest.mark.parametrize("field", ["timeout", "termination_grace"])
def test_process_group_rejects_non_positive_bounds(tmp_path, field):
    kwargs = {"timeout": 1.0, "termination_grace": 1.0}
    kwargs[field] = 0

    with pytest.raises(ValueError, match=field):
        run_process_group(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=os.environ,
            **kwargs,
        )
