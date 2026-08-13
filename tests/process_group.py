"""Bounded subprocess execution with scoped descendant cleanup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import signal
import subprocess


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> tuple[str, str]:
    """Terminate only the new session led by ``process``."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def run_process_group(
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None,
    timeout: float,
    termination_grace: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    """Run a command in a new session and kill that session on timeout.

    A timeout from ``subprocess.run`` kills only its direct child. vLLM tests
    can also own engine and tensor-parallel descendants, so release gates need
    an isolated process group with scoped TERM/KILL cleanup.
    """

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if termination_grace <= 0:
        raise ValueError("termination_grace must be positive")

    argv = list(command)
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process_group(
            process,
            grace_seconds=termination_grace,
        )
        raise subprocess.TimeoutExpired(
            argv,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from None
    except BaseException:
        _terminate_process_group(process, grace_seconds=termination_grace)
        raise

    return subprocess.CompletedProcess(
        argv,
        process.returncode,
        stdout,
        stderr,
    )
