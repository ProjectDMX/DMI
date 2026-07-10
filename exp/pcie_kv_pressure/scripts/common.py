#!/usr/bin/env python3
"""Shared helpers for PCIe KV pressure experiment scripts."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def now_ns() -> int:
    return time.time_ns()


def mono_ns() -> int:
    return time.monotonic_ns()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            if n == 0:
                raise OSError(f"zero-byte write while appending {path}")
            view = view[n:]
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def git_commit(repo: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def create_manifest(
    run_dir: str | Path,
    *,
    run_id: str,
    mode: str,
    config_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "config_path": str(config_path) if config_path else None,
        "run_start_wall_ns": now_ns(),
        "run_start_mono_ns": mono_ns(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "git_commit": git_commit(repo),
    }
    if extra:
        manifest.update(extra)
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def load_manifest(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "run_manifest.json"
    return read_json(path)


def event_base(manifest: dict[str, Any], *, source: str, event_type: str) -> dict[str, Any]:
    wall = now_ns()
    mono = mono_ns()
    return {
        "run_id": manifest["run_id"],
        "source": source,
        "event_type": event_type,
        "ts_ns": wall,
        "t_rel_ms": (mono - int(manifest["run_start_mono_ns"])) / 1e6,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(data),
        "mean": mean(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": max(data),
    }
