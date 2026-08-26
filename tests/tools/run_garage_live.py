#!/usr/bin/env python3
"""Run the manual Garage integration test against an ephemeral local server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ports(count: int) -> list[int]:
    listeners = [socket.socket() for _ in range(count)]
    try:
        for listener in listeners:
            listener.bind(("127.0.0.1", 0))
        return [listener.getsockname()[1] for listener in listeners]
    finally:
        for listener in listeners:
            listener.close()


def _log_tail(log_path, limit: int = 2000) -> str:
    try:
        return log_path.read_text(errors="replace")[-limit:]
    except OSError:
        return "(no Garage log)"


def _wait_for_port(
    port: int, process: subprocess.Popen, log_path, timeout: float = 20
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Garage exited with status {process.returncode}\n"
                f"{_log_tail(log_path)}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("Garage did not open its S3 port")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    binary = os.environ.get("DMI_GARAGE_BINARY") or shutil.which("garage")
    if binary is None:
        raise RuntimeError(
            "set DMI_GARAGE_BINARY or install Garage from "
            "https://garagehq.deuxfleurs.fr/download/"
        )
    with tempfile.TemporaryDirectory(prefix="dmi-garage-live-") as directory:
        root = Path(directory)
        version = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=True
        ).stdout
        expected_version = os.environ.get("DMI_GARAGE_VERSION", "2.3.0")
        if f"cargo:{expected_version}" not in version:
            raise RuntimeError(
                f"expected Garage {expected_version}, got {version.strip()}"
            )
        ports = _ports(4)
        s3_port, rpc_port, web_port, admin_port = ports
        config = root / "garage.toml"
        config.write_text(
            f'''metadata_dir = "{root / "meta"}"
data_dir = "{root / "data"}"
db_engine = "sqlite"
replication_factor = 1
rpc_bind_addr = "127.0.0.1:{rpc_port}"
rpc_public_addr = "127.0.0.1:{rpc_port}"
rpc_secret = "{secrets.token_hex(32)}"

[s3_api]
s3_region = "garage"
api_bind_addr = "127.0.0.1:{s3_port}"
root_domain = ".s3.garage.localhost"

[s3_web]
bind_addr = "127.0.0.1:{web_port}"
root_domain = ".web.garage.localhost"
index = "index.html"

[admin]
api_bind_addr = "127.0.0.1:{admin_port}"
admin_token = "{secrets.token_urlsafe(32)}"
metrics_token = "{secrets.token_urlsafe(32)}"
'''
        )
        access_key = "GK" + secrets.token_hex(16)
        secret_key = secrets.token_hex(32)
        environment = os.environ.copy()
        environment.update(
            {
                "GARAGE_CONFIG_FILE": str(config),
                "GARAGE_DEFAULT_ACCESS_KEY": access_key,
                "GARAGE_DEFAULT_SECRET_KEY": secret_key,
                "GARAGE_DEFAULT_BUCKET": "dmi-test",
                "DMI_S3_ENDPOINT": f"http://127.0.0.1:{s3_port}",
                "DMI_S3_BUCKET": "dmi-test",
                "DMI_S3_REGION": "garage",
                "DMI_S3_ACCESS_KEY_ID": access_key,
                "DMI_S3_SECRET_ACCESS_KEY": secret_key,
                "DMI_S3_ALLOW_HTTP": "1",
                "PYTHONPATH": str(REPO_ROOT / "src"),
            }
        )
        # Garage logs to stderr for the whole run. A PIPE nobody reads fills its
        # buffer and blocks the server mid-benchmark, so send it to a file that
        # has no such limit and can still be shown on failure.
        log_path = Path(root) / "garage.log"
        log_handle = log_path.open("w")
        process = subprocess.Popen(
            [binary, "server", "--single-node", "--default-bucket"],
            cwd=root,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
            text=True,
        )
        try:
            _wait_for_port(s3_port, process, log_path)
            command = (
                [
                    sys.executable,
                    "-m",
                    "benchmarks.bench_garage_upload",
                    *(
                        args.benchmark_args[1:]
                        if args.benchmark_args[:1] == ["--"]
                        else args.benchmark_args
                    ),
                ]
                if args.benchmark
                else [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_garage_live.py",
                    "-m",
                    "garage and manual",
                    "-q",
                ]
            )
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
            )
            return completed.returncode
        finally:
            log_handle.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
