#!/usr/bin/env python3
"""Build and smoke-test DMI as an installed wheel.

Pytest adds ``src`` to ``sys.path``, which is useful for development but can
hide broken package discovery after a layout change.  This check builds a real
wheel, validates its archive layout, installs it into a temporary virtual
environment, and imports it from outside the repository.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import venv
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_MEMBERS = {
    "dmi/__init__.py",
    "dmi/adapters/base.py",
    "dmi/adapters/huggingface/adapter.py",
    "dmi/api/v1/__init__.py",
    "dmi/hooks/dispatch.py",
    "dmi/hooks/point.py",
    "dmi/storage/internals.py",
    "dmi/transport/native.py",
    "dmi/transport/ring.py",
}

FORBIDDEN_PREFIXES = (
    "monitoring/",
    "integration/",
    "benchmark/",
    "example/",
    "Figures/",
)


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _validate_archive(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())

    missing = sorted(REQUIRED_MEMBERS - members)
    if missing:
        raise RuntimeError(
            "wheel is missing canonical package files: " + ", ".join(missing)
        )

    legacy = sorted(
        member
        for member in members
        if member.startswith(FORBIDDEN_PREFIXES)
    )
    if legacy:
        raise RuntimeError(
            "wheel contains legacy top-level paths: " + ", ".join(legacy)
        )


def _smoke_test_install(wheel: Path, audit_root: Path) -> None:
    venv_root = audit_root / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_root)

    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    python = venv_root / scripts_dir / ("python.exe" if os.name == "nt" else "python")
    _run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)])

    smoke_code = """
import importlib
import importlib.util
from pathlib import Path
import sys

import dmi

package_file = Path(dmi.__file__).resolve()
venv_root = Path(sys.argv[1]).resolve()
assert package_file.is_relative_to(venv_root), package_file
assert dmi.MonitoringEngine.__module__ == "dmi.engine"

for module_name in (
    "dmi.adapters.base",
    "dmi.adapters.huggingface.adapter",
    "dmi.api.v1",
    "dmi.hooks.dispatch",
    "dmi.storage.internals",
    "dmi.transport.native",
):
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    assert module_file.is_relative_to(venv_root), module_file

for legacy_name in ("monitoring", "integration", "benchmark", "example"):
    spec = importlib.util.find_spec(legacy_name)
    if spec is None:
        continue
    locations = [spec.origin, *(spec.submodule_search_locations or ())]
    assert not any(
        location and Path(location).resolve().is_relative_to(venv_root)
        for location in locations
    ), (legacy_name, locations)

print(f"installed package smoke test passed: {package_file}")
"""
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    _run(
        [
            str(python),
            "-I",
            "-c",
            smoke_code,
            str(venv_root),
        ],
        cwd=audit_root,
        env=clean_env,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dmi-package-check-") as directory:
        audit_root = Path(directory)
        wheel_dir = audit_root / "wheel"
        wheel_dir.mkdir()

        build_command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
        ]
        if importlib.util.find_spec("wheel") is not None:
            build_command.append("--no-build-isolation")
        build_command.extend(
            ["--wheel-dir", str(wheel_dir), str(REPO_ROOT)]
        )
        _run(build_command)

        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {len(wheels)}")

        _validate_archive(wheels[0])
        _smoke_test_install(wheels[0], audit_root)

    print("package distribution check passed")


if __name__ == "__main__":
    main()
