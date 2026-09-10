"""Missing native drivers must be per-test skips, not collection skips.

Module-level `pytest.skip(...)` makes the whole module report `tests=0`
in the JUnit file, hiding that any coverage exists. A `skipif` mark on
`pytestmark` collects every test and reports each as
`<testcase><skipped/></testcase>` — what a CI skip guard needs to name
the gap. Modules that import their built artifact at module scope
(test_native_rollback, test_native_adapter_torch) cannot convert and
keep the module-level skip.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

MODULES = [
    "tests/test_native_pack_conformance.py",
    "tests/test_native_pack_sink.py",
    "tests/test_native_s3_client.py",
    "tests/test_native_s3_sign.py",
    "tests/test_native_spool.py",
    "tests/test_native_uploader.py",
]

pytestmark = pytest.mark.cpu


def test_missing_drivers_yield_test_skips_not_collection_skips(tmp_path):
    build = REPO / "native" / "build"
    stash = tmp_path / "drivers"
    stash.mkdir()
    moved = []
    if build.exists():
        for child in build.iterdir():
            if child.is_file():
                shutil.move(child, stash / child.name)
                moved.append(child.name)
    try:
        junit = tmp_path / "skips.xml"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *MODULES,
             "-q", f"--junit-xml={junit}"],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
    finally:
        for name in moved:
            shutil.move(stash / name, build / name)

    root = ET.parse(junit).getroot()
    suites = root.findall("testsuite")
    assert suites, "JUnit parsed no testsuite at all"
    tests = sum(int(s.get("tests", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    assert tests > 0, f"collection skip: no tests collected: {proc.stdout[-2000:]}"
    assert "collection skipped" not in junit.read_text(), (
        "a module-level skip survived the skipif conversion"
    )
