"""Producer row tables are read only inside their physical storage.

A row table is ``[n, src0, len0, ...]`` int64.  The kernels re-read ``n``
every replay, so the host never reads it (no per-launch sync, nothing baked
into a graph); it passes the table's element count instead, and the device
loader clamps the segment count by both ``GATHER_MAX_SEGS`` and that
capacity.  A one-word table whose count word is 1 used to read ``table[1]``
and ``table[2]`` past its allocation.

The clamp itself (``native/csrc/ring/row_table.h``) is plain C++ shared by
host and device code, so it is compiled and run here on the CPU; the static
checks pin the wiring from the torch ops to the shared loader.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[1]
RING = ROOT / "native" / "csrc" / "ring"


def test_segment_count_is_bounded_by_table_storage(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    source = ROOT / "tests" / "native" / "test_row_table_clamp.cpp"
    executable = tmp_path / "test_row_table_clamp"
    built = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT / 'native' / 'csrc'}", str(source), "-o", str(executable)],
        capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True,
                         check=False, timeout=60)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert ran.stdout.strip() == "ok"


def test_python_and_native_segment_limits_agree():
    from dmi.hook_rows import GATHER_MAX_SEGS

    header = (RING / "row_table.h").read_text()
    match = re.search(r"GATHER_MAX_SEGS\s*=\s*(\d+)", header)
    assert match and int(match.group(1)) == GATHER_MAX_SEGS


def test_every_table_reader_receives_the_table_capacity():
    producer = (RING / "producer.cu").read_text()
    loader = producer[producer.index("load_row_table(const int64_t* table"):]
    loader = loader[:loader.index("__syncthreads")]
    # The count comes from the shared clamp, bounded by table_numel.
    assert "row_table_segment_count(table[0], table_numel)" in loader
    assert "GATHER_MAX_SEGS) n_raw" not in loader
    # Each kernel passes its own table's capacity to the loader.
    assert producer.count("load_row_table(seg_table_dev_ptr, table_numel,") == 2
    assert "load_row_table(gtab, gtab_numel," in producer
    assert "load_row_table(htab, htab_numel," in producer

    op = (RING / "ring_torch_op.cpp").read_text()
    for impl, needles in (
        ("ring_producer_gather_impl", ["seg_table.numel()"]),
        ("ring_producer_head_impl", ["seg_table.numel()"]),
        ("ring_producer_fused_impl", ["gather_table->numel()",
                                      "head_table->numel()"]),
    ):
        body = op[op.index(f"void {impl}("):]
        body = body[:body.index("\n}\n")]
        for needle in needles:
            assert needle in body, (impl, needle)
        # The count word is never read on the host (no sync, nothing baked
        # into a captured graph).
        assert ".item" not in body and "[0]" not in body, impl
        assert "check_row_table(" in body, impl
    # A table must at least hold its count word.
    assert "table.numel() >= 1" in op
