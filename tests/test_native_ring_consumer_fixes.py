"""Legacy-ring consumer handoff: CPU-only native coverage.

* tests/native/test_ring_meta_fifo_ownership.cpp -- TensorMetaFifo owns its
  pending StepContexts: clearing or destroying it releases their capture
  leases exactly once (standard library only).
* tests/native/test_p2p_batch_handoff.cpp -- the real p2p_thread.cpp with a
  batch-only consumer delivers HF batched token/attention captures as
  one-request items; queues charge the pooled allocation a payload keeps
  alive; p2p_pool_cap_bytes == 0 allocates exact-size buffers; the delivery
  wait honours a deadline.  Needs ATen only: the CUDA types come from the
  CPU stand-in tests/native/ring/cpu_cuda_stub/ (no CUDA API is called and
  the drain is never started), used even where a CUDA toolkit is installed
  so every machine builds what the CUDA-free CI runner builds.
* tests/native/test_buffer_pool_failure.cpp -- BufferPool::acquire with each
  of its allocations failed in turn: std::bad_alloc propagates and the block
  returns to the pool exactly once.
* tests/native/test_batch_queue_shared_eviction.cpp -- BatchQueue eviction
  with views of a shared allocation frees only what the remaining references
  release, and evicts nothing unless the arrival is then admitted.
* The delivery barrier's finite timeout covers producer-stream completion,
  the flush and the delivery wait with one deadline (source check: the
  CUDA path itself needs a GPU).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "native" / "csrc"
CPU_CUDA_STUB = ROOT / "tests" / "native" / "ring" / "cpu_cuda_stub"


def _compiler():
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    return compiler


def _run(executable: Path) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([str(executable)], capture_output=True, text=True,
                            check=False, env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


@pytest.mark.cpu
def test_meta_fifo_owns_pending_step_contexts(tmp_path):
    executable = tmp_path / "test_ring_meta_fifo_ownership"
    result = subprocess.run(
        [_compiler(), "-std=c++17", "-O0", "-g", "-pthread", f"-I{CSRC}",
         str(ROOT / "tests" / "native" / "test_ring_meta_fifo_ownership.cpp"),
         "-o", str(executable)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    _run(executable)


def _build_with_aten(tmp_path: Path, name: str, *sources: Path) -> Path:
    """Compile ``sources`` against CPU-only ATen and the CPU CUDA stub (never
    a CUDA toolkit's headers), C++20 as the Torch headers require."""
    torch = pytest.importorskip("torch")
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib = torch_root / "lib"
    abi = int(torch.compiled_with_cxx11_abi())
    executable = tmp_path / name
    result = subprocess.run(
        [_compiler(), "-std=c++20", "-O0", "-w", "-pthread",
         f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
         f"-I{CSRC}", f"-I{CPU_CUDA_STUB}",
         f"-I{torch_root / 'include'}",
         f"-I{torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
         *[str(src) for src in sources],
         "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
         # RPATH, not RUNPATH: the interpreter's torch libraries win over any
         # other torch on LD_LIBRARY_PATH.
         f"-L{torch_lib}", f"-Wl,-rpath,{torch_lib}", "-Wl,--disable-new-dtags",
         "-ltorch_cpu", "-lc10", "-o", str(executable)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return executable


@pytest.mark.cpu
def test_p2p_batch_handoff_and_pooled_retention(tmp_path):
    _run(_build_with_aten(
        tmp_path, "test_p2p_batch_handoff",
        ROOT / "tests" / "native" / "test_p2p_batch_handoff.cpp",
        CSRC / "ring" / "p2p_thread.cpp"))


@pytest.mark.cpu
def test_buffer_pool_acquire_failure_releases_the_block_once(tmp_path):
    _run(_build_with_aten(
        tmp_path, "test_buffer_pool_failure",
        ROOT / "tests" / "native" / "test_buffer_pool_failure.cpp"))


@pytest.mark.cpu
def test_batch_queue_eviction_with_shared_allocations(tmp_path):
    _run(_build_with_aten(
        tmp_path, "test_batch_queue_shared_eviction",
        ROOT / "tests" / "native" / "test_batch_queue_shared_eviction.cpp"))


@pytest.mark.cpu
def test_cpu_cuda_stub_stands_in_for_the_toolkit():
    """The host-only builds above must not depend on a CUDA toolkit: none of
    their sources may reach a CUDA header the stub does not provide."""
    provided = {p.relative_to(CPU_CUDA_STUB).as_posix()
                for p in CPU_CUDA_STUB.rglob("*.h")}
    assert "cuda_runtime.h" in provided
    seen, pending = set(), [
        ROOT / "tests" / "native" / "test_p2p_batch_handoff.cpp",
        ROOT / "tests" / "native" / "test_buffer_pool_failure.cpp",
        ROOT / "tests" / "native" / "test_batch_queue_shared_eviction.cpp",
        CSRC / "ring" / "p2p_thread.cpp",
    ]
    include = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in include.findall(path.read_text()):
            if name.startswith(("cuda", "cub/", "thrust/", "ATen/cuda/",
                                "c10/cuda/")) or name.endswith(".cuh"):
                assert name in provided, f"{path.name} includes {name}"
            for base in (path.parent, CSRC):
                if (base / name).is_file():
                    pending.append(base / name)
                    break


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[brace:i + 1]
    raise AssertionError(f"unterminated body for {signature}")


@pytest.mark.cpu
def test_delivery_barrier_timeout_bounds_every_phase():
    py = (CSRC / "ring" / "ring_engine_py.cu").read_text()
    body = _function_body(py, "bool RingEnginePy::drain_to_sink_and_wait(")
    finite = body[body.index("const auto deadline"):]
    # A finite timeout never synchronizes the stream unboundedly: it polls
    # an event against the deadline, then hands the same deadline on.
    assert "cudaStreamSynchronize" not in finite
    assert "wait_stream_until(ms, deadline)" in finite
    assert "drain_to_sink_and_wait_until(deadline)" in finite
    poll = _function_body(py, "bool wait_stream_until(")
    assert "cudaEventDisableTiming" in poll and "cudaEventQuery" in poll
    assert "cudaEventDestroy" in poll and "deadline" in poll

    engine = (CSRC / "ring" / "ring_engine.cu").read_text()
    until = _function_body(engine, "bool RingEngine::drain_to_sink_and_wait_until(")
    bounded = until[until.index("force_flush_and_wait_until"):]
    assert re.search(r"if \(!drain_->force_flush_and_wait_until\(\*deadline\)\)\s*return false;",
                     until)
    assert "wait_until_processed_until(drain_->tasks_enqueued(), *deadline)" in bounded
    assert "force_flush_and_wait()" not in bounded
