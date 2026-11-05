"""Host-only tests of the ring producer ops' launch counter (no CUDA toolkit
or GPU).

native/csrc/ring/ring_torch_op.cpp is compiled unchanged against CPU-only
Torch and tests/native/ring/cpu_cuda_stub/, with the RingEnginePy double in
tests/native/ring/test_ring_torch_op_host.cpp, into a shared library.  A fresh
interpreter loads it (the ops register the ``ring`` Torch library, which must
not collide with a native extension loaded by another test) and:

* calls every producer op and reads the real ``ring_producer_launch_count``;
* drives the real ``dmi.node_toggle`` capture lifecycle with an engine whose
  producer *is* the compiled op, so whether a capture counts as carrying
  producers comes from the op's own counter, not from a test double.

Device-side behavior (the uploaded exec, node state on the GPU) is covered by
the GPU tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RING_DIR = ROOT / "native" / "csrc" / "ring"
TESTS_DIR = ROOT / "tests" / "native" / "ring"

NEW_OPS = ("gather", "head", "fused", "fused_rows", "fused_heads")
ALL_OPS = ("static", "prefix", "chunked") + NEW_OPS

_DRIVER = r"""
import ctypes, json, runpy, sys
import pytest
import torch

lib = ctypes.CDLL(sys.argv[1])
for name in ("dmx_host_launch_count", "dmx_host_launcher_calls",
             "dmx_host_anomalies", "dmx_host_node_count", "dmx_host_node_key"):
    getattr(lib, name).restype = ctypes.c_uint64
lib.dmx_host_node_count.argtypes = [ctypes.c_uint64]
lib.dmx_host_node_key.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
lib.dmx_host_op.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
lib.dmx_host_last_error.restype = ctypes.c_char_p
ops = json.loads(sys.argv[3])
out = {"ops": {}, "captured": {}, "excluded": {}, "late": {}}


def run(op, graph=0):
    c0, l0 = lib.dmx_host_launch_count(), lib.dmx_host_launcher_calls()
    rc = lib.dmx_host_op(op.encode(), graph)
    return {"rc": rc, "counter_delta": lib.dmx_host_launch_count() - c0,
            "launcher_calls": lib.dmx_host_launcher_calls() - l0,
            "error": lib.dmx_host_last_error().decode().splitlines()[:1]}


for op in ops + ["fused_empty", "gather_host_tensor", "gather_bad_table"]:
    out["ops"][op] = run(op)

# Under stream capture with node recording on.
lib.dmx_host_set_toggle_capture(1)
for i, op in enumerate(ops):
    lib.dmx_host_clear_nodes()
    graph = 0x7000 + i
    res = run(op, graph)
    res["keys"] = [divmod(lib.dmx_host_node_key(graph, j), 1 << 32)
                   for j in range(lib.dmx_host_node_count(graph))]
    res["anomalies"] = lib.dmx_host_anomalies()
    out["captured"][op] = res
lib.dmx_host_set_toggle_capture(0)
lib.dmx_host_clear_nodes()

# The real node_toggle lifecycle, the compiled op as the producer.  The ring
# ops are this library's; a locally built native extension must not register
# them a second time.
import dmi.transport.native as native


def _no_extension():
    raise ImportError("the host harness provides the ring ops")


native._load_extension = _no_extension
fx = runpy.run_path(sys.argv[2])
from dmi import node_toggle as nt
from dmi.transport import ring


class NativeOpEngine(fx["LifecycleEngine"]):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def produce(self, layer=0, *, counted=True):
        lib.dmx_host_set_toggle_capture(int(bool(self.capture)))
        raw = self.capturing.raw if self.capturing is not None else 0
        try:
            res = run(self.op, raw)
        finally:
            lib.dmx_host_set_toggle_capture(0)
        assert res["rc"] == 0, res

    def producer_launch_count(self):
        return lib.dmx_host_launch_count()

    def toggle_node_count(self):
        return lib.dmx_host_node_count(0)

    def toggle_node_count_for_graph(self, raw):
        return lib.dmx_host_node_count(raw)

    def clear_toggle_registry(self):
        super().clear_toggle_registry()
        lib.dmx_host_clear_nodes()

    def is_graph_ready(self, raw):
        return lib.dmx_host_node_count(raw) > 0 and raw in self.bound


def capture(engine):
    graph = torch.cuda.CUDAGraph()
    graph.capture_begin()
    engine.capturing = graph
    engine.produce()
    engine.capturing = None
    graph.capture_end()
    return graph


for op in ops:
    # Excluded capture in device-toggle mode: no node is recorded, so only
    # the launch counter can tell that it holds a producer that needs the
    # upload before its first replay.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "CUDAGraph", fx["LifecycleGraph"])
        fx["_reset_patch_state"](mp, nt)
        mp.setattr(ring, "_active_transport", None)
        engine = NativeOpEngine(op)
        nt.begin_capture_window(engine, exclude_capture=lambda: True,
                                device_toggle=True)
        graph = capture(engine)
        trace_at_capture_end = list(graph.trace)
        uploads_before_replay = list(engine.uploads)
        nt.end_capture_window(engine)
        nt.set_replay_guard(True)
        mp.setattr(ring, "_active_transport",
                   fx["GatedTransport"](engine, requires_registry=False))
        graph.replay()
        out["excluded"][op] = {
            "producers": graph._dmi_producers,
            "trace_at_capture_end": trace_at_capture_end,
            "uploaded_before_replay": uploads_before_replay == [
                (graph.raw, graph.exec_ptr)],
            "replays": graph.replays,
        }
        nt._WINDOW.engine = None

    # A capture after the window closed: no node, no bind.  The guard must
    # refuse its replay.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "CUDAGraph", fx["LifecycleGraph"])
        fx["_reset_patch_state"](mp, nt)
        mp.setattr(ring, "_active_transport", None)
        engine = NativeOpEngine("static")
        nt.begin_capture_window(engine)
        monitored = capture(engine)
        nt.bind_graph(engine, monitored)
        nt.end_capture_window(engine)
        nt.set_replay_guard(True)
        engine.op = op
        late = capture(engine)
        mp.setattr(ring, "_active_transport", fx["GatedTransport"](engine))
        error = None
        try:
            late.replay()
        except RuntimeError as exc:
            error = str(exc)
        monitored.replay()
        out["late"][op] = {"producers": late._dmi_producers,
                           "replays": late.replays, "error": error,
                           "monitored_replays": monitored.replays}
        nt._WINDOW.engine = None

out["cuda_initialized"] = torch.cuda.is_initialized()
print("RESULT " + json.dumps(out))
"""


def _torch_root() -> Path:
    import torch

    return Path(torch.__file__).resolve().parent


@pytest.fixture(scope="module")
def op_results(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++20 compiler is required")
    import torch

    torch_root = _torch_root()
    build_dir = tmp_path_factory.mktemp("ring_torch_op_host")
    library = build_dir / "libring_torch_op_host.so"
    build = subprocess.run(
        [compiler, "-std=c++20", "-O0", "-fPIC", "-shared",
         "-Wl,--no-undefined",
         f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
         # The stub directory comes first: it shadows the CUDA runtime and
         # <ATen/cuda/CUDAContext.h>, so no CUDA toolkit header is used.
         f"-I{TESTS_DIR / 'cpu_cuda_stub'}", f"-I{RING_DIR}",
         f"-I{RING_DIR.parent}", f"-I{torch_root / 'include'}",
         f"-I{torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
         str(RING_DIR / "ring_torch_op.cpp"),
         str(TESTS_DIR / "test_ring_torch_op_host.cpp"),
         f"-L{torch_root / 'lib'}", f"-Wl,-rpath,{torch_root / 'lib'}",
         "-ltorch_cpu", "-lc10", "-o", str(library)],
        capture_output=True, text=True, check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    run = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(library),
         str(ROOT / "tests" / "test_node_toggle.py"), json.dumps(list(ALL_OPS))],
        capture_output=True, text=True, check=False, timeout=300, env=env,
        cwd=build_dir,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    line = [ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, run.stdout + run.stderr
    results = json.loads(line[-1][len("RESULT "):])
    assert results["cuda_initialized"] is False
    return results


@pytest.mark.cpu
@pytest.mark.parametrize("op", ALL_OPS)
def test_every_producer_op_counts_its_launch_once(op_results, op):
    res = op_results["ops"][op]
    assert res["rc"] == 0, res
    assert res["launcher_calls"] == 1
    assert res["counter_delta"] == 1


@pytest.mark.cpu
@pytest.mark.parametrize("op", ["fused_empty", "gather_host_tensor"])
def test_an_op_that_launches_nothing_counts_nothing(op_results, op):
    res = op_results["ops"][op]
    assert res["rc"] == 0
    assert (res["launcher_calls"], res["counter_delta"]) == (0, 0)


@pytest.mark.cpu
def test_a_refused_op_counts_nothing(op_results):
    res = op_results["ops"]["gather_bad_table"]
    assert res["rc"] == 1 and "row table" in res["error"][0]
    assert (res["launcher_calls"], res["counter_delta"]) == (0, 0)


@pytest.mark.cpu
@pytest.mark.parametrize("op,keys,anomalies", [
    ("static", [[3, 0]], 0),
    ("prefix", [[3, 0]], 0),
    ("chunked", [], 1),                  # not toggle-managed: fails closed
    ("gather", [[3, 0]], 0),
    ("head", [[24, 0]], 0),
    ("fused", [[3, 0], [24, 0]], 0),     # one node, both keys
    ("fused_rows", [[3, 0]], 0),
    ("fused_heads", [[24, 0]], 0),
])
def test_a_captured_op_counts_one_launch_and_records_its_keys(
        op_results, op, keys, anomalies):
    res = op_results["captured"][op]
    assert res["rc"] == 0 and res["counter_delta"] == 1
    assert res["keys"] == keys
    assert res["anomalies"] == anomalies


@pytest.mark.cpu
@pytest.mark.parametrize("op", ALL_OPS)
def test_an_excluded_device_toggle_capture_is_uploaded_before_it_replays(
        op_results, op):
    res = op_results["excluded"][op]
    assert res["producers"] is True
    assert res["trace_at_capture_end"] == [
        "capture_begin", "capture_end", "instantiate"]
    assert res["uploaded_before_replay"] is True
    assert res["replays"] == 1


@pytest.mark.cpu
@pytest.mark.parametrize("op", ALL_OPS)
def test_a_late_unbound_capture_fails_closed(op_results, op):
    res = op_results["late"][op]
    assert res["producers"] is True
    assert res["replays"] == 0
    assert "not registered and bound" in res["error"]
    assert res["monitored_replays"] == 1
