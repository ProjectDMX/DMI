"""The native sink's admission policy, from NativeSinkConfig to the C++ sink.

A ring-fed sink admits on the record worker, so its policy decides what a
burst does: under drop_newest a burst larger than the queue loses records
(and latches the record ring), under block the worker waits for room, up to
``admission_timeout_s``. The C++ ``SinkConfig`` keeps drop_newest with no
timeout, the oracle's default; ``NativeSinkConfig``, the ring-fed entry
point, defaults to block with a 2 s timeout.

``NativeSinkConfig`` lives in :mod:`dmi.storage.native_capture`, live code,
so the public config no longer loads the backup capture package; the old
import path re-exports it.

Build: make -C native cpu-goals PYTHON=<venv>/bin/python
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"

pytestmark = pytest.mark.cpu

LAYOUT = "capture_pack_reference_v1"
MiB = 1 << 20


@pytest.fixture(scope="module")
def native_sink_module():
    # Not a skip: a CPU host builds this module (`make -C native cpu-goals`),
    # and an unbuilt one is missing coverage, not missing hardware.
    if not sorted(BUILD.glob("_dmi_native_sink*.so")):
        pytest.fail("native/build/_dmi_native_sink*.so is not built; run "
                    "`make -C native cpu-goals PYTHON=<venv>/bin/python`")
    from dmi.storage.capture.native_sink import _load_native_sink_extension

    return _load_native_sink_extension()


# --- NativeSinkConfig --------------------------------------------------------


def test_ring_fed_default_is_block_with_a_two_second_timeout(tmp_path):
    from dmi.storage.native_capture import NativeSinkConfig

    config = NativeSinkConfig(spool_root=str(tmp_path))
    assert config.overload == "block"
    assert config.admission_timeout_s == 2.0


@pytest.mark.parametrize("fields, error, match", [
    ({"overload": "drop_oldest"}, ValueError, "overload"),
    ({"overload": None}, ValueError, "overload"),
    ({"admission_timeout_s": 0}, ValueError, "admission_timeout_s"),
    ({"admission_timeout_s": -1.0}, ValueError, "admission_timeout_s"),
    ({"admission_timeout_s": math.nan}, ValueError, "admission_timeout_s"),
    ({"admission_timeout_s": math.inf}, ValueError, "admission_timeout_s"),
    ({"admission_timeout_s": "2"}, TypeError, "admission_timeout_s"),
    ({"admission_timeout_s": True}, TypeError, "admission_timeout_s"),
])
def test_admission_fields_are_validated(tmp_path, fields, error, match):
    from dmi.storage.native_capture import NativeSinkConfig

    with pytest.raises(error, match=match):
        NativeSinkConfig(spool_root=str(tmp_path), **fields)


def test_block_may_wait_without_bound_when_asked(tmp_path):
    from dmi.storage.native_capture import NativeSinkConfig

    config = NativeSinkConfig(spool_root=str(tmp_path),
                              admission_timeout_s=None)
    assert config.admission_timeout_s is None


def test_the_old_import_path_re_exports_the_same_type():
    from dmi.storage.capture import native_sink
    from dmi.storage import native_capture

    assert native_sink.NativeSinkConfig is native_capture.NativeSinkConfig


def test_the_config_type_does_not_load_the_backup_capture_package():
    # A fresh interpreter: this process has long since imported everything.
    probe = (
        "import sys\n"
        "from dmi.storage.native_capture import NativeSinkConfig\n"
        "from dmi.config import MonitoringConfig\n"
        "MonitoringConfig(storage_backend='persistent',\n"
        "                 capture_sink_config=NativeSinkConfig(spool_root='/x'))\n"
        "print(sorted(m for m in sys.modules\n"
        "             if m.startswith('dmi.storage.capture')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        check=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": ""})
    assert result.stdout.strip() == "[]", result.stdout


# --- the native binding ------------------------------------------------------


def test_the_binding_keeps_the_cxx_default(native_sink_module, tmp_path):
    sink = native_sink_module.NativePackSink(
        spool_root=str(tmp_path), layout=LAYOUT)
    assert sink.overload == "drop_newest"
    assert sink.admission_timeout_s is None


def test_the_binding_accepts_block_with_a_timeout(native_sink_module,
                                                  tmp_path):
    sink = native_sink_module.NativePackSink(
        spool_root=str(tmp_path), layout=LAYOUT, overload="block",
        admission_timeout_s=0.25)
    assert sink.overload == "block"
    assert sink.admission_timeout_s == 0.25


@pytest.mark.parametrize("fields", [
    {"overload": "drop_oldest"},
    {"overload": "block", "admission_timeout_s": -0.5},
    {"overload": "block", "admission_timeout_s": math.nan},
])
def test_the_binding_refuses_a_bad_policy(native_sink_module, tmp_path,
                                          fields):
    with pytest.raises(ValueError):
        native_sink_module.NativePackSink(
            spool_root=str(tmp_path), layout=LAYOUT, **fields)


def test_the_handle_passes_the_policy_to_the_native_sink(native_sink_module,
                                                         tmp_path):
    from dmi.storage.capture.native_sink import create_native_pack_sink
    from dmi.storage.native_capture import NativeSinkConfig

    default = create_native_pack_sink(
        NativeSinkConfig(spool_root=str(tmp_path / "a"))).native_sink
    assert (default.overload, default.admission_timeout_s) == ("block", 2.0)
    dropping = create_native_pack_sink(NativeSinkConfig(
        spool_root=str(tmp_path / "b"), overload="drop_newest")).native_sink
    assert dropping.overload == "drop_newest"


def _metadata(index: int) -> dict:
    from dmi.storage.capture import CaptureMetadata

    return CaptureMetadata(
        capture_id=f"burst-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v",
        hook_name="resid_post", layer_number=0, producer_rank=0,
        step_number=index, token_start=index, token_end=index + 1,
        batch_position=0, dtype="float32", shape=(MiB // 4,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    ).to_mapping()


def test_a_burst_four_times_the_queue_is_admitted_under_the_default(
        native_sink_module, tmp_path):
    """64 x 1 MiB rows in one envelope, against the default 16 MiB queue:
    the record worker waits for room instead of dropping."""
    import torch

    from dmi.storage.capture.native_sink import create_native_pack_sink
    from dmi.storage.native_capture import NativeSinkConfig

    sink = create_native_pack_sink(
        NativeSinkConfig(spool_root=str(tmp_path))).native_sink
    lease = sink.attach()
    rows = [{
        "metadata_json": json.dumps(_metadata(index)),
        "offset": index * MiB, "length": MiB, "dtype": 6,  # at::kFloat
        "shape": [MiB // 4],
    } for index in range(64)]
    payload = torch.arange(64 * MiB // 4, dtype=torch.float32).view(
        torch.uint8)
    sink.submit_envelope(LAYOUT, rows, payload)
    assert sink.flush_and_wait(60.0)
    sink.rethrow_if_failed()
    snapshot = sink.snapshot()
    assert snapshot["persisted_records"] == 64, snapshot
    assert snapshot["dropped_records"] == 0, snapshot
    assert snapshot["timed_out_records"] == 0, snapshot
    assert snapshot["rejected_closed_records"] == 0, snapshot
    del lease
