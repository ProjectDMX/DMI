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


# A NativeSinkConfig pickled before the move and the admission fields: the
# old class at the old path (whose name now re-exports the new class), with
# spool_root="/spool/old" and max_queue_bytes=32 MiB, protocol 2. Its state
# is the eight fields the old class had.
_OLD_FORMAT_PICKLE = (
    b"\x80\x02cdmi.storage.capture.native_sink\nNativeSinkConfig\nq\x00)"
    b"\x81q\x01]q\x02(X\n\x00\x00\x00/spool/oldq\x03\x8a\x06\x00\x00\x00"
    b"\x00\x00\x01K\x01M\x00\x01J\x00\x00\x00\x02J\x00\x00\x00\x08M\x10'J"
    b"\x00\xca\x9a;eb.")


def test_an_old_format_pickle_loads_with_the_new_defaults():
    import pickle

    from dmi.storage.native_capture import NativeSinkConfig

    config = pickle.loads(_OLD_FORMAT_PICKLE)
    assert type(config) is NativeSinkConfig
    assert config == NativeSinkConfig(spool_root="/spool/old",
                                      max_queue_bytes=32 * MiB)
    assert (config.overload, config.admission_timeout_s) == ("block", 2.0)


def test_a_new_pickle_round_trips(tmp_path):
    import pickle

    from dmi.storage.native_capture import NativeSinkConfig

    config = NativeSinkConfig(spool_root=str(tmp_path),
                              overload="drop_newest")
    assert pickle.loads(pickle.dumps(config)) == config


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


# --- the admission bound a record ring reads ---------------------------------


@pytest.mark.parametrize("fields, bound", [
    ({}, 0.0),  # the C++ default, drop_newest: refused at once
    ({"overload": "block", "admission_timeout_s": 0.25}, 0.25),
    ({"overload": "block"}, None),  # waits without bound
])
def test_the_sink_reports_its_admission_bound(native_sink_module, tmp_path,
                                              fields, bound):
    """A record ring with a stall budget waits for one in-flight submit
    after the budget is spent; it reads this bound to refuse a sink whose
    admission has none."""
    sink = native_sink_module.NativePackSink(
        spool_root=str(tmp_path), layout=LAYOUT, **fields)
    assert sink.admission_bound_s == bound


# --- a record lost after admission -------------------------------------------


def _one_row_envelope(index: int, nbytes: int):
    import torch

    mapping = _metadata(index)
    mapping["shape"] = [nbytes // 4]
    rows = [{"metadata_json": json.dumps(mapping), "offset": 0,
             "length": nbytes, "dtype": 6, "shape": [nbytes // 4]}]
    return rows, torch.zeros(nbytes // 4, dtype=torch.float32).view(
        torch.uint8)


def test_a_record_that_fits_no_pack_fails_the_flush(native_sink_module,
                                                     tmp_path):
    """max_pack_bytes = one 1 MiB record: admission screens the payload
    alone, so the record is admitted, and the pack worker then drops it as
    oversized (it fits no empty pack with its framing). The reference
    adapter counts oversized_records as a loss and raises at flush and
    rethrow; the native sink used to report success with nothing
    persisted."""
    from dmi.storage.capture.native_sink import create_native_pack_sink
    from dmi.storage.native_capture import NativeSinkConfig

    sink = create_native_pack_sink(NativeSinkConfig(
        spool_root=str(tmp_path), max_pack_bytes=MiB,
        max_queue_bytes=4 * MiB)).native_sink
    lease = sink.attach()
    rows, payload = _one_row_envelope(0, MiB)
    sink.submit_envelope(LAYOUT, rows, payload)
    with pytest.raises(RuntimeError, match="oversized_records"):
        sink.flush_and_wait(30.0)
    with pytest.raises(RuntimeError, match="oversized_records"):
        sink.rethrow_if_failed()
    snapshot = sink.snapshot()
    assert (snapshot["admitted_records"], snapshot["persisted_records"],
            snapshot["oversized_records"]) == (1, 0, 1), snapshot
    # The next envelope is refused too: capture stops and says why, on the
    # record worker, instead of losing records until someone flushes.
    rows, payload = _one_row_envelope(1, 1024)
    with pytest.raises(RuntimeError, match="oversized_records"):
        sink.submit_envelope(LAYOUT, rows, payload)
    del lease


def test_a_stored_record_still_flushes_clean(native_sink_module, tmp_path):
    from dmi.storage.capture.native_sink import create_native_pack_sink
    from dmi.storage.native_capture import NativeSinkConfig

    sink = create_native_pack_sink(
        NativeSinkConfig(spool_root=str(tmp_path))).native_sink
    lease = sink.attach()
    rows, payload = _one_row_envelope(0, 1024)
    sink.submit_envelope(LAYOUT, rows, payload)
    assert sink.flush_and_wait(30.0)
    sink.rethrow_if_failed()
    assert sink.snapshot()["persisted_records"] == 1
    del lease
