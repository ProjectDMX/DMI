"""The native pack sink stages its open pack when a ring releases it.

A RingEngine stops by draining its record worker into the sink and then
releasing the sink's lease -- without flushing it. The sink seals a pack
only when it fills, when it has lingered ``max_linger_ns``, or on a flush,
so the records of the pack still open at that moment stayed in memory: the
audit's probe counted 0 ready packs right after the ring stopped (1 once the
linger fired), and a process that exited before then lost them. Nothing
drains a spool a pack never reached, whatever runs after the sink.

``on_engine_release`` now flushes the sink, bounded and without throwing,
so the release leaves the open pack in the spool as one ``.ready`` file.

Build: make -C native cpu-goals PYTHON=<venv>/bin/python
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "native" / "build"

pytestmark = pytest.mark.cpu

LAYOUT = "capture_pack_reference_v1"
MiB = 1 << 20
MINUTE_NS = 60_000_000_000


@pytest.fixture(scope="module")
def native_sink_module():
    if not sorted(BUILD.glob("_dmi_native_sink*.so")):
        pytest.fail("native/build/_dmi_native_sink*.so is not built; run "
                    "`make -C native cpu-goals PYTHON=<venv>/bin/python`")
    from dmi.storage.capture.native_sink import _load_native_sink_extension

    return _load_native_sink_extension()


def _sink(root: Path, **fields):
    from dmi.storage.capture.native_sink import create_native_pack_sink
    from dmi.storage.native_capture import NativeSinkConfig

    return create_native_pack_sink(
        NativeSinkConfig(spool_root=str(root), **fields)).native_sink


def _envelope(index: int, nbytes: int = 1024):
    import torch
    from dmi.storage.capture import CaptureMetadata

    mapping = CaptureMetadata(
        capture_id=f"release-{index:04d}", tenant_id="t", experiment_id="e",
        run_id="r", session_id="s", request_id=f"q{index}",
        sequence_id=f"n{index}", model_id="m", model_revision="mr",
        adapter_revision=None, capture_policy_version="v",
        hook_name="resid_post", layer_number=0, producer_rank=0,
        step_number=index, token_start=index, token_end=index + 1,
        batch_position=0, dtype="float32", shape=(nbytes // 4,),
        captured_at_ns=1_700_000_000_000_000_000 + index,
    ).to_mapping()
    rows = [{"metadata_json": json.dumps(mapping), "offset": 0,
             "length": nbytes, "dtype": 6, "shape": [nbytes // 4]}]
    return rows, torch.full((nbytes // 4,), float(index)).view(torch.uint8)


def _ready(root: Path) -> list[Path]:
    return sorted(root.rglob("*.dmi-pack.ready"))


def _wait_until_admitted(sink, records: int) -> None:
    deadline = time.monotonic() + 10.0
    while sink.snapshot()["admitted_records"] < records:
        assert time.monotonic() < deadline, sink.snapshot()
        time.sleep(0.01)


def test_the_open_pack_is_staged_when_the_engine_releases_the_sink(
        native_sink_module, tmp_path):
    """A 60 s linger: nothing but a flush can seal the open pack before
    the release, so the release must be what stages it."""
    sink = _sink(tmp_path, max_linger_ns=MINUTE_NS)
    lease = sink.attach()
    for index in range(3):
        sink.submit_envelope(LAYOUT, *_envelope(index))
    _wait_until_admitted(sink, 3)
    time.sleep(0.2)
    assert _ready(tmp_path) == []  # still in memory, as the audit found

    del lease  # what RingEngine::stop does once its record worker is joined

    assert len(_ready(tmp_path)) == 1
    snapshot = sink.snapshot()
    assert snapshot["persisted_records"] == 3, snapshot
    assert snapshot["packs_persisted"] == 1, snapshot
    # The sink object is still alive: its destructor, which also seals the
    # open pack, has not run and so is not what staged it.
    assert sink.layout == LAYOUT


def test_a_release_with_nothing_open_stages_nothing(native_sink_module,
                                                    tmp_path):
    sink = _sink(tmp_path, max_linger_ns=MINUTE_NS)
    lease = sink.attach()
    sink.submit_envelope(LAYOUT, *_envelope(0))
    assert sink.flush_and_wait(30.0)
    assert len(_ready(tmp_path)) == 1

    del lease

    assert len(_ready(tmp_path)) == 1
    assert sink.snapshot()["packs_persisted"] == 1


def _binding_sink(module, root: Path, **fields):
    """The binding's own NativePackSink, for the knobs NativeSinkConfig
    does not carry (release_flush_timeout_s)."""
    fields.setdefault("max_linger_ns", MINUTE_NS)
    return module.NativePackSink(str(root), LAYOUT, **fields)


RELEASE_LINE = ("NativePackSink: the open pack did not reach the spool when "
                "the engine released the sink: ")


def test_a_sink_that_cannot_stage_is_released_and_says_why(
        native_sink_module, tmp_path, capfd):
    """The release runs inside RingEngine::stop and its destructor, which
    cannot throw: a sink whose release flush fails -- here the spool's byte
    limit refuses the open pack -- still releases, at once, and says why on
    stderr, the only report it can make once released."""
    sink = _binding_sink(native_sink_module, tmp_path, spool_max_bytes=4096)
    lease = sink.attach()
    sink.submit_envelope(LAYOUT, *_envelope(0, 8192))
    _wait_until_admitted(sink, 1)
    capfd.readouterr()

    started = time.monotonic()
    del lease
    assert time.monotonic() - started < 5.0

    err = capfd.readouterr().err
    assert RELEASE_LINE in err, err
    assert "spool byte limit exceeded" in err, err
    assert _ready(tmp_path) == []
    assert sink.snapshot()["failures"] == 1, sink.snapshot()
    # Released: another engine may take the sink, and then it reports the
    # failure the release could only print.
    lease = sink.attach()
    with pytest.raises(RuntimeError, match="spool byte limit exceeded"):
        sink.rethrow_if_failed()
    del lease


def test_a_record_the_pipeline_dropped_does_not_hold_the_release(
        native_sink_module, tmp_path):
    """A record dropped as oversized is a loss the sink counts, not a
    pipeline failure: flush_and_wait reports it, and the release that
    follows flushes an empty pipeline, at once."""
    sink = _sink(tmp_path, max_linger_ns=MINUTE_NS, max_pack_bytes=MiB,
                 max_queue_bytes=4 * MiB)
    lease = sink.attach()
    # Admitted (the payload alone fits the queue), then dropped by the pack
    # worker as oversized: it fits no empty pack with its framing.
    sink.submit_envelope(LAYOUT, *_envelope(0, MiB))
    with pytest.raises(RuntimeError, match="oversized_records"):
        sink.flush_and_wait(30.0)

    started = time.monotonic()
    del lease
    assert time.monotonic() - started < 5.0

    assert _ready(tmp_path) == []
    # Released: another engine may take the sink.
    lease = sink.attach()
    del lease


def test_the_release_flush_is_bounded_by_its_timeout(native_sink_module,
                                                     tmp_path, capfd):
    """A sink whose pipeline is stuck -- its stager parked, as on a hung
    filesystem -- must not hold RingEngine::stop, and so engine.close(),
    for longer than release_flush_timeout_s. The release gives up then,
    says so on stderr, and leaves the pack to the pipeline: it reaches the
    spool once the stager moves again."""
    sink = _binding_sink(native_sink_module, tmp_path,
                         release_flush_timeout_s=0.5)
    assert sink.release_flush_timeout_s == 0.5
    release_stages = sink._hold_stages_for_testing(10.0)
    lease = sink.attach()
    for index in range(3):
        sink.submit_envelope(LAYOUT, *_envelope(index))
    _wait_until_admitted(sink, 3)
    capfd.readouterr()

    started = time.monotonic()
    del lease
    elapsed = time.monotonic() - started
    assert 0.4 < elapsed < 2.0, elapsed

    err = capfd.readouterr().err
    assert RELEASE_LINE + "timed out after 500 ms" in err, err
    assert _ready(tmp_path) == []  # parked

    release_stages()
    deadline = time.monotonic() + 10.0
    while sink.snapshot()["persisted_records"] < 3:
        assert time.monotonic() < deadline, sink.snapshot()
        time.sleep(0.01)
    assert len(_ready(tmp_path)) == 1


def test_a_zero_release_timeout_leaves_the_open_pack_to_the_pipeline(
        native_sink_module, tmp_path):
    """0 turns the backstop off: the release stages nothing, and the open
    pack is the pipeline's, for the next flush, the linger or the sink's
    own close."""
    sink = _binding_sink(native_sink_module, tmp_path,
                         release_flush_timeout_s=0.0)
    assert sink.release_flush_timeout_s == 0.0
    lease = sink.attach()
    sink.submit_envelope(LAYOUT, *_envelope(0))
    _wait_until_admitted(sink, 1)

    del lease
    time.sleep(0.2)
    assert _ready(tmp_path) == []

    lease = sink.attach()
    assert sink.flush_and_wait(30.0)
    assert len(_ready(tmp_path)) == 1
    del lease


def test_sealed_on_release_says_whether_the_sink_can_still_stage(
        native_sink_module, tmp_path):
    """The engine lets go of its spool directory's owner lock only once
    nothing can stage into the directory any more. sealed_on_release says
    so: true once a release's flush has persisted everything admitted --
    a released sink admits nothing -- and false while the sink is
    attached, after a release whose flush timed out (a stage is still on
    its way, and lands later), and with the backstop off."""
    sink = _binding_sink(native_sink_module, tmp_path / "sealed")
    assert sink.sealed_on_release is False
    lease = sink.attach()
    sink.submit_envelope(LAYOUT, *_envelope(0))
    _wait_until_admitted(sink, 1)
    assert sink.sealed_on_release is False  # attached: it may take more
    del lease
    assert sink.sealed_on_release is True
    assert len(_ready(tmp_path / "sealed")) == 1
    lease = sink.attach()
    assert sink.sealed_on_release is False  # attached again
    del lease
    assert sink.sealed_on_release is True  # nothing left to flush

    wedged = _binding_sink(native_sink_module, tmp_path / "wedged",
                           release_flush_timeout_s=0.5)
    release_stages = wedged._hold_stages_for_testing(10.0)
    lease = wedged.attach()
    wedged.submit_envelope(LAYOUT, *_envelope(1))
    _wait_until_admitted(wedged, 1)
    del lease
    assert wedged.sealed_on_release is False
    assert _ready(tmp_path / "wedged") == []
    release_stages()  # the stage the release gave up on still lands
    deadline = time.monotonic() + 10.0
    while not _ready(tmp_path / "wedged"):
        assert time.monotonic() < deadline, wedged.snapshot()
        time.sleep(0.01)
    assert wedged.sealed_on_release is False

    off = _binding_sink(native_sink_module, tmp_path / "off",
                        release_flush_timeout_s=0.0)
    lease = off.attach()
    del lease
    assert off.sealed_on_release is False


@pytest.mark.parametrize("timeout_s", [-1.0, float("nan"), float("inf")])
def test_the_release_timeout_must_be_finite_and_not_negative(
        native_sink_module, tmp_path, timeout_s):
    with pytest.raises(ValueError, match="release_flush_timeout_s"):
        _binding_sink(native_sink_module, tmp_path,
                      release_flush_timeout_s=timeout_s)
