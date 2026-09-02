"""The estimate must describe what the runtime will actually do.

Reproduction suite for two ways the payload estimate could mislead:

**An unenforced schedule.** ``CaptureSchedule`` is authored by the
configurator and serialized into the YAML, but no shipped adapter applies it:
``should_capture_step`` / ``should_capture_request`` have no callers outside
``dmi.config`` itself, and nothing in ``dmi.adapters`` reads
``CompiledDMIConfig.schedule``. An estimate that divided its figures by
``step_stride`` would therefore promise a reduction the capture never
delivers, and someone sizing storage from it would under-provision by exactly
that factor. Silence is worse than a wrong number here: the stride control
looks like it works.

**A layer range that selects nothing.** A range outside the model's layers
resolves to an empty set, and the estimate for per-layer hooks collapses to
zero bytes. Zero reads as "this is free" rather than "this selects nothing",
so the panel has to say which it means.

These are small tests: pure arithmetic over a synthesized descriptor, no I/O.
"""
from __future__ import annotations

import pytest

from dmi.config import CaptureSchedule
from dmi.configuration import (
    DMIConfig,
    LayerSelection,
    ModelDescriptor,
    ModelIdentity,
    ModelTopology,
    ObservationConfig,
)
from dmi.configuration.estimate import Workload, estimate_config

pytestmark = pytest.mark.cpu

NUM_LAYERS = 8


def _descriptor(num_layers: int = NUM_LAYERS) -> ModelDescriptor:
    return ModelDescriptor(
        model=ModelIdentity(
            id="fidelity-test", name="Fidelity Test", architecture="decoder_transformer"
        ),
        topology=ModelTopology(
            num_layers=num_layers,
            hidden_size=512,
            num_attention_heads=8,
            num_kv_heads=8,
            head_dim=64,
            intermediate_size=1376,
            vocab_size=32000,
        ),
    )


def _workload(**overrides) -> Workload:
    base = dict(
        batch_size=1,
        prompt_tokens=128,
        decode_tokens=64,
        decode_steps_per_second=10.0,
        packed=True,
    )
    base.update(overrides)
    return Workload(**base)


def _config(hooks=("resid_pre",), layers=None, **schedule) -> DMIConfig:
    return DMIConfig(
        observations=ObservationConfig(hooks=list(hooks), layers=layers),
        schedule=CaptureSchedule(**schedule),
    )


def _notes(estimate) -> str:
    return " ".join(estimate.warnings + estimate.assumptions).lower()


# ---------------------------------------------------------------------------
# An unenforced schedule must not shrink the estimate
# ---------------------------------------------------------------------------


def test_step_stride_does_not_shrink_the_sustained_rate():
    """No adapter enforces the stride, so the bytes still arrive."""
    dense = estimate_config(_config(step_stride=1), _descriptor(), _workload())
    strided = estimate_config(_config(step_stride=4), _descriptor(), _workload())

    assert strided.sustained_bytes_per_second == dense.sustained_bytes_per_second


def test_step_stride_does_not_shrink_the_daily_volume():
    dense = estimate_config(_config(step_stride=1), _descriptor(), _workload())
    strided = estimate_config(_config(step_stride=8), _descriptor(), _workload())

    assert strided.bytes_per_day == dense.bytes_per_day


def test_step_stride_does_not_shrink_the_per_request_total():
    dense = estimate_config(_config(step_stride=1), _descriptor(), _workload())
    strided = estimate_config(_config(step_stride=4), _descriptor(), _workload())

    assert strided.bytes_per_request == dense.bytes_per_request


def test_a_large_stride_still_reports_the_full_volume():
    """The pathological case: stride 1000 previously reported ~0.1% of reality."""
    dense = estimate_config(_config(step_stride=1), _descriptor(), _workload())
    strided = estimate_config(_config(step_stride=1000), _descriptor(), _workload())

    assert strided.sustained_bytes_per_second == dense.sustained_bytes_per_second


def test_a_set_step_stride_says_it_is_not_enforced():
    """The control looks functional; the estimate has to say it is not."""
    estimate = estimate_config(_config(step_stride=4), _descriptor(), _workload())

    assert "not enforced" in _notes(estimate)


def test_a_set_request_stride_says_it_is_not_enforced():
    estimate = estimate_config(_config(request_stride=4), _descriptor(), _workload())

    assert "not enforced" in _notes(estimate)


def test_a_default_schedule_is_not_warned_about():
    """Nothing was asked for, so there is nothing to disclaim."""
    estimate = estimate_config(_config(), _descriptor(), _workload())

    assert "not enforced" not in _notes(estimate)


def test_prefill_and_decode_toggles_keep_working():
    """Only the sampling knobs are unenforced; the phase toggles gate shapes."""
    both = estimate_config(_config(), _descriptor(), _workload())
    decode_only = estimate_config(
        _config(capture_prefill=False), _descriptor(), _workload()
    )

    assert decode_only.peak_step_bytes < both.peak_step_bytes


# ---------------------------------------------------------------------------
# A layer range that selects nothing must say so
# ---------------------------------------------------------------------------


def test_a_layer_range_beyond_the_model_reports_zero_with_a_reason():
    estimate = estimate_config(
        _config(layers=LayerSelection(40, 50)), _descriptor(), _workload()
    )

    assert estimate.peak_step_bytes == 0
    assert "selects no layer" in _notes(estimate)


def test_a_layer_range_clipped_by_the_model_says_what_survived():
    """Layers 6-50 on an 8-layer model is 2 layers, not 45."""
    estimate = estimate_config(
        _config(layers=LayerSelection(6, 50)), _descriptor(), _workload()
    )

    notes = _notes(estimate)
    assert "clipped" in notes
    assert "2 of" in notes


def test_a_layer_range_inside_the_model_is_not_warned_about():
    estimate = estimate_config(
        _config(layers=LayerSelection(2, 4)), _descriptor(), _workload()
    )

    notes = _notes(estimate)
    assert "selects no layer" not in notes
    assert "clipped" not in notes


def test_selecting_every_layer_is_not_reported_as_clipped():
    estimate = estimate_config(
        _config(layers=LayerSelection(0, NUM_LAYERS - 1)),
        _descriptor(),
        _workload(),
    )

    assert "clipped" not in _notes(estimate)


def test_an_empty_layer_range_does_not_suppress_global_hooks():
    """A global hook ignores the layer range, so it must still be counted."""
    estimate = estimate_config(
        _config(hooks=("resid_pre", "final_logits"), layers=LayerSelection(40, 50)),
        _descriptor(),
        _workload(),
    )

    assert estimate.peak_step_bytes > 0


# ---------------------------------------------------------------------------
# The UI must carry the same disclaimer the policy control carries
# ---------------------------------------------------------------------------


def test_the_capture_panel_discloses_that_sampling_is_not_enforced():
    """Guard against the notice silently disappearing from the markup."""
    from pathlib import Path

    from dmi.ui.app import STATIC_DIR

    markup = (Path(STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    capture_panel = markup.split('aria-label="Capture schedule"', 1)[-1].split(
        "</section>", 1
    )[0]

    assert "not enforced" in capture_panel.lower()
