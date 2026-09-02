"""Payload estimation.

The estimator's whole value is that it agrees with what the runtime reserves,
so most of these tests are hand-computed byte counts rather than golden
values: ``elem_size * prod(compute_hook_shape(...))`` rounded up to
``PAYLOAD_ALIGN``, summed over firing hooks.

The rank tests matter as much as the totals. Aggregate bytes are roughly
TP-invariant while per-rank pressure is not, and it is per-rank pressure that
decides whether ``prepare_step`` falls off its fast path.
"""
from __future__ import annotations

import pytest

from dmi.configuration import (
    DMIConfig,
    LayerSelection,
    ModelDescriptor,
    ModelIdentity,
    ModelTopology,
    ObservationConfig,
)
from dmi.configuration.estimate import (
    PAYLOAD_ALIGN,
    Estimate,
    Workload,
    check_ring_fit,
    estimate_config,
    estimate_payload,
)

pytestmark = pytest.mark.cpu


NUM_LAYERS = 4
HIDDEN = 512
HEADS = 8
HEAD_DIM = 64
VOCAB = 32000
INTERMEDIATE = 1376
FP16 = 2
INT64 = 8


def _descriptor(**overrides) -> ModelDescriptor:
    topology = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        intermediate_size=INTERMEDIATE,
    )
    topology.update(overrides)
    return ModelDescriptor(
        model=ModelIdentity(
            id="test", name="Test", architecture="decoder_transformer"
        ),
        topology=ModelTopology(**topology),
    )


def _config(hooks, layers=None, **schedule_kwargs) -> DMIConfig:
    from dmi.config import CaptureSchedule

    return DMIConfig(
        observations=ObservationConfig(hooks=list(hooks), layers=layers),
        schedule=CaptureSchedule(**schedule_kwargs),
    )


def _workload(**overrides) -> Workload:
    base = dict(batch_size=1, prompt_tokens=128, decode_tokens=0, packed=True)
    base.update(overrides)
    return Workload(**base)


# ---------------------------------------------------------------------------
# Byte totals against hand computation
# ---------------------------------------------------------------------------


def test_hidden_state_hook_matches_hand_computed_bytes():
    """resid_pre packs to [q_len, hidden] on every selected layer."""
    estimate = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload()
    )

    assert estimate.peak_step_bytes == 128 * HIDDEN * FP16 * NUM_LAYERS


def test_layer_range_scales_the_estimate_linearly():
    full = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())
    half = estimate_config(
        _config(["resid_pre"], layers=LayerSelection(0, 1)),
        _descriptor(),
        _workload(),
    )

    assert half.peak_step_bytes == full.peak_step_bytes // 2


def test_global_hook_is_counted_once_not_per_layer():
    estimate = estimate_config(
        _config(["final_logits"]), _descriptor(), _workload()
    )

    # Packed: one logit row per request, so [num_reqs, vocab].
    assert estimate.peak_step_bytes == 1 * VOCAB * FP16


def test_token_ids_uses_int64_not_the_model_dtype():
    """ring.py overrides token_ids' dtype from the framework's input_ids."""
    estimate = estimate_config(
        _config(["token_ids"]), _descriptor(), _workload()
    )

    assert estimate.peak_step_bytes == 128 * INT64


def test_layer_range_does_not_reduce_a_global_hook():
    wide = estimate_config(
        _config(["final_logits"]), _descriptor(), _workload()
    )
    narrow = estimate_config(
        _config(["final_logits"], layers=LayerSelection(0, 0)),
        _descriptor(),
        _workload(),
    )

    assert narrow.peak_step_bytes == wide.peak_step_bytes


def test_unavailable_hook_contributes_nothing():
    """A dense model has no router; the hook cannot fire, so it costs 0."""
    dense = estimate_config(
        _config(["router_logits"]), _descriptor(num_experts=0), _workload()
    )

    assert dense.peak_step_bytes == 0


def test_moe_hook_fires_when_the_topology_supports_it():
    moe = estimate_config(
        _config(["router_logits"]),
        _descriptor(num_experts=8, top_k=2),
        _workload(),
    )

    assert moe.peak_step_bytes == 128 * 8 * FP16 * NUM_LAYERS


def test_every_total_is_payload_aligned():
    """Reservations are rounded up to PAYLOAD_ALIGN; estimates must be too."""
    estimate = estimate_config(
        _config(["resid_pre", "q", "k", "v", "token_ids", "final_logits"]),
        _descriptor(),
        _workload(prompt_tokens=127),
    )

    assert estimate.peak_step_bytes % PAYLOAD_ALIGN == 0


# ---------------------------------------------------------------------------
# Prefill sets the peak
# ---------------------------------------------------------------------------


def test_prefill_dominates_the_peak_step():
    estimate = estimate_config(
        _config(["resid_pre"]),
        _descriptor(),
        _workload(prompt_tokens=2048, decode_tokens=128),
    )

    assert estimate.peak_step_bytes > estimate.decode_step_bytes


def test_disabling_prefill_drops_the_peak_to_decode():
    estimate = estimate_config(
        _config(["resid_pre"], capture_prefill=False),
        _descriptor(),
        _workload(prompt_tokens=2048, decode_tokens=128),
    )

    assert estimate.peak_step_bytes == estimate.decode_step_bytes


def test_capturing_nothing_is_reported_as_a_warning():
    estimate = estimate_config(
        _config(["resid_pre"], capture_prefill=False, capture_decode=False),
        _descriptor(),
        _workload(),
    )

    assert estimate.peak_step_bytes == 0
    assert any("captures nothing" in w for w in estimate.warnings)


# ---------------------------------------------------------------------------
# Rank awareness
# ---------------------------------------------------------------------------


def test_tensor_parallel_rank_zero_is_the_busiest():
    """Rank 0 carries every unsharded hook plus its own shard."""
    estimate = estimate_config(
        _config(["resid_pre", "q"]),
        _descriptor(),
        _workload(tensor_parallel_size=2),
    )

    by_rank = {load.label: load for load in estimate.ranks}
    assert by_rank["pp0/tp0"].prefill_step_bytes > (
        by_rank["pp0/tp1"].prefill_step_bytes
    )
    assert estimate.peak_step_rank == "pp0/tp0"


def test_sharded_hook_is_divided_across_tensor_parallel_ranks():
    single = estimate_config(
        _config(["q"]), _descriptor(), _workload(tensor_parallel_size=1)
    )
    sharded = estimate_config(
        _config(["q"]), _descriptor(), _workload(tensor_parallel_size=2)
    )

    assert sharded.peak_step_bytes == single.peak_step_bytes // 2


def test_unsharded_hook_is_not_divided_and_stays_on_rank_zero():
    single = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(tensor_parallel_size=1)
    )
    sharded = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(tensor_parallel_size=2)
    )

    assert sharded.peak_step_bytes == single.peak_step_bytes
    by_rank = {load.label: load for load in sharded.ranks}
    assert by_rank["pp0/tp1"].prefill_step_bytes == 0


def test_pp_last_hook_lands_on_the_final_stage():
    estimate = estimate_config(
        _config(["resid_pre", "final_logits"]),
        _descriptor(),
        _workload(pipeline_parallel_size=2),
    )

    by_stage = {load.pp_stage: load for load in estimate.ranks}
    # Both stages carry two layers of resid_pre; only the last adds logits.
    assert by_stage[1].prefill_step_bytes > by_stage[0].prefill_step_bytes
    assert estimate.peak_step_rank == "pp1/tp0"


def test_pp_first_hook_lands_on_the_first_stage():
    estimate = estimate_config(
        _config(["token_ids"]), _descriptor(), _workload(pipeline_parallel_size=2)
    )

    by_stage = {load.pp_stage: load for load in estimate.ranks}
    assert by_stage[0].prefill_step_bytes == 128 * INT64
    assert by_stage[1].prefill_step_bytes == 0


def test_pipeline_stages_split_layers_between_them():
    single = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload()
    )
    staged = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(pipeline_parallel_size=2)
    )

    assert staged.peak_step_bytes == single.peak_step_bytes // 2


def test_rank_report_covers_every_pipeline_stage():
    estimate = estimate_config(
        _config(["resid_pre"]),
        _descriptor(),
        _workload(pipeline_parallel_size=4, tensor_parallel_size=2),
    )

    assert {load.pp_stage for load in estimate.ranks} == {0, 1, 2, 3}
    # Rank 0 and one representative non-zero rank per stage.
    assert {load.tp_rank for load in estimate.ranks} == {0, 1}


# ---------------------------------------------------------------------------
# Schedule and rate
# ---------------------------------------------------------------------------


def test_step_stride_thins_the_sustained_rate():
    dense = estimate_config(
        _config(["resid_pre"], step_stride=1),
        _descriptor(),
        _workload(decode_tokens=64, decode_steps_per_second=20.0),
    )
    strided = estimate_config(
        _config(["resid_pre"], step_stride=4),
        _descriptor(),
        _workload(decode_tokens=64, decode_steps_per_second=20.0),
    )

    assert strided.sustained_bytes_per_second == pytest.approx(
        dense.sustained_bytes_per_second / 4
    )


def test_step_stride_does_not_change_the_peak_step():
    """Sampling fewer steps does not make any single step smaller."""
    dense = estimate_config(
        _config(["resid_pre"], step_stride=1), _descriptor(), _workload()
    )
    strided = estimate_config(
        _config(["resid_pre"], step_stride=8), _descriptor(), _workload()
    )

    assert strided.peak_step_bytes == dense.peak_step_bytes


def test_per_day_volume_follows_the_sustained_rate():
    estimate = estimate_config(
        _config(["resid_pre"]),
        _descriptor(),
        _workload(decode_tokens=64, decode_steps_per_second=10.0),
    )

    assert estimate.bytes_per_day == pytest.approx(
        estimate.sustained_bytes_per_second * 86_400
    )


def test_missing_rate_is_reported_rather_than_guessed():
    estimate = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())

    assert estimate.sustained_bytes_per_second is None
    assert estimate.bytes_per_day is None
    assert any("decode_steps_per_second" in w for w in estimate.warnings)


# ---------------------------------------------------------------------------
# Convention handling
# ---------------------------------------------------------------------------


def test_attention_weights_are_refused_in_the_packed_convention():
    estimate = estimate_config(
        _config(["pattern"]), _descriptor(), _workload(packed=True)
    )

    assert estimate.peak_step_bytes == 0
    assert any("packed convention" in w for w in estimate.warnings)


def test_attention_weights_are_counted_in_the_batched_convention():
    estimate = estimate_config(
        _config(["pattern"]), _descriptor(), _workload(packed=False)
    )

    # [batch, heads, q_len, kv_dim] per layer.
    expected = 1 * HEADS * 128 * 128 * FP16 * NUM_LAYERS
    assert estimate.peak_step_bytes == expected


def test_convention_is_recorded_as_an_assumption():
    packed = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(packed=True)
    )
    batched = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(packed=False)
    )

    assert any("packed" in a for a in packed.assumptions)
    assert any("batched" in a for a in batched.assumptions)


# ---------------------------------------------------------------------------
# Ring fit
# ---------------------------------------------------------------------------


def test_a_step_within_capacity_fits():
    estimate = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())

    fit = check_ring_fit(estimate, payload_bytes=64 * 1024 * 1024)

    assert fit.fits is True
    assert fit.occupancy_percent < 100


def test_an_oversized_step_explains_the_eager_fallback():
    estimate = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())

    fit = check_ring_fit(estimate, payload_bytes=1024)

    assert fit.fits is False
    assert "STEP_OVERSIZED" in fit.detail


def test_the_smaller_ring_is_the_binding_limit():
    """prepare_step compares against min(payload_cap, staging_cap)."""
    estimate = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())

    fit = check_ring_fit(
        estimate, payload_bytes=64 * 1024 * 1024, pinned_bytes=1 * 1024 * 1024
    )

    assert fit.effective_bytes == 1 * 1024 * 1024
    assert "Pinned staging" in fit.detail


def test_zero_payload_capacity_is_rejected():
    estimate = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())

    with pytest.raises(ValueError, match="payload_bytes must be >= 1"):
        check_ring_fit(estimate, payload_bytes=0)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_adding_an_observation_never_lowers_the_estimate():
    fewer = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())
    more = estimate_config(
        _config(["resid_pre", "q", "k", "v"]), _descriptor(), _workload()
    )

    assert more.peak_step_bytes >= fewer.peak_step_bytes


def test_widening_a_layer_range_never_lowers_the_estimate():
    narrow = estimate_config(
        _config(["resid_pre"], layers=LayerSelection(0, 0)),
        _descriptor(),
        _workload(),
    )
    wide = estimate_config(
        _config(["resid_pre"], layers=LayerSelection(0, 3)),
        _descriptor(),
        _workload(),
    )

    assert wide.peak_step_bytes >= narrow.peak_step_bytes


def test_longer_prompts_never_lower_the_estimate():
    short = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(prompt_tokens=128)
    )
    long = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(prompt_tokens=4096)
    )

    assert long.peak_step_bytes > short.peak_step_bytes


# ---------------------------------------------------------------------------
# Workload validation and serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"prompt_tokens": 0}, "prompt_tokens must be >= 1"),
        ({"decode_tokens": -1}, "decode_tokens must be >= 0"),
        ({"tensor_parallel_size": 0}, "tensor_parallel_size must be >= 1"),
        ({"pipeline_parallel_size": 0}, "pipeline_parallel_size must be >= 1"),
        ({"decode_steps_per_second": -1.0}, "decode_steps_per_second must be >= 0"),
    ],
)
def test_invalid_workload_is_rejected_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Workload(**kwargs)


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="Unknown dtype"):
        estimate_config(
            _config(["resid_pre"]), _descriptor(), _workload(dtype="float13")
        )


def test_decode_step_comes_from_the_same_rank_as_the_peak():
    """One result must not describe two different ranks.

    Taking a max over ranks for decode while choosing the reported rank by
    peak lets the two disagree whenever the prefill-heaviest stage is not the
    decode-heaviest one.
    """
    estimate = estimate_config(
        _config(["resid_pre", "final_logits"]),
        _descriptor(),
        _workload(decode_tokens=32, pipeline_parallel_size=2),
    )

    worst = [r for r in estimate.ranks if r.label == estimate.peak_step_rank][0]
    assert estimate.decode_step_bytes == worst.decode_step_bytes


def test_payload_is_json_safe():
    import json

    estimate = estimate_config(
        _config(["resid_pre"]),
        _descriptor(),
        _workload(tensor_parallel_size=2, decode_steps_per_second=10.0),
    )

    payload = estimate_payload(estimate)

    json.dumps(payload)  # must not raise
    assert payload["peak_step_rank"] == "pp0/tp0"
    assert len(payload["ranks"]) == len(estimate.ranks)


def test_estimate_is_immutable():
    estimate = estimate_config(_config(["resid_pre"]), _descriptor(), _workload())

    assert isinstance(estimate, Estimate)
    with pytest.raises(Exception):
        estimate.peak_step_bytes = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-request accounting
# ---------------------------------------------------------------------------


def test_per_request_bytes_are_per_sequence_in_the_packed_convention():
    single = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(batch_size=1)
    )
    batch = estimate_config(
        _config(["resid_pre"]), _descriptor(), _workload(batch_size=8)
    )

    assert batch.bytes_per_request == single.bytes_per_request


def test_per_request_bytes_are_per_sequence_in_the_batched_convention():
    """Batched shapes carry the leading batch dimension, so the whole-batch
    step total divides by batch_size exactly like the packed convention --
    8 identical sequences must not report 8x the cost of one."""
    single = estimate_config(
        _config(["resid_pre"]),
        _descriptor(),
        _workload(packed=False, batch_size=1),
    )
    batch = estimate_config(
        _config(["resid_pre"]),
        _descriptor(),
        _workload(packed=False, batch_size=8),
    )

    assert batch.bytes_per_request == single.bytes_per_request


def test_aggregate_peak_follows_the_enabled_phases():
    """The aggregate sums each rank's peak step over whichever phases are
    enabled; with prefill capture off it is a decode figure, and the field is
    named for that."""
    estimate = estimate_config(
        _config(["resid_pre"], capture_prefill=False),
        _descriptor(),
        _workload(prompt_tokens=2048, decode_tokens=128),
    )

    assert estimate.aggregate_peak_step_bytes == estimate.decode_step_bytes
    assert estimate_payload(estimate)["aggregate_peak_step_bytes"] == (
        estimate.decode_step_bytes
    )
