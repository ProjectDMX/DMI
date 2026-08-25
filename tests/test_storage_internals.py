"""Unit tests for dmi.storage.internals -- pure reassembly logic, no DB."""
from pathlib import Path

import pytest
import torch

from dmi.storage.internals import (
    _TRAILING_RANGE_ACTS,
    IncompleteInternalError,
    InternalRequirements,
    get_internal,
    make_lazy_internal,
)

pytestmark = pytest.mark.cpu

ACT = "blocks.hook_resid_pre"


class FakeReader:
    """Returns canned (key, tensor) rows regardless of the prefix."""
    def __init__(self, rows):
        self._rows = rows
        self.calls = 0

    def prefix_get(self, prefix):
        self.calls += 1
        return self._rows


class FailingOnceReader:
    def __init__(self, rows):
        self._rows = rows
        self.calls = 0

    def prefix_get(self, prefix):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return self._rows


class SequenceReader:
    def __init__(self, row_sets):
        self._row_sets = list(row_sets)
        self.calls = 0

    def prefix_get(self, prefix):
        self.calls += 1
        idx = min(self.calls - 1, len(self._row_sets) - 1)
        rows = self._row_sets[idx]
        if isinstance(rows, Exception):
            raise rows
        return rows


class PrefixReader:
    def __init__(self, rows):
        self._rows = rows
        self.prefixes = []

    @property
    def calls(self):
        return len(self.prefixes)

    def prefix_get(self, prefix):
        self.prefixes.append(prefix)
        return [
            (key, tensor)
            for key, tensor in self._rows
            if key[:len(prefix)] == prefix
        ]


def _row(req, layer, start, tensor):
    # key = (model_id, request_id, act_name, layer_no, shard_rank, start, end)
    return ((("m", req, ACT, layer, 0, start, start + tensor.shape[0])), tensor)


def _row_act(req, act, layer, start, end, tensor):
    return ((("m", req, act, layer, 0, start, end)), tensor)


def test_per_layer_tuple_and_shape():
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4)),
        _row("0:0", 1, 0, torch.ones(3, 4) * 5),
    ]
    internal = get_internal("m", FakeReader(rows))
    assert internal.available == ["hidden_states"]
    hs = internal.hidden_states
    assert len(hs) == 2                       # two layers
    assert tuple(hs[0].shape) == (1, 3, 4)    # [batch, seq, hidden]


def test_available_includes_hf_mapped_fields():
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4)),
        _row_act("0:0", "blocks.attn.hook_pattern", 0, 0, 3, torch.ones(2, 3, 3)),
        _row_act("0:0", "final_logits", -1, 0, 3, torch.ones(3, 10)),
        _row_act("0:0", "token_ids", -1, 0, 3, torch.arange(3)),
        _row_act("0:0", "hook_embed", -1, 0, 3, torch.ones(3, 4)),
        _row_act("0:0", "blocks.attn.hook_q", 0, 0, 3, torch.ones(3, 2, 2)),
    ]
    internal = get_internal("m", FakeReader(rows))

    assert internal.available == [
        "attentions",
        "embeddings",
        "hidden_states",
        "logits",
        "q",
        "token_ids",
    ]


def test_logits_reassemble_global_batch_by_numeric_request_id():
    rows = [
        _row_act("0:10", "final_logits", -1, 0, 1, torch.full((1, 2), 10.0)),
        _row_act("0:2", "final_logits", -1, 0, 1, torch.full((1, 2), 2.0)),
    ]
    logits = get_internal("m", FakeReader(rows)).logits

    assert tuple(logits.shape) == (2, 1, 2)
    assert torch.equal(logits[0, 0], torch.full((2,), 2.0))
    assert torch.equal(logits[1, 0], torch.full((2,), 10.0))


def test_token_ids_reassemble_global_1d():
    rows = [
        _row_act("0:0", "token_ids", -1, 0, 2, torch.tensor([10, 11])),
        _row_act("0:1", "token_ids", -1, 0, 1, torch.tensor([20])),
    ]
    token_ids = get_internal("m", FakeReader(rows)).token_ids

    assert torch.equal(token_ids, torch.tensor([[10, 11], [0, 20]]))


def test_embeddings_reassemble_global_hidden_field():
    rows = [
        _row_act("0:0", "hook_embed", -1, 0, 2, torch.ones(2, 4)),
        _row_act("0:1", "hook_embed", -1, 0, 1, torch.ones(1, 4) * 2),
    ]
    embeddings = get_internal("m", FakeReader(rows)).embeddings

    assert tuple(embeddings.shape) == (2, 2, 4)
    assert torch.equal(embeddings[1, 0], torch.zeros(4))
    assert torch.equal(embeddings[1, 1], torch.ones(4) * 2)


def test_q_reassemble_per_layer():
    rows = [
        _row_act("0:0", "blocks.attn.hook_q", 0, 0, 2, torch.ones(2, 2, 3)),
        _row_act("0:0", "blocks.attn.hook_q", 1, 0, 2, torch.ones(2, 2, 3) * 2),
    ]
    q = get_internal("m", FakeReader(rows)).q

    assert len(q) == 2
    assert tuple(q[0].shape) == (1, 2, 2, 3)
    assert torch.equal(q[1], torch.ones(1, 2, 2, 3) * 2)


def test_attentions_reassemble_per_layer_with_decode_growth():
    prefill = torch.ones(2, 2, 2)
    decode = torch.ones(2, 1, 3) * 2
    rows = [
        _row_act("0:0", "blocks.attn.hook_pattern", 0, 0, 2, prefill),
        _row_act("0:0", "blocks.attn.hook_pattern", 0, 2, 3, decode),
    ]
    attentions = get_internal("m", FakeReader(rows)).attentions

    assert len(attentions) == 1
    assert tuple(attentions[0].shape) == (1, 2, 3, 3)
    assert torch.equal(attentions[0][0, :, :2, :2], prefill)
    assert torch.equal(attentions[0][0, :, 2:, :], decode)


def test_attention_scores_reassemble_with_negative_inf_padding():
    prefill = torch.ones(1, 2, 2)
    decode = torch.ones(1, 1, 3) * 2
    rows = [
        _row_act("0:0", "blocks.attn.hook_attn_scores", 0, 0, 2, prefill),
        _row_act("0:0", "blocks.attn.hook_attn_scores", 0, 2, 3, decode),
    ]
    scores = get_internal("m", FakeReader(rows)).attention_scores

    assert tuple(scores[0].shape) == (1, 1, 3, 3)
    assert torch.isneginf(scores[0][0, 0, 0, 2])
    assert torch.equal(scores[0][0, :, :2, :2], prefill)
    assert torch.equal(scores[0][0, :, 2:, :], decode)


def test_chunks_concat_by_start():
    rows = [
        _row("0:0", 0, 2, torch.full((1, 4), 9.0)),  # out-of-order on purpose
        _row("0:0", 0, 0, torch.full((2, 4), 1.0)),
    ]
    hs = get_internal("m", FakeReader(rows)).hidden_states
    assert tuple(hs[0].shape) == (1, 3, 4)
    assert torch.equal(hs[0][0, :2], torch.full((2, 4), 1.0))
    assert torch.equal(hs[0][0, 2], torch.full((4,), 9.0))


def test_ragged_batch_left_pads():
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4)),   # 3 tokens
        _row("0:1", 0, 0, torch.ones(2, 4) * 2),  # 2 tokens -> left-pad to 3
    ]
    hs = get_internal("m", FakeReader(rows)).hidden_states
    assert tuple(hs[0].shape) == (2, 3, 4)
    assert torch.equal(hs[0][1, 0], torch.zeros(4))        # front row padded
    assert torch.equal(hs[0][1, 1:], torch.ones(2, 4) * 2)  # real tokens right-aligned


def test_uncaptured_field_raises():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    internal = get_internal("m", FakeReader(rows))
    with pytest.raises(AttributeError, match="not captured"):
        internal.attention


def test_get_internal_accepts_model_id():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FakeReader(rows)

    assert get_internal("m", reader).available == ["hidden_states"]


def test_lazy_internal_loads_once_after_success():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FakeReader(rows)
    internal = make_lazy_internal("m", reader)

    assert "pending" in repr(internal)
    assert internal.hidden_states[0].shape == (1, 3, 4)
    assert reader.calls == 1
    assert internal.hidden_states[0].shape == (1, 3, 4)
    assert reader.calls == 1
    assert "cached=['hidden_states']" in repr(internal)


def test_lazy_internal_with_request_ids_reads_only_requested_act_prefixes():
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4)),
        _row("0:1", 0, 0, torch.ones(2, 4) * 2),
        _row_act("0:0", "final_logits", -1, 0, 3, torch.ones(3, 10)),
        _row_act("other:0", ACT, 0, 0, 1, torch.ones(1, 4) * 9),
    ]
    reader = PrefixReader(rows)
    internal = make_lazy_internal("m", reader, request_ids=("0:0", "0:1"))

    hs = internal.hidden_states

    assert tuple(hs[0].shape) == (2, 3, 4)
    assert reader.prefixes == [
        ("m", "0:0", ACT),
        ("m", "0:1", ACT),
    ]


def test_lazy_internal_token_mask_uses_recorded_ranges_without_reader():
    reader = PrefixReader([])
    internal = make_lazy_internal(
        "m",
        reader,
        request_ids=("0:0", "0:1"),
        token_ranges={
            "0:0": ((0, 3), (3, 4)),
            "0:1": ((0, 2), (2, 2)),
        },
    )

    mask = internal.token_mask

    assert mask.dtype == torch.bool
    assert torch.equal(
        mask,
        torch.tensor([
            [True, True, True, True],
            [False, False, True, True],
        ]),
    )
    assert reader.calls == 0
    assert internal.token_mask is mask


def test_lazy_internal_retries_after_failed_load():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FailingOnceReader(rows)
    internal = make_lazy_internal("m", reader)

    with pytest.raises(RuntimeError, match="temporary failure"):
        internal.hidden_states

    assert internal.hidden_states[0].shape == (1, 3, 4)
    assert reader.calls == 2


def test_lazy_internal_clear_cache_one_field():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FakeReader(rows)
    internal = make_lazy_internal("m", reader)

    assert internal.hidden_states[0].shape == (1, 3, 4)
    assert internal.hidden_states[0].shape == (1, 3, 4)
    assert reader.calls == 1

    internal.clear_cache("hidden_states")
    assert internal.hidden_states[0].shape == (1, 3, 4)
    assert reader.calls == 2


def test_lazy_internal_clear_cache_all_fields():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FakeReader(rows)
    internal = make_lazy_internal("m", reader)

    assert internal.hidden_states[0].shape == (1, 3, 4)
    internal.clear_cache()
    assert internal.hidden_states[0].shape == (1, 3, 4)
    assert reader.calls == 2


def test_lazy_internal_requirement_blocks_incomplete_cache():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FakeReader(rows)
    internal = make_lazy_internal("m", reader)
    internal.require("hidden_states", count=2)

    with pytest.raises(IncompleteInternalError, match="expected 2 entries, found 1"):
        internal.hidden_states

    with pytest.raises(IncompleteInternalError, match="expected 2 entries, found 1"):
        internal.hidden_states

    assert reader.calls == 2


def test_lazy_internal_requirement_retries_missing_until_success():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = SequenceReader([[], rows])
    internal = make_lazy_internal("m", reader)
    internal.require("hidden_states", count=1, retry=True, timeout_s=1.0, poll_s=0.001)

    assert len(internal.hidden_states) == 1
    assert reader.calls == 2


def test_lazy_internal_requirement_retries_incomplete_until_success():
    one_layer = [_row("0:0", 0, 0, torch.ones(3, 4))]
    two_layers = [
        _row("0:0", 0, 0, torch.ones(3, 4)),
        _row("0:0", 1, 0, torch.ones(3, 4)),
    ]
    reader = SequenceReader([one_layer, two_layers])
    internal = make_lazy_internal("m", reader)
    internal.require("hidden_states", count=2, retry=True, timeout_s=1.0, poll_s=0.001)

    assert len(internal.hidden_states) == 2
    assert reader.calls == 2


def test_lazy_internal_requirement_detects_missing_token_ranges():
    rows = [_row("0:0", 0, 0, torch.ones(2, 4))]
    reader = PrefixReader(rows)
    internal = make_lazy_internal(
        "m",
        reader,
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 2), (2, 3))},
    )
    internal.require("hidden_states", count=1, match_token_ranges=True)

    with pytest.raises(IncompleteInternalError, match="token ranges are incomplete"):
        internal.hidden_states


def test_lazy_internal_validates_token_ranges_by_default_when_request_metadata_exists():
    rows = [_row("0:0", 0, 0, torch.ones(2, 4))]
    internal = make_lazy_internal(
        "m",
        PrefixReader(rows),
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 2), (2, 3))},
    )

    with pytest.raises(IncompleteInternalError, match="token ranges are incomplete"):
        internal.hidden_states


def test_pipeline_parallel_lag_is_reported_and_clears_on_retry():
    """PP splits layers across ranks that flush independently.

    A lagging rank must be reported rather than silently yielding a short
    tuple, and ``retry=True`` must clear it once the rank flushes.
    """
    ranges = ((0, 2), (2, 3))
    def rows(partial=()):
        return [
            _row("0:0", layer, start, torch.ones(n, 4))
            for layer in range(8)
            for start, n in ((0, 2), (2, 1))
            if not (layer in partial and start == 2)
        ]

    lagging = rows(partial=(4, 5, 6, 7))       # rank1 decode rows not flushed
    internal = make_lazy_internal(
        "m", PrefixReader(lagging),
        request_ids=("0:0",), token_ranges={"0:0": ranges},
    )
    with pytest.raises(IncompleteInternalError, match="layer=4"):
        internal.hidden_states

    internal = make_lazy_internal(
        "m", SequenceReader([lagging, rows()]),
        request_ids=("0:0",), token_ranges={"0:0": ranges},
    )
    internal.require(
        "hidden_states", count=8, retry=True, timeout_s=2.0, poll_s=0.01
    )
    assert len(internal.hidden_states) == 8


def test_trailing_range_acts_matches_the_native_rewrite_sites():
    """Guard the hardcoded exemption against native drift.

    ``_TRAILING_RANGE_ACTS`` only stays correct while ``final_logits`` is the
    single act whose stored row range is re-based away from the step's token
    range.  If someone narrows another act in p2p_thread.cpp, this fails.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "native" / "csrc" / "ring" / "p2p_thread.cpp"
    )
    if not source.is_file():
        pytest.skip("native sources not present")
    text = source.read_text()

    rewrites = [
        line.strip() for line in text.splitlines()
        if "db_start =" in line and "req.start_token" not in line
    ]
    assert len(rewrites) == 1, (
        f"native db_start rewrite sites changed: {rewrites}. "
        "Update _TRAILING_RANGE_ACTS in dmi/storage/internals.py to match."
    )
    guard = text.split("db_start = req.end_token")[0].splitlines()[-3:]
    assert any("HOOK_TYPE_FINAL_LOGITS" in line for line in guard), (
        "the db_start rewrite is no longer guarded by HOOK_TYPE_FINAL_LOGITS"
    )
    assert _TRAILING_RANGE_ACTS == frozenset({"final_logits"})


def test_token_range_check_covers_every_layer_not_just_the_highest():
    # Layer 0 is missing its decode row; layers 1 and 2 are complete.  A
    # representative-layer check that only looked at the highest layer would
    # pass this.
    rows = [
        _row("0:0", layer, start, torch.ones(n, 4))
        for layer in (0, 1, 2)
        for start, n in ((0, 2), (2, 1))
        if not (layer == 0 and start == 2)
    ]
    internal = make_lazy_internal(
        "m",
        PrefixReader(rows),
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 2), (2, 3))},
    )

    with pytest.raises(IncompleteInternalError, match="layer=0"):
        internal.hidden_states


def test_token_range_check_names_the_offending_layer():
    rows = [
        _row("0:0", layer, start, torch.ones(n, 4))
        for layer in (0, 1)
        for start, n in ((0, 2), (2, 1))
        if not (layer == 1 and start == 0)
    ]
    internal = make_lazy_internal(
        "m",
        PrefixReader(rows),
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 2), (2, 3))},
    )

    with pytest.raises(IncompleteInternalError) as excinfo:
        internal.hidden_states

    assert "layer=1" in str(excinfo.value)


def test_global_field_errors_do_not_mention_a_layer():
    rows = [_row_act("0:0", "hook_embed", -1, 0, 2, torch.ones(2, 4))]
    internal = make_lazy_internal(
        "m",
        PrefixReader(rows),
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 2), (2, 3))},
    )

    with pytest.raises(IncompleteInternalError) as excinfo:
        internal.embeddings

    assert "layer=" not in str(excinfo.value)


def test_a_layer_that_left_no_rows_at_all_is_not_detectable_by_range_check():
    """Documents the one loss the range check inherently cannot see.

    If a layer never wrote a single row it is absent from the data, so there
    is nothing to compare.  Only a declared ``count`` closes this.
    """
    rows = [
        _row("0:0", layer, start, torch.ones(n, 4))
        for layer in (0, 1)
        for start, n in ((0, 2), (2, 1))
    ]
    internal = make_lazy_internal(
        "m",
        PrefixReader(rows),
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 2), (2, 3))},
    )

    assert len(internal.hidden_states) == 2  # silent: layer 2 never landed

    internal.clear_cache()
    internal.require("hidden_states", count=3)
    with pytest.raises(IncompleteInternalError, match="expected 3 entries"):
        internal.hidden_states


LOGITS_ACT = "final_logits"


def _logits_internal(rows, ranges):
    return make_lazy_internal(
        "m",
        PrefixReader(rows),
        request_ids=("0:0",),
        token_ranges={"0:0": ranges},
    )


def test_lazy_internal_accepts_narrowed_final_logits_rows():
    """HF generate passes logits_to_keep=1, so the prefill logits row covers
    only the last prompt position while the step range covers the whole
    prompt.  Comparing full ranges would fail on every real generate call."""
    rows = [
        _row_act("0:0", LOGITS_ACT, -1, 3, 4, torch.ones(1, 8)),   # prefill
        _row_act("0:0", LOGITS_ACT, -1, 4, 5, torch.ones(1, 8)),   # decode
        _row_act("0:0", LOGITS_ACT, -1, 5, 6, torch.ones(1, 8)),   # decode
    ]

    logits = _logits_internal(rows, ((0, 4), (4, 5), (5, 6))).logits

    assert logits.shape == (1, 3, 8)


def test_lazy_internal_still_detects_a_missing_final_logits_step():
    rows = [
        _row_act("0:0", LOGITS_ACT, -1, 3, 4, torch.ones(1, 8)),
        _row_act("0:0", LOGITS_ACT, -1, 4, 5, torch.ones(1, 8)),
    ]

    with pytest.raises(IncompleteInternalError, match="token range ends"):
        _logits_internal(rows, ((0, 4), (4, 5), (5, 6))).logits


def test_lazy_internal_accepts_full_width_prefill_logits():
    # logits_to_keep=0 keeps every prefill position; the relaxed check must
    # still pass for the unnarrowed case.
    rows = [
        _row_act("0:0", LOGITS_ACT, -1, 0, 4, torch.ones(4, 8)),
        _row_act("0:0", LOGITS_ACT, -1, 4, 5, torch.ones(1, 8)),
    ]

    assert _logits_internal(rows, ((0, 4), (4, 5))).logits.shape == (1, 5, 8)


def test_lazy_internal_keeps_strict_range_matching_for_non_logits_fields():
    rows = [_row("0:0", 0, 3, torch.ones(1, 4))]
    internal = make_lazy_internal(
        "m",
        PrefixReader(rows),
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 4),)},
    )

    with pytest.raises(IncompleteInternalError, match=r"expected \[\(0, 4\)\]"):
        internal.hidden_states


def test_retry_timeout_reports_the_token_range_reason_not_a_count_mismatch():
    rows = [_row_act("0:0", LOGITS_ACT, -1, 3, 4, torch.ones(1, 8))]
    internal = _logits_internal(rows, ((0, 4), (4, 5)))
    internal.require("logits", count=1, retry=True, timeout_s=0.02, poll_s=0.001)

    with pytest.raises(IncompleteInternalError) as excinfo:
        internal.logits

    message = str(excinfo.value)
    assert "token range ends" in message
    assert "found none" not in message
    assert isinstance(excinfo.value.__cause__, IncompleteInternalError)


def test_lazy_internal_requirement_retries_missing_token_ranges_until_success():
    partial = [_row("0:0", 0, 0, torch.ones(2, 4))]
    complete = [
        _row("0:0", 0, 0, torch.ones(2, 4)),
        _row("0:0", 0, 2, torch.ones(1, 4)),
    ]
    reader = SequenceReader([partial, complete])
    internal = make_lazy_internal(
        "m",
        reader,
        request_ids=("0:0",),
        token_ranges={"0:0": ((0, 2), (2, 3))},
    )
    internal.require(
        "hidden_states",
        count=1,
        retry=True,
        timeout_s=1.0,
        poll_s=0.001,
        match_token_ranges=True,
    )

    assert tuple(internal.hidden_states[0].shape) == (1, 3, 4)
    assert reader.calls == 2


def test_lazy_internal_requirement_retry_timeout_raises_incomplete():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = SequenceReader([rows])
    internal = make_lazy_internal("m", reader)
    internal.require("hidden_states", count=2, retry=True, timeout_s=0.0, poll_s=0.001)

    with pytest.raises(IncompleteInternalError, match="expected 2 entries, found 1"):
        internal.hidden_states
    assert reader.calls == 1


def test_lazy_internal_requirement_can_retry_without_timeout():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = SequenceReader([[], [], rows])
    internal = make_lazy_internal("m", reader)
    internal.require("hidden_states", count=1, retry=True, timeout_s=None, poll_s=0.001)

    assert len(internal.hidden_states) == 1
    assert reader.calls == 3


def test_lazy_internal_retry_does_not_swallow_unexpected_errors():
    reader = SequenceReader([RuntimeError("boom")])
    internal = make_lazy_internal("m", reader)
    internal.require("hidden_states", count=1, retry=True, timeout_s=1.0, poll_s=0.001)

    with pytest.raises(RuntimeError, match="boom"):
        internal.hidden_states
    assert reader.calls == 1


def test_lazy_internal_requirement_revalidates_existing_cache():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FakeReader(rows)
    internal = make_lazy_internal("m", reader)

    assert len(internal.hidden_states) == 1
    internal.require("hidden_states", count=2)

    with pytest.raises(IncompleteInternalError, match="expected 2 entries, found 1"):
        internal.hidden_states

    assert reader.calls == 1
    with pytest.raises(IncompleteInternalError, match="expected 2 entries, found 1"):
        internal.hidden_states
    assert reader.calls == 2


def test_lazy_internal_copies_reusable_requirements():
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4)),
        _row("0:0", 1, 0, torch.ones(3, 4)),
    ]
    reqs = InternalRequirements().require("hidden_states", count=2)
    first = make_lazy_internal("m", FakeReader(rows), requirements=reqs)
    second = make_lazy_internal("m", FakeReader(rows), requirements=reqs)

    first.require("hidden_states", count=1)

    with pytest.raises(IncompleteInternalError, match="expected 1 entries, found 2"):
        first.hidden_states
    assert len(second.hidden_states) == 2
