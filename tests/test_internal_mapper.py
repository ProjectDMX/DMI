"""Unit tests for monitoring.internal_mapper -- pure reassembly logic, no DB."""
import pytest
import torch

from monitoring.internal_mapper import (
    IncompleteInternalError,
    InternalRequirements,
    get_internal,
    make_lazy_internal,
)

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


def _row(req, layer, start, tensor):
    # key = (model_id, request_id, act_name, layer_no, shard_rank, start, end)
    return ((("m", req, ACT, layer, 0, start, start + tensor.shape[0])), tensor)


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
