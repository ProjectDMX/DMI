"""Unit tests for monitoring.internal_mapper -- pure reassembly logic, no DB."""
import pytest
import torch

from monitoring.internal_mapper import get_internal

ACT = "blocks.hook_resid_pre"


class FakeReader:
    """Returns canned (key, tensor) rows regardless of the prefix."""
    def __init__(self, rows):
        self._rows = rows

    def prefix_get(self, prefix):
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


def test_source_accepts_out_or_model_id():
    rows = [_row("0:0", 0, 0, torch.ones(3, 4))]
    reader = FakeReader(rows)

    class Out:
        model_id = "m"

    assert get_internal(Out(), reader).available == ["hidden_states"]
    assert get_internal("m", reader).available == ["hidden_states"]
