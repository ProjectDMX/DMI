"""Reassembly must refuse colliding chunks instead of comparing tensors.

``_reassemble_*`` grouped a request's chunks by ``(request, layer)`` only, then
called ``sorted(chunks)`` on ``(start_token, tensor)`` pairs. When two rows
share a start token Python falls through to comparing the TENSORS, so:

  * multi-element payloads raised ``RuntimeError: Boolean value of Tensor with
    more than one value is ambiguous`` -- an error naming nothing about the
    duplicate, the request or the shard; and
  * single-element payloads compared fine and were SILENTLY concatenated,
    merging two TP shards (or two runs) into one wrong tensor.

The collision is production-shaped: ``filter_by_tp_rank`` deliberately keeps
TP-sharded hooks on every rank, and the repo writes all ranks to one table
with ``shard_rank`` distinguishing the rows (tests/hf_compare_runner.py).
"""
import pytest
import torch

from dmi.storage.internals import get_internal

pytestmark = pytest.mark.cpu

ACT = "blocks.hook_resid_pre"


class _FakeReader:
    def __init__(self, rows):
        self._rows = rows

    def prefix_get(self, prefix):
        return self._rows


def _row(req, layer, start, tensor, shard_rank=0, act=ACT):
    key = ("m", req, act, layer, shard_rank, start, start + tensor.shape[0])
    return (key, tensor)


def test_two_tp_shards_at_one_start_token_are_refused_by_name():
    """The common case: multi-element payloads. Used to raise an opaque
    torch error; must now name what actually collided."""
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4), shard_rank=0),
        _row("0:0", 0, 0, torch.ones(3, 4) * 2, shard_rank=1),
    ]

    with pytest.raises(RuntimeError, match="duplicate") as excinfo:
        get_internal("m", _FakeReader(rows)).hidden_states

    message = str(excinfo.value)
    assert "0:0" in message, "must name the request"
    assert "shard_rank" in message, "must point at the likely cause"
    assert "ambiguous" not in message, "the opaque torch error must be gone"


def test_single_element_chunks_are_refused_rather_than_silently_merged():
    """The dangerous case. One-element tensors compare fine, so the old code
    concatenated two shards into a double-length tensor and returned it."""
    rows = [
        _row("0:0", 0, 0, torch.ones(1, 4), shard_rank=0),
        _row("0:0", 0, 0, torch.ones(1, 4) * 2, shard_rank=1),
    ]

    with pytest.raises(RuntimeError, match="duplicate"):
        get_internal("m", _FakeReader(rows)).hidden_states


def test_a_non_layered_field_refuses_the_same_collision():
    """_reassemble_global carries its own copy of the sort."""
    rows = [
        _row("0:0", 0, 0, torch.ones(2, 4), shard_rank=0, act="hook_embed"),
        _row("0:0", 0, 0, torch.ones(2, 4) * 2, shard_rank=1, act="hook_embed"),
    ]

    with pytest.raises(RuntimeError, match="duplicate"):
        get_internal("m", _FakeReader(rows)).embeddings


def test_distinct_start_tokens_still_reassemble_in_token_order():
    """Positive control: ordinary chunked capture is untouched, and chunks
    arriving out of order are still ordered by start token."""
    rows = [
        _row("0:0", 0, 2, torch.ones(2, 4) * 9),
        _row("0:0", 0, 0, torch.ones(2, 4)),
    ]

    hs = get_internal("m", _FakeReader(rows)).hidden_states

    assert tuple(hs[0].shape) == (1, 4, 4)
    assert torch.equal(hs[0][0, :2], torch.ones(2, 4))
    assert torch.equal(hs[0][0, 2:], torch.ones(2, 4) * 9)


def test_one_shard_alone_is_still_readable():
    """Refusing a collision must not refuse a single-shard TP run."""
    rows = [_row("0:0", 0, 0, torch.ones(3, 4), shard_rank=1)]

    hs = get_internal("m", _FakeReader(rows)).hidden_states

    assert tuple(hs[0].shape) == (1, 3, 4)
