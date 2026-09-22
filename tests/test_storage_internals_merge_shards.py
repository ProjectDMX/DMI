"""``merge_shards=True`` joins a TP-sharded activation back into one tensor.

Under tensor parallelism a sharded hook is written once per rank, each rank
holding a different slice of the SAME tokens (``filter_by_tp_rank`` keeps
sharded hooks on every rank; all ranks write one table with ``shard_rank``
telling the rows apart).  Reassembly concatenates along the TOKEN axis, so it
cannot merge those rows and refuses the collision by name.

``merge_shards`` is the opt-in that joins them along the axis TP actually
split, leaving the refusal as the default for callers who do not know their
data is sharded.  The axis differs by layout, which is the substance of these
tests: an attention matrix row is ``[heads, q, kv]`` and splits on dim 0,
while q/k/v/z/mlp_post are ``[tokens, features]`` and split on the trailing
axis.

What must NOT be merged is tested here too: two rows from the SAME rank are a
duplicate capture, not a TP split, and joining them would fabricate a tensor
wider than the model ever produced.
"""
import pytest
import torch

from dmi.storage.internals import get_internal

pytestmark = pytest.mark.cpu

Z_ACT = "blocks.attn.hook_z"
PATTERN_ACT = "blocks.attn.hook_pattern"


class _FakeReader:
    def __init__(self, rows):
        self._rows = rows

    def prefix_get(self, prefix):
        return self._rows


def _row(req, layer, start, tensor, *, shard_rank, act, token_len=None):
    span = tensor.shape[0] if token_len is None else token_len
    key = ("m", req, act, layer, shard_rank, start, start + span)
    return (key, tensor)


def test_two_shards_of_a_feature_split_join_on_the_trailing_axis():
    """z is [tokens, features]; TP splits features, so the join is dim -1."""
    rank0 = torch.ones(3, 4)
    rank1 = torch.ones(3, 4) * 2
    rows = [
        _row("0:0", 0, 0, rank0, shard_rank=0, act=Z_ACT),
        _row("0:0", 0, 0, rank1, shard_rank=1, act=Z_ACT),
    ]

    value = get_internal("m", _FakeReader(rows), merge_shards=True).attention_values

    merged = value[0]  # tuple is per layer; [batch, tokens, features]
    assert merged.shape == (1, 3, 8), "features must double, tokens must not"
    assert torch.equal(merged[0, :, :4], rank0)
    assert torch.equal(merged[0, :, 4:], rank1)


def test_shards_join_in_rank_order_not_arrival_order():
    """Rank r holds the r-th slice, so the sort is what makes the join the
    inverse of the split.  Rows arrive here in the wrong order on purpose."""
    rank0 = torch.ones(2, 3)
    rank1 = torch.ones(2, 3) * 2
    rows = [
        _row("0:0", 0, 0, rank1, shard_rank=1, act=Z_ACT),
        _row("0:0", 0, 0, rank0, shard_rank=0, act=Z_ACT),
    ]

    merged = get_internal(
        "m", _FakeReader(rows), merge_shards=True).attention_values[0]

    assert torch.equal(merged[0, :, :3], rank0), "rank 0 must come first"
    assert torch.equal(merged[0, :, 3:], rank1)


def test_an_attention_matrix_joins_on_the_head_axis():
    """[heads, q, kv]: TP splits heads, so the join is dim 0, not dim -1."""
    rank0 = torch.ones(2, 3, 3)
    rank1 = torch.ones(2, 3, 3) * 2
    rows = [
        _row("0:0", 0, 0, rank0, shard_rank=0, act=PATTERN_ACT, token_len=3),
        _row("0:0", 0, 0, rank1, shard_rank=1, act=PATTERN_ACT, token_len=3),
    ]

    merged = get_internal(
        "m", _FakeReader(rows), merge_shards=True).attentions[0]

    # [batch, heads, q, kv] -- heads doubled, the token axes untouched.
    assert merged.shape == (1, 4, 3, 3)
    assert torch.equal(merged[0, :2], rank0)
    assert torch.equal(merged[0, 2:], rank1)


def test_merged_shards_still_concatenate_along_tokens():
    """Both joins compose: shards widen features, chunks extend tokens."""
    rows = [
        _row("0:0", 0, 0, torch.ones(2, 4), shard_rank=0, act=Z_ACT),
        _row("0:0", 0, 0, torch.ones(2, 4) * 2, shard_rank=1, act=Z_ACT),
        _row("0:0", 0, 2, torch.ones(3, 4) * 3, shard_rank=0, act=Z_ACT),
        _row("0:0", 0, 2, torch.ones(3, 4) * 4, shard_rank=1, act=Z_ACT),
    ]

    merged = get_internal(
        "m", _FakeReader(rows), merge_shards=True).attention_values[0]

    assert merged.shape == (1, 5, 8)
    assert torch.equal(merged[0, :2, :4], torch.ones(2, 4))
    assert torch.equal(merged[0, 2:, 4:], torch.ones(3, 4) * 4)


def test_an_unsharded_field_is_unchanged_by_merge_shards():
    """No collision means nothing to join; the flag must be inert."""
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4), shard_rank=0, act=Z_ACT),
    ]

    plain = get_internal("m", _FakeReader(rows)).attention_values[0]
    merged = get_internal(
        "m", _FakeReader(rows), merge_shards=True).attention_values[0]

    assert torch.equal(plain, merged)


def test_merge_shards_and_shard_rank_together_are_refused():
    """They are opposite answers to the same question -- selecting one rank's
    slice versus joining them all -- so silently letting one win is wrong."""
    rows = [_row("0:0", 0, 0, torch.ones(3, 4), shard_rank=0, act=Z_ACT)]

    with pytest.raises(ValueError, match="mutually exclusive"):
        get_internal("m", _FakeReader(rows), shard_rank=0, merge_shards=True)


def test_a_repeated_shard_rank_is_still_refused_under_merge_shards():
    """Two rows from ONE rank are a duplicate capture, not a TP split.
    Joining them would fabricate a tensor wider than the model produced."""
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4), shard_rank=0, act=Z_ACT),
        _row("0:0", 0, 0, torch.ones(3, 4) * 2, shard_rank=0, act=Z_ACT),
    ]

    with pytest.raises(RuntimeError, match="repeated shard_rank") as excinfo:
        get_internal("m", _FakeReader(rows), merge_shards=True).attention_values

    assert "0:0" in str(excinfo.value), "must name the request"


def test_inconsistent_rank_sets_across_start_tokens_are_refused():
    """If one start token is missing a rank's rows the merged tensors differ
    in width, and the token concatenation would throw or ragged-join."""
    rows = [
        _row("0:0", 0, 0, torch.ones(2, 4), shard_rank=0, act=Z_ACT),
        _row("0:0", 0, 0, torch.ones(2, 4) * 2, shard_rank=1, act=Z_ACT),
        _row("0:0", 0, 2, torch.ones(2, 4) * 3, shard_rank=0, act=Z_ACT),
    ]

    with pytest.raises(RuntimeError, match="inconsistent shard_rank"):
        get_internal("m", _FakeReader(rows), merge_shards=True).attention_values


def test_the_default_still_refuses_the_collision():
    """merge_shards must be opt-in: the refusal #132 added is the default."""
    rows = [
        _row("0:0", 0, 0, torch.ones(3, 4), shard_rank=0, act=Z_ACT),
        _row("0:0", 0, 0, torch.ones(3, 4) * 2, shard_rank=1, act=Z_ACT),
    ]

    with pytest.raises(RuntimeError, match="duplicate"):
        get_internal("m", _FakeReader(rows)).attention_values
