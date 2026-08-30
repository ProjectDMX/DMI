"""Unit tests for dmi.storage.reassembly -- pure CPU torch, no DB."""
import pytest
import torch

from dmi.storage.reassembly import (
    OffloadedSegments,
    TorchOffloadedSegmentsAttnMatrix,
    TorchOffloadedSegmentsOnDim,
    get_delta_token_len,
    merge_segments,
    parse_internal_id,
    segment_manager,
)

pytestmark = pytest.mark.cpu


# --- parse_internal_id ----------------------------------------------------------


def test_parse_internal_id_blocks_prefix():
    assert parse_internal_id("blocks.3.rest") == (3, "blocks.rest")


def test_parse_internal_id_blocks_prefix_multi_segment():
    assert parse_internal_id("blocks.12.attn.hook_pattern") == (12, "blocks.attn.hook_pattern")


def test_parse_internal_id_fallback():
    assert parse_internal_id("hook_embed") == (-1, "hook_embed")


# --- get_delta_token_len ----------------------------------------------------------

_SHAPE = (7, 11, 13, 17)


@pytest.mark.parametrize(
    "act_name, have_batch_dim, expected",
    [
        ("blocks.attn.hook_attn_scores", False, _SHAPE[1]),
        ("blocks.attn.hook_attn_scores", True, _SHAPE[2]),
        ("blocks.attn.hook_pattern", False, _SHAPE[1]),
        ("blocks.attn.hook_pattern", True, _SHAPE[2]),
        ("blocks.hook_resid_pre", False, _SHAPE[0]),
        ("blocks.hook_resid_pre", True, _SHAPE[1]),
    ],
)
def test_get_delta_token_len(act_name, have_batch_dim, expected):
    assert get_delta_token_len(_SHAPE, act_name, have_batch_dim) == expected


# --- OffloadedSegments ABC pass bodies --------------------------------------------


class _Passthrough(OffloadedSegments):
    """Concrete subclass delegating to the abstract pass bodies."""

    def append(self, value):
        return super().append(value)

    def extend(self, segments):
        return super().extend(segments)

    def read_and_merge(self):
        return super().read_and_merge()

    @property
    def size(self):
        return super().size


def test_abstract_pass_bodies_are_reachable_via_super():
    seg = _Passthrough()

    assert seg.append(torch.ones(1)) is None
    assert seg.extend([]) is None
    assert seg.read_and_merge() is None
    assert seg.size is None


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        OffloadedSegments()


# --- TorchOffloadedSegmentsOnDim ----------------------------------------------------


def test_on_dim_rejects_negative_drop_token_cnt():
    with pytest.raises(AssertionError):
        TorchOffloadedSegmentsOnDim(token_dim=0, drop_token_cnt_to=-1)


def test_on_dim_append_returns_byte_size():
    seg = TorchOffloadedSegmentsOnDim(token_dim=0)

    assert seg.append(torch.ones(2, 3, dtype=torch.float32)) == 2 * 3 * 4
    assert seg.append(torch.ones(1, 3, dtype=torch.float16)) == 1 * 3 * 2


def test_on_dim_extend_from_segments_object_and_list():
    donor = TorchOffloadedSegmentsOnDim(token_dim=0)
    donor.append(torch.ones(1, 2))

    seg = TorchOffloadedSegmentsOnDim(token_dim=0)
    seg.extend(donor)
    seg.extend([torch.ones(2, 2) * 3])

    merged = seg.read_and_merge()
    assert tuple(merged.shape) == (3, 2)
    assert torch.equal(merged[0], torch.ones(2))
    assert torch.equal(merged[1:], torch.ones(2, 2) * 3)


def test_on_dim_read_and_merge_empty_returns_none():
    assert TorchOffloadedSegmentsOnDim(token_dim=0).read_and_merge() is None


def test_on_dim_read_and_merge_plain_concat_replaces_list():
    seg = TorchOffloadedSegmentsOnDim(token_dim=0)
    seg.append(torch.ones(2, 4))
    seg.append(torch.ones(3, 4) * 2)

    merged = seg.read_and_merge()

    assert tuple(merged.shape) == (5, 4)
    assert seg._tensor_list == [merged]
    # Merging again is a no-op concat of the single kept tensor.
    assert torch.equal(seg.read_and_merge(), merged)


def test_on_dim_read_and_merge_drops_tokens_past_cap():
    seg = TorchOffloadedSegmentsOnDim(token_dim=0, drop_token_cnt_to=2)
    seg.append(torch.arange(4, dtype=torch.float32).reshape(4, 1))

    merged = seg.read_and_merge()

    assert tuple(merged.shape) == (2, 1)
    assert torch.equal(merged, torch.tensor([[0.0], [1.0]]))


def test_on_dim_cap_larger_than_tokens_keeps_everything():
    seg = TorchOffloadedSegmentsOnDim(token_dim=0, drop_token_cnt_to=10)
    seg.append(torch.ones(3, 2))

    assert tuple(seg.read_and_merge().shape) == (3, 2)


def test_on_dim_size_sums_pending_bytes():
    seg = TorchOffloadedSegmentsOnDim(token_dim=0)
    assert seg.size == 0
    seg.append(torch.ones(2, 3, dtype=torch.float32))
    seg.append(torch.ones(1, 3, dtype=torch.float32))
    assert seg.size == (6 + 3) * 4


# --- TorchOffloadedSegmentsAttnMatrix -----------------------------------------------


def _attn(**kwargs) -> TorchOffloadedSegmentsAttnMatrix:
    # No batch dim: [heads, query_tokens, key_tokens].
    return TorchOffloadedSegmentsAttnMatrix(
        token_dim_incremental=1,
        token_dim_sum_to_now=2,
        fill_nan_value=0.0,
        **kwargs,
    )


def test_attn_rejects_negative_drop_token_cnt():
    with pytest.raises(AssertionError):
        _attn(drop_token_cnt_to=-1)


def test_attn_append_returns_byte_size():
    seg = _attn()

    assert seg.append(torch.ones(2, 1, 3, dtype=torch.float32)) == 2 * 1 * 3 * 4


def test_attn_extend_from_segments_object_and_list():
    donor = _attn()
    donor.append(torch.ones(1, 2, 2))

    seg = _attn()
    seg.extend(donor)
    seg.extend([torch.ones(1, 1, 3) * 2])

    assert len(seg._tensor_list) == 2


def test_attn_read_and_merge_empty_returns_none():
    assert _attn().read_and_merge() is None


def test_attn_read_and_merge_pads_earlier_chunks_to_full_width():
    prefill = torch.ones(1, 2, 2)
    decode = torch.ones(1, 1, 3) * 2
    seg = _attn()
    seg.extend([prefill, decode])

    merged = seg.read_and_merge()

    assert tuple(merged.shape) == (1, 3, 3)
    assert torch.equal(merged[:, :2, :2], prefill)
    assert torch.equal(merged[:, :2, 2], torch.zeros(1, 2))  # fill value pads keys
    assert torch.equal(merged[:, 2:, :], decode)
    assert seg._tensor_list == [merged]


def test_attn_read_and_merge_uses_fill_value_for_padding():
    seg = TorchOffloadedSegmentsAttnMatrix(
        token_dim_incremental=1,
        token_dim_sum_to_now=2,
        fill_nan_value=float("-inf"),
    )
    seg.extend([torch.ones(1, 1, 1), torch.ones(1, 1, 2)])

    merged = seg.read_and_merge()

    assert torch.isneginf(merged[0, 0, 1])


def test_attn_read_and_merge_narrows_chunk_wider_than_cap():
    # Single chunk wider than the cap on the sum-to-now dim: it is narrowed
    # there, then the merged matrix is narrowed on the incremental dim too.
    seg = _attn(drop_token_cnt_to=2)
    seg.append(torch.arange(9, dtype=torch.float32).reshape(1, 3, 3))

    merged = seg.read_and_merge()

    assert tuple(merged.shape) == (1, 2, 2)
    assert torch.equal(merged, torch.tensor([[[0.0, 1.0], [3.0, 4.0]]]))


def test_attn_read_and_merge_capped_growth_across_chunks():
    prefill = torch.ones(1, 2, 2)
    decode = torch.ones(1, 1, 3) * 2
    seg = _attn(drop_token_cnt_to=2)
    seg.extend([prefill, decode])

    merged = seg.read_and_merge()

    assert tuple(merged.shape) == (1, 2, 2)
    assert torch.equal(merged, prefill)


def test_attn_size_sums_pending_bytes():
    seg = _attn()
    assert seg.size == 0
    seg.append(torch.ones(1, 2, 2, dtype=torch.float32))
    seg.append(torch.ones(1, 1, 3, dtype=torch.float32))
    assert seg.size == (4 + 3) * 4


# --- segment_manager and merge_segments ---------------------------------------------


def test_segment_manager_attn_scores_branch():
    manager = segment_manager("blocks.attn.hook_attn_scores")

    assert isinstance(manager, TorchOffloadedSegmentsAttnMatrix)
    assert manager._td_inc == 1
    assert manager._td_sum == 2
    assert manager._fill_val == float("-inf")


def test_segment_manager_pattern_branch():
    manager = segment_manager("blocks.attn.hook_pattern", have_batch_dim=True)

    assert isinstance(manager, TorchOffloadedSegmentsAttnMatrix)
    assert manager._td_inc == 2
    assert manager._td_sum == 3
    assert manager._fill_val == 0.0


def test_segment_manager_else_branch_on_dim():
    manager = segment_manager("blocks.hook_resid_pre", drop_token_cnt_to=5)

    assert isinstance(manager, TorchOffloadedSegmentsOnDim)
    assert manager._token_dim == 0
    assert manager._drop_token_cnt_to == 5


def test_segment_manager_else_branch_with_batch_dim():
    manager = segment_manager("blocks.hook_mlp_out", have_batch_dim=True)

    assert isinstance(manager, TorchOffloadedSegmentsOnDim)
    assert manager._token_dim == 1


def test_merge_segments_end_to_end():
    merged = merge_segments(
        [torch.ones(2, 4), torch.ones(1, 4) * 2],
        "blocks.hook_resid_pre",
    )

    assert tuple(merged.shape) == (3, 4)
    assert torch.equal(merged[2], torch.full((4,), 2.0))
