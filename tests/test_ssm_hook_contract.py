"""CPU contracts for architecture-neutral SSM branch hook types."""

from __future__ import annotations

import torch

import monitoring.ring_transport as ring
from monitoring.selection import resolve_hook_selection
from tests.compare_disk_vs_ch import _BUF_TO_CH_ACT


def _model_shape() -> ring.ModelShapeConfig:
    return ring.ModelShapeConfig(
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.bfloat16,
        vocab_size=128,
        intermediate_dim=96,
    )


def test_ssm_hooks_are_native_hidden_shape_branch_boundaries() -> None:
    definitions = {row[2]: row for row in ring._HOOK_DEFS}

    for short_name, hook_type in (
        ("ssm_in", ring.HOOK_TYPE_SSM_IN),
        ("ssm_out", ring.HOOK_TYPE_SSM_OUT),
    ):
        definition = definitions[short_name]
        assert definition[0] == hook_type
        assert definition[3] is True
        assert definition[4] == ring.GROUP_SSM
        assert definition[5] is False
        assert definition[6] == ring.SHAPE_HIDDEN
        assert definition[7] == ring.PP_ANY


def test_ssm_storage_names_are_derived_from_native_definitions() -> None:
    assert _BUF_TO_CH_ACT["ssm_in"] == "blocks.ssm.hook_in"
    assert _BUF_TO_CH_ACT["ssm_out"] == "blocks.ssm.hook_out"


def test_ssm_hooks_use_full_hidden_rows_in_both_layouts() -> None:
    config = _model_shape()

    for hook_type in (ring.HOOK_TYPE_SSM_IN, ring.HOOK_TYPE_SSM_OUT):
        assert ring._compute_hook_shape(hook_type, config, 0, 7, 0) == [
            7,
            64,
        ]
        assert ring._compute_hook_shape(hook_type, config, 3, 7, 0) == [
            3,
            7,
            64,
        ]


def test_full_selects_ssm_hooks_without_reclassifying_attention() -> None:
    selected = resolve_hook_selection("full")

    assert ring.HOOK_TYPE_SSM_IN in selected
    assert ring.HOOK_TYPE_SSM_OUT in selected
    assert ring._SSM_SUFFIXES == ("ssm.hook_in", "ssm.hook_out")
    assert set(ring._SSM_SUFFIXES).isdisjoint(ring._ATTN_SUFFIXES)
