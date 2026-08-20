"""CPU contracts for architecture-neutral short-convolution hooks."""

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


def test_conv_hooks_are_native_hidden_shape_layer_boundaries() -> None:
    definitions = {row[2]: row for row in ring.HOOK_DEFINITIONS}

    for short_name, hook_type in (
        ("conv_in", ring.HOOK_TYPE_CONV_IN),
        ("conv_out", ring.HOOK_TYPE_CONV_OUT),
    ):
        definition = definitions[short_name]
        assert definition[0] == hook_type
        assert definition[3] is True
        assert definition[4] == ring.GROUP_CONV
        assert definition[5] is False
        assert definition[6] == ring.SHAPE_HIDDEN
        assert definition[7] == ring.PP_ANY


def test_conv_hooks_use_full_hidden_rows_in_both_layouts() -> None:
    config = _model_shape()

    for hook_type in (ring.HOOK_TYPE_CONV_IN, ring.HOOK_TYPE_CONV_OUT):
        assert ring._compute_hook_shape(hook_type, config, 0, 7, 0) == [
            7,
            64,
        ]
        assert ring._compute_hook_shape(hook_type, config, 3, 7, 0) == [
            3,
            7,
            64,
        ]


def test_full_selects_conv_without_reclassifying_ssm() -> None:
    selected = resolve_hook_selection("full")

    assert ring.HOOK_TYPE_CONV_IN in selected
    assert ring.HOOK_TYPE_CONV_OUT in selected
    assert ring._CONV_SUFFIXES == ("conv.hook_in", "conv.hook_out")
    assert set(ring._CONV_SUFFIXES).isdisjoint(ring._SSM_SUFFIXES)
    assert set(ring._CONV_SUFFIXES).isdisjoint(ring._ATTN_SUFFIXES)


def test_conv_storage_names_are_derived_from_native_definitions() -> None:
    assert _BUF_TO_CH_ACT["conv_in"] == "blocks.conv.hook_in"
    assert _BUF_TO_CH_ACT["conv_out"] == "blocks.conv.hook_out"
