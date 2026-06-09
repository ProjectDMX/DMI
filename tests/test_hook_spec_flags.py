"""Tests for HookSpec.allow_token_cnt_mismatch and its plumbing into
TensorMeta.flags via RingTransport.pre_push_all_metas.

The runtime exercise of this flag (consumer-side dim-0 recovery in
p2p_thread.cpp) is covered by end-to-end paths once a spec sets
allow_token_cnt_mismatch=True.  No dense / non-EP spec sets it today,
so these tests guard the plumbing only -- the flag's payload effect
will be exercised when MoE/EP specs that need it land.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from monitoring.ring_transport import HookSpec, ModelShapeConfig, RingTransport

pytestmark = pytest.mark.cpu


def test_hook_spec_flag_defaults_false():
    spec = HookSpec(hook_type=0, module=nn.Identity())
    assert spec.allow_token_cnt_mismatch is False


def test_hook_spec_flag_settable():
    spec = HookSpec(
        hook_type=0, module=nn.Identity(),
        allow_token_cnt_mismatch=True,
    )
    assert spec.allow_token_cnt_mismatch is True


def test_pre_push_all_metas_passes_flags_list():
    """pre_push_all_metas must build a flags list parallel to hook_types,
    with 1 for specs whose allow_token_cnt_mismatch=True and 0 otherwise."""
    transport = RingTransport.__new__(RingTransport)
    transport.null_offload = False
    transport._current_model_id = "test"
    transport._current_tp_rank = 0
    transport._current_dp_rank = 0
    transport._current_ep_rank = 0
    transport._current_pp_rank = 0
    transport._current_flattened = False
    transport._current_req_ids = ["r0"]
    transport._current_token_ranges = [(0, 4)]
    transport._current_dim0_offsets = [0]
    transport._current_kv_offsets = [0]
    transport._model_cfg = ModelShapeConfig(
        hidden_dim=8,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        intermediate_dim=16,
        vocab_size=10,
    )
    transport._active_specs = [
        HookSpec(hook_type=0, module=nn.Identity(), layer_no=0,
                 allow_token_cnt_mismatch=False),
        HookSpec(hook_type=0, module=nn.Identity(), layer_no=1,
                 allow_token_cnt_mismatch=True),
    ]
    transport._ring_engine = MagicMock()

    transport.pre_push_all_metas(batch=1, q_len=4, kv_dim=4)

    transport._ring_engine.push_all_metas.assert_called_once()
    call = transport._ring_engine.push_all_metas.call_args
    flags = call.args[4] if len(call.args) > 4 else call.kwargs["flags"]
    assert flags == [0, 1]
