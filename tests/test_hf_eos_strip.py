"""Unit tests for HFAdaptor's post-EOS strip in ``build_step_context``.

Phase 4.B verification gate.  Pure unit test against the strip logic --
no DB, no GPU, no real model.  Reuses the FakeEngine pattern from
``tests/test_adapter_protocol.py``.

The strip semantic under test:
  * Detection runs one step late by construction -- ``input_ids[:, -1]``
    at decode step k is the token *appended* by step k-1's argmax.
  * The activation that produced the first EOS is captured (prefill's
    last position predicted EOS; decode step 0 is the first one whose
    fed input is EOS, and that's the first stripped step).
  * Once a request latches, subsequent steps continue to emit
    zero-length ranges regardless of what HF feeds (HF overrides
    finished requests' output to pad).
"""
from __future__ import annotations

from typing import Optional

import pytest
import torch

from integration.hf_adapter import HFAdaptor


# ---------------------------------------------------------------------------
# Fakes (minimal -- HFAdaptor only touches a few fields)
# ---------------------------------------------------------------------------


class FakeTransport:
    null_offload = False
    force_eager = False
    _model_cfg = None
    _active_specs: list = []
    _using_forward_hooks = False


class FakeEngine:
    def __init__(self) -> None:
        self._ring_transport = FakeTransport()
        self._ring_engine = None
        self._auto_batch_group_id = 0

    def next_auto_group_id(self) -> int:
        gid = int(self._auto_batch_group_id)
        self._auto_batch_group_id += 1
        return gid


def _make_adaptor(eos_token_id=None, no_strip_right_pad=False) -> HFAdaptor:
    a = HFAdaptor(
        FakeEngine(), "test-model",
        no_strip_right_pad=no_strip_right_pad,
        eos_token_id=eos_token_id,
    )
    if eos_token_id is not None:
        a._eos_token_ids = HFAdaptor._normalize_eos(eos_token_id)
    return a


def _prefill_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "cache_position": None,
    }


def _decode_inputs(token_ids_per_req: list[int]) -> dict:
    return {
        "input_ids": torch.tensor([[t] for t in token_ids_per_req], dtype=torch.long),
        "attention_mask": None,
        "past_key_values": object(),  # any non-None
        "cache_position": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eos_normalize_int():
    assert HFAdaptor._normalize_eos(2) == frozenset({2})


def test_eos_normalize_list():
    assert HFAdaptor._normalize_eos([2, 5, 7]) == frozenset({2, 5, 7})


def test_eos_normalize_tensor():
    t = torch.tensor([2, 5, 7], dtype=torch.long)
    assert HFAdaptor._normalize_eos(t) == frozenset({2, 5, 7})


def test_eos_normalize_none():
    assert HFAdaptor._normalize_eos(None) == frozenset()


def test_post_eos_strip_one_step_late():
    """3-request batch: req 0 hits EOS at first decode step; req 1, 2 don't.

    Expected:
      * Prefill: all three get ``(0, real_len_i)`` ranges.
      * Decode step 0 (input = [EOS, t, t]): req 0 latches, gets ``(r0, r0)``.
        req 1, 2 advance normally.
      * Decode step 1 (input = anything for req 0; HF would feed pad):
        req 0 stays latched, still ``(r0, r0)``.  req 1, 2 continue.
    """
    EOS = 2
    a = _make_adaptor(eos_token_id=EOS)

    # Prefill: real lengths [1, 3, 4] -- request 0 has 3 left-pads.
    input_ids = torch.tensor(
        [[0, 0, 0, 9], [0, 5, 5, 5], [4, 4, 4, 4]], dtype=torch.long
    )
    mask = torch.tensor(
        [[0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long
    )
    ctx = a.build_step_context(_prefill_inputs(input_ids, mask))
    assert ctx is not None
    assert ctx.token_ranges == [(0, 1), (0, 3), (0, 4)]
    assert a._batch_starts == [1, 3, 4]
    assert a._batch_finished == [False, False, False]

    # Decode step 0: req 0 fed EOS (the first EOS). Latch for req 0.
    ctx = a.build_step_context(_decode_inputs([EOS, 100, 200]))
    assert ctx is not None
    assert ctx.token_ranges == [(1, 1), (3, 4), (4, 5)]
    assert a._batch_starts == [1, 4, 5]   # req 0's start unchanged
    assert a._batch_finished == [True, False, False]

    # Decode step 1: req 0 latched -> stays stripped.  Other requests advance.
    ctx = a.build_step_context(_decode_inputs([0, 101, 201]))  # any input for req 0
    assert ctx is not None
    assert ctx.token_ranges == [(1, 1), (4, 5), (5, 6)]
    assert a._batch_starts == [1, 5, 6]
    assert a._batch_finished == [True, False, False]

    # Decode step 2: still latched.
    ctx = a.build_step_context(_decode_inputs([0, 102, 202]))
    assert ctx is not None
    assert ctx.token_ranges == [(1, 1), (5, 6), (6, 7)]
    assert a._batch_finished == [True, False, False]


def test_no_strip_right_pad_disables_strip():
    """no_strip_right_pad=True keeps every decode row regardless of EOS."""
    EOS = 2
    a = _make_adaptor(eos_token_id=EOS, no_strip_right_pad=True)

    input_ids = torch.tensor(
        [[0, 0, 0, 9], [0, 5, 5, 5], [4, 4, 4, 4]], dtype=torch.long
    )
    mask = torch.tensor(
        [[0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long
    )
    a.build_step_context(_prefill_inputs(input_ids, mask))

    # Decode step 0: req 0 fed EOS, but no_strip_right_pad=True keeps it.
    ctx = a.build_step_context(_decode_inputs([EOS, 100, 200]))
    assert ctx is not None
    assert ctx.token_ranges == [(1, 2), (3, 4), (4, 5)]
    assert a._batch_starts == [2, 4, 5]


def test_empty_eos_set_skips_detection():
    """Empty _eos_token_ids -> no .tolist() sync, no latch, never strips."""
    a = _make_adaptor(eos_token_id=None)  # no EOS configured
    assert a._eos_token_ids == frozenset()

    input_ids = torch.tensor(
        [[0, 9], [4, 4]], dtype=torch.long
    )
    mask = torch.tensor(
        [[0, 1], [1, 1]], dtype=torch.long
    )
    a.build_step_context(_prefill_inputs(input_ids, mask))

    # Feed token id 2 (would be EOS if configured) -- with empty set, no latch.
    ctx = a.build_step_context(_decode_inputs([2, 5]))
    assert ctx is not None
    assert ctx.token_ranges == [(1, 2), (2, 3)]
    assert a._batch_finished == [False, False]


def test_multi_eos_any_match_latches():
    """Multi-EOS (frozenset of multiple ids): any match latches."""
    a = _make_adaptor(eos_token_id=[2, 7, 11])

    input_ids = torch.tensor([[9], [4], [3]], dtype=torch.long)
    mask = torch.tensor([[1], [1], [1]], dtype=torch.long)
    a.build_step_context(_prefill_inputs(input_ids, mask))

    # Step 0: req 0 fed 2, req 1 fed 7, req 2 fed something else.
    ctx = a.build_step_context(_decode_inputs([2, 7, 5]))
    assert ctx is not None
    assert a._batch_finished == [True, True, False]
    assert ctx.token_ranges == [(1, 1), (1, 1), (1, 2)]


def test_resolve_eos_chain():
    """attach-arg > constructor-arg > generation_config > config > empty."""
    class FakeCfg:
        eos_token_id = 99

    class FakeGenCfg:
        eos_token_id = 50

    class FakeModel:
        config = FakeCfg()
        generation_config = FakeGenCfg()

    a = HFAdaptor(FakeEngine(), "x")  # constructor arg = None
    model = FakeModel()

    # Auto-detect: generation_config wins over config.
    assert a._resolve_eos_token_ids(model, None) == frozenset({50})

    # attach-arg explicit wins over auto-detect.
    assert a._resolve_eos_token_ids(model, [11, 22]) == frozenset({11, 22})

    # Constructor-arg wins over auto-detect.
    a2 = HFAdaptor(FakeEngine(), "x", eos_token_id=33)
    assert a2._resolve_eos_token_ids(model, None) == frozenset({33})

    # attach-arg overrides constructor-arg.
    assert a2._resolve_eos_token_ids(model, 44) == frozenset({44})

    # No generation_config: fall back to config.
    class FakeModelNoGen:
        config = FakeCfg()

    assert a._resolve_eos_token_ids(FakeModelNoGen(), None) == frozenset({99})

    # Generation_config exists but eos is None: fall through to config.
    class FakeGenCfgNone:
        eos_token_id = None

    class FakeModelGenNone:
        config = FakeCfg()
        generation_config = FakeGenCfgNone()

    assert a._resolve_eos_token_ids(FakeModelGenNone(), None) == frozenset({99})

    # Nothing anywhere: empty frozenset.
    class FakeCfgNone:
        eos_token_id = None

    class FakeModelEmpty:
        config = FakeCfgNone()
        generation_config = FakeGenCfgNone()

    assert a._resolve_eos_token_ids(FakeModelEmpty(), None) == frozenset()


def test_batch_resize_resets_finished_latch():
    """Mid-call batch shrink -> reset clears _batch_finished, fresh detection."""
    EOS = 2
    a = _make_adaptor(eos_token_id=EOS)

    # B=3 prefill + decode where req 0 latches.
    input_ids = torch.tensor(
        [[0, 9], [4, 4], [4, 4]], dtype=torch.long
    )
    mask = torch.tensor(
        [[0, 1], [1, 1], [1, 1]], dtype=torch.long
    )
    a.build_step_context(_prefill_inputs(input_ids, mask))
    a.build_step_context(_decode_inputs([EOS, 100, 200]))
    assert a._batch_finished == [True, False, False]

    # B=2 decode (batch shrink mid-call) -- triggers reset because
    # len(current_ids) != batch_size.  After reset starts begin at 0,
    # then the decode step itself advances both to 1.
    ctx = a.build_step_context(_decode_inputs([100, 200]))
    assert ctx is not None
    assert a._batch_finished == [False, False]   # latch cleared
    assert a._batch_starts == [1, 1]              # advanced by this decode
