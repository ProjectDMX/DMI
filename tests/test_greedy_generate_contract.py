"""Contract tests for the manual HuggingFace greedy-generation loop.

The scripted model returns deterministic logits on CPU.  CUDA synchronization
is replaced with a no-op, keeping token-count, EOS, validation, and timing
semantics independent from model weights and GPU availability.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

try:
    from integration.hf_adapter import (
        GreedyGenerateTimings,
        generate_greedy_with_monitoring,
    )
    _NATIVE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on build environment
    GreedyGenerateTimings = None
    generate_greedy_with_monitoring = None
    _NATIVE_IMPORT_ERROR = exc


pytestmark = [
    pytest.mark.native_backend,
    pytest.mark.skipif(
        _NATIVE_IMPORT_ERROR is not None,
        reason=f"DMI native backend required: {_NATIVE_IMPORT_ERROR}",
    ),
]


class _ScriptedModel(torch.nn.Module):
    def __init__(self, token_steps: list[list[int]], vocab_size: int = 32) -> None:
        super().__init__()
        self.token_steps = token_steps
        self.vocab_size = vocab_size
        self.calls = []
        self.dtype = torch.float32
        self.config = SimpleNamespace()

    def forward(self, input_ids, **kwargs):
        call_index = len(self.calls)
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "kwargs": dict(kwargs),
            }
        )
        if call_index >= len(self.token_steps):
            raise AssertionError("model was called beyond its scripted steps")
        next_tokens = self.token_steps[call_index]
        if len(next_tokens) != int(input_ids.shape[0]):
            raise AssertionError("scripted batch size does not match input")
        logits = torch.full(
            (len(next_tokens), 1, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
        )
        for row, token_id in enumerate(next_tokens):
            logits[row, 0, token_id] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=object())


@pytest.fixture(autouse=True)
def no_cuda_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def _generate(
    token_steps,
    *,
    max_new_tokens,
    min_new_tokens=0,
    eos_token_id=None,
    logits_to_keep=0,
    timings=None,
):
    model = _ScriptedModel(token_steps)
    batch = len(token_steps[0]) if token_steps else 1
    input_ids = torch.ones((batch, 2), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = generate_greedy_with_monitoring(
        model,
        input_ids,
        attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=eos_token_id,
        logits_to_keep=logits_to_keep,
        monitoring=False,
        timings=timings,
    )
    return model, [tokens.tolist() for tokens in result]


def test_generates_exactly_one_token():
    model, result = _generate([[7]], max_new_tokens=1)

    assert result == [[7]]
    assert len(model.calls) == 1


def test_without_eos_generates_exact_maximum():
    model, result = _generate([[3], [4], [5]], max_new_tokens=3)

    assert result == [[3, 4, 5]]
    assert len(model.calls) == 3


def test_logits_to_keep_is_forwarded_to_every_model_call():
    model, result = _generate(
        [[3], [4], [5]], max_new_tokens=3, logits_to_keep=1
    )

    assert result == [[3, 4, 5]]
    assert [call["kwargs"]["logits_to_keep"] for call in model.calls] == [1, 1, 1]


def test_timings_describe_the_completed_generation():
    timings = GreedyGenerateTimings()
    model, result = _generate(
        [[3, 4], [5, 6]],
        max_new_tokens=2,
        timings=timings,
    )

    assert result == [[3, 5], [4, 6]]
    assert len(model.calls) == 2
    assert timings.batch_size == 2
    assert timings.prefill_tokens == 2
    assert timings.decode_steps == 1
    assert len(timings.step_ms) == 1
    assert timings.total_ms >= timings.prefill_ms >= 0
    assert timings.total_ms >= timings.decode_ms >= 0


@pytest.mark.parametrize(
    ("max_new_tokens", "min_new_tokens", "message"),
    [
        (-1, 0, "max_new_tokens"),
        (1, -1, "min_new_tokens"),
        (1, 2, "min_new_tokens"),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason="known bug: greedy generation does not validate token limits",
)
def test_invalid_token_limits_are_rejected(
    max_new_tokens, min_new_tokens, message
):
    with pytest.raises(ValueError, match=message):
        _generate(
            [[7]],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )


@pytest.mark.xfail(
    strict=True,
    reason="known bug: max_new_tokens=0 still performs prefill and returns a token",
)
def test_zero_maximum_returns_empty_sequences_without_calling_model():
    model, result = _generate([[7]], max_new_tokens=0)

    assert result == [[]]
    assert model.calls == []


@pytest.mark.xfail(
    strict=True,
    reason="known bug: EOS from the prefill result is not checked before decode",
)
def test_first_generated_eos_stops_without_an_extra_model_call():
    model, result = _generate(
        [[2], [9]],
        max_new_tokens=2,
        eos_token_id=2,
    )

    assert result == [[2]]
    assert len(model.calls) == 1


@pytest.mark.xfail(
    strict=True,
    reason="known bug: EOS at min_new_tokens has an off-by-one stop check",
)
def test_eos_at_minimum_boundary_stops_immediately():
    model, result = _generate(
        [[5], [2], [9]],
        max_new_tokens=3,
        min_new_tokens=2,
        eos_token_id=2,
    )

    assert result == [[5, 2]]
    assert len(model.calls) == 2


@pytest.mark.xfail(
    strict=True,
    reason="known bug: final truncation honors EOS that occurred before the minimum",
)
def test_eos_before_minimum_does_not_truncate_the_result():
    model, result = _generate(
        [[2], [6], [7]],
        max_new_tokens=3,
        min_new_tokens=3,
        eos_token_id=2,
    )

    assert result == [[2, 6, 7]]
    assert len(model.calls) == 3


@pytest.mark.xfail(
    strict=True,
    reason="known bug: first-step EOS is not latched per batch row",
)
def test_batch_rows_finish_independently_without_extra_decode():
    model, result = _generate(
        [[2, 4], [9, 2], [9, 9]],
        max_new_tokens=3,
        eos_token_id=2,
    )

    assert result == [[2], [4, 2]]
    assert len(model.calls) == 2
