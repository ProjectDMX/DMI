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

    def forward(self, input_ids, position_ids=None, **kwargs):
        # position_ids must be a declared parameter: the loop probes
        # inspect.signature(model.forward) to decide whether to pass it at all,
        # so a **kwargs-only double silently disables every position_ids path.
        call_index = len(self.calls)
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "position_ids": (
                    None if position_ids is None
                    else position_ids.detach().clone()
                ),
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
    attention_mask=None,
):
    model = _ScriptedModel(token_steps)
    batch = len(token_steps[0]) if token_steps else 1
    width = 2 if attention_mask is None else int(attention_mask.shape[1])
    input_ids = torch.ones((batch, width), dtype=torch.long)
    if attention_mask is None:
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
    reason=(
        "requested hardening, not a known defect: the loop silently accepts "
        "invalid limits (max_new_tokens<0 and max_new_tokens=0 both return "
        "exactly one token; min_new_tokens>max_new_tokens is ignored) instead "
        "of rejecting them.  Nobody has signed off on raising ValueError, so "
        "this pins a proposal -- it flips to XPASS if validation lands."
    ),
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


def test_eos_at_the_minimum_does_not_stop_generation():
    """The ``>`` in the stop check is correct; do not "fix" it to ``>=``.

    HuggingFace bans EOS outright until more than ``min_new_tokens`` tokens
    exist.  ``MinNewTokensLengthLogitsProcessor.__call__`` is::

        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        if new_tokens_length < self.min_new_tokens:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)

    and ``prompt_length_to_skip`` is the prompt length.  Before generating
    token *k* the prompt holds ``k-1`` new tokens, so EOS is suppressed while
    ``k <= min_new_tokens`` and the earliest legal EOS is token
    ``min_new_tokens + 1``.  ``tokens_generated > min_new_tokens`` encodes
    exactly that.  Stopping at ``>=`` would end one token earlier than the
    reference this repo is measured against.

    Verified against transformers 5.15.1.  This started life as a strict xfail
    calling the ``>`` an off-by-one bug -- it is not; it is HF parity.

    Only the model-call count is asserted.  The returned sequence is still
    truncated at the pre-minimum EOS, which is a real and separate defect --
    see ``test_result_is_never_shorter_than_min_new_tokens``.
    """
    model, _result = _generate(
        [[5], [2], [9]],
        max_new_tokens=3,
        min_new_tokens=2,
        eos_token_id=2,
    )

    # EOS lands at token 2 == min_new_tokens, so it must NOT stop: all three
    # steps run.
    assert len(model.calls) == 3


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


@pytest.mark.parametrize(
    "eos_token_id",
    [
        pytest.param([2, 5], id="list"),
        pytest.param(torch.tensor([2, 5]), id="tensor"),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug: the decode loop and the final truncation compare tokens "
        "against eos_token_id with scalar !=/==, so a multi-EOS set raises "
        "instead of stopping -- AttributeError for a list ('bool' has no "
        "'long'), RuntimeError for a Tensor (size mismatch).  "
        "HFAdaptor._normalize_eos accepts int, list[int] and Tensor, and "
        "generate_greedy_with_monitoring forwards eos_token_id straight into "
        "it, so the two halves disagree on the accepted type."
    ),
)
def test_multi_eos_token_ids_stop_generation(eos_token_id):
    model, result = _generate(
        [[4], [5], [9]],
        max_new_tokens=3,
        eos_token_id=eos_token_id,
    )

    assert result == [[4, 5]]
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    ("token_steps", "max_new_tokens", "min_new_tokens"),
    [
        # min == max: the configuration benchmark/bench_hf_transport.py uses.
        ([[4], [2], [9]], 3, 3),
        ([[2], [6], [7]], 3, 2),
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug: the stop check honors min_new_tokens but the final "
        "truncation does not, so a result can be shorter than the requested "
        "minimum.  Every generated step is still forwarded (and monitored), "
        "so the returned token count desyncs from the captured activation "
        "rows."
    ),
)
def test_result_is_never_shorter_than_min_new_tokens(
    token_steps, max_new_tokens, min_new_tokens
):
    model, result = _generate(
        token_steps,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=2,
    )

    assert len(model.calls) == max_new_tokens
    for row in result:
        assert len(row) >= min_new_tokens


def test_prefill_position_ids_come_from_the_attention_mask():
    """Left-padded rows must start at 0 on their first real token."""
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long)
    model, _result = _generate(
        [[3, 4], [5, 6]], max_new_tokens=2, attention_mask=mask
    )

    # cumsum(-1) - 1, with masked positions pinned to 0.
    assert model.calls[0]["position_ids"].tolist() == [[0, 0, 1], [0, 1, 2]]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known bug: eager decode computes seq_pos = Pmax + step + 1, so the "
        "first decode token is placed at Pmax+1 and skips a position.  The "
        "cuda_graphs path is correct -- it starts cache_pos at [Pmax] -- so "
        "the two decode paths disagree.  Only reachable with cuda_graphs=False; "
        "the sole caller (benchmark/bench_hf_transport.py) passes True."
    ),
)
def test_first_decode_position_id_continues_from_the_prompt():
    model, _result = _generate([[3], [4], [5]], max_new_tokens=3)

    # Prompt occupies positions 0..Pmax-1, so the first generated token is at
    # Pmax, and each later step advances by exactly one.
    assert model.calls[1]["position_ids"].tolist() == [[2]]
    assert model.calls[2]["position_ids"].tolist() == [[3]]
