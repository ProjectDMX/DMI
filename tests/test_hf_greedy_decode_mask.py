"""Decode steps of ``generate_greedy_with_monitoring`` must carry a mask.

Prefill was given ``attention_mask``; the decode kwargs were not. HF builds an
all-ones causal mask over the whole cache when none is supplied, so for a
left-padded batch every decode step attended to the pad positions' K/V and the
greedy tokens diverged from ``model.generate()``. Verified previously on a
real model: supplying the grown mask at decode restored exact parity, while
correcting ``position_ids`` alone did not.

These tests pin the mechanism -- what the decode step is handed -- with a
scripted stub, so they need no weights.
"""
import pytest
import torch

from dmi.adapters.huggingface.generation import generate_greedy_with_monitoring


class _RecordingCausalLM(torch.nn.Module):
    """Emits a fixed token per forward and records every call's kwargs."""

    def __init__(self, script, vocab_size=16):
        super().__init__()
        self._script = list(script)
        self._vocab_size = vocab_size
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
        logits_to_keep=0,
        cache_position=None,
    ):
        self.calls.append({
            "input_ids": input_ids,
            "attention_mask": None if attention_mask is None else attention_mask.clone(),
        })
        batch, seq_len = input_ids.shape
        token = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        logits = torch.zeros(batch, seq_len, self._vocab_size, device=input_ids.device)
        logits[:, -1, token] = 1.0
        return type("Out", (), {"logits": logits, "past_key_values": None})()


def _left_padded_batch():
    """Row 0 carries two left-pad positions; row 1 is full width."""
    ids = torch.tensor([[0, 0, 5, 6], [1, 2, 3, 4]], device="cuda")
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], device="cuda")
    return ids, mask


@pytest.mark.gpu
def test_every_decode_step_receives_a_mask():
    """The regression: decode was called with no mask at all."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = _RecordingCausalLM([3, 4, 5, 6]).cuda()
    ids, mask = _left_padded_batch()

    generate_greedy_with_monitoring(model, ids, mask, max_new_tokens=4)

    decode_calls = model.calls[1:]
    assert len(decode_calls) == 3
    for call in decode_calls:
        assert call["attention_mask"] is not None, "decode step ran with no mask"


@pytest.mark.gpu
def test_the_decode_mask_grows_by_one_position_per_step():
    """Width must track the cache, else the mask does not describe it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = _RecordingCausalLM([3, 4, 5, 6]).cuda()
    ids, mask = _left_padded_batch()
    prompt_width = ids.shape[1]

    generate_greedy_with_monitoring(model, ids, mask, max_new_tokens=4)

    widths = [c["attention_mask"].shape[1] for c in model.calls[1:]]
    assert widths == [prompt_width + 1, prompt_width + 2, prompt_width + 3]


@pytest.mark.gpu
def test_the_decode_mask_keeps_the_prompt_padding_and_admits_new_tokens():
    """The point of the fix: pad positions stay masked OUT while every
    generated position is masked IN."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = _RecordingCausalLM([3, 4, 5, 6]).cuda()
    ids, mask = _left_padded_batch()
    prompt_width = ids.shape[1]

    generate_greedy_with_monitoring(model, ids, mask, max_new_tokens=4)

    for call in model.calls[1:]:
        grown = call["attention_mask"]
        # Prompt columns are the caller's mask, untouched.
        assert torch.equal(grown[:, :prompt_width], mask)
        # Every generated column is attended by both rows.
        assert bool(grown[:, prompt_width:].all())
