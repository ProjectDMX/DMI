"""EOS handling in ``generate_greedy_with_monitoring``.

The loop compared the freshly sampled token against ``eos_token_id`` with
``!=`` and the finished sequences against it with ``==``.  Both are correct
only while ``eos_token_id`` is a scalar: torch returns a plain Python ``bool``
for ``tensor != list``, with no broadcasting and no error, so ``.long()`` and
``.nonzero()`` were called on a ``bool`` the moment a caller passed the list
form -- which is what ``generation_config.eos_token_id`` holds for Qwen and
Llama-3, and what the adapter this function forwards to documents accepting.
"""
import inspect

import pytest
import torch

from dmi.adapters.huggingface.generation import (
    _eos_id_tensor,
    generate_greedy_with_monitoring,
)


# --- the normalisation itself: pure CPU ---------------------------------------


@pytest.mark.cpu
def test_eos_id_tensor_normalises_every_documented_spelling():
    """int, list and tensor must all become one 1-D tensor of ids."""
    device = torch.device("cpu")

    assert _eos_id_tensor(None, device=device, dtype=torch.long) is None

    scalar = _eos_id_tensor(7, device=device, dtype=torch.long)
    assert scalar.tolist() == [7] and scalar.ndim == 1

    listed = _eos_id_tensor([7, 9], device=device, dtype=torch.long)
    assert listed.tolist() == [7, 9] and listed.ndim == 1

    tensored = _eos_id_tensor(
        torch.tensor([7, 9], dtype=torch.int32), device=device, dtype=torch.long)
    assert tensored.tolist() == [7, 9] and tensored.dtype is torch.long


@pytest.mark.cpu
def test_membership_against_the_normalised_ids_is_elementwise():
    """The property the old ``!=`` lacked: one answer per batch row.

    ``torch.tensor([5]) != [1, 2]`` is the Python bool ``True``; a bool has no
    ``.long()``, which is the crash.  Pinned here because it is a torch
    behaviour the fix depends on, not something this module controls.
    """
    tokens = torch.tensor([5, 9])
    ids = _eos_id_tensor([7, 9], device=torch.device("cpu"), dtype=torch.long)

    assert torch.isin(tokens, ids).tolist() == [False, True]
    assert isinstance(tokens != [7, 9], bool)  # the old expression, still a bool


# --- the real decode loop -----------------------------------------------------


class _ScriptedCausalLM(torch.nn.Module):
    """Emits a fixed token per forward, so the loop's stopping is observable."""

    def __init__(self, script, vocab_size=16):
        super().__init__()
        self._script = list(script)
        self._vocab_size = vocab_size
        self.calls = 0

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
        batch, seq_len = input_ids.shape
        token = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        logits = torch.zeros(
            batch, seq_len, self._vocab_size, device=input_ids.device)
        logits[:, -1, token] = 1.0
        return type("Out", (), {"logits": logits, "past_key_values": None})()


def _run(script, eos_token_id, max_new_tokens=5):
    model = _ScriptedCausalLM(script).cuda()
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    mask = torch.ones_like(ids)
    return generate_greedy_with_monitoring(
        model, ids, mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


@pytest.mark.gpu
def test_a_list_eos_token_id_stops_exactly_where_the_scalar_form_does():
    """The regression: the list form used to raise on the first decode step.

    Asserted against the scalar form rather than a hand-computed sequence, so
    the test pins equivalence rather than a transcription of the loop.
    """
    script = [3, 9, 4, 4, 4]

    scalar = _run(script, 9)
    listed = _run(script, [7, 9])

    assert [t.tolist() for t in listed] == [t.tolist() for t in scalar]
    assert scalar[0].tolist() == [3, 9]


@pytest.mark.gpu
def test_an_unreachable_eos_in_the_list_runs_to_max_new_tokens():
    """A list whose ids never appear must not stop early."""
    generated = _run([3, 4, 4, 4, 4], [7], max_new_tokens=5)

    assert generated[0].tolist() == [3, 4, 4, 4, 4]


@pytest.mark.gpu
def test_a_multi_element_tensor_eos_matches_the_list_form():
    """The tensor spelling the adapter documents, on the same script."""
    script = [3, 9, 4, 4, 4]

    listed = _run(script, [7, 9])
    tensored = _run(script, torch.tensor([7, 9]))

    assert [t.tolist() for t in tensored] == [t.tolist() for t in listed]


@pytest.mark.cpu
def test_generate_greedy_still_advertises_the_eos_argument():
    """Guards the signature the adapter forwards into (adapter.py accepts
    ``int``, ``list[int]`` or ``torch.Tensor``)."""
    params = inspect.signature(generate_greedy_with_monitoring).parameters
    assert "eos_token_id" in params
