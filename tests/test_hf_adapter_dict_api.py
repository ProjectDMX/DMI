from types import SimpleNamespace

import pytest

from dmi.adapters.huggingface import generation as hf_generation
from dmi.storage.internals import InternalRequirements

pytestmark = pytest.mark.cpu


def test_generate_with_monitoring_returns_impl_output_unchanged(monkeypatch):
    expected = object()

    def fake_impl(model, *args, **kwargs):
        return expected

    monkeypatch.setattr(hf_generation, "_generate_with_monitoring_impl", fake_impl)

    assert hf_generation.generate_with_monitoring(object(), max_new_tokens=1) is expected


def test_generate_with_monitoring_dict_forces_dict_and_attaches_lazy(monkeypatch):
    output = SimpleNamespace(sequences="tokens")
    captured = {}
    reader = object()
    requirements = InternalRequirements().require("hidden_states", count=2)

    def fake_impl(model, *args, **kwargs):
        captured.update(kwargs)
        return output, "run-id", ("0:0",), {"0:0": ((0, 2),)}

    monkeypatch.setattr(hf_generation, "_generate_with_monitoring_impl", fake_impl)

    result = hf_generation.generate_with_monitoring_dict(
        object(),
        max_new_tokens=1,
        reader=reader,
        internal_requirements=requirements,
    )

    assert result is output
    assert captured["return_dict_in_generate"] is True
    assert captured["_return_internal_metadata"] is True
    assert "reader" not in captured
    assert "internal_requirements" not in captured
    assert repr(result.dmi_internal) == "<DMIInternal model_id='run-id' state=pending>"
    assert result.dmi_internal._request_ids == ("0:0",)
    assert result.dmi_internal._token_ranges == {"0:0": ((0, 2),)}
    result.dmi_internal.require("hidden_states", count=1)
    assert requirements.expected_count("hidden_states") == 2


def test_generate_with_monitoring_dict_warns_when_false_is_overridden(monkeypatch):
    output = SimpleNamespace(sequences="tokens")
    captured = {}

    def fake_impl(model, *args, **kwargs):
        captured.update(kwargs)
        return output, "run-id", (), {}

    monkeypatch.setattr(hf_generation, "_generate_with_monitoring_impl", fake_impl)

    with pytest.warns(UserWarning, match="overriding the supplied False value"):
        hf_generation.generate_with_monitoring_dict(
            object(),
            return_dict_in_generate=False,
        )

    assert captured["return_dict_in_generate"] is True


class _FakeRingEngine:
    def payload_cap(self):
        return 1

    def staging_cap(self):
        return 1


class _FakeAdaptor:
    """Minimal stand-in for HuggingFaceAdapter, forced over ring capacity."""

    def __init__(self, engine, model_id, **_):
        self.model_cfg = object()
        self.active_specs = ("spec",)
        self.ring_engine = _FakeRingEngine()
        self.request_ids_in_this_generate = []
        self.token_ranges_in_this_generate = {}

    def attach_model(self, target, **_):
        pass

    def detach_model(self, target):
        pass

    def decode_step_bytes(self, batch, kv_dim):
        return 1 << 30  # always overflows the fake caps


def _overflowing_generate_kwargs(monkeypatch, **extra):
    """Run the impl down the capacity-overflow branch; return generate()'s kwargs."""
    import torch

    seen = {}

    def fake_generate(*args, **kwargs):
        seen.update(kwargs)
        return "sequences"

    model = SimpleNamespace(
        generate=fake_generate,
        monitoring_engine=SimpleNamespace(
            _ring_transport=object(), _model_id="run-id",
        ),
        generation_config=None,
    )

    monkeypatch.setattr(hf_generation, "HuggingFaceAdapter", _FakeAdaptor)

    with pytest.warns(UserWarning):
        hf_generation._generate_with_monitoring_impl(
            model,
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            max_new_tokens=4,
            **extra,
        )
    return seen


def test_capacity_overflow_strips_both_compile_kwargs(monkeypatch):
    # Both pops must run even though compile_config alone already satisfies
    # `had_compile`; a short-circuiting `or` leaves cache_implementation in
    # kwargs, which contradicts disable_compile=True.
    seen = _overflowing_generate_kwargs(
        monkeypatch,
        compile_config=object(),
        cache_implementation="static",
    )

    assert "compile_config" not in seen
    assert "cache_implementation" not in seen
    assert seen["disable_compile"] is True


def test_capacity_overflow_strips_cache_implementation_alone(monkeypatch):
    seen = _overflowing_generate_kwargs(monkeypatch, cache_implementation="static")

    assert "cache_implementation" not in seen
    assert seen["disable_compile"] is True
