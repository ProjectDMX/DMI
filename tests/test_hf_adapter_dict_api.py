from types import SimpleNamespace

import pytest

from integration import hf_adapter
from monitoring.internal_mapper import InternalRequirements


def test_generate_with_monitoring_returns_impl_output_unchanged(monkeypatch):
    expected = object()

    def fake_impl(model, *args, **kwargs):
        return expected

    monkeypatch.setattr(hf_adapter, "_generate_with_monitoring_impl", fake_impl)

    assert hf_adapter.generate_with_monitoring(object(), max_new_tokens=1) is expected


def test_generate_with_monitoring_dict_forces_dict_and_attaches_lazy(monkeypatch):
    output = SimpleNamespace(sequences="tokens")
    captured = {}
    reader = object()
    requirements = InternalRequirements().require("hidden_states", count=2)

    def fake_impl(model, *args, **kwargs):
        captured.update(kwargs)
        return output, "run-id"

    monkeypatch.setattr(hf_adapter, "_generate_with_monitoring_impl", fake_impl)

    result = hf_adapter.generate_with_monitoring_dict(
        object(),
        max_new_tokens=1,
        reader=reader,
        internal_requirements=requirements,
    )

    assert result is output
    assert captured["return_dict_in_generate"] is True
    assert captured["_return_model_id"] is True
    assert "reader" not in captured
    assert "internal_requirements" not in captured
    assert repr(result.dmi_internal) == "<DMIInternal model_id='run-id' state=pending>"
    result.dmi_internal.require("hidden_states", count=1)
    assert requirements.expected_count("hidden_states") == 2


def test_generate_with_monitoring_dict_warns_when_false_is_overridden(monkeypatch):
    output = SimpleNamespace(sequences="tokens")
    captured = {}

    def fake_impl(model, *args, **kwargs):
        captured.update(kwargs)
        return output, "run-id"

    monkeypatch.setattr(hf_adapter, "_generate_with_monitoring_impl", fake_impl)

    with pytest.warns(UserWarning, match="overriding the supplied False value"):
        hf_adapter.generate_with_monitoring_dict(
            object(),
            return_dict_in_generate=False,
        )

    assert captured["return_dict_in_generate"] is True
