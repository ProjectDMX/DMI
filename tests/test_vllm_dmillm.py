"""Unit tests for DMILLM's per-request .dmi_internal tagging.

Covers the tagging logic only -- no GPU, ClickHouse, or vLLM engine. The lazy
proxy is constructed but never read, so nothing touches the store.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")  # vllm_adapter imports vllm at module load

from integration.vllm_adapter import _attach_dmi_internal, normalize_vllm_request_id


def _fake_outputs(*request_ids):
    return [SimpleNamespace(request_id=rid) for rid in request_ids]


def test_tags_each_output_with_lazy_internal():
    outs = _fake_outputs("req-a", "req-b")
    _attach_dmi_internal(outs, "demo_vllm", reader=None)
    assert all(hasattr(o, "dmi_internal") for o in outs)
    assert outs[0].dmi_internal._model_id == "demo_vllm"


def test_per_request_isolation():
    outs = _fake_outputs("req-a", "req-b")
    _attach_dmi_internal(outs, "m", reader=None)
    # Each output's internal is keyed only by its own (normalized) request_id.
    assert outs[0].dmi_internal._request_ids == (normalize_vllm_request_id("req-a"),)
    assert outs[1].dmi_internal._request_ids == (normalize_vllm_request_id("req-b"),)


def test_reader_passed_through():
    outs = _fake_outputs("req-a")
    sentinel = object()
    _attach_dmi_internal(outs, "m", reader=sentinel)
    assert outs[0].dmi_internal._reader is sentinel
