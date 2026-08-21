"""Independent lifecycle and failure-containment tests for ``HFAdaptor``.

These tests deliberately use a tiny model and transport doubles: no model
weights, database, or CUDA device are required.  Importing the adapter still
requires the native hook-definition table, so the module skips cleanly when
the DMI extension has not been built.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

try:
    from integration.hf_adapter import HFAdaptor
    from monitoring import hook_points, ring_transport
    _NATIVE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on build environment
    HFAdaptor = None
    hook_points = None
    ring_transport = None
    _NATIVE_IMPORT_ERROR = exc


pytestmark = [
    pytest.mark.native_backend,
    pytest.mark.skipif(
        _NATIVE_IMPORT_ERROR is not None,
        reason=f"DMI native backend required: {_NATIVE_IMPORT_ERROR}",
    ),
]


class _FakeRingEngine:
    def payload_cap(self) -> int:
        return 4096

    def staging_cap(self) -> int:
        return 4096


class _FakeTransport:
    def __init__(self) -> None:
        self._ring_engine = _FakeRingEngine()
        self._ring_payload = object()
        self._active_specs = []
        self._using_forward_hooks = False
        self._model_cfg = None
        self.null_offload = False
        self.force_eager = False

    def set_model_cfg(self, cfg) -> None:
        self._model_cfg = cfg


class _FakeEngine:
    def __init__(self) -> None:
        self._ring_transport = _FakeTransport()
        self._ring_engine = self._ring_transport._ring_engine
        self._auto_batch_group_id = 0
        self._model_id = "test-model"

    def next_auto_group_id(self) -> int:
        result = self._auto_batch_group_id
        self._auto_batch_group_id += 1
        return result


class _FakeModel:
    dtype = torch.float16
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        vocab_size=32,
        eos_token_id=None,
    )
    generation_config = SimpleNamespace(eos_token_id=None)

    def __init__(self) -> None:
        self.hidden = hook_points.HookPoint()
        self.logits = hook_points.HookPoint()
        self.prepare_calls = 0

    def prepare_inputs_for_generation(self, *args, **kwargs):
        self.prepare_calls += 1
        return {"prepared": self.prepare_calls, **kwargs}

    def get_hook_specs(self):
        return [
            ring_transport.HookSpec(
                ring_transport.HOOK_TYPE_RESID_PRE,
                self.hidden,
                layer_no=0,
            ),
            ring_transport.HookSpec(
                ring_transport.HOOK_TYPE_FINAL_LOGITS,
                self.logits,
                layer_no=-1,
            ),
        ]


class _FakeCudaTensor:
    is_cuda = True
    nbytes = 16

    def contiguous(self):
        return self


def _make_attached(*, selection: str = "full"):
    model = _FakeModel()
    engine = _FakeEngine()
    adaptor = HFAdaptor(engine, "test-model")
    original_prepare = model.prepare_inputs_for_generation
    adaptor.attach_model(model, hook_selection=selection)
    return model, engine, adaptor, original_prepare


def test_detach_restores_original_prepare_and_transport_state():
    model, engine, adaptor, original_prepare = _make_attached()

    assert model.prepare_inputs_for_generation != original_prepare
    assert engine._ring_transport._using_forward_hooks is True
    assert len(engine._ring_transport._active_specs) == 2

    adaptor.detach_model(model)

    assert model.prepare_inputs_for_generation == original_prepare
    assert model._monitoring_orig_prepare is None
    assert engine._ring_transport._using_forward_hooks is False
    assert engine._ring_transport._active_specs == []


def test_detach_is_idempotent():
    model, _engine, adaptor, original_prepare = _make_attached()

    adaptor.detach_model(model)
    adaptor.detach_model(model)

    assert model.prepare_inputs_for_generation == original_prepare


def test_attach_detach_attach_does_not_stack_prepare_wrappers():
    model, _engine, adaptor, original_prepare = _make_attached()
    first_wrapper = model.prepare_inputs_for_generation
    adaptor.detach_model(model)

    adaptor.attach_model(model)
    second_wrapper = model.prepare_inputs_for_generation

    assert second_wrapper != original_prepare
    assert second_wrapper != first_wrapper
    assert model._monitoring_orig_prepare == original_prepare
    assert model.prepare_inputs_for_generation(example=1)["prepared"] == 1


def test_reattach_with_different_selection_disables_old_selection():
    model, engine, adaptor, _original_prepare = _make_attached(
        selection="hidden-states"
    )
    assert model.hidden.enabled is True
    assert model.logits.enabled is False

    adaptor.detach_model(model)
    adaptor.attach_model(model, hook_selection="logits")

    assert model.hidden.enabled is False
    assert model.logits.enabled is True
    assert [s.module for s in engine._ring_transport._active_specs] == [
        model.logits
    ]


@pytest.mark.xfail(
    strict=True,
    reason="known bug: detach leaves producer attributes installed on HookPoints",
)
def test_detach_prevents_plain_forward_from_dispatching(monkeypatch):
    model, _engine, adaptor, _original_prepare = _make_attached(
        selection="hidden-states"
    )
    dispatched = []
    monkeypatch.setattr(
        hook_points,
        "_dispatch_producer",
        lambda *args: dispatched.append(args),
    )

    adaptor.detach_model(model)
    result = model.hidden(_FakeCudaTensor())

    assert result is not None
    assert dispatched == []
    assert model.hidden._ring_hook_type is None
    assert model.hidden._ring_hook_id is None
    assert model.hidden._ring_payload is None


@pytest.mark.parametrize("phase", ["build", "plan", "commit"])
@pytest.mark.xfail(
    strict=True,
    reason="known bug: the HF prepare wrapper swallows monitoring failures",
)
def test_prepare_wrapper_propagates_monitoring_protocol_failure(
    monkeypatch, phase
):
    model, _engine, adaptor, _original_prepare = _make_attached()
    failure = RuntimeError(f"{phase} failed")
    context = object()
    plan = object()

    if phase == "build":
        monkeypatch.setattr(
            adaptor, "build_step_context", lambda *_args: (_ for _ in ()).throw(failure)
        )
    else:
        monkeypatch.setattr(adaptor, "build_step_context", lambda *_args: context)
        if phase == "plan":
            monkeypatch.setattr(
                adaptor, "plan_step", lambda *_args: (_ for _ in ()).throw(failure)
            )
        else:
            monkeypatch.setattr(adaptor, "plan_step", lambda *_args: plan)
            monkeypatch.setattr(
                adaptor, "commit_step", lambda *_args: (_ for _ in ()).throw(failure)
            )

    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        model.prepare_inputs_for_generation(input_ids=object())


def test_prepare_wrapper_propagates_original_model_failure(monkeypatch):
    model = _FakeModel()
    engine = _FakeEngine()
    adaptor = HFAdaptor(engine, "test-model")

    def fail_original(*_args, **_kwargs):
        raise ValueError("model preparation failed")

    monkeypatch.setattr(model, "prepare_inputs_for_generation", fail_original)
    adaptor.attach_model(model)

    with pytest.raises(ValueError, match="model preparation failed"):
        model.prepare_inputs_for_generation(input_ids=object())


def test_null_offload_bypasses_monitoring_protocol(monkeypatch):
    model, engine, adaptor, _original_prepare = _make_attached()
    engine._ring_transport.null_offload = True
    monkeypatch.setattr(
        adaptor,
        "before_forward",
        lambda *_args: pytest.fail("monitoring protocol should be bypassed"),
    )

    result = model.prepare_inputs_for_generation(input_ids="tokens")

    assert result["input_ids"] == "tokens"


# ---------------------------------------------------------------------------
# monitoring=True path of generate_greedy_with_monitoring
#
# generate_greedy_with_monitoring builds its own HFAdaptor, attaches with
# install_prepare_wrapper=False, drives before_forward_manual per forward, and
# detaches in a finally block.  That wiring has no other coverage: the
# prepare-wrapper tests above all go through attach_model directly.
# ---------------------------------------------------------------------------


class _GeneratingFakeModel(_FakeModel):
    """``_FakeModel`` that can also serve the manual greedy loop."""

    def __init__(self, engine, token_steps: list[list[int]]) -> None:
        super().__init__()
        self.monitoring_engine = engine
        self.token_steps = token_steps
        self.forward_calls = 0
        self.raise_on_call: int | None = None

    def forward(self, input_ids, **kwargs):
        index = self.forward_calls
        self.forward_calls += 1
        if self.raise_on_call is not None and index == self.raise_on_call:
            raise RuntimeError("model exploded mid-generation")
        next_tokens = self.token_steps[index]
        logits = torch.full((len(next_tokens), 1, 32), -1000.0)
        for row, token_id in enumerate(next_tokens):
            logits[row, 0, token_id] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=object())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@pytest.fixture
def greedy_monitoring(monkeypatch):
    """Patch out CUDA sync and record every before_forward_manual call."""
    from integration import hf_adapter

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls = []
    monkeypatch.setattr(
        HFAdaptor,
        "before_forward_manual",
        lambda self, *args, **kwargs: calls.append(kwargs),
    )
    return hf_adapter.generate_greedy_with_monitoring, calls


def _run_greedy(generate, model, *, max_new_tokens):
    input_ids = torch.ones((1, 2), dtype=torch.long)
    return generate(
        model,
        input_ids,
        torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        monitoring=True,
    )


def test_monitoring_generation_drives_one_step_per_forward(greedy_monitoring):
    generate, calls = greedy_monitoring
    engine = _FakeEngine()
    model = _GeneratingFakeModel(engine, [[3], [4], [5]])

    result = _run_greedy(generate, model, max_new_tokens=3)

    assert [tokens.tolist() for tokens in result] == [[3, 4, 5]]
    assert model.forward_calls == 3
    assert len(calls) == 3
    assert engine._hf_adaptor is not None


def test_monitoring_generation_detaches_after_success(greedy_monitoring):
    generate, _calls = greedy_monitoring
    engine = _FakeEngine()
    model = _GeneratingFakeModel(engine, [[3], [4]])

    _run_greedy(generate, model, max_new_tokens=2)

    assert engine._ring_transport._using_forward_hooks is False
    assert engine._ring_transport._active_specs == []


@pytest.mark.parametrize("raise_on_call", [0, 1])
def test_monitoring_generation_detaches_when_the_model_raises(
    greedy_monitoring, raise_on_call
):
    generate, _calls = greedy_monitoring
    engine = _FakeEngine()
    model = _GeneratingFakeModel(engine, [[3], [4], [5]])
    model.raise_on_call = raise_on_call

    with pytest.raises(RuntimeError, match="model exploded mid-generation"):
        _run_greedy(generate, model, max_new_tokens=3)

    assert engine._ring_transport._using_forward_hooks is False
    assert engine._ring_transport._active_specs == []
