"""HF under ``storage_backend="persistent"`` must fail loudly, never store nothing.

Two reproductions from the capture-path audit, both of which "succeeded":

* Link 6. The HF adapter drives legacy HookPoints. Under the capture config
  those run on the engine's legacy ring, which has no host (the capture
  backend refuses one), so its P2P thread drops every capture. Generation
  returns normally and the catalog stays empty. Capture storage is not wired
  to the HF adapter yet, so every HF entry point refuses the config instead.
* Link 7. Once ``create_record_runtime`` has run, the engine's ring is a
  record ring, and ``commit_step`` -> ``push_all_metas`` throws "legacy
  metadata cannot be pushed to a record ring". ``generate_with_monitoring``
  swallowed that in its prepare wrapper, so generation "succeeded" with
  nothing captured, while ``generate_greedy_with_monitoring`` (which has no
  wrapper) propagated it. The wrapper now re-raises in capture or record
  mode and, on the legacy path where swallowing is deliberate, logs the
  first failure at WARNING instead of passing silently. ``commit_step``
  itself refuses record mode, because an adapter attached before
  ``create_record_runtime`` still holds the stopped legacy ring, which
  accepts the step without any error.

The forked ``HookedLlama`` classes live in the transformers submodule, which
the CPU job does not install, so the model here is the smallest stand-in the
adapter accepts: a real HookPoint, an HF-shaped config, and HF's
``prepare_inputs_for_generation`` / ``generate`` protocol. The ring is a spy
that behaves like the native engine it replaces.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from dmi.adapters.base import BackendAdapter, StepPlan, StepReservation
from dmi.adapters.huggingface.adapter import HuggingFaceAdapter
from dmi.adapters.huggingface.generation import (
    generate_greedy_with_monitoring,
    generate_with_monitoring,
)
from dmi.adapters.types import StepContext
from dmi.configuration import DMIConfig, ObservationConfig
from dmi.configuration.compiler import attach_config
from dmi.configuration.errors import ConfigurationError
from dmi.hooks.dispatch import install_ring_hooks
from dmi.hooks.point import HookPoint
from dmi.hooks.specs import HOOK_TYPE_RESID_PRE, HookSpec, ModelShapeConfig
from dmi.transport.ring import RingTransport

pytestmark = pytest.mark.cpu

ADAPTER_LOGGER = "dmi.adapters.huggingface.adapter"
RECORD_RING_ERROR = "legacy metadata cannot be pushed to a record ring"
RECORD_MODE_REFUSAL = (
    r"(before_forward|commit_step)\(\): the engine is in record mode")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SpyRingEngine:
    """The native RingEngine surface the legacy HF path touches.

    ``record_ring=True`` behaves like a ring made by ``create_record``: the
    legacy metadata push throws, exactly as ``RingEnginePy::push_step`` does.
    ``prepare_step_error`` makes the reservation itself fail.
    """

    def __init__(self, *, record_ring=False, prepare_step_error=None):
        self.record_ring = record_ring
        self.prepare_step_error = prepare_step_error
        self.prepare_step_calls: list = []
        self.push_all_metas_calls = 0

    def payload_tensor(self):
        return torch.zeros(16, dtype=torch.uint8)

    def payload_cap(self):
        return 1 << 20

    def staging_cap(self):
        return 1 << 20

    def prepare_step(self, total_bytes, num_hooks):
        self.prepare_step_calls.append((total_bytes, num_hooks))
        if self.prepare_step_error is not None:
            raise self.prepare_step_error
        return 0

    def push_all_metas(self, *args):
        self.push_all_metas_calls += 1
        if self.record_ring:
            raise RuntimeError(RECORD_RING_ERROR)


class _SpyEngine:
    """The MonitoringEngine attributes the HF adapter and entry points read."""

    def __init__(self, storage_backend="auto", *, record_mode=False,
                 ring_engine=None):
        self._storage_backend = storage_backend
        self._record_mode = record_mode
        self._ring_engine = ring_engine or _SpyRingEngine(
            record_ring=record_mode)
        self._ring_transport = RingTransport(self._ring_engine)
        self._model_id = "tiny"
        self.config = None
        self._group = 0

    def next_auto_group_id(self):
        self._group += 1
        return self._group


class _TinyHookedLM(torch.nn.Module):
    """One real HookPoint behind HF's generation protocol."""

    def __init__(self, engine):
        super().__init__()
        self.hook_resid_pre = HookPoint()
        self.config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=16,
            intermediate_size=16,
        )
        self.monitoring_engine = engine
        self.generate_calls = 0
        self.forward_calls = 0

    def get_hook_specs(self):
        return [HookSpec(hook_type=HOOK_TYPE_RESID_PRE,
                         module=self.hook_resid_pre, layer_no=0)]

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None,
                                      **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(self, input_ids=None, **kwargs):
        self.forward_calls += 1
        raise AssertionError("a refused entry point must not run the model")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=2,
                 **kwargs):
        """HF's loop, reduced to what DMI hooks: one prepare call per step."""
        self.generate_calls += 1
        for _ in range(max_new_tokens):
            self.prepare_inputs_for_generation(
                input_ids, attention_mask=attention_mask)
        return input_ids


class _StubAdapter(BackendAdapter):
    """A non-HF adapter: the refusal lives in the shared base attach."""

    def detect_model_shape(self, model):
        return ModelShapeConfig(
            hidden_dim=8, num_heads=2, num_kv_heads=2, head_dim=4,
            dtype=torch.float16, vocab_size=16, intermediate_dim=16)

    def detect_parallel_ranks(self):
        return (0, 0, 0, 0)

    def is_pp_first(self):
        return True

    def is_pp_last(self):
        return True

    def build_step_context(self, *raw):
        return None

    def on_capacity_exceeded(self, ctx):
        return None


def _inputs():
    input_ids = torch.tensor([[1, 2, 3]])
    return input_ids, torch.ones_like(input_ids)


def _assert_untouched(model, engine):
    """A refusal happens before anything is armed, wrapped or reserved."""
    assert model.hook_resid_pre._ring_hook_type is None
    assert getattr(model, "_dmi_active_adapter", None) is None
    assert getattr(model, "_monitoring_orig_prepare", None) is None
    assert "prepare_inputs_for_generation" not in vars(model)
    assert engine._ring_engine.prepare_step_calls == []
    assert engine._ring_engine.push_all_metas_calls == 0


# ---------------------------------------------------------------------------
# Link 6: the persistent config is refused at every HF entry point
# ---------------------------------------------------------------------------


def test_hf_attach_model_refuses_capture_storage():
    engine = _SpyEngine("persistent")
    model = _TinyHookedLM(engine)

    with pytest.raises(ConfigurationError,
                       match="capture storage is not wired to "
                             "HuggingFaceAdapter yet"):
        HuggingFaceAdapter(engine, "tiny").attach_model(model)

    _assert_untouched(model, engine)


@pytest.mark.parametrize("via_attach_config", [False, True],
                         ids=["attach_model", "attach_config"])
def test_base_attach_model_refuses_capture_storage_for_any_adapter(
        via_attach_config):
    """The legacy HookPoint path is what every adapter's attach installs, so
    the refusal is in the base class, not only in the HF override."""
    engine = _SpyEngine("persistent")
    model = _TinyHookedLM(engine)
    adapter = _StubAdapter(engine, "tiny")

    with pytest.raises(ConfigurationError,
                       match="capture storage is not wired to _StubAdapter"):
        if via_attach_config:
            attach_config(adapter, model, DMIConfig(
                observations=ObservationConfig(hooks=["resid_pre"])))
        else:
            adapter.attach_model(model)

    _assert_untouched(model, engine)


def test_generate_with_monitoring_refuses_capture_storage():
    engine = _SpyEngine("persistent")
    model = _TinyHookedLM(engine)
    input_ids, attention_mask = _inputs()

    with pytest.raises(ConfigurationError,
                       match=r"generate_with_monitoring\(\).*capture storage "
                             "is not wired"):
        generate_with_monitoring(model, input_ids,
                                 attention_mask=attention_mask)

    assert model.generate_calls == 0
    assert not hasattr(engine, "_hf_adaptor")
    _assert_untouched(model, engine)


def test_generate_greedy_with_monitoring_refuses_capture_storage():
    engine = _SpyEngine("persistent")
    model = _TinyHookedLM(engine)
    input_ids, attention_mask = _inputs()

    with pytest.raises(ConfigurationError,
                       match=r"generate_greedy_with_monitoring\(\).*capture "
                             "storage is not wired"):
        generate_greedy_with_monitoring(
            model, input_ids, attention_mask,
            max_new_tokens=2, monitoring=True)

    assert model.forward_calls == 0
    assert not hasattr(engine, "_hf_adaptor")
    _assert_untouched(model, engine)


@pytest.mark.parametrize("entry", [
    "hf_attach_model", "base_attach_model", "attach_config",
    "generate_with_monitoring", "generate_greedy_with_monitoring"])
def test_every_hf_entry_point_refuses_when_capture_is_off(entry):
    """storage_backend="none" turns capture off, so there is no ring for the
    hooks, and each entry point says so before touching the model."""
    engine = _SpyEngine("none")
    model = _TinyHookedLM(engine)
    input_ids, attention_mask = _inputs()

    with pytest.raises(ConfigurationError, match="capture is off"):
        if entry == "hf_attach_model":
            HuggingFaceAdapter(engine, "tiny").attach_model(model)
        elif entry == "base_attach_model":
            _StubAdapter(engine, "tiny").attach_model(model)
        elif entry == "attach_config":
            attach_config(_StubAdapter(engine, "tiny"), model, DMIConfig(
                observations=ObservationConfig(hooks=["resid_pre"])))
        elif entry == "generate_with_monitoring":
            generate_with_monitoring(model, input_ids,
                                     attention_mask=attention_mask)
        else:
            generate_greedy_with_monitoring(
                model, input_ids, attention_mask,
                max_new_tokens=2, monitoring=True)

    assert model.generate_calls == 0
    assert model.forward_calls == 0
    _assert_untouched(model, engine)


@pytest.mark.parametrize("backend", ["auto", "in-memory"])
def test_other_storage_backends_still_attach(backend):
    """Only 'persistent' and 'none' are refused; the rest attach as before."""
    engine = _SpyEngine(backend)
    model = _TinyHookedLM(engine)
    adapter = HuggingFaceAdapter(engine, "tiny")

    adapter.attach_model(model)
    try:
        assert model.hook_resid_pre._ring_hook_type == HOOK_TYPE_RESID_PRE
        assert model._dmi_active_adapter is adapter
    finally:
        adapter.detach_model(model)


# ---------------------------------------------------------------------------
# Link 7: a failed step is not swallowed where it means nothing is stored
# ---------------------------------------------------------------------------


def test_record_ring_failure_propagates_out_of_generate_with_monitoring():
    """The audit's exact path: the engine is on a record ring, so the legacy
    metadata push threw inside HF's prepare step. It used to be swallowed
    and generate() returned as if it had captured. commit_step now refuses
    record mode before reserving or pushing anything."""
    engine = _SpyEngine("auto", record_mode=True)
    model = _TinyHookedLM(engine)
    input_ids, attention_mask = _inputs()

    with pytest.raises(RuntimeError, match=RECORD_MODE_REFUSAL):
        generate_with_monitoring(model, input_ids,
                                 attention_mask=attention_mask)

    assert engine._ring_engine.prepare_step_calls == []
    assert engine._ring_engine.push_all_metas_calls == 0
    # The finally still detaches: the wrapper is gone and the hooks disarmed.
    assert model._monitoring_orig_prepare is None
    assert not hasattr(model.prepare_inputs_for_generation, "__wrapped__")
    assert model.hook_resid_pre._ring_hook_type is None


def test_capture_mode_failure_propagates_out_of_the_prepare_wrapper():
    """Capture mode re-raises too, even for an adapter attached earlier.
    commit_step refuses the capture config itself, so the failure the
    wrapper sees is that refusal, raised before anything is reserved."""
    engine = _SpyEngine("auto")
    model = _TinyHookedLM(engine)
    adapter = HuggingFaceAdapter(engine, "tiny")
    adapter.attach_model(model)
    engine._storage_backend = "persistent"
    input_ids, attention_mask = _inputs()

    try:
        with pytest.raises(ConfigurationError,
                           match=r"HuggingFaceAdapter\.commit_step\(\).*"
                                 "capture storage is not wired"):
            model.prepare_inputs_for_generation(
                input_ids, attention_mask=attention_mask)
    finally:
        adapter.detach_model(model)

    assert engine._ring_engine.prepare_step_calls == []
    assert engine._ring_engine.push_all_metas_calls == 0


def test_legacy_driver_failure_is_logged_once_not_swallowed(caplog):
    """On the legacy ring, swallowing stays: capture is best-effort there and
    must not abort the user's generate(). It is no longer silent."""
    engine = _SpyEngine("auto", ring_engine=_SpyRingEngine(
        prepare_step_error=RuntimeError("ring hiccup")))
    model = _TinyHookedLM(engine)
    input_ids, attention_mask = _inputs()

    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER):
        out = generate_with_monitoring(model, input_ids,
                                       attention_mask=attention_mask,
                                       max_new_tokens=3)

    assert out is input_ids
    assert len(engine._ring_engine.prepare_step_calls) == 3
    warnings = [r for r in caplog.records
                if r.name == ADAPTER_LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "ring hiccup" in warnings[0].getMessage()


def _switch_to_record_ring(engine):
    """What ``create_record_runtime`` does to an engine an adapter is
    already attached to: it stops the legacy ring and replaces the engine's
    ring and transport. The adapter keeps the references it took at attach."""
    engine._ring_engine = _SpyRingEngine(record_ring=True)
    engine._ring_transport = RingTransport(engine._ring_engine)
    engine._record_mode = True


def test_adapter_attached_before_create_record_runtime_refuses_the_step():
    """attach, then create_record_runtime: the adapter's ring reference is
    the stopped legacy ring, which accepts the step without complaint, so no
    record-ring refusal fires and the step would pass silently. commit_step
    refuses record mode itself, before touching either ring."""
    engine = _SpyEngine("auto")
    model = _TinyHookedLM(engine)
    adapter = HuggingFaceAdapter(engine, "tiny")
    adapter.attach_model(model)
    stale_ring = engine._ring_engine
    _switch_to_record_ring(engine)
    input_ids, attention_mask = _inputs()

    try:
        with pytest.raises(RuntimeError, match=RECORD_MODE_REFUSAL):
            model.prepare_inputs_for_generation(
                input_ids, attention_mask=attention_mask)
    finally:
        adapter.detach_model(model)

    for ring in (stale_ring, engine._ring_engine):
        assert ring.prepare_step_calls == []
        assert ring.push_all_metas_calls == 0


# ---------------------------------------------------------------------------
# The step itself refuses the capture config, whatever attach path ran
# ---------------------------------------------------------------------------


class _NoSuperAttachAdapter(_StubAdapter):
    """An adapter whose attach_model skips super(), as attach_config allows:
    it arms its HookPoints through the exported install_ring_hooks, so the
    base attach refusal never runs."""

    def attach_model(self, model, hook_selection=None):
        self.model_cfg = self.detect_model_shape(model)
        self.transport.set_model_cfg(self.model_cfg)
        specs = model.get_hook_specs()
        install_ring_hooks(specs, ring_payload=self.transport._ring_payload)
        self.active_specs = specs
        self.transport._active_specs = specs


def _step_context():
    return StepContext(
        model_id="tiny", flattened=False, req_ids=["1:0"],
        token_ranges=[(0, 3)], dim0_offsets=[0], kv_offsets=[0],
        batch=1, q_len=3)


@pytest.mark.parametrize("via_attach_config", [False, True],
                         ids=["attach_model", "attach_config"])
def test_commit_step_refuses_capture_storage_when_attach_skipped_super(
        via_attach_config):
    """The reviewer's reproduction: with the base attach bypassed, the step
    reserved and published into a ring with no host, and commit_step
    returned RESERVED under storage_backend="persistent"."""
    engine = _SpyEngine("persistent")
    model = _TinyHookedLM(engine)
    adapter = _NoSuperAttachAdapter(engine, "tiny")
    if via_attach_config:
        attach_config(adapter, model, DMIConfig(
            observations=ObservationConfig(hooks=["resid_pre"])))
    else:
        adapter.attach_model(model)
    plan = StepPlan(total_bytes=64, hook_count=1, needs_eager=False)

    with pytest.raises(ConfigurationError,
                       match=r"_NoSuperAttachAdapter\.commit_step\(\).*"
                             "capture storage is not wired to "
                             "_NoSuperAttachAdapter"):
        adapter.commit_step(_step_context(), plan)

    assert engine._ring_engine.prepare_step_calls == []
    assert engine._ring_engine.push_all_metas_calls == 0


# ---------------------------------------------------------------------------
# Record mode is refused even while capture is disabled
# ---------------------------------------------------------------------------


def _record_engine_with_capture_disabled():
    """Record mode after set_capture_enabled(False): the transport is in null
    mode, but the HookPoints stay armed and still dispatch into the record
    ring, whose native guard then raises from inside the model forward."""
    engine = _SpyEngine("auto", record_mode=True)
    engine._ring_transport.null_offload = True
    return engine


def test_commit_step_refuses_record_mode_while_capture_is_disabled():
    engine = _record_engine_with_capture_disabled()
    adapter = _NoSuperAttachAdapter(engine, "tiny")
    adapter.attach_model(_TinyHookedLM(engine))
    plan = StepPlan(total_bytes=64, hook_count=1, needs_eager=False)

    with pytest.raises(RuntimeError, match=RECORD_MODE_REFUSAL):
        adapter.commit_step(_step_context(), plan)

    assert engine._ring_engine.prepare_step_calls == []
    assert engine._ring_engine.push_all_metas_calls == 0


def test_before_forward_refuses_record_mode_while_capture_is_disabled():
    engine = _record_engine_with_capture_disabled()
    adapter = _NoSuperAttachAdapter(engine, "tiny")
    adapter.attach_model(_TinyHookedLM(engine))

    with pytest.raises(RuntimeError, match=RECORD_MODE_REFUSAL):
        adapter.before_forward(None)

    assert engine._ring_engine.prepare_step_calls == []
    assert engine._ring_engine.push_all_metas_calls == 0


def test_generate_with_monitoring_refuses_record_mode_while_capture_is_disabled():
    """The HF prepare wrapper also skipped the driver in null mode, so the
    first thing to fail was a HookPoint inside the forward."""
    engine = _record_engine_with_capture_disabled()
    model = _TinyHookedLM(engine)
    input_ids, attention_mask = _inputs()

    with pytest.raises(RuntimeError, match=RECORD_MODE_REFUSAL):
        generate_with_monitoring(model, input_ids,
                                 attention_mask=attention_mask)

    assert engine._ring_engine.prepare_step_calls == []
    assert engine._ring_engine.push_all_metas_calls == 0
    assert model._monitoring_orig_prepare is None
    assert model.hook_resid_pre._ring_hook_type is None


@pytest.mark.parametrize("entry", ["before_forward", "commit_step"])
def test_capture_disabled_outside_record_mode_still_skips(entry):
    """Null mode on the legacy ring is a legitimate pause: it still skips."""
    engine = _SpyEngine("auto")
    engine._ring_transport.null_offload = True
    adapter = _NoSuperAttachAdapter(engine, "tiny")
    adapter.attach_model(_TinyHookedLM(engine))

    if entry == "before_forward":
        assert adapter.before_forward(None) is None
    else:
        plan = StepPlan(total_bytes=64, hook_count=1, needs_eager=False)
        assert adapter.commit_step(_step_context(), plan) is (
            StepReservation.SKIPPED)
    assert engine._ring_engine.prepare_step_calls == []
