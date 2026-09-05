"""Reproduction tests for the XbzOnGit review findings on PR #122.

Findings covered here (runtime/estimator side):

- 3919326685  the phase flags must gate real capture, not sit unenforced.
- 3919326694  per-request / sustained volume must sum every rank.
- 3919326736  dtype accounting must match what the runtime reserves.
- 3919326740  StaticCache prefill attention must be representable.

The API/serialization findings live in ``test_pr122_review_findings.py`` and
``test_configuration_*``; UI-only findings are exercised through the API
contract in ``test_configurator_api.py`` where one exists.
"""
from __future__ import annotations

import dataclasses

import pytest
from types import SimpleNamespace

from dmi.adapters.base import BackendAdapter, StepPlan
from dmi.adapters.types import StepContext
from dmi.config import CaptureSchedule, MonitoringConfig
from dmi.configuration import (
    DMIConfig,
    LayerSelection,
    ModelDescriptor,
    ObservationConfig,
    Workload,
    estimate_config,
)
from dmi.configuration.schema import ModelTopology

pytestmark = pytest.mark.cpu

REPO_ROOT = pytest.importorskip("pathlib").Path(__file__).resolve().parents[1]
DENSE = REPO_ROOT / "examples" / "model_descriptors" / "llama3-8b.yaml"


# ---------------------------------------------------------------------------
# Shared doubles (mirrors test_adapter_protocol.py, kept DAMP on purpose:
# each suite reads standalone)
# ---------------------------------------------------------------------------


class FakeTransport:
    def __init__(self) -> None:
        self.null_offload = False
        self.force_eager = False
        self.set_step_context_calls: list = []
        self.pre_push_all_metas_calls: list = []

    def set_step_context(self, **kwargs):
        self.set_step_context_calls.append(kwargs)

    def pre_push_all_metas(self, **kwargs):
        self.pre_push_all_metas_calls.append(kwargs)


class FakeRingEngine:
    def __init__(self) -> None:
        self.prepare_step_calls: list = []

    def prepare_step(self, total_bytes: int, n_hooks: int) -> int:
        self.prepare_step_calls.append((total_bytes, n_hooks))
        return 0


class FakeEngine:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self._ring_transport = FakeTransport()
        self._ring_engine = FakeRingEngine()
        self.config = config


class StubAdapter(BackendAdapter):
    def __init__(self, engine, model_id, ctx):
        super().__init__(engine, model_id)
        self._ctx = ctx

    def detect_model_shape(self, model):
        raise NotImplementedError

    def detect_parallel_ranks(self):
        return (0, 0, 0, 0)

    def is_pp_first(self):
        return True

    def is_pp_last(self):
        return True

    def build_step_context(self, *raw):
        return self._ctx

    def on_capacity_exceeded(self, ctx):
        pass

    def plan_step(self, ctx):
        return StepPlan(total_bytes=1024, hook_count=1, needs_eager=False)


def _make_ctx(**overrides) -> StepContext:
    values = dict(
        model_id="test_model",
        flattened=False,
        req_ids=["0:0", "0:1"],
        token_ranges=[(0, 4), (0, 4)],
        dim0_offsets=[0, 1],
        kv_offsets=[0, 0],
        batch=2,
        q_len=4,
        kv_dim=4,
        logits_to_keep=0,
    )
    values.update(overrides)
    return StepContext(**values)


# ---------------------------------------------------------------------------
# Finding 3919326685: `capture_decode: false` must stop decode capture.
# ---------------------------------------------------------------------------


class TestScheduleGatesRealCapture:
    def _adapter(self, schedule: CaptureSchedule, phase: str | None):
        engine = FakeEngine(MonitoringConfig(schedule=schedule))
        ctx = _make_ctx(phase=phase)
        return StubAdapter(engine, "test_model", ctx)

    def test_capture_decode_false_skips_decode_steps(self):
        adapter = self._adapter(
            CaptureSchedule(capture_prefill=True, capture_decode=False),
            phase="decode",
        )
        adapter.before_forward(None)

        assert adapter.transport.set_step_context_calls == [], (
            "a decode step under capture_decode=false must not be captured"
        )

    def test_capture_decode_false_still_captures_prefill(self):
        adapter = self._adapter(
            CaptureSchedule(capture_prefill=True, capture_decode=False),
            phase="prefill",
        )
        adapter.before_forward(None)

        assert len(adapter.transport.set_step_context_calls) == 1

    def test_capture_prefill_false_skips_prefill_steps(self):
        adapter = self._adapter(
            CaptureSchedule(capture_prefill=False, capture_decode=True),
            phase="prefill",
        )
        adapter.before_forward(None)

        assert adapter.transport.set_step_context_calls == []

    def test_step_stride_skips_steps(self):
        adapter = self._adapter(CaptureSchedule(step_stride=3), phase="decode")
        for _ in range(6):
            adapter.before_forward(None)

        # Steps 0, 3 capture; 1, 2, 4, 5 do not.
        assert len(adapter.transport.set_step_context_calls) == 2

    def test_warmup_steps_delay_capture(self):
        adapter = self._adapter(CaptureSchedule(warmup_steps=2), phase="decode")
        for _ in range(4):
            adapter.before_forward(None)

        assert len(adapter.transport.set_step_context_calls) == 2

    def test_request_stride_skips_whole_requests(self):
        adapter = self._adapter(CaptureSchedule(request_stride=2), phase="decode")
        for group in range(4):
            adapter._ctx = _make_ctx(
                req_ids=[f"{group}:0", f"{group}:1"], phase="decode"
            )
            adapter.before_forward(None)

        # Groups 0 and 2 capture; 1 and 3 do not.
        assert len(adapter.transport.set_step_context_calls) == 2

    def test_no_phase_reported_captures_as_before(self):
        # An adapter that does not report a phase must keep capturing: the
        # schedule gates what it knows about, not what it does not.
        adapter = self._adapter(
            CaptureSchedule(capture_prefill=False, capture_decode=False),
            phase=None,
        )
        adapter.before_forward(None)

        assert len(adapter.transport.set_step_context_calls) == 1

    def test_engine_without_a_config_captures_as_before(self):
        engine = FakeEngine(config=None)
        adapter = StubAdapter(engine, "test_model", _make_ctx(phase="decode"))
        adapter.before_forward(None)

        assert len(adapter.transport.set_step_context_calls) == 1

    def test_default_schedule_changes_nothing(self):
        adapter = self._adapter(CaptureSchedule(), phase="prefill")
        adapter.before_forward(None)

        assert len(adapter.transport.set_step_context_calls) == 1


# ---------------------------------------------------------------------------
# Finding 3919326694: per-request volume must sum every rank, not read the
# peak-pressure rank only.
# ---------------------------------------------------------------------------


def _descriptor() -> ModelDescriptor:
    import yaml

    from dmi.configuration import parse_descriptor

    return parse_descriptor(yaml.safe_load(DENSE.read_text()))


class TestPerRequestVolumeSumsAllRanks:
    def test_tp_growth_does_not_deflate_per_request_volume(self):
        descriptor = _descriptor()
        config = DMIConfig(observations=ObservationConfig(hooks=["q"]))
        baseline = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=32,
                tensor_parallel_size=1, packed=True,
            )
        )
        sharded = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=32,
                tensor_parallel_size=2, packed=True,
            )
        )

        assert baseline.bytes_per_request > 0
        # Two TP shards each emit half-sized q tensors; the captured VOLUME
        # is the same total as one unsharded rank emits.
        assert sharded.bytes_per_request == pytest.approx(
            baseline.bytes_per_request, rel=0.01
        ), (
            "bytes_per_request is a volume figure: it must not shrink when "
            "the same capture is split across more ranks"
        )

    def test_pp_growth_does_not_deflate_per_request_volume(self):
        descriptor = _descriptor()
        config = DMIConfig(observations=ObservationConfig(hooks=["resid_pre"]))
        baseline = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=32,
                pipeline_parallel_size=1, packed=True,
            )
        )
        split = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=32,
                pipeline_parallel_size=4, packed=True,
            )
        )

        assert split.bytes_per_request == pytest.approx(
            baseline.bytes_per_request, rel=0.01
        )

    def test_peak_stays_per_rank(self):
        descriptor = _descriptor()
        config = DMIConfig(observations=ObservationConfig(hooks=["q"]))
        estimate = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=32,
                tensor_parallel_size=2, packed=True,
            )
        )

        # Ring capacity is judged per rank, so the peak figure must stay the
        # single-rank pressure even though volume sums.
        single = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=32,
                tensor_parallel_size=1, packed=True,
            )
        )
        assert estimate.peak_step_bytes * 2 == pytest.approx(
            single.peak_step_bytes, rel=0.01
        )

    def test_disabled_phase_contributes_zero_volume(self):
        descriptor = _descriptor()
        config = DMIConfig(
            observations=ObservationConfig(hooks=["q"]),
            schedule=CaptureSchedule(capture_prefill=False),
        )
        estimate = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=32, packed=True,
            )
        )

        assert estimate.bytes_per_request > 0


# ---------------------------------------------------------------------------
# Finding 3919326736: dtype accounting must match what the runtime reserves.
# ---------------------------------------------------------------------------


class TestDtypeAccounting:
    def test_packed_convention_counts_token_ids_at_int32(self):
        descriptor = _descriptor()
        hooks = DMIConfig(observations=ObservationConfig(hooks=["token_ids"]))
        workload = Workload(
            batch_size=1, prompt_tokens=128, decode_tokens=0, packed=True,
        )

        estimate = estimate_config(hooks, descriptor, workload)

        # vLLM token ids are int32: vocab < 2**31, the runtime meta carries
        # the tensor dtype it sees. Two bytes/elem of the int64 figure must go.
        expected = 128 * 4  # 128 positions * 4 bytes
        assert estimate.peak_step_bytes == expected

    def test_batched_convention_counts_token_ids_at_int64(self):
        descriptor = _descriptor()
        hooks = DMIConfig(observations=ObservationConfig(hooks=["token_ids"]))
        workload = Workload(
            batch_size=1, prompt_tokens=128, decode_tokens=0, packed=False,
        )

        estimate = estimate_config(hooks, descriptor, workload)

        expected = 128 * 8  # HF input_ids are int64
        assert estimate.peak_step_bytes == expected


# ---------------------------------------------------------------------------
# Finding 3919326740: StaticCache prefill attention is shaped by the physical
# KV width, not the prompt length.
# ---------------------------------------------------------------------------


class TestStaticCachePrefillWidth:
    def test_cache_max_len_widens_prefill_attention(self):
        descriptor = _descriptor()
        config = DMIConfig(observations=ObservationConfig(hooks=["pattern"]))
        dynamic = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=2048,
                packed=False,
            )
        )
        static = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=2048,
                packed=False, cache_max_len=128 + 2048,
            )
        )

        assert static.peak_step_bytes > dynamic.peak_step_bytes
        # [H, prompt, cache_len] vs [H, prompt, prompt]: the ratio is
        # cache_len / prompt = (128+2048)/128 = 17.
        assert static.peak_step_bytes == pytest.approx(
            dynamic.peak_step_bytes * (128 + 2048) / 128, rel=0.01
        )

    def test_absent_cache_max_len_says_which_cache_is_assumed(self):
        descriptor = _descriptor()
        config = DMIConfig(observations=ObservationConfig(hooks=["pattern"]))
        estimate = estimate_config(
            config, descriptor, Workload(
                batch_size=1, prompt_tokens=128, decode_tokens=2048,
                packed=False,
            )
        )

        assert any("StaticCache" in warning or "cache" in warning.lower()
                   for warning in estimate.warnings)


# ---------------------------------------------------------------------------
# Finding 3919326691: "valid" must mean executable. The shipped Llama
# descriptor marks pos_embed/mlp_post available, but the attached model
# produces no such spec -- a config selecting only those compiled to an
# empty spec list with no complaint.
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass

from dmi.configuration import compile_config
from dmi.configuration.compiler import ModelContext as _ModelContext
from dmi.configuration.errors import ConfigValidationError
from dmi.hooks.specs import HOOK_TYPE_Q, HOOK_TYPE_RESID_PRE


@_dataclass
class _LiveSpec:
    """Duck-typed HookSpec: only hook_type and layer_no are consulted."""

    hook_type: int
    layer_no: int


def _llama_like_context() -> _ModelContext:
    """What a live Llama decoder actually exposes: no pos_embed, no mlp_post."""
    return _ModelContext(
        specs=[
            _LiveSpec(HOOK_TYPE_RESID_PRE, layer)
            for layer in range(32)
        ]
        + [
            _LiveSpec(HOOK_TYPE_Q, layer)
            for layer in range(32)
        ]
    )


class TestValidMeansExecutable:
    def test_selecting_only_absent_hooks_is_refused(self):
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["pos_embed"],
                layers=LayerSelection(8, 15),
            )
        )

        with pytest.raises(ConfigValidationError, match="pos_embed"):
            compile_config(config, _llama_like_context())

    def test_partially_absent_hooks_name_the_missing_one(self):
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["q", "pos_embed"],
                layers=LayerSelection(8, 15),
            )
        )

        with pytest.raises(ConfigValidationError, match="pos_embed") as excinfo:
            compile_config(config, _llama_like_context())
        assert "q" not in str(excinfo.value)

    def test_present_hooks_compile(self):
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["resid_pre"],
                layers=LayerSelection(8, 15),
            )
        )

        compiled = compile_config(config, _llama_like_context())

        assert compiled.selected_layers == list(range(8, 16))

    def test_layer_clipping_cannot_reintroduce_absent_hooks(self):
        # A range the model covers selects present hooks fine; the absent
        # hook is refused regardless of the range.
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["resid_pre", "mlp_post"],
                layers=LayerSelection(0, 3),
            )
        )

        with pytest.raises(ConfigValidationError, match="mlp_post"):
            compile_config(config, _llama_like_context())


# ---------------------------------------------------------------------------
# Finding 3919326699: the pinned vLLM adapter's attach_model takes only
# (model, hook_selection), so a configured layer range raised a raw TypeError
# from inside the integration. The submodule update lands in its own repo;
# in this repo the failure must at least be diagnosable.
# ---------------------------------------------------------------------------


class _LegacyVLLMStyleAdapter:
    """attach_model without a layers keyword, like the pinned integration."""

    def __init__(self) -> None:
        self.calls: list = []

    def attach_model(self, model, hook_selection: str = "full") -> None:
        self.calls.append((model, hook_selection))


class _KwargsAdapter:
    """attach_model forwarding **kwargs accepts layers without declaring it."""

    def __init__(self) -> None:
        self.calls: list = []

    def attach_model(self, model, hook_selection: str = "full", **kwargs) -> None:
        self.calls.append((model, hook_selection, kwargs))


class _LayerAwareAdapter:
    def attach_model(self, model, hook_selection: str = "full", *, layers=None):
        pass


class TestAttachConfigDiagnosesUnsupportedLayers:
    def test_range_on_a_legacy_adapter_is_a_configuration_error(self):
        from dmi.configuration.errors import ConfigurationError

        adapter = _LegacyVLLMStyleAdapter()
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["q"], layers=LayerSelection(8, 15)
            )
        )

        with pytest.raises(
            ConfigurationError, match="does not accept.*layers"
        ):
            _attach(adapter, config)

    def test_no_range_on_a_legacy_adapter_still_attaches(self):
        adapter = _LegacyVLLMStyleAdapter()
        config = DMIConfig(observations=ObservationConfig(hooks=["q"]))

        _attach(adapter, config)

        assert len(adapter.calls) == 1

    def test_kwargs_adapter_is_left_alone(self):
        adapter = _KwargsAdapter()
        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["q"], layers=LayerSelection(8, 15)
            )
        )

        _attach(adapter, config)

        assert adapter.calls[0][2] == {"layers": LayerSelection(8, 15)}

    def test_layer_aware_adapter_receives_the_range(self):
        from dmi.configuration.compiler import attach_config as real_attach

        real_attach(_LayerAwareAdapter(), object(), DMIConfig(
            observations=ObservationConfig(
                hooks=["q"], layers=LayerSelection(8, 15)
            )
        ))


def _attach(adapter, config) -> None:
    from dmi.configuration.compiler import attach_config

    attach_config(adapter, object(), config)


# ---------------------------------------------------------------------------
# Review follow-up on 3919326685: gating plan/commit is NOT enough. The
# model's HookPoints still dispatch producers during the refused step's
# forward -- unreserved ring writes and meta/FIFO desync. The refusal has to
# reach the producer launch decision via a transport-level per-step flag.
# ---------------------------------------------------------------------------


class TestRefusedStepsSilenceProducers:
    def test_refused_step_sets_capture_flag_false_on_the_transport(self):
        adapter = StubAdapter(
            FakeEngine(MonitoringConfig(
                schedule=CaptureSchedule(capture_prefill=True, capture_decode=False),
            )),
            "test_model",
            _make_ctx(phase="decode"),
        )

        adapter.before_forward(None)

        assert adapter.transport.capture_step is False

    def test_captured_step_sets_capture_flag_true(self):
        adapter = StubAdapter(
            FakeEngine(MonitoringConfig(schedule=CaptureSchedule())),
            "test_model",
            _make_ctx(phase="decode"),
        )

        adapter.before_forward(None)

        assert adapter.transport.capture_step is True

    def test_engine_without_a_config_leaves_capture_flag_true(self):
        adapter = StubAdapter(FakeEngine(config=None), "test_model", _make_ctx())

        adapter.before_forward(None)

        assert adapter.transport.capture_step is True

    def test_hook_point_does_not_dispatch_on_a_refused_step(self):
        """The whole point: during a refused step's forward, the fast path in
        HookPoint.forward must not launch a producer kernel.

        A MagicMock tensor poses as CUDA input (truthy is_cuda), so the
        re-armed path is observable on a CPU-only box: past the gate it
        reaches dispatch_producer; through the gate it must not.
        """
        from unittest.mock import MagicMock

        import torch

        import dmi.hooks.point as hook_point
        from dmi.hooks.point import HookPoint

        dispatched: list = []
        original = hook_point.dispatch_producer

        def recording_producer(*args, **kwargs):
            dispatched.append(args)

        hp = HookPoint()
        hp._ring_hook_type = 6
        hp._ring_hook_id = 0
        hp._strip_tensor = None
        hp._strip_row_bytes = 0
        hp._ring_payload = torch.zeros(16, dtype=torch.uint8)
        x = MagicMock()

        hook_point.dispatch_producer = recording_producer
        try:
            from dmi.transport import ring as _rt

            class _T:
                force_eager = False
                capture_step = False
                _ring_engine = None

            _rt._active_transport = _T()
            hp.forward(x)
            assert dispatched == [], "producer fired on a refused step"

            # A captured step re-arms the hook and dispatches.
            _rt._active_transport.capture_step = True
            hp.forward(x)
            assert len(dispatched) == 1
        finally:
            hook_point.dispatch_producer = original
            from dmi.transport import ring as _rt

            _rt._active_transport = None


# ---------------------------------------------------------------------------
# Review follow-ups, minor: sustained rate must apply both strides (the
# request gate drops whole requests too); the estimator must say when its
# reductions are only conditionally enforced; the architecture refusal must
# match real HF spellings.
# ---------------------------------------------------------------------------


class TestSustainedRateAppliesBothStrides:
    """Batched (Hugging Face) only: that is the driver that gates on them.

    Sampling divides where ``BackendAdapter.before_forward`` runs. The pinned
    vLLM integrations commit steps directly with ``config=None``, so the
    packed side must NOT divide -- pinned in
    ``test_pr122_round3_review_findings.py``.
    """

    def test_request_stride_thins_the_sustained_rate(self):
        dense = estimate_config(
            _config(request_stride=1), _descriptor(), _workload(packed=False)
        )
        strided = estimate_config(
            _config(request_stride=4), _descriptor(), _workload(packed=False)
        )

        assert strided.sustained_bytes_per_second == pytest.approx(
            dense.sustained_bytes_per_second / 4
        )


class TestEstimateDisclosesConditionalEnforcement:
    def test_strides_on_batched_carry_an_enforcement_assumption(self):
        """The HF gate is conditional on what the adapter reports."""
        estimate = estimate_config(
            _config(step_stride=4), _descriptor(), _workload(packed=False)
        )

        notes = " ".join(estimate.assumptions).lower()
        assert "assumes" in notes and ("phase" in notes or "request ids" in notes)

    def test_strides_on_packed_say_the_schedule_is_not_applied(self):
        """A packed estimate must not imply a reduction vLLM never performs."""
        estimate = estimate_config(
            _config(step_stride=4), _descriptor(), _workload(packed=True)
        )

        assert any(
            "vLLM" in warning and "not applied" in warning.lower()
            for warning in estimate.warnings
        ), estimate.warnings

    def test_default_schedule_makes_no_enforcement_claim(self):
        estimate = estimate_config(_config(), _descriptor(), _workload())

        assert not any("assumes" in a and "phase" in a for a in estimate.assumptions)


class TestMaskedLMArchitecturesAreRefused:
    def test_bertformaskedlm_architecture_is_refused(self):
        from types import SimpleNamespace

        # An unknown model_type whose architectures name the masked-LM head:
        # HF names carry no underscores, so the lowered suffix check must
        # match "formaskedlm".
        masked = SimpleNamespace(
            model_type="some_future_family",
            architectures=["BertForMaskedLM"],
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=2,
            vocab_size=30522,
            is_encoder_decoder=False,
        )

        from dmi.configuration.errors import DescriptorError
        from dmi.configuration.introspect import descriptor_from_hf_config

        with pytest.raises(DescriptorError, match="decoder_transformer|causal"):
            descriptor_from_hf_config(masked, "some-future-family")


def _config(hooks=("resid_pre",), layers=None, **schedule) -> DMIConfig:
    return DMIConfig(
        observations=ObservationConfig(hooks=list(hooks), layers=layers),
        schedule=CaptureSchedule(**schedule),
    )


def _workload(**overrides) -> Workload:
    base = dict(
        batch_size=1,
        prompt_tokens=128,
        decode_tokens=64,
        decode_steps_per_second=10.0,
        packed=True,
    )
    base.update(overrides)
    return Workload(**base)


# ---------------------------------------------------------------------------
# Independent-review round: validation/compile/estimate all certified
# final_logits on a descriptor with vocab_size=0 (the default for a
# hand-written descriptor), while compute_hook_shape returns [] for it --
# the exact "valid certifies a capture that never fires" class the
# absent-hook rejection exists for.
# ---------------------------------------------------------------------------


class TestFinalLogitsRequiresVocabSize:
    def test_final_logits_is_unavailable_without_vocab_size(self):
        from dmi.configuration import describe_hooks

        infos = {i.id: i for i in describe_hooks(
            ModelTopology(num_layers=4, hidden_size=64, num_attention_heads=4,
                          num_kv_heads=4, vocab_size=0)
        )}
        assert infos["final_logits"].available is False
        assert "vocab_size" in (infos["final_logits"].reason or "")

    def test_final_logits_is_available_with_vocab_size(self):
        from dmi.configuration import describe_hooks

        infos = {i.id: i for i in describe_hooks(
            ModelTopology(num_layers=4, hidden_size=64, num_attention_heads=4,
                          num_kv_heads=4, vocab_size=32000)
        )}
        assert infos["final_logits"].available is True


# ---------------------------------------------------------------------------
# Independent-review round on the estimator: (1) check_ring_fit ignored the
# task-entry cap prepare_step also enforces (a wide full-preset model "fit"
# on bytes while every runtime step returned OVERSIZED); (2) the PP layer
# split put the remainder on the last stages while vLLM's get_pp_indices
# puts it on the first, mislocating the peak rank; (3) batched final_logits
# counted every prefill row while HF generate() materializes one.
# ---------------------------------------------------------------------------


class TestRingFitCountsTaskEntries:
    def test_hook_count_over_the_task_cap_does_not_fit(self):
        from dmi.configuration.estimate import RankLoad, RingFit, check_ring_fit

        estimate = estimate_config.__wrapped__ if hasattr(estimate_config, "__wrapped__") else None
        from dmi.configuration.estimate import Estimate

        est = Estimate(
            peak_step_bytes=1024,
            peak_step_rank="pp0/tp0",
            decode_step_bytes=512,
            bytes_per_request=1024,
            aggregate_peak_step_bytes=2048,
            sustained_bytes_per_second=None,
            bytes_per_day=None,
            ranks=(
                RankLoad(label="pp0/tp0", pp_stage=0, tp_rank=0,
                         prefill_step_bytes=1024, decode_step_bytes=512,
                         prefill_hooks=1126, decode_hooks=1126),
            ),
        )
        fit = check_ring_fit(est, payload_bytes=1024 * 1024, task_entries=1024)
        assert fit.fits is False, (
            "1126 hooks cannot fit 1024 task entries no matter how many "
            "bytes are free -- prepare_step returns OVERSIZED"
        )
        assert "task" in fit.detail.lower()

    def test_hooks_within_the_cap_fit_on_bytes_alone(self):
        from dmi.configuration.estimate import Estimate, RankLoad, check_ring_fit

        est = Estimate(
            peak_step_bytes=1024,
            peak_step_rank="pp0/tp0",
            decode_step_bytes=512,
            bytes_per_request=1024,
            aggregate_peak_step_bytes=2048,
            sustained_bytes_per_second=None,
            bytes_per_day=None,
            ranks=(
                RankLoad(label="pp0/tp0", pp_stage=0, tp_rank=0,
                         prefill_step_bytes=1024, decode_step_bytes=512,
                         prefill_hooks=900, decode_hooks=900),
            ),
        )
        fit = check_ring_fit(est, payload_bytes=1024 * 1024, task_entries=1024)
        assert fit.fits is True


class TestPPSplitMatchesVLLM:
    def test_remainder_layers_land_on_the_first_stages(self):
        from dmi.configuration.estimate import _stage_layers

        # vLLM's get_pp_indices: 8 layers over 3 stages -> [3, 3, 2].
        sizes = [len(_stage_layers(8, 3, stage)) for stage in range(3)]
        assert sizes == [3, 3, 2], sizes


# ---------------------------------------------------------------------------
# Independent-review round on the gate: the no-phase tail (warmup/stride for
# adapters that report no phase) had no pin; the docstring claimed no-phase
# steps "always capture" while they are stride-gated; a foreign phase value
# killed capture silently via the HF wrapper's bare except.
# ---------------------------------------------------------------------------


class TestNoPhaseGateTail:
    def test_no_phase_steps_respect_warmup_and_stride(self):
        """A phase-less adapter must still be schedule-gated on the
        unconditional parts -- the estimator divides by the stride, so an
        adapter ignoring it would over-capture what the estimate promises."""
        adapter = StubAdapter(
            FakeEngine(MonitoringConfig(
                schedule=CaptureSchedule(step_stride=2, warmup_steps=1),
            )),
            "test_model",
            _make_ctx(phase=None),
        )
        for _ in range(4):
            adapter.before_forward(None)

        # Step 0 warmup, 1 first captured, 2 skipped, 3 captured.
        assert len(adapter.transport.set_step_context_calls) == 2

    def test_foreign_phase_value_does_not_kill_capture(self):
        """An unknown phase string must capture-as-unreported, not raise
        through the HF wrapper's exception guard leaving capture disarmed."""
        adapter = StubAdapter(
            FakeEngine(MonitoringConfig(schedule=CaptureSchedule())),
            "test_model",
            _make_ctx(phase="speculative-decode"),
        )

        adapter.before_forward(None)

        assert len(adapter.transport.set_step_context_calls) == 1
        assert adapter.transport.capture_step is True


# ---------------------------------------------------------------------------
# Independent-review round: the schedule gate is a Python branch inside the
# compiled region. Under torch.compile + CUDA graphs, replay never runs
# Python -- the flag is baked at the FIRST decode trace. A first-step
# refusal would compile producers out of every later captured step; a
# first-step capture would leave every later refused step dispatching. The
# runtime fix mirrors the existing capacity-overflow path: a non-default
# schedule disables compilation for the generate() call.
# ---------------------------------------------------------------------------


class TestCompileDisabledUnderNonDefaultSchedule:
    def _run(self, schedule):
        """Drive generate_with_monitoring far enough to inspect kwargs.

        Stubs the model and adapter so the capacity preflight runs with a
        tiny ring and the compile-config phase is observable without a GPU.
        """
        import torch

        from dmi.adapters.huggingface import generation as gen_module

        seen = {}

        class _FakeEngine:
            config = MonitoringConfig(schedule=schedule) if schedule is not None else None
            _model_id = "m"

            def next_auto_group_id(self):
                return 0

        class _FakeAdaptor:
            engine = _FakeEngine()
            model_cfg = SimpleNamespace(dtype=torch.float16)
            active_specs = [object()]
            ring_engine = SimpleNamespace(
                payload_cap=lambda: 1 << 20,
                staging_cap=lambda: 1 << 20,
            )

            def decode_step_bytes(self, batch, kv_dim):
                return 16

            def attach_model(self, *a, **k):
                pass

            def detach_model(self, *a):
                pass

        adaptor = _FakeAdaptor()

        class _FakeTransport:
            pass

        class _FakeEngine2:
            _ring_transport = _FakeTransport()
            _model_id = "m"

        class _FakeModel:
            generation_config = None
            monitoring_engine = _FakeEngine2()

            def generate(self, *a, **kw):
                seen.update(kw)
                return "gen"

        model = _FakeModel()
        monkey = pytest.MonkeyPatch()
        monkey.setattr(gen_module, "HuggingFaceAdapter", lambda *a, **k: adaptor)
        try:
            gen_module.generate_with_monitoring(
                model,
                input_ids=torch.zeros(1, 4, dtype=torch.long),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
                max_new_tokens=2,
            )
        finally:
            monkey.undo()
        return seen

    def test_non_default_schedule_disables_compilation(self):
        seen = self._run(CaptureSchedule(step_stride=2))
        assert seen.get("disable_compile") is True, (
            "the schedule gate is a Python branch; CUDA-graph replay bakes it "
            "at the first decode trace, so a non-default schedule must "
            "disable compilation for this generate() call"
        )

    def test_default_schedule_keeps_compilation(self):
        seen = self._run(CaptureSchedule())
        assert "disable_compile" not in seen
