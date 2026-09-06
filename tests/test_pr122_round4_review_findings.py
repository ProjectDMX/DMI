"""Round-4 review residuals on PR #122 (Samfisheryu follow-up, still open).

Each class below pins one residual finding. All fail against the current
tree; the fixes land with them.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmi.configuration import (
    DMIConfig,
    LayerSelection,
    ObservationConfig,
)
from dmi.configuration.compiler import attach_config
from dmi.configuration.errors import ConfigurationError

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# 1. One-owner invariant: attach_config(A) then attach_config(B) must fail
#    BEFORE mutating anything; same-owner re-attach is fine; a non-owner's
#    detach must not clear the marker or the wrapper.
# ---------------------------------------------------------------------------


class _OwnerFakeEngine:
    def __init__(self):
        self.config = None


class _OwnerFakeAdapter:
    def __init__(self, engine):
        self.engine = engine
        self.attach_calls: list = []

    def attach_model(self, model, hook_selection: str = "full", *, layers=None):
        self.attach_calls.append((model, hook_selection, layers))


def _owner_config() -> DMIConfig:
    return DMIConfig(observations=ObservationConfig(hooks=["resid_pre"]))


class TestOneOwnerEnforcement:
    def test_second_attach_by_another_adapter_is_refused(self):
        engine = _OwnerFakeEngine()
        first, second = _OwnerFakeAdapter(engine), _OwnerFakeAdapter(engine)
        model = SimpleNamespace()

        attach_config(first, model, _owner_config())
        with pytest.raises(ConfigurationError, match="already attached|owner"):
            attach_config(second, model, _owner_config())

        assert model._dmi_active_adapter is first
        assert second.attach_calls == [], "refused attach must not touch the model"

    def test_same_adapter_may_reattach(self):
        engine = _OwnerFakeEngine()
        adapter = _OwnerFakeAdapter(engine)
        model = SimpleNamespace()

        attach_config(adapter, model, _owner_config())
        attach_config(adapter, model, _owner_config())

        assert len(adapter.attach_calls) == 2

    def test_refused_attach_does_not_install_its_schedule(self):
        from dmi.config import CaptureSchedule

        engine = _OwnerFakeEngine()
        first, second = _OwnerFakeAdapter(engine), _OwnerFakeAdapter(engine)
        model = SimpleNamespace()
        attach_config(first, model, _owner_config())

        other = DMIConfig(
            observations=ObservationConfig(hooks=["resid_pre"]),
            schedule=CaptureSchedule(step_stride=9),
        )
        with pytest.raises(ConfigurationError, match="already attached|owner"):
            attach_config(second, model, other)

        assert engine.config.schedule.step_stride == 1

    def test_detach_by_a_non_owner_leaves_the_marker(self):
        from dmi.adapters.huggingface.adapter import HuggingFaceAdapter

        owner = HuggingFaceAdapter(SimpleNamespace(_ring_transport=None), "m")
        stranger = HuggingFaceAdapter(SimpleNamespace(_ring_transport=None), "m")
        model = SimpleNamespace()
        model._dmi_active_adapter = owner

        stranger.detach_model(model)

        assert model._dmi_active_adapter is owner


# ---------------------------------------------------------------------------
# 2. Packed schedule honesty: the pinned vLLM runtime executes no schedule,
#    so phase flags must not shrink packed figures (strides are already
#    handled); the authored intent stays visible as a warning.
# ---------------------------------------------------------------------------


def _descriptor():
    import yaml
    from pathlib import Path
    from dmi.configuration import parse_descriptor

    dense = Path("examples/model_descriptors/llama3-8b.yaml")
    return parse_descriptor(yaml.safe_load(dense.read_text()))


def _estimate(config, **workload_overrides):
    from dmi.configuration import Workload, estimate_config

    base = dict(batch_size=1, prompt_tokens=128, decode_tokens=8, packed=True)
    base.update(workload_overrides)
    return estimate_config(config, _descriptor(), Workload(**base))


class TestPackedIgnoresUnsupportedSchedule:
    def test_capture_decode_false_does_not_shrink_packed_volume(self):
        from dmi.config import CaptureSchedule

        both = _estimate(DMIConfig(observations=ObservationConfig(hooks=["q"])))
        no_decode = _estimate(
            DMIConfig(
                observations=ObservationConfig(hooks=["q"]),
                schedule=CaptureSchedule(capture_decode=False),
            )
        )

        assert no_decode.bytes_per_request == both.bytes_per_request
        assert no_decode.sustained_bytes_per_second == both.sustained_bytes_per_second

    def test_capture_prefill_false_does_not_shrink_packed_peak(self):
        from dmi.config import CaptureSchedule

        both = _estimate(DMIConfig(observations=ObservationConfig(hooks=["q"])))
        no_prefill = _estimate(
            DMIConfig(
                observations=ObservationConfig(hooks=["q"]),
                schedule=CaptureSchedule(capture_prefill=False),
            )
        )

        assert no_prefill.peak_step_bytes == both.peak_step_bytes

    def test_packed_phase_flags_carry_an_explicit_warning(self):
        from dmi.config import CaptureSchedule

        estimate = _estimate(
            DMIConfig(
                observations=ObservationConfig(hooks=["q"]),
                schedule=CaptureSchedule(capture_decode=False),
            )
        )

        assert any("vLLM" in w or "packed" in w.lower() for w in estimate.warnings)


# ---------------------------------------------------------------------------
# 3. vLLM PP partition rule: remainder goes backward from the second-to-last
#    stage, never the last; VLLM_PP_LAYER_PARTITION-style explicit counts
#    are accepted and validated.
# ---------------------------------------------------------------------------


class TestVLLMPartitionRule:
    @pytest.mark.parametrize(
        "num_layers, pp_size, expected",
        [
            (4, 3, [1, 2, 1]),
            (10, 4, [2, 3, 3, 2]),
            (8, 3, [3, 3, 2]),
            (8, 2, [4, 4]),
        ],
    )
    def test_remainder_matches_vllm_get_pp_indices(self, num_layers, pp_size, expected):
        from dmi.configuration.estimate import _stage_layers

        assert [len(_stage_layers(num_layers, pp_size, s)) for s in range(pp_size)] == expected

    def test_small_model_many_stages_leaves_the_first_empty(self):
        from dmi.configuration.estimate import _stage_layers

        assert [len(_stage_layers(32, 40, s)) for s in range(40)][:8] == [0] * 7 + [1]

    def test_explicit_partition_is_accepted_and_validated(self):
        from dmi.configuration import Workload

        good = Workload(pp_layer_counts=(3, 3, 2), pipeline_parallel_size=3)
        assert good.pp_layer_counts == (3, 3, 2)
        with pytest.raises(ValueError, match="pp_layer_counts"):
            Workload(pp_layer_counts=(3, 3), pipeline_parallel_size=3)

    def test_partition_that_does_not_cover_the_model_is_refused(self):
        from dmi.configuration import Workload

        # A (2, 2, 2) partition is well-formed but covers 6 layers, not the
        # descriptor's 32: the sum check needs the topology, so it fires at
        # estimate time, naming the mismatch.
        with pytest.raises(ValueError, match="pp_layer_counts"):
            _estimate(
                DMIConfig(observations=ObservationConfig(hooks=["resid_pre"])),
                pipeline_parallel_size=3,
                pp_layer_counts=(2, 2, 2),
            )


# ---------------------------------------------------------------------------
# 4. Positive decoder identification must fail closed on encoder subtypes
#    (qwen2_audio_encoder slips through the qwen2 family prefix).
# ---------------------------------------------------------------------------


class TestEncoderSubtypesRefused:
    def test_audio_encoder_subtype_is_refused(self):
        from types import SimpleNamespace

        from dmi.configuration.errors import DescriptorError
        from dmi.configuration.introspect import descriptor_from_hf_config

        audio = SimpleNamespace(
            model_type="qwen2_audio_encoder",
            architectures=["Qwen2AudioEncoder"],
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=32,
            intermediate_size=2816,
            vocab_size=152064,
            is_encoder_decoder=False,
        )
        with pytest.raises(DescriptorError, match="decoder"):
            descriptor_from_hf_config(audio, "qwen2-audio")

    def test_decoder_with_classification_head_still_accepted(self):
        from types import SimpleNamespace

        from dmi.configuration.introspect import descriptor_from_hf_config

        clf = SimpleNamespace(
            model_type="llama",
            architectures=["LlamaForSequenceClassification"],
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=4,
            num_key_value_heads=8,
            intermediate_size=11008,
            vocab_size=32000,
            is_encoder_decoder=False,
        )
        descriptor = descriptor_from_hf_config(clf, "llama-clf")
        assert descriptor.model.architecture == "decoder_transformer"


# ---------------------------------------------------------------------------
# 5. Ring + workload boundary: falsy/negative pinned bytes, non-integer
#    cache_max_len, and unhashable YAML keys must all be 400s, not 500s.
# ---------------------------------------------------------------------------


class TestRingBoundaryExactness:
    @pytest.mark.parametrize("bad", [False, 0.0, "", []])
    def test_falsy_pinned_bytes_is_not_silently_defaulted(self, bad, client):
        response = client.post(
            "/api/estimate",
            json={"config": client.valid_config, "ring": {"payload_bytes": 1024, "pinned_bytes": bad}},
        )
        assert response.status_code == 400, bad

    def test_negative_pinned_bytes_is_rejected(self, client):
        response = client.post(
            "/api/estimate",
            json={"config": client.valid_config, "ring": {"payload_bytes": 1024, "pinned_bytes": -1}},
        )
        assert response.status_code == 400


@pytest.fixture
def client():
    pytest.importorskip("fastapi", reason="DMI-configurator UI extra not installed")
    from fastapi.testclient import TestClient
    from pathlib import Path
    from dmi.ui.app import create_app

    dense = Path("examples/model_descriptors/llama3-8b.yaml")
    app = create_app(dense)
    valid_config = {
        "version": 1,
        "observations": {"hooks": ["resid_pre"]},
    }

    class _Client:
        def __init__(self):
            self._client = TestClient(app, base_url="http://127.0.0.1")
            self.valid_config = valid_config

        def post(self, *a, **k):
            return self._client.post(*a, **k)

    return _Client()


class TestCacheMaxLenExactness:
    @pytest.mark.parametrize("bad", [True, 1.5])
    def test_cache_max_len_must_be_an_exact_integer(self, bad):
        from dmi.configuration import Workload

        with pytest.raises(ValueError, match="cache_max_len"):
            Workload(cache_max_len=bad)


class TestMalformedYamlKeysAre400:
    @pytest.mark.parametrize(
        "body",
        [
            "? [a, b]\n: 1\n",
            "{[1, 2]: 3}\n",
        ],
    )
    def test_unhashable_keys_are_a_400_not_a_500(self, body, client):
        response = client.post("/api/config/parse", json={"yaml": body})
        assert response.status_code == 400, body


# ---------------------------------------------------------------------------
# 6. Offset/warmup disclosure must not depend on a stride being set.
# ---------------------------------------------------------------------------


class TestOffsetWarmupDisclosure:
    def test_offset_without_stride_is_disclosed(self):
        from dmi.config import CaptureSchedule

        estimate = _estimate(
            DMIConfig(
                observations=ObservationConfig(hooks=["q"]),
                schedule=CaptureSchedule(step_offset=99),
            ),
            packed=False,
        )
        notes = " ".join(estimate.warnings + estimate.assumptions).lower()
        assert "offset" in notes or "warmup" in notes

    def test_warmup_without_stride_is_disclosed(self):
        from dmi.config import CaptureSchedule

        estimate = _estimate(
            DMIConfig(
                observations=ObservationConfig(hooks=["q"]),
                schedule=CaptureSchedule(warmup_steps=99),
            ),
            packed=False,
        )
        notes = " ".join(estimate.warnings + estimate.assumptions).lower()
        assert "offset" in notes or "warmup" in notes
