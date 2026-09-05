"""Reproduction tests for the Samfisheryu review round on PR #122.

The review was written against ``c19eba5``; ``3dc68ca`` landed minutes later
and had already closed some of it. Every finding is encoded here as the
behaviour the review asks for, so the suite says which is which: a failure is
a live finding, a pass is one the tree already satisfies and now pins.

Findings covered, in the review's own severity order:

Blocker
- compiler.py:98   an executing API must APPLY the schedule, not just return it.
- compiler.py:192  ``generate_with_monitoring`` must not silently re-own an
  already-attached model and reconfigure the shared transport.
- estimate.py:543  packed (vLLM) figures must not be divided by a schedule the
  pinned vLLM integration does not execute.

High
- estimate.py:551  per-request cost under a stride is a long-run average, and
  offsets/warmups must not be silently ignored.
- app.py:376       ``/api/config/save`` validates against a descriptor, so it
  must not present the verdict as runtime-ready.
- compiler.py:94   a live layer range that empties the selection must fail,
  not compile to a silent zero-capture configuration.
- introspect.py:101 architecture detection must fail CLOSED for anything that
  is not a known causal decoder.

Major
- app.py:388       a save must commit file and server state together.
- app.py:360       ``/api/config/parse`` must reject duplicate mapping keys.
- yaml.py:253      ``observations.hooks`` must reject non-strings, not
  stringify them into invented hook names.
- estimate.py:101  ``Workload`` must be type-strict on ``packed`` and the
  decode rate, as it already is on the integer counts.
- app.py (ring)    ring byte counts must not be coerced with ``int()``.

Minor
- server.py:59     an explicit ``--port`` outside 1-65535 must be refused.
- cli.py:161       ``describe-model -o`` failures must reach the CLI's error
  boundary instead of leaking a traceback.

Plus estimate.py's PP partition, which the review marked outdated: pinned here
against vLLM's ``get_pp_indices`` remainder rule.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml as pyyaml

from dmi.adapters.base import BackendAdapter, StepPlan
from dmi.config import CaptureSchedule, MonitoringConfig
from dmi.configuration import (
    ConfigurationError,
    ConfigValidationError,
    DescriptorError,
    DMIConfig,
    LayerSelection,
    ModelContext,
    ObservationConfig,
    Workload,
    attach_config,
    compile_config,
    estimate_config,
    load_descriptor,
    parse_config,
)
from dmi.hooks.specs import HOOK_TYPE_Q, HookSpec

pytestmark = pytest.mark.cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
DENSE_DESCRIPTOR = REPO_ROOT / "examples" / "model_descriptors" / "llama3-8b.yaml"


# ---------------------------------------------------------------------------
# Shared doubles. Kept DAMP: this file reads standalone.
# ---------------------------------------------------------------------------


class _Point:
    """Stand-in for a HookPoint: the filters only touch ``enabled``."""

    def __init__(self) -> None:
        self.enabled = True


class FakeTransport:
    def __init__(self) -> None:
        self.null_offload = False
        self.force_eager = False
        self._ring_payload = None
        self._active_specs: list = []
        self._using_forward_hooks = False
        self.model_cfgs: list = []

    def set_model_cfg(self, cfg) -> None:
        self.model_cfgs.append(cfg)


class FakeEngine:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self._ring_transport = FakeTransport()
        self._model_id = "test-model"
        self.config = config


class RecordingAdapter(BackendAdapter):
    """Adapter whose ``attach_model`` records rather than installs."""

    def __init__(self, engine) -> None:
        super().__init__(engine, engine._model_id)
        self.attach_calls: list = []

    def attach_model(self, model, hook_selection="full", *, layers=None):
        self.attach_calls.append((model, hook_selection, layers))

    def detect_model_shape(self, model):
        raise NotImplementedError

    def detect_parallel_ranks(self):
        return (0, 0, 0, 0)

    def is_pp_first(self):
        return True

    def is_pp_last(self):
        return True

    def build_step_context(self, *raw):
        return None

    def on_capacity_exceeded(self, ctx):
        pass

    def plan_step(self, ctx):
        return StepPlan(total_bytes=0, hook_count=0, needs_eager=False)


def _config(**overrides) -> DMIConfig:
    values = dict(
        observations=ObservationConfig(hooks=["q"]),
        schedule=CaptureSchedule(),
    )
    values.update(overrides)
    return DMIConfig(**values)


# ---------------------------------------------------------------------------
# Blocker -- compiler.py:98: the schedule must actually be applied.
# ---------------------------------------------------------------------------


class TestAttachConfigAppliesTheSchedule:
    """``load_config -> attach_config`` must install the schedule it carries.

    The review's repro: a YAML saying ``capture_prefill: false`` and
    ``step_stride: 17`` attaches hooks and leaves ``adapter.engine.config``
    untouched, so the runtime captures everything.
    """

    def test_schedule_reaches_an_engine_that_had_no_config(self):
        engine = FakeEngine(config=None)
        adapter = RecordingAdapter(engine)
        config = _config(
            schedule=CaptureSchedule(step_stride=17, capture_prefill=False)
        )

        attach_config(adapter, SimpleNamespace(), config)

        assert engine.config is not None, (
            "attach_config left engine.config as None: the schedule the YAML "
            "declared is never consulted by _schedule_allows."
        )
        assert engine.config.schedule == config.schedule

    def test_schedule_replaces_an_existing_default_schedule(self):
        engine = FakeEngine(config=MonitoringConfig(schedule=CaptureSchedule()))
        adapter = RecordingAdapter(engine)
        config = _config(
            schedule=CaptureSchedule(request_stride=19, capture_decode=False)
        )

        attach_config(adapter, SimpleNamespace(), config)

        assert engine.config.schedule.request_stride == 19
        assert engine.config.schedule.capture_decode is False

    def test_the_documented_yaml_to_runtime_flow_applies_both_halves(
        self, tmp_path
    ):
        """load_config(file) -> attach_config: where AND when, in one call."""
        from dmi.configuration import load_config

        document = tmp_path / "capture.dmi.yaml"
        document.write_text(
            "version: 1\n"
            "observations:\n"
            "  hooks: [q]\n"
            "schedule:\n"
            "  capture_prefill: false\n"
            "  step_stride: 17\n"
            "  request_stride: 19\n",
            encoding="utf-8",
        )
        engine = FakeEngine(config=None)
        adapter = RecordingAdapter(engine)
        model = SimpleNamespace()

        attach_config(adapter, model, load_config(document))

        schedule = engine.config.schedule
        assert schedule.capture_prefill is False
        assert (schedule.step_stride, schedule.request_stride) == (17, 19)
        assert adapter.attach_calls[0][1] == "q"

    def test_hooks_are_still_installed_through_the_adapter(self):
        engine = FakeEngine(config=None)
        adapter = RecordingAdapter(engine)
        model = SimpleNamespace()

        attach_config(adapter, model, _config())

        assert adapter.attach_calls, "attach_config must still attach hooks"
        assert adapter.attach_calls[0][0] is model


# ---------------------------------------------------------------------------
# Blocker -- compiler.py:192: one owner for attachment.
# ---------------------------------------------------------------------------


class TestOneOwnerForModelAttachment:
    """``generate_with_monitoring`` must not re-own an attached model.

    The review's repro: attach a narrow configuration, then call the public
    generation helper. It builds a second adapter, re-attaches ``full``, and
    its ``finally`` detaches the first one -- one reservation, five producers.
    """

    def test_generate_with_monitoring_refuses_an_already_attached_model(self):
        from dmi.adapters.huggingface.generation import generate_with_monitoring

        engine = FakeEngine(config=None)
        model = SimpleNamespace(monitoring_engine=engine)
        adapter = RecordingAdapter(engine)
        adapter.attach_model = lambda *a, **k: None  # noqa: E731 - record only
        attach_config(adapter, model, _config())

        with pytest.raises(RuntimeError, match="already attached|already owns"):
            generate_with_monitoring(model)

    def test_the_greedy_loop_refuses_an_already_attached_model_too(self):
        """Same hazard, same shape: it builds its own adapter and detaches."""
        import torch

        from dmi.adapters.huggingface.generation import (
            generate_greedy_with_monitoring,
        )

        engine = FakeEngine(config=None)
        model = SimpleNamespace(
            monitoring_engine=engine, forward=lambda **kwargs: None
        )
        adapter = RecordingAdapter(engine)
        attach_config(adapter, model, _config())

        with pytest.raises(RuntimeError, match="already attached"):
            generate_greedy_with_monitoring(
                model,
                torch.zeros((1, 2), dtype=torch.long),
                torch.ones((1, 2), dtype=torch.long),
                max_new_tokens=1,
                monitoring=True,
            )

    def test_attach_config_records_the_owning_adapter(self):
        """The marker the guard reads is set by whoever attached."""
        engine = FakeEngine(config=None)
        adapter = RecordingAdapter(engine)
        model = SimpleNamespace()

        attach_config(adapter, model, _config())

        assert getattr(model, "_dmi_active_adapter", None) is adapter


# ---------------------------------------------------------------------------
# Blocker -- estimate.py:543: packed figures must not divide by the schedule.
# ---------------------------------------------------------------------------


class TestPackedEstimatesDoNotApplyAnUnenforcedSchedule:
    """The pinned vLLM integrations call ``commit_step`` directly.

    They construct ``MonitoringEngine(config=None)`` and never reach
    ``BackendAdapter._schedule_allows``, so dividing packed byte figures by
    ``step_stride``/``request_stride`` reports a reduction production does not
    perform. Packed is the UI default.
    """

    def _estimate(self, stride: int, packed: bool):
        descriptor = load_descriptor(DENSE_DESCRIPTOR)
        config = _config(
            schedule=CaptureSchedule(step_stride=stride, request_stride=stride)
        )
        workload = Workload(
            batch_size=1,
            prompt_tokens=128,
            decode_tokens=8,
            packed=packed,
            decode_steps_per_second=10.0,
        )
        return estimate_config(config, descriptor, workload)

    def test_packed_per_request_bytes_ignore_the_stride(self):
        unstrided = self._estimate(1, packed=True)
        strided = self._estimate(1000, packed=True)

        assert strided.bytes_per_request == unstrided.bytes_per_request, (
            "packed figures were divided by a schedule the shipped vLLM "
            "runtime does not enforce"
        )

    def test_packed_sustained_rate_ignores_the_stride(self):
        unstrided = self._estimate(1, packed=True)
        strided = self._estimate(1000, packed=True)

        assert strided.sustained_bytes_per_second == pytest.approx(
            unstrided.sustained_bytes_per_second
        )

    def test_packed_estimate_says_why_the_schedule_was_not_applied(self):
        strided = self._estimate(1000, packed=True)

        assert any(
            "vLLM" in warning and "schedule" in warning
            for warning in strided.warnings
        ), strided.warnings

    def test_batched_hugging_face_estimates_still_divide(self):
        """The HF driver DOES gate on the schedule, so batched must divide."""
        unstrided = self._estimate(1, packed=False)
        strided = self._estimate(1000, packed=False)

        assert strided.bytes_per_request < unstrided.bytes_per_request


# ---------------------------------------------------------------------------
# High -- estimate.py:551: a stride is a long-run average, and offsets count.
# ---------------------------------------------------------------------------


class TestScheduleCostIsDisclosedAsALongRunAverage:
    """A finite request does not see the uniform division.

    With ``step_stride=1000`` step 0 is still accepted and its prefill alone
    dwarfs the reported per-request figure; adding ``step_offset=999`` changes
    the early request to zero captures and does not move the estimate.
    """

    def _estimate(self, schedule: CaptureSchedule):
        descriptor = load_descriptor(DENSE_DESCRIPTOR)
        workload = Workload(
            batch_size=1, prompt_tokens=2048, decode_tokens=1, packed=False
        )
        return estimate_config(
            _config(schedule=schedule), descriptor, workload
        )

    def test_a_strided_estimate_is_labelled_a_long_run_average(self):
        estimate = self._estimate(CaptureSchedule(step_stride=1000))

        assert any(
            "average" in text.lower() for text in estimate.assumptions
        ), estimate.assumptions

    def test_a_strided_estimate_bounds_the_single_captured_step(self):
        """The peak step is what a single accepted request actually costs."""
        estimate = self._estimate(CaptureSchedule(step_stride=1000))

        assert estimate.peak_step_bytes > estimate.bytes_per_request
        assert any(
            "peak" in text.lower() and "request" in text.lower()
            for text in estimate.assumptions + estimate.warnings
        ), (estimate.assumptions, estimate.warnings)

    def test_offsets_and_warmups_are_disclosed_rather_than_ignored(self):
        estimate = self._estimate(
            CaptureSchedule(step_stride=1000, step_offset=999, warmup_steps=5)
        )

        assert any(
            "offset" in text.lower() or "warmup" in text.lower()
            for text in estimate.warnings
        ), estimate.warnings


# ---------------------------------------------------------------------------
# High -- compiler.py:94: a range that empties the selection must fail.
# ---------------------------------------------------------------------------


class TestLayerRangeCannotSilentlyEmptyTheSelection:
    """A stale descriptor must not compile to zero capture.

    Live ``q`` specs exist on layers 0-15; the configuration asks for layers
    20-25. The type-presence check passes (``q`` is exposed) and the layer
    filter then returns nothing.
    """

    def _live_specs(self, layers: int = 16):
        return [
            HookSpec(hook_type=HOOK_TYPE_Q, module=_Point(), layer_no=layer)
            for layer in range(layers)
        ]

    def test_a_range_beyond_the_live_layers_raises(self):
        config = _config(
            observations=ObservationConfig(
                hooks=["q"], layers=LayerSelection(start=20, end=25)
            )
        )
        context = ModelContext(specs=self._live_specs(), shape=None)

        with pytest.raises(ConfigValidationError) as excinfo:
            compile_config(config, context)

        assert "20" in str(excinfo.value) and "25" in str(excinfo.value)

    def test_a_partially_overlapping_range_still_compiles(self):
        config = _config(
            observations=ObservationConfig(
                hooks=["q"], layers=LayerSelection(start=12, end=25)
            )
        )
        context = ModelContext(specs=self._live_specs(), shape=None)

        compiled = compile_config(config, context)

        assert compiled.selected_layers == [12, 13, 14, 15]


# ---------------------------------------------------------------------------
# High -- introspect.py:101: architecture detection must fail closed.
# ---------------------------------------------------------------------------


class TestArchitectureDetectionFailsClosed:
    """A blacklist cannot establish causal-decoder support.

    ``dinov2`` and ``modernbert`` are absent from the list entirely, and
    ``bert-generation`` escapes the ``bert_`` prefix rule the comment promises.
    """

    def _hf_config(self, model_type: str, **extra):
        values = dict(
            model_type=model_type,
            num_hidden_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=128,
        )
        values.update(extra)
        return SimpleNamespace(**values)

    @pytest.mark.parametrize(
        "model_type", ["dinov2", "modernbert", "bert-generation"]
    )
    def test_non_decoder_families_are_refused(self, model_type):
        from dmi.configuration.introspect import descriptor_from_hf_config

        with pytest.raises(DescriptorError):
            descriptor_from_hf_config(self._hf_config(model_type), model_type)

    def test_an_unknown_model_type_is_refused_rather_than_guessed(self):
        from dmi.configuration.introspect import descriptor_from_hf_config

        with pytest.raises(DescriptorError):
            descriptor_from_hf_config(
                self._hf_config("wildly-unknown-arch"), "acme/unknown"
            )

    @pytest.mark.parametrize("model_type", ["llama", "qwen3", "mistral", "gpt2"])
    def test_known_decoders_still_describe(self, model_type):
        from dmi.configuration.introspect import descriptor_from_hf_config

        descriptor = descriptor_from_hf_config(
            self._hf_config(model_type), f"acme/{model_type}"
        )

        assert descriptor.topology.num_layers == 4


# ---------------------------------------------------------------------------
# Major -- yaml.py:253: a typed list[str] rejects non-strings.
# ---------------------------------------------------------------------------


class TestHookListIsStrictlyStrings:
    """``[q, 1, true, null]`` must not become ``["q","1","True","None"]``."""

    def _document(self, hooks):
        return {"version": 1, "observations": {"hooks": hooks}}

    @pytest.mark.parametrize("bad", [1, True, None, 3.5, ["q"]])
    def test_a_non_string_entry_is_refused(self, bad):
        with pytest.raises(ConfigurationError):
            parse_config(self._document(["q", bad]))

    def test_the_message_names_the_offending_index_and_type(self):
        with pytest.raises(ConfigurationError) as excinfo:
            parse_config(self._document(["q", 1]))

        message = str(excinfo.value)
        assert "1" in message and "int" in message

    def test_a_well_formed_list_still_parses(self):
        config = parse_config(self._document(["q", "k"]))

        assert config.observations.hooks == ["q", "k"]


# ---------------------------------------------------------------------------
# Major -- estimate.py:101: Workload type strictness.
# ---------------------------------------------------------------------------


class TestWorkloadIsTypeStrict:
    """Integer counts are already strict; ``packed`` and the rate are not."""

    def test_packed_must_be_an_actual_boolean(self):
        with pytest.raises(ValueError, match="packed"):
            Workload(packed="false")

    def test_decode_rate_must_be_a_number(self):
        with pytest.raises(ValueError, match="decode_steps_per_second"):
            Workload(decode_steps_per_second="fast")

    def test_decode_rate_rejects_a_boolean(self):
        with pytest.raises(ValueError, match="decode_steps_per_second"):
            Workload(decode_steps_per_second=True)

    def test_fractional_counts_are_still_refused(self):
        with pytest.raises(ValueError, match="batch_size"):
            Workload(batch_size=1.5)


# ---------------------------------------------------------------------------
# Minor -- server.py:59: explicit ports need range validation.
# ---------------------------------------------------------------------------


class TestExplicitPortRange:
    def test_port_zero_is_refused(self):
        from dmi.ui.server import resolve_port

        with pytest.raises(ConfigurationError, match="port"):
            resolve_port("127.0.0.1", 0)

    def test_port_above_the_range_is_refused(self):
        from dmi.ui.server import resolve_port

        with pytest.raises(ConfigurationError, match="port"):
            resolve_port("127.0.0.1", 65536)

    def test_a_normal_port_is_returned_unchanged(self):
        from dmi.ui.server import resolve_port

        assert resolve_port("127.0.0.1", 8123) == 8123


# ---------------------------------------------------------------------------
# Minor -- cli.py:161: describe-model output failures use the error boundary.
# ---------------------------------------------------------------------------


class TestDescribeModelWriteFailureIsACleanError:
    def _model_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "acme-model"
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 4,
                    "hidden_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "vocab_size": 128,
                }
            ),
            encoding="utf-8",
        )
        return directory

    def test_an_unwritable_output_path_exits_one_without_a_traceback(
        self, tmp_path, capsys
    ):
        from dmi.cli import main

        model = self._model_dir(tmp_path)
        target = tmp_path / "missing" / "parent" / "out.yaml"

        code = main(["describe-model", str(model), "-o", str(target)])

        assert code == 1
        assert "dmi describe-model:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Regression pins for the two threads the review marked outdated.
# ---------------------------------------------------------------------------


class TestPipelinePartitionMatchesVLLM:
    def test_eleven_layers_over_four_stages_puts_the_remainder_first(self):
        from dmi.configuration.estimate import _stage_layers

        split = [len(_stage_layers(11, 4, stage)) for stage in range(4)]

        assert split == [3, 3, 3, 2]

    def test_a_pipelined_estimate_discloses_the_partition_override(self):
        descriptor = load_descriptor(DENSE_DESCRIPTOR)
        estimate = estimate_config(
            _config(),
            descriptor,
            Workload(pipeline_parallel_size=2, packed=True),
        )

        assert any(
            "VLLM_PP_LAYER_PARTITION" in text
            for text in estimate.assumptions + estimate.warnings
        ), (estimate.assumptions, estimate.warnings)


# ---------------------------------------------------------------------------
# HTTP surface. Skipped whole when the optional UI stack is absent.
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="DMI[ui] not installed")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    from dmi.ui.app import create_app

    descriptor = tmp_path / "llama3-8b.yaml"
    descriptor.write_text(DENSE_DESCRIPTOR.read_text(encoding="utf-8"), "utf-8")
    app = create_app(descriptor, tmp_path / "out.dmi.yaml")
    # base_url matters: TestClient's default ``testserver`` Host is refused by
    # the loopback TrustedHostMiddleware, which would turn every assertion
    # about a 400 into a pass for the wrong reason.
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        test_client.save_path = tmp_path / "out.dmi.yaml"
        yield test_client


def _body(hooks=("q",), step_stride=1):
    return {
        "config": {
            "version": 1,
            "observations": {"hooks": list(hooks)},
            "schedule": {"step_stride": step_stride},
        }
    }


class TestSaveIsLabelledDesignTime:
    """``/api/config/save`` validates against a portable descriptor.

    ``hooks: [pos_embed]`` passes descriptor validation and then fails to
    compile against a live Llama, so the endpoint must not present its verdict
    as runtime-ready.
    """

    def test_save_response_states_the_scope_of_its_validation(self, client):
        response = client.post("/api/config/save", json=_body())

        assert response.status_code == 200
        assert response.json().get("validated") == "design-time"

    def test_validate_response_states_the_scope_too(self, client):
        response = client.post("/api/validate", json=_body())

        assert response.status_code == 200
        assert response.json().get("scope") == "design-time"

    def test_the_note_names_the_descriptor_as_the_authority(self, client):
        note = client.post("/api/validate", json=_body()).json().get("note", "")

        assert "descriptor" in note.lower()


class TestConcurrentSavesCommitOnce:
    """Server state and the file on disk must never disagree."""

    def test_the_reported_config_matches_the_file_after_racing_saves(
        self, client
    ):
        errors: list = []

        def save(index: int) -> None:
            try:
                hooks = ("q",) if index % 2 == 0 else ("k",)
                stride = 1 if index % 2 == 0 else 2
                response = client.post(
                    "/api/config/save", json=_body(hooks, stride)
                )
                assert response.status_code == 200, response.text
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=save, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        on_disk = pyyaml.safe_load(client.save_path.read_text(encoding="utf-8"))
        served = client.get("/api/config").json()["config"]

        assert served["observations"]["hooks"] == on_disk["observations"]["hooks"]
        assert served["schedule"]["step_stride"] == on_disk["schedule"]["step_stride"]


class TestParseRejectsDuplicateKeys:
    """A merge conflict must not silently change capture scope."""

    def test_duplicate_observation_keys_are_a_client_error(self, client):
        document = (
            "version: 1\n"
            "observations:\n"
            "  hooks: [q]\n"
            "  hooks: [k, v, mlp_out]\n"
        )

        response = client.post("/api/config/parse", json={"yaml": document})

        assert response.status_code == 400
        assert "duplicate" in response.json()["detail"].lower()

    def test_duplicate_top_level_keys_are_a_client_error(self, client):
        document = "version: 1\nschedule:\n  step_stride: 1\nschedule:\n  step_stride: 9\n"

        response = client.post("/api/config/parse", json={"yaml": document})

        assert response.status_code == 400

    def test_a_clean_document_still_parses(self, client):
        document = "version: 1\nobservations:\n  hooks: [q]\n"

        response = client.post("/api/config/parse", json={"yaml": document})

        assert response.status_code == 200

    @pytest.mark.parametrize("bad", [None, True, 7, ["q"], {"hooks": []}])
    def test_a_non_string_yaml_field_is_a_client_error(self, client, bad):
        response = client.post("/api/config/parse", json={"yaml": bad})

        assert response.status_code == 400


class TestEstimateBoundaryIsTypeStrict:
    @pytest.mark.parametrize(
        "ring",
        [
            {"payload_bytes": 1024.0},
            {"payload_bytes": True},
            {"payload_bytes": 1024, "pinned_bytes": 2.5},
        ],
    )
    def test_non_integer_ring_sizes_are_refused(self, client, ring):
        payload = dict(_body())
        payload["ring"] = ring

        response = client.post("/api/estimate", json=payload)

        assert response.status_code == 400

    @pytest.mark.parametrize(
        "workload",
        [
            {"batch_size": 1.5},
            {"packed": "false"},
            {"decode_steps_per_second": "fast"},
            {"tensor_parallel_size": True},
        ],
    )
    def test_malformed_workload_scalars_are_refused(self, client, workload):
        payload = dict(_body())
        payload["workload"] = workload

        response = client.post("/api/estimate", json=payload)

        assert response.status_code == 400
