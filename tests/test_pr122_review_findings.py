"""Reproduction harness for the Copilot review on PR #122.

Every test class maps to one Copilot review comment and encodes the behaviour
the review asks for. Run it against the unfixed tree: failures are confirmed
bugs; passes are findings Copilot raised that the build had already addressed
(kept as regression pins). The same harness runs after the fix.

Comments covered:

- 3910708530  ``parse_config`` must require an explicit integer ``version``.
- 3910708559  plan doc snippets must match the implemented schema/compiler.
- 3910708590  ``python-multipart`` must not ship in the ``[ui]`` extra.
- 3917511836  ``or {}``/``or []`` defaulting must not bypass type checks.
- 3917511917  ``schedule or {}`` must not treat ``schedule: []`` as missing.
- 3917511962  ``/api/config/save`` must translate the ``ConfigurationError``
  that ``save_config`` actually raises, not ``OSError``.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml as pyyaml

from dmi.config import CaptureSchedule
from dmi.configuration.schema import ModelDescriptor, ModelTopology
from dmi.configuration.manifest import parse_descriptor
from dmi.configuration import (
    ConfigurationError,
    DMIConfig,
    LayerSelection,
    ObservationConfig,
    RuntimePolicy,
    UnsupportedConfigVersion,
    compile_config,
    parse_config,
)
from dmi.configuration.compiler import ModelContext
from dmi.hooks.selection import (
    filter_by_layers,
    hook_belongs_to_layers,
    select_hook_specs,
)
from dmi.configuration.compatibility import to_legacy_hook_selection
from dmi.configuration.introspect import descriptor_from_hf_config
from dmi.configuration.errors import DescriptorError
from dmi.hooks.specs import HOOK_TYPE_RESID_PRE

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
DENSE = REPO / "examples" / "model_descriptors" / "llama3-8b.yaml"
PLAN = REPO / "docs" / "dmi-configurator-plan.md"


def _test_client():
    """Skip, not error, when the UI extra or its test-only transport is absent.

    fastapi.testclient needs httpx2 (or the deprecated httpx), which the [ui]
    extra deliberately does not carry: the UI never makes HTTP requests.
    """
    pytest.importorskip("fastapi", reason="DMI-configurator UI extra not installed")
    if not any(importlib.util.find_spec(name) for name in ("httpx2", "httpx")):
        pytest.skip("fastapi.testclient needs httpx2 (or httpx); test-only, see CI install step")
    from fastapi.testclient import TestClient

    return TestClient

UNVERSIONED = {"observations": {"hooks": ["resid_pre"]}}
VALID_CONFIG = {
    "version": 1,
    "observations": {
        "layers": {"start": 8, "end": 15},
        "hooks": ["q", "k"],
    },
    "schedule": {
        "step_stride": 4,
        "request_stride": 1,
        "capture_prefill": True,
        "capture_decode": True,
    },
}


# ---------------------------------------------------------------------------
# Comment 3910708530: version is mandatory and must be an integer.
# Already addressed in this build; pinned so it cannot regress.
# ---------------------------------------------------------------------------


class TestVersionEnforcement:
    def test_missing_version_is_refused(self):
        with pytest.raises(ConfigurationError, match="missing 'version'"):
            parse_config(dict(UNVERSIONED))

    def test_string_version_is_malformed_not_unsupported(self):
        with pytest.raises(ConfigurationError, match="must be an integer"):
            parse_config({**UNVERSIONED, "version": "1"})

    def test_boolean_version_is_malformed(self):
        with pytest.raises(ConfigurationError, match="must be an integer"):
            parse_config({**UNVERSIONED, "version": True})

    def test_wrong_version_is_unsupported(self):
        with pytest.raises(UnsupportedConfigVersion):
            parse_config({**UNVERSIONED, "version": 99})


# ---------------------------------------------------------------------------
# Comments 3917511836 / 3917511917: `or {}` / `or []` treats a falsy
# malformed value as "missing" and bypasses the type check. Each falsy
# document below must be refused, not silently defaulted.
# ---------------------------------------------------------------------------


class TestFalsySectionsAreRejected:
    @pytest.mark.parametrize("bad", [[], "", 0, False])
    def test_falsy_observations_is_not_silently_defaulted(self, bad):
        with pytest.raises(ConfigurationError, match="'observations' must be a mapping"):
            parse_config({"version": 1, "observations": bad})

    @pytest.mark.parametrize("bad", ["", 0, False, {}])
    def test_falsy_hooks_is_not_silently_defaulted(self, bad):
        with pytest.raises(ConfigurationError, match="must be a list"):
            parse_config(
                {"version": 1, "observations": {"hooks": bad}}
            )

    @pytest.mark.parametrize("bad", [[], "", 0, False])
    def test_falsy_schedule_is_not_silently_defaulted(self, bad):
        with pytest.raises(
            ConfigurationError, match="'schedule' must be a mapping"
        ):
            parse_config(
                {"version": 1, "observations": {"hooks": ["resid_pre"]}, "schedule": bad}
            )

    def test_absent_sections_still_default(self):
        # The fix must tighten types, not require the keys.
        assert parse_config({"version": 1}) == DMIConfig()


# ---------------------------------------------------------------------------
# Comment 3917511962: save_config reports filesystem failure as
# ConfigurationError; /api/config/save must catch that, not OSError.
# ---------------------------------------------------------------------------


class TestSaveEndpointTranslatesConfigurationError:
    @pytest.fixture
    def doomed_client(self, tmp_path):
        TestClient = _test_client()
        from dmi.ui.app import create_app

        # A save path whose parent does not exist: startup does not touch it
        # (so build_state succeeds) but the write fails inside save_config.
        target = tmp_path / "no-such-dir" / "out.dmi.yaml"
        return TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")

    def test_write_failure_is_a_translated_500(self, doomed_client):
        # Before the fix this raises ConfigurationError straight through the
        # TestClient: the endpoint's `except OSError` is dead code because
        # save_config never lets an OSError escape.
        response = doomed_client.post(
            "/api/config/save", json={"config": VALID_CONFIG}
        )
        assert response.status_code == 500
        assert "Could not write" in response.json()["detail"]

    def test_write_failure_message_names_the_path(self, doomed_client):
        response = doomed_client.post(
            "/api/config/save", json={"config": VALID_CONFIG}
        )
        assert "no-such-dir" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Same `or {}` class, API surface: `Workload(**(payload.get("workload") or {}))`
# silently defaults a falsy non-mapping. The request body must be type-checked.
# ---------------------------------------------------------------------------


class TestWorkloadTypeIsChecked:
    @pytest.fixture
    def client(self):
        TestClient = _test_client()
        from dmi.ui.app import create_app

        return TestClient(create_app(DENSE), base_url="http://127.0.0.1")

    @pytest.mark.parametrize("bad", [[], "", 0, False])
    def test_falsy_workload_is_a_400(self, client, bad):
        response = client.post(
            "/api/estimate", json={"config": VALID_CONFIG, "workload": bad}
        )
        assert response.status_code == 400
        assert "must be a mapping" in response.json()["detail"]

    def test_truthy_non_mapping_workload_is_a_400(self, client):
        response = client.post(
            "/api/estimate", json={"config": VALID_CONFIG, "workload": "prompt_tokens"}
        )
        assert response.status_code == 400

    def test_absent_workload_still_defaults(self, client):
        response = client.post("/api/estimate", json={"config": VALID_CONFIG})
        assert response.status_code == 200
        assert response.json()["peak_step_bytes"] > 0


# ---------------------------------------------------------------------------
# Comment 3910708559: the plan doc's schema/compiler snippets must match the
# implementation. The DMIConfig/CompiledDMIConfig `runtime:` drift was already
# fixed; the compile_config snippet still calls
# filter_by_layers(specs, config.observations.layers), which raises TypeError
# against the real (specs, start, end) signature.
# ---------------------------------------------------------------------------


@dataclass
class _StubSpec:
    """Duck-typed HookSpec: the selection path only reads these two fields.

    ``hook_type`` is the integer constant from the catalog, not the short
    name -- that is what ``select_hook_specs`` unions against.
    """

    hook_type: int
    layer_no: int


def _compilation_snippets() -> list[str]:
    text = PLAN.read_text(encoding="utf-8")
    section = text.split("## 7. Compilation", 1)[1].split("## 8.", 1)[0]
    return [block.split("```", 1)[0] for block in section.split("```python")[1:]]


class TestPlanDocSnippetsMatchImplementation:
    def test_no_schema_snippet_carries_a_runtime_field(self):
        text = PLAN.read_text(encoding="utf-8")
        assert "runtime: RuntimeConfig" not in text

    def test_no_snippet_passes_a_layer_selection_to_filter_by_layers(self):
        text = PLAN.read_text(encoding="utf-8")
        assert "filter_by_layers(specs, config.observations.layers)" not in text

    def test_compilation_snippet_executes_and_matches_the_compiler(self):
        snippets = _compilation_snippets()
        assert snippets, "section 7 lost its code blocks"

        namespace: dict = {
            "dataclass": dataclass,
            "HookSpec": object,  # only appears in the dataclass annotations
            "CaptureSchedule": CaptureSchedule,
            "RuntimePolicy": RuntimePolicy,
            "select_hook_specs": select_hook_specs,
            "to_legacy_hook_selection": to_legacy_hook_selection,
            "hook_belongs_to_layers": hook_belongs_to_layers,
            "filter_by_layers": filter_by_layers,
        }
        for snippet in snippets:
            exec(snippet, namespace)  # noqa: S102 - harness executes repo docs

        config = DMIConfig(
            observations=ObservationConfig(
                hooks=["resid_pre"], layers=LayerSelection(8, 15)
            )
        )
        context = ModelContext(
            specs=[
                _StubSpec(HOOK_TYPE_RESID_PRE, 8),
                _StubSpec(HOOK_TYPE_RESID_PRE, 12),
                _StubSpec(HOOK_TYPE_RESID_PRE, 30),
            ]
        )

        compiled = namespace["compile_config"](config, context)
        reference = compile_config(config, context)

        assert [spec.layer_no for spec in compiled.hook_specs] == [8, 12]
        assert [spec.layer_no for spec in compiled.hook_specs] == [
            spec.layer_no for spec in reference.hook_specs
        ]


# ---------------------------------------------------------------------------
# Comment 3910708590: python-multipart was in the [ui] extra although the
# backend is JSON-only. Already removed; pinned so it stays out.
# ---------------------------------------------------------------------------


class TestUiExtraDependencySurface:
    def test_ui_extra_does_not_declare_python_multipart(self):
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert "python-multipart" not in text

    def test_ui_code_uses_no_multipart_reception(self):
        for path in (REPO / "src" / "dmi" / "ui").rglob("*.py"):
            body = path.read_text(encoding="utf-8")
            assert not re.search(r"UploadFile|File\(|Form\(", body), path


# ---------------------------------------------------------------------------
# Finding 3919326717 (XbzOnGit): the YAML boundary is not type-strict.
# `int(2.9)` truncates, `int(True)` is 1, quoted "false" stays a truthy
# string, and unknown keys inside `layers` and `policy` are ignored.
# ---------------------------------------------------------------------------


class TestScalarTypesAreExact:
    @pytest.mark.parametrize("bad", [2.9, True, "8", "8.0"])
    def test_layer_bounds_must_be_exact_integers(self, bad):
        with pytest.raises(ConfigurationError, match="must be an integer"):
            parse_config(
                {
                    "version": 1,
                    "observations": {
                        "hooks": ["resid_pre"],
                        "layers": {"start": bad, "end": 15},
                    },
                }
            )

    @pytest.mark.parametrize("field", ["step_stride", "request_stride"])
    def test_schedule_ints_must_be_exact_integers(self, field):
        with pytest.raises(ConfigurationError, match="must be an integer"):
            parse_config(
                {
                    "version": 1,
                    "observations": {"hooks": ["resid_pre"]},
                    "schedule": {field: 2.5},
                }
            )

    @pytest.mark.parametrize(
        "field", ["capture_prefill", "capture_decode"]
    )
    def test_schedule_bools_must_be_real_bools(self, field):
        with pytest.raises(ConfigurationError, match="must be a boolean"):
            parse_config(
                {
                    "version": 1,
                    "observations": {"hooks": ["resid_pre"]},
                    "schedule": {field: "false"},
                }
            )

    def test_unknown_keys_inside_layers_are_refused(self):
        with pytest.raises(ConfigurationError, match="Unknown field"):
            parse_config(
                {
                    "version": 1,
                    "observations": {
                        "hooks": ["resid_pre"],
                        "layers": {"start": 8, "end": 15, "inclusive": True},
                    },
                }
            )

    def test_unknown_keys_inside_policy_are_refused(self):
        with pytest.raises(ConfigurationError, match="Unknown field"):
            parse_config(
                {
                    "version": 1,
                    "observations": {"hooks": ["resid_pre"]},
                    "policy": {"objective": "balanced", "weight": 0.5},
                }
            )

    def test_schedule_offset_fields_must_be_exact_integers(self):
        with pytest.raises(ConfigurationError, match="must be an integer"):
            parse_config(
                {
                    "version": 1,
                    "observations": {"hooks": ["resid_pre"]},
                    "schedule": {"warmup_steps": True},
                }
            )


# ---------------------------------------------------------------------------
# Finding 3919326719 (XbzOnGit): descriptor topology validation accepts
# values that fail downstream -- `num_layers: 1.5` survives these checks and
# raises TypeError in range() through the estimate API; head_dim is not
# checked at all.
# ---------------------------------------------------------------------------


class TestTopologyTypesAreExact:
    def _topology(self, **overrides):
        base = dict(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )
        base.update(overrides)
        return ModelTopology(**base)

    @pytest.mark.parametrize("field", ["num_layers", "hidden_size", "num_attention_heads", "num_kv_heads"])
    def test_integral_geometry_must_be_exact_integers(self, field):
        with pytest.raises(TypeError, match="must be an integer"):
            self._topology(**{field: 1.5})

    def test_boolean_geometry_is_not_an_integer(self):
        with pytest.raises(TypeError, match="must be an integer"):
            self._topology(num_layers=True)

    @pytest.mark.parametrize("bad", [0, -8, 1.5, True])
    def test_head_dim_must_be_a_positive_exact_integer(self, bad):
        with pytest.raises((TypeError, ValueError), match="head_dim"):
            self._topology(head_dim=bad)

    def test_one_and_a_half_layers_cannot_reach_the_estimate_api(self):
        import pytest as _pytest
        import yaml

        from dmi.configuration import estimate_config, Workload
        from dmi.configuration.errors import DescriptorError

        document = {
            "schema_version": 1,
            "model": {"id": "half", "name": "Half", "architecture": "decoder_transformer"},
            "topology": {
                "num_layers": 1.5,
                "hidden_size": 512,
                "num_attention_heads": 8,
                "num_kv_heads": 8,
            },
        }
        # Refused at the descriptor boundary -- it can no longer survive
        # parsing and explode later as TypeError inside range().
        with _pytest.raises(DescriptorError, match="must be an integer"):
            parse_descriptor(document)


# ---------------------------------------------------------------------------
# Finding 3919326723 (XbzOnGit): `is_encoder_decoder == False` does not imply
# causal decoder-only. BERT- and ViT-shaped configs were labeled
# decoder_transformer and rendered with decoder observations.
# ---------------------------------------------------------------------------


class TestArchitectureIsPositivelyIdentified:
    def _bert_shaped(self, **extra):
        from types import SimpleNamespace

        base = dict(
            model_type="bert",
            architectures=["BertModel"],
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            num_key_value_heads=None,
            intermediate_size=3072,
            vocab_size=30522,
            is_encoder_decoder=False,
        )
        base.update(extra)
        return SimpleNamespace(**base)

    def _vit_shaped(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            model_type="vit",
            architectures=["ViTModel"],
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            num_key_value_heads=None,
            intermediate_size=3072,
            vocab_size=0,
            is_encoder_decoder=False,
            patch_size=16,
        )

    def test_bert_shaped_config_is_refused(self):
        with pytest.raises(DescriptorError, match="decoder_transformer|decoder"):
            descriptor_from_hf_config(self._bert_shaped(), "bert-base")

    def test_vit_shaped_config_is_refused(self):
        with pytest.raises(DescriptorError, match="decoder_transformer|decoder"):
            descriptor_from_hf_config(self._vit_shaped(), "vit-base")

    def test_causal_lm_architectures_are_accepted(self):
        from types import SimpleNamespace

        causal = SimpleNamespace(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=32,
            num_key_value_heads=8,
            intermediate_size=11008,
            vocab_size=32000,
            is_encoder_decoder=False,
        )
        descriptor = descriptor_from_hf_config(causal, "llama")
        assert descriptor.model.architecture == "decoder_transformer"


# ---------------------------------------------------------------------------
# Copilot's third pass (suppressed, "previously missed"): fastapi.testclient
# needs an HTTP transport (httpx2, or the deprecated httpx) that is not part
# of the [ui] extra -- the UI itself never makes HTTP requests. With the extra
# installed but no transport, `from fastapi.testclient import TestClient`
# raises at collection time and the whole API contract suite errors instead
# of skipping like every other optional-dependency suite in this tree.
# ---------------------------------------------------------------------------


class TestApiSuiteSkipsWithoutTestClientTransport:
    # Selected so the run covers every module that imports TestClient.
    TARGETS = (
        "tests/test_configurator_api.py",
        "tests/test_pr122_review_findings.py::TestSaveEndpointTranslatesConfigurationError",
        "tests/test_pr122_review_findings.py::TestWorkloadTypeIsChecked",
    )

    def test_missing_transport_skips_instead_of_erroring(self):
        pytest.importorskip("fastapi", reason="DMI-configurator UI extra not installed")
        # `sys.modules[name] = None` makes both `import name` fail and
        # `importlib.util.find_spec(name)` return None, so the child sees a
        # machine with the [ui] extra but neither transport installed.
        program = (
            "import sys, pytest\n"
            "sys.modules['httpx'] = None\n"
            "sys.modules['httpx2'] = None\n"
            "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', "
            "'-o', 'addopts=', *sys.argv[1:]]))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program, *self.TARGETS],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=300,
        )
        summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        detail = f"exit={result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        assert result.returncode == 0, detail
        assert "skipped" in summary and "error" not in summary, detail
        # A pass would mean the transport was not actually blocked.
        assert "passed" not in summary, detail


# ---------------------------------------------------------------------------
# vLLM integration follow-up (#20 / DMI-vLLM-Integration#21): the facade
# must export the layer-range pieces the integration imports through
# dmi.api.v1. Internal import paths are not integration surface.
# ---------------------------------------------------------------------------


class TestFacadeExportsLayerRangeAPI:
    def test_facade_exports_layer_range_api(self):
        import dmi.api.v1 as facade

        for name in (
            "filter_by_layers",
            "hook_belongs_to_layers",
            "LayerSelection",
        ):
            assert name in facade.__all__, f"{name} must be in the facade __all__"
            assert hasattr(facade, name)

    def test_exports_are_the_real_implementations(self):
        from dmi.api import v1 as facade
        from dmi.configuration.schema import LayerSelection as _Real
        from dmi.hooks.selection import (
            filter_by_layers as _filter,
            hook_belongs_to_layers as _belongs,
        )

        assert facade.filter_by_layers is _filter
        assert facade.hook_belongs_to_layers is _belongs
        assert facade.LayerSelection is _Real


# ---------------------------------------------------------------------------
# Independent-review round: save_config truncates the old file before
# writing, so a mid-write failure destroys the user's config; the
# descriptor parser accepts bool/float schema_version, ignores unknown keys
# in `model` and at the root, and lets a bad model.id escape as ValueError.
# ---------------------------------------------------------------------------


class TestSaveIsAtomic:
    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path, monkeypatch):
        target = tmp_path / "config.dmi.yaml"
        target.write_text("version: 1\nobservations:\n  hooks: [q]\n", encoding="utf-8")
        original = target.read_text()

        import dmi.configuration.yaml as config_yaml

        real_dump = config_yaml.dump_config

        def failing_dump(config):
            data = real_dump(config)
            # Simulate ENOSPC mid-write: full payload, then boom.
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(config_yaml, "dump_config", failing_dump)
        config = parse_config({"version": 1, "observations": {"hooks": ["k"]}})

        with pytest.raises(config_yaml.ConfigurationError):
            config_yaml.save_config(config, target)

        assert target.read_text() == original, (
            "a failed save must not destroy the previous configuration"
        )


class TestDescriptorBoundaryMatchesConfigStrictness:
    def test_schema_version_must_be_exact_int(self):
        for bad in (True, 1.0):
            with pytest.raises(DescriptorError, match="schema_version"):
                parse_descriptor({
                    "schema_version": bad,
                    "model": {"id": "m", "name": "M", "architecture": "decoder_transformer"},
                    "topology": {"num_layers": 2, "hidden_size": 8,
                                 "num_attention_heads": 2, "num_kv_heads": 2},
                })

    def test_unknown_keys_in_model_section_are_refused(self):
        with pytest.raises(DescriptorError, match="Unknown field"):
            parse_descriptor({
                "model": {"id": "m", "name": "M", "architecture": "decoder_transformer",
                          "context_length": 131072},
                "topology": {"num_layers": 2, "hidden_size": 8,
                             "num_attention_heads": 2, "num_kv_heads": 2},
            })

    def test_unknown_root_keys_are_refused(self):
        with pytest.raises(DescriptorError, match="Unknown field"):
            parse_descriptor({
                "model": {"id": "m", "name": "M", "architecture": "decoder_transformer"},
                "topology": {"num_layers": 2, "hidden_size": 8,
                             "num_attention_heads": 2, "num_kv_heads": 2},
                "quantization": "fp8",
            })

    def test_bad_model_id_is_a_descriptor_error_not_valueerror(self):
        with pytest.raises(DescriptorError, match="path segment"):
            parse_descriptor({
                "model": {"id": "org/model", "name": "M",
                          "architecture": "decoder_transformer"},
                "topology": {"num_layers": 2, "hidden_size": 8,
                             "num_attention_heads": 2, "num_kv_heads": 2},
            })

    def test_num_experts_without_top_k_is_refused(self):
        with pytest.raises((DescriptorError, ValueError), match="top_k"):
            parse_descriptor({
                "model": {"id": "m", "name": "M", "architecture": "decoder_transformer"},
                "topology": {"num_layers": 2, "hidden_size": 8,
                             "num_attention_heads": 2, "num_kv_heads": 2,
                             "num_experts": 8},
            })
