"""HTTP contract for the DMI-configurator backend.

The point of these tests is that the browser never decides anything: every
answer comes from the same Python the runtime uses.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="DMI-configurator UI extra not installed")
# fastapi.testclient needs an HTTP transport (httpx2, or the deprecated httpx)
# that the [ui] extra deliberately does not carry: the UI itself never makes
# HTTP requests. Without one the import raises RuntimeError at collection,
# so check for it here and skip like every other optional-dependency suite.
if not any(importlib.util.find_spec(name) for name in ("httpx2", "httpx")):
    pytest.skip(
        "fastapi.testclient needs httpx2 (or httpx); test-only, see CI install step",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from dmi.configuration import (  # noqa: E402
    config_to_dict,
    load_config,
    normalize_config,
    parse_config,
)
from dmi.ui.app import build_state, create_app, default_save_path  # noqa: E402

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[1]
DENSE = REPO / "examples" / "model_descriptors" / "llama3-8b.yaml"
MOE = REPO / "tests" / "data" / "moe-decoder.model.yaml"

VALID_CONFIG = {
    "version": 1,
    "observations": {
        "layers": {"start": 8, "end": 15},
        "hooks": ["q", "k", "v", "pattern"],
    },
    "schedule": {
        "step_stride": 4,
        "request_stride": 1,
        "capture_prefill": True,
        "capture_decode": True,
    },
    "policy": {"objective": "balanced"},
}


@pytest.fixture
def client():
    return TestClient(create_app(DENSE), base_url="http://127.0.0.1")


@pytest.fixture
def moe_client():
    return TestClient(create_app(MOE), base_url="http://127.0.0.1")


class TestStaticSurface:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "DMI-configurator" in response.text

    @pytest.mark.parametrize(
        "asset", ["app.js", "architecture.js", "styles.css"]
    )
    def test_assets_are_served_without_a_build_step(self, client, asset):
        assert client.get(f"/static/{asset}").status_code == 200


class TestModelEndpoint:
    def test_reports_topology_and_layout(self, client):
        payload = client.get("/api/model").json()
        assert payload["name"] == "Llama 3 8B"
        assert payload["topology"]["num_layers"] == 32
        assert payload["topology"]["head_dim"] == 128

    def test_layout_nodes_carry_scope_and_hooks(self, client):
        layout = client.get("/api/model").json()["architecture_layout"]
        assert layout["num_layers"] == 32
        node_ids = {node["id"] for node in layout["nodes"]}
        assert {"input", "attention", "mlp", "output"} <= node_ids
        for node in layout["nodes"]:
            assert node["scope"] in ("global", "layer")
            assert node["hooks"]

    def test_moe_node_is_unavailable_on_a_dense_model(self, client):
        layout = client.get("/api/model").json()["architecture_layout"]
        moe = [node for node in layout["nodes"] if node["id"] == "moe"][0]
        # Shown but greyed, so the absence is explained rather than mysterious.
        assert moe["available"] is False
        # Reasons differ per hook ("no experts" vs "no top-k expert routing"),
        # but every one must explain itself.
        assert all("expert" in hook["reason"] for hook in moe["hooks"])

    def test_moe_node_is_available_on_a_moe_model(self, moe_client):
        layout = moe_client.get("/api/model").json()["architecture_layout"]
        moe = [node for node in layout["nodes"] if node["id"] == "moe"][0]
        assert moe["available"] is True
        assert layout["is_moe"] is True


class TestCatalogEndpoint:
    def test_groups_are_ordered_and_populated(self, client):
        payload = client.get("/api/catalog").json()
        assert [group["id"] for group in payload["groups"]] == [
            "attention",
            "mlp",
            "moe",
            "residual",
            "global",
        ]

    def test_unavailable_hooks_carry_a_reason(self, client):
        payload = client.get("/api/catalog").json()
        moe = [group for group in payload["groups"] if group["id"] == "moe"][0]
        assert all(not hook["available"] and hook["reason"] for hook in moe["hooks"])


class TestValidateEndpoint:
    def test_valid_configuration(self, client):
        payload = client.post("/api/validate", json={"config": VALID_CONFIG}).json()
        assert payload["valid"] is True
        assert payload["issues"] == []

    def test_a_verdict_names_the_scope_it_was_reached_in(self, client):
        """Validation runs against the descriptor, not a loaded model."""
        payload = client.post("/api/validate", json={"config": VALID_CONFIG}).json()
        assert payload["scope"] == "design-time"
        assert "descriptor" in payload["note"].lower()

    def test_issues_name_the_control_that_caused_them(self, client):
        bad = {
            **VALID_CONFIG,
            "observations": {
                "hooks": ["router_logits", "nope"],
                "layers": {"start": 0, "end": 99},
            },
        }
        payload = client.post("/api/validate", json={"config": bad}).json()
        assert payload["valid"] is False
        fields = {issue["field"] for issue in payload["issues"]}
        assert "observations.hooks.router_logits" in fields
        assert "observations.hooks.nope" in fields
        assert "observations.layers" in fields

    def test_warnings_do_not_block(self, client):
        warned = {
            **VALID_CONFIG,
            "observations": {
                "hooks": ["final_logits"],
                "layers": {"start": 0, "end": 3},
            },
        }
        payload = client.post("/api/validate", json={"config": warned}).json()
        assert payload["valid"] is True
        assert payload["issues"][0]["severity"] == "warning"

    def test_ui_validity_matches_library_validity(self, client):
        # The whole reason the schema lives in DMI rather than in JavaScript.
        from dmi.configuration import is_valid, load_descriptor

        descriptor = load_descriptor(DENSE)
        for candidate in (VALID_CONFIG, {**VALID_CONFIG, "observations": {"hooks": []}}):
            served = client.post("/api/validate", json={"config": candidate}).json()
            assert served["valid"] == is_valid(parse_config(candidate), descriptor)

    def test_malformed_body_is_a_400(self, client):
        assert client.post("/api/validate", json={"nope": 1}).status_code == 400

    def test_unsupported_version_is_a_400(self, client):
        response = client.post("/api/validate", json={"config": {"version": 99}})
        assert response.status_code == 400


class TestSerializationEndpoints:
    def test_serialize_produces_canonical_yaml(self, client):
        payload = client.post(
            "/api/config/serialize", json={"config": VALID_CONFIG}
        ).json()
        # Catalog order, not click order.
        assert "- pattern\n  - q\n  - k\n  - v" in payload["yaml"]

    def test_parse_of_serialize_is_the_canonical_form(self, client):
        text = client.post(
            "/api/config/serialize", json={"config": VALID_CONFIG}
        ).json()["yaml"]
        reparsed = client.post("/api/config/parse", json={"yaml": text}).json()["config"]
        assert reparsed == config_to_dict(parse_config(VALID_CONFIG))

    def test_serialization_is_stable_across_a_second_pass(self, client):
        first = client.post(
            "/api/config/serialize", json={"config": VALID_CONFIG}
        ).json()["yaml"]
        reparsed = client.post("/api/config/parse", json={"yaml": first}).json()["config"]
        second = client.post("/api/config/serialize", json={"config": reparsed}).json()[
            "yaml"
        ]
        assert first == second

    def test_invalid_yaml_is_a_400(self, client):
        response = client.post("/api/config/parse", json={"yaml": "a: [1,"})
        assert response.status_code == 400

    def test_parse_requires_a_yaml_key(self, client):
        assert client.post("/api/config/parse", json={"config": {}}).status_code == 400


class TestConfigLoadAndSave:
    def test_no_starting_config_by_default(self, client):
        payload = client.get("/api/config").json()
        assert payload["config"] is None
        assert payload["path"].endswith("llama3-8b.dmi.yaml")

    def test_starting_config_is_loaded_when_given(self, tmp_path):
        target = tmp_path / "start.dmi.yaml"
        target.write_text(
            (REPO / "tests" / "golden" / "qwen3-attention.yaml").read_text(),
            encoding="utf-8",
        )
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")
        payload = client.get("/api/config").json()
        assert payload["config"]["observations"]["layers"] == {"start": 8, "end": 15}

    def test_save_writes_to_the_server_side_path(self, tmp_path):
        target = tmp_path / "out.dmi.yaml"
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")
        response = client.post("/api/config/save", json={"config": VALID_CONFIG})
        assert response.status_code == 200
        assert response.json()["path"] == str(target)
        # Saved files are canonical, so compare against the normalized form.
        assert load_config(target) == normalize_config(parse_config(VALID_CONFIG))

    def test_client_cannot_choose_the_save_path(self, tmp_path):
        # The browser gets no filesystem access; a path in the body is ignored.
        target = tmp_path / "allowed.yaml"
        attacker = tmp_path / "elsewhere.yaml"
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")
        client.post(
            "/api/config/save",
            json={"config": VALID_CONFIG, "path": str(attacker)},
        )
        assert target.exists()
        assert not attacker.exists()

    def test_saved_file_reloads_through_open(self, tmp_path):
        target = tmp_path / "cycle.dmi.yaml"
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")
        client.post("/api/config/save", json={"config": VALID_CONFIG})
        reopened = client.post(
            "/api/config/parse", json={"yaml": target.read_text()}
        ).json()["config"]
        assert reopened == config_to_dict(parse_config(VALID_CONFIG))


class TestServerState:
    def test_default_save_path_sits_beside_the_descriptor(self):
        state = build_state(DENSE)
        assert state.save_path == default_save_path(DENSE, state.descriptor)
        assert state.save_path.name == "llama3-8b.dmi.yaml"

    def test_missing_starting_config_is_not_an_error(self, tmp_path):
        state = build_state(DENSE, tmp_path / "not-created-yet.yaml")
        assert state.initial_config is None
        assert state.save_path == tmp_path / "not-created-yet.yaml"


# ---------------------------------------------------------------------------
# POST /api/estimate
# ---------------------------------------------------------------------------


def _estimate(client, config=None, **body):
    request = {"config": config or VALID_CONFIG}
    request.update(body)
    return client.post("/api/estimate", json=request)


def test_estimate_returns_a_peak_step_and_the_rank_that_carries_it():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = _estimate(client)

    assert response.status_code == 200
    body = response.json()
    assert body["peak_step_bytes"] > 0
    assert body["peak_step_rank"] == "pp0/tp0"
    assert body["assumptions"]


def test_estimate_defaults_the_workload_when_none_is_given():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = _estimate(client)

    assert response.status_code == 200
    assert response.json()["peak_step_bytes"] > 0


def test_estimate_honours_a_supplied_workload():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        short = _estimate(client, workload={"prompt_tokens": 128}).json()
        long = _estimate(client, workload={"prompt_tokens": 4096}).json()

    assert long["peak_step_bytes"] > short["peak_step_bytes"]


def test_estimate_reports_every_rank_under_tensor_parallelism():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        body = _estimate(client, workload={"tensor_parallel_size": 4}).json()

    labels = {rank["label"] for rank in body["ranks"]}
    assert labels == {"pp0/tp0", "pp0/tp1"}
    by_label = {rank["label"]: rank for rank in body["ranks"]}
    assert (
        by_label["pp0/tp0"]["prefill_step_bytes"]
        >= by_label["pp0/tp1"]["prefill_step_bytes"]
    )


def test_estimate_rejects_an_invalid_workload():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = _estimate(client, workload={"batch_size": 0})

    assert response.status_code == 400
    assert "Invalid workload" in response.json()["detail"]


def test_estimate_rejects_an_unknown_dtype():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = _estimate(client, workload={"dtype": "float13"})

    assert response.status_code == 400
    assert "Unknown dtype" in response.json()["detail"]


def test_estimate_rejects_a_malformed_body():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = client.post("/api/estimate", json={"nope": 1})

    assert response.status_code == 400


def test_estimate_omits_ring_fit_when_no_ring_size_is_given():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        body = _estimate(client).json()

    assert "ring_fit" not in body


def test_estimate_includes_ring_fit_when_a_ring_size_is_given():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        body = _estimate(
            client, ring={"payload_bytes": 4096 * 1024 * 1024}
        ).json()

    assert body["ring_fit"]["fits"] is True
    assert body["ring_fit"]["occupancy_percent"] >= 0


def test_estimate_flags_a_step_that_will_not_fit_the_ring():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        body = _estimate(client, ring={"payload_bytes": 1024}).json()

    assert body["ring_fit"]["fits"] is False
    assert "STEP_OVERSIZED" in body["ring_fit"]["detail"]


def test_estimate_uses_the_smaller_of_payload_and_pinned():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        body = _estimate(
            client,
            ring={
                "payload_bytes": 4096 * 1024 * 1024,
                "pinned_bytes": 64 * 1024 * 1024,
            },
        ).json()

    assert body["ring_fit"]["effective_bytes"] == 64 * 1024 * 1024


def test_estimate_rejects_a_zero_ring_size():
    """0 must be rejected, not silently treated as "no ring given"."""
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = _estimate(client, ring={"payload_bytes": 0})

    assert response.status_code == 400


def test_estimate_rejects_a_negative_ring_size():
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = _estimate(client, ring={"payload_bytes": -5})

    assert response.status_code == 400


@pytest.mark.parametrize("bad", [[], [4096], "4096", 4096])
def test_estimate_rejects_a_ring_that_is_not_a_mapping(bad):
    """A malformed body is a 400, not an AttributeError behind a 500."""
    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        response = _estimate(client, ring=bad)

    assert response.status_code == 400
    assert "must be a mapping" in response.json()["detail"]


def test_estimate_matches_the_library_for_the_same_inputs():
    """The UI must not be able to disagree with the Python API."""
    from dmi.configuration import Workload, estimate_config, load_descriptor

    descriptor = load_descriptor(DENSE)
    config = parse_config(VALID_CONFIG)
    direct = estimate_config(config, descriptor, Workload(prompt_tokens=512))

    with TestClient(create_app(DENSE), base_url="http://127.0.0.1") as client:
        body = _estimate(client, workload={"prompt_tokens": 512}).json()

    assert body["peak_step_bytes"] == direct.peak_step_bytes
    assert body["decode_step_bytes"] == direct.decode_step_bytes


class TestHostValidation:
    """DNS-rebinding guard: a loopback server answers loopback names only."""

    def test_loopback_hosts_are_served(self, client):
        assert client.get("/api/model").status_code == 200

    def test_localhost_is_served(self):
        client = TestClient(create_app(DENSE), base_url="http://localhost")
        assert client.get("/api/model").status_code == 200

    def test_a_rebound_hostname_is_rejected(self):
        client = TestClient(
            create_app(DENSE), base_url="http://rebound.attacker.example"
        )
        response = client.get("/api/model")
        assert response.status_code == 400

    def test_a_rebound_hostname_cannot_save(self, tmp_path):
        target = tmp_path / "victim.dmi.yaml"
        client = TestClient(
            create_app(DENSE, target),
            base_url="http://rebound.attacker.example",
        )
        response = client.post(
            "/api/config/save", json={"config": VALID_CONFIG}
        )
        assert response.status_code == 400
        assert not target.exists()

    def test_a_non_loopback_bind_serves_any_host(self):
        # Serving the network is an explicit choice; the valid names cannot
        # be enumerated, so the Host filter only guards loopback binds.
        client = TestClient(
            create_app(DENSE, bind_host="0.0.0.0"),
            base_url="http://some.internal.name",
        )
        assert client.get("/api/model").status_code == 200


class TestRelaunch:
    def test_relaunch_without_config_reloads_the_saved_file(self, tmp_path):
        """A second `dmi ui MODEL` must reopen what the first session saved,
        not a blank default that Save would clobber it with."""
        import shutil

        descriptor = tmp_path / "llama3-8b.yaml"
        shutil.copyfile(DENSE, descriptor)

        first = TestClient(create_app(descriptor), base_url="http://127.0.0.1")
        saved_to = Path(
            first.post(
                "/api/config/save", json={"config": VALID_CONFIG}
            ).json()["path"]
        )
        assert saved_to.parent == tmp_path

        state = build_state(descriptor)
        assert state.initial_config is not None
        assert state.initial_config == normalize_config(
            parse_config(VALID_CONFIG)
        )


# ---------------------------------------------------------------------------
# Blocking finding 3919326709: a non-loopback bind serves an unauthenticated
# file-writing API to every reachable peer. Mutating endpoints must demand a
# per-launch token whenever the server is bound beyond loopback.
# ---------------------------------------------------------------------------


class TestNetworkBindRequiresToken:
    def _network_app(self):
        return create_app(DENSE, bind_host="0.0.0.0")

    def test_create_app_mints_a_token_for_non_loopback_binds(self):
        app = self._network_app()
        token = app.state.launch_token
        assert token, "a network bind must mint a token to print at startup"

    def test_loopback_bind_mints_no_token(self):
        app = create_app(DENSE, bind_host="127.0.0.1")
        assert app.state.launch_token is None, (
            "loopback needs no token and must not print one"
        )

    def test_save_without_token_is_refused_on_a_network_bind(self):
        app = self._network_app()
        with TestClient(app, base_url="http://0.0.0.0") as client:
            response = client.post("/api/config/save", json={"config": VALID_CONFIG})
            assert response.status_code == 401

    def test_save_with_wrong_token_is_refused(self):
        app = self._network_app()
        with TestClient(app, base_url="http://0.0.0.0") as client:
            response = client.post(
                "/api/config/save",
                json={"config": VALID_CONFIG},
                headers={"X-DMI-Token": "not-the-token"},
            )
            assert response.status_code == 401

    def test_save_with_the_token_writes(self, tmp_path):
        app = create_app(DENSE, tmp_path / "out.dmi.yaml", bind_host="0.0.0.0")
        with TestClient(app, base_url="http://0.0.0.0") as client:
            response = client.post(
                "/api/config/save",
                json={"config": VALID_CONFIG},
                headers={"X-DMI-Token": app.state.launch_token},
            )
            assert response.status_code == 200

    def test_token_gates_every_mutating_endpoint(self):
        app = self._network_app()
        with TestClient(app, base_url="http://0.0.0.0") as client:
            for method, path, body in [
                ("post", "/api/config/save", {"config": VALID_CONFIG}),
                ("post", "/api/config/parse", {"yaml": "version: 1"}),
                ("post", "/api/validate", {"config": VALID_CONFIG}),
                ("post", "/api/estimate", {"config": VALID_CONFIG}),
            ]:
                response = getattr(client, method)(path, json=body)
                assert response.status_code == 401, (method, path)

    def test_read_endpoints_stay_open_on_a_network_bind(self):
        app = self._network_app()
        with TestClient(app, base_url="http://0.0.0.0") as client:
            assert client.get("/api/model").status_code == 200
            assert client.get("/api/catalog").status_code == 200
            assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# Blocking finding 3919326705: a successful save left `GET /api/config`
# serving the launch-time config, so a refresh restored stale YAML and a
# second Save overwrote the just-saved file with it.
# ---------------------------------------------------------------------------


class TestSaveRefreshesReloadState:
    def test_saved_config_is_what_a_refresh_gets(self, tmp_path):
        target = tmp_path / "out.dmi.yaml"
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")

        client.post("/api/config/save", json={"config": VALID_CONFIG})
        payload = client.get("/api/config").json()

        assert payload["config"] == config_to_dict(parse_config(VALID_CONFIG)), (
            "a refresh after Save must return the saved configuration, not "
            "the launch-time one"
        )

    def test_second_save_round_trips_through_a_refresh(self, tmp_path):
        target = tmp_path / "out.dmi.yaml"
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")

        first = {"version": 1, "observations": {"hooks": ["q"]}}
        second = {"version": 1, "observations": {"hooks": ["k"], "layers": {"start": 1, "end": 2}}}
        client.post("/api/config/save", json={"config": first})
        client.post("/api/config/save", json={"config": second})
        reloaded = client.get("/api/config").json()["config"]

        # The pre-fix failure mode: refresh serves the LAUNCH config, and a
        # client that then saves it overwrites B with A.
        assert reloaded["observations"]["hooks"] == ["k"]
        saved = load_config(target)
        assert [h for h in saved.observations.hooks] == ["k"]

    def test_failed_save_does_not_touch_the_reload_state(self, tmp_path, monkeypatch):
        import dmi.ui.app as mod

        app = create_app(DENSE, tmp_path / "fresh.yaml")
        established = {"done": False}

        real_save = mod.save_config

        def save_once_then_fail(config, path):
            if not established["done"]:
                established["done"] = True
                return real_save(config, path)
            raise mod.ConfigurationError("disk on fire")

        monkeypatch.setattr(mod, "save_config", save_once_then_fail)

        with TestClient(app, base_url="http://127.0.0.1") as client:
            assert client.post("/api/config/save", json={"config": VALID_CONFIG}).status_code == 200
            response = client.post("/api/config/save", json={"config": VALID_CONFIG})
            assert response.status_code == 500
            payload = client.get("/api/config").json()
            # The failed second save must not publish itself as the reload state.
            assert payload["config"] == config_to_dict(parse_config(VALID_CONFIG))


# ---------------------------------------------------------------------------
# Blocking finding 3919326712: Save only parsed the document, so an empty
# hook set or out-of-range layers was persisted with a success response even
# though /api/validate calls it invalid.
# ---------------------------------------------------------------------------


class TestSaveValidatesAgainstTheModel:
    def test_save_of_an_invalid_config_is_a_400(self, tmp_path):
        target = tmp_path / "out.dmi.yaml"
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")
        invalid = {"version": 1, "observations": {"hooks": []}}

        response = client.post("/api/config/save", json={"config": invalid})

        assert response.status_code == 400
        assert not target.exists(), "nothing may be written for an invalid config"

    def test_save_names_the_validation_issues(self, tmp_path):
        client = TestClient(
            create_app(DENSE, tmp_path / "out.dmi.yaml"), base_url="http://127.0.0.1"
        )
        out_of_range = {
            "version": 1,
            "observations": {"hooks": ["q"], "layers": {"start": 999, "end": 1000}},
        }
        response = client.post("/api/config/save", json={"config": out_of_range})
        assert response.status_code == 400
        assert "999" in response.json()["detail"] or "layer" in response.json()["detail"].lower()

    def test_save_of_a_valid_config_still_writes(self, tmp_path):
        target = tmp_path / "out.dmi.yaml"
        client = TestClient(create_app(DENSE, target), base_url="http://127.0.0.1")
        response = client.post("/api/config/save", json={"config": VALID_CONFIG})
        assert response.status_code == 200
        assert target.exists()


# ---------------------------------------------------------------------------
# Independent-review round: malformed JSON bodies surfaced as raw 500s, the
# token compare crashed on non-ASCII, ring fit ignored pinned-only input,
# and the loopback bind had no Origin defense (sendBeacon CSRF).
# ---------------------------------------------------------------------------


class TestMalformedBodiesAre400Not500:
    def test_parse_rejects_non_string_yaml(self, client):
        for bad in ({"a": 1}, 42, ["x"], None):
            response = client.post("/api/config/parse", json={"yaml": bad})
            assert response.status_code == 400, bad

    def test_estimate_rejects_float_and_bool_scalars(self, client):
        base = {
            "batch_size": 1, "prompt_tokens": 8, "decode_tokens": 2,
            "packed": True, "tensor_parallel_size": 1,
        }
        for field, value in (
            ("batch_size", 2.5),
            ("prompt_tokens", 2048.0),
            ("tensor_parallel_size", True),
            ("decode_tokens", 1e3),
            ("decode_steps_per_second", "fast"),
            ("dtype", 123),
        ):
            workload = {**base, field: value}
            response = _estimate(client, workload=workload)
            assert response.status_code == 400, (field, value, response.text)

    def test_estimate_rejects_nonfinite_numbers(self, client):
        # Raw bodies: Python json.loads accepts Infinity/NaN literals that
        # a hostile client can send, but this venv's httpx refuses to
        # encode them -- so the body goes as text.
        for literal in ("Infinity", "NaN"):
            body = (
                '{"config": {"version": 1, "observations": {"hooks": '
                '["resid_pre"]}}, "workload": {"batch_size": 1, '
                '"prompt_tokens": 8, "decode_tokens": 2, "packed": true, '
                f'"tensor_parallel_size": {literal}}}}}'
            )
            response = client.post(
                "/api/estimate", content=body,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 400, literal

    def test_estimate_rejects_overflowing_ring_bytes(self, client):
        response = _estimate(client, ring={"payload_bytes": 10**400})
        assert response.status_code == 400
        body = (
            '{"config": {"version": 1, "observations": {"hooks": '
            '["resid_pre"]}}, "ring": {"payload_bytes": Infinity}}'
        )
        response = client.post(
            "/api/estimate", content=body,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_ring_pinned_only_is_an_error_not_silence(self, client):
        response = _estimate(client, ring={"pinned_bytes": 1024})
        assert response.status_code == 400
        assert "payload_bytes" in response.json()["detail"]


class TestTokenCompareNeverCrashes:
    def test_non_ascii_token_header_is_a_401_not_a_500(self):
        app = create_app(DENSE, bind_host="0.0.0.0")
        with TestClient(app, base_url="http://0.0.0.0") as client:
            # Raw header bytes: HTTP headers are latin-1 on the wire, and
            # httpx refuses to encode non-ASCII str headers.
            response = client.post(
                "/api/config/save",
                json={"config": VALID_CONFIG},
                headers={"X-DMI-Token": b"t\xf3k\xe9n-\xfc"},
            )
            assert response.status_code == 401


class TestOriginDefenseOnLoopback:
    def test_foreign_origin_post_is_refused(self, tmp_path):
        client = TestClient(
            create_app(DENSE, tmp_path / "out.yaml"), base_url="http://127.0.0.1"
        )
        response = client.post(
            "/api/config/save",
            json={"config": VALID_CONFIG},
            headers={"Origin": "https://evil.example"},
        )
        assert response.status_code == 403
        assert not (tmp_path / "out.yaml").exists()

    def test_loopback_origin_post_is_accepted(self, tmp_path):
        client = TestClient(
            create_app(DENSE, tmp_path / "out.yaml"), base_url="http://127.0.0.1"
        )
        response = client.post(
            "/api/config/save",
            json={"config": VALID_CONFIG},
            headers={"Origin": "http://127.0.0.1"},
        )
        assert response.status_code == 200

    def test_no_origin_header_still_works(self, tmp_path):
        """curl and same-origin fetches often omit Origin; only a FOREIGN
        origin is evidence of a cross-site write."""
        client = TestClient(
            create_app(DENSE, tmp_path / "out.yaml"), base_url="http://127.0.0.1"
        )
        response = client.post("/api/config/save", json={"config": VALID_CONFIG})
        assert response.status_code == 200


class TestNetworkBindGatesConfigRead:
    def test_api_config_requires_the_token_on_a_network_bind(self):
        app = create_app(DENSE, bind_host="0.0.0.0")
        with TestClient(app, base_url="http://0.0.0.0") as client:
            response = client.get("/api/config")
            assert response.status_code == 401
            authorized = client.get(
                "/api/config", headers={"X-DMI-Token": app.state.launch_token}
            )
            assert authorized.status_code == 200
