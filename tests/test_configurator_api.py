"""HTTP contract for the DMI-configurator backend.

The point of these tests is that the browser never decides anything: every
answer comes from the same Python the runtime uses.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="DMI-configurator UI extra not installed")

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
    return TestClient(create_app(DENSE))


@pytest.fixture
def moe_client():
    return TestClient(create_app(MOE))


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
        assert payload == {"valid": True, "issues": []}

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
        client = TestClient(create_app(DENSE, target))
        payload = client.get("/api/config").json()
        assert payload["config"]["observations"]["layers"] == {"start": 8, "end": 15}

    def test_save_writes_to_the_server_side_path(self, tmp_path):
        target = tmp_path / "out.dmi.yaml"
        client = TestClient(create_app(DENSE, target))
        response = client.post("/api/config/save", json={"config": VALID_CONFIG})
        assert response.status_code == 200
        assert response.json()["path"] == str(target)
        # Saved files are canonical, so compare against the normalized form.
        assert load_config(target) == normalize_config(parse_config(VALID_CONFIG))

    def test_client_cannot_choose_the_save_path(self, tmp_path):
        # The browser gets no filesystem access; a path in the body is ignored.
        target = tmp_path / "allowed.yaml"
        attacker = tmp_path / "elsewhere.yaml"
        client = TestClient(create_app(DENSE, target))
        client.post(
            "/api/config/save",
            json={"config": VALID_CONFIG, "path": str(attacker)},
        )
        assert target.exists()
        assert not attacker.exists()

    def test_saved_file_reloads_through_open(self, tmp_path):
        target = tmp_path / "cycle.dmi.yaml"
        client = TestClient(create_app(DENSE, target))
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
