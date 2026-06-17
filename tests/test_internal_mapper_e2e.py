"""End-to-end tests for the get_internal read path.

Runs Qwen3-0.6B through generate_with_monitoring (writing captures to
ClickHouse via the native ring), then reads them back with
monitoring.internal_mapper.get_internal and checks the result lines up with
HuggingFace's native output.

Requires a CUDA device, a reachable ClickHouse, the patched transformers fork,
and the model in the local cache; skips cleanly otherwise.

ClickHouse connection: DMX_DB_HOST / DMX_DB_PORT (default localhost:9000).
"""
from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
import torch

from tests._requirements import require_cuda, require_clickhouse, require_model_cache

MODEL = "Qwen/Qwen3-0.6B"
MAX_NEW = 8

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.e2e,
    pytest.mark.clickhouse,
    pytest.mark.hf,
    require_cuda(),
    require_clickhouse(),
    require_model_cache(MODEL),
]


@pytest.fixture(scope="module")
def deps():
    """Import the DMI HF stack; skip the module if any piece is missing."""
    try:
        from transformers import AutoTokenizer
        from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM
        from monitoring import (
            CaptureSchedule, HostEngineConfig, MonitoringConfig, MonitoringEngine,
        )
        from monitoring._native_engine import ClickHouseClientConfig, StageConfig
        from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
        from monitoring.internal_mapper import get_internal
        from integration.hf_adapter import generate_with_monitoring_dict
        import clickhouse_driver
    except Exception as exc:
        pytest.skip(f"DMI HF stack unavailable: {exc}")
    return SimpleNamespace(**{k: v for k, v in locals().items() if k != "exc"})


def _db_host_port():
    return (os.environ.get("DMX_DB_HOST", "localhost"),
            int(os.environ.get("DMX_DB_PORT", "9000")))


def _wipe(deps, model_id):
    host, port = _db_host_port()
    deps.clickhouse_driver.Client(host=host, port=port).execute(
        "ALTER TABLE default.offload DELETE WHERE model_id = %(m)s", {"m": model_id})


def _monitored(deps, model, inputs, model_id, hook_selection="resid_pre"):
    """Run generate_with_monitoring_dict into a fresh ClickHouse slot."""
    _wipe(deps, model_id)
    host, port = _db_host_port()
    db = deps.ClickHouseClientConfig()
    db.host, db.port = host, port
    db.username, db.password = "default", ""
    db.database, db.table = "default", "offload"
    db.create_database_if_missing = True
    stage = deps.StageConfig.clickhouse_insert(db, parallelism=4, name="ch_insert")
    engine = deps.MonitoringEngine(
        config=deps.MonitoringConfig(schedule=deps.CaptureSchedule()),
        model_id=model_id,
        db_config=deps.HostEngineConfig(stages=[stage]),
    )
    model.monitoring_engine = engine
    try:
        with torch.no_grad():
            return deps.generate_with_monitoring_dict(
                model, **inputs, max_new_tokens=MAX_NEW, do_sample=False,
                hook_selection=hook_selection,
            )
    finally:
        engine.close()


@pytest.fixture(scope="module")
def model_tok(deps):
    tok = deps.AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = deps.HookedQwen3ForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="eager",
    ).to("cuda").eval()
    return model, tok


@pytest.fixture(scope="module")
def single(deps, model_tok):
    """One prompt (batch=1): monitored output + a plain-HF reference."""
    model, tok = model_tok
    inputs = tok(["The capital of France is"], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        ref = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    model_id = f"test_internal::single::{uuid.uuid4().hex}"[:120]
    out = _monitored(deps, model, inputs, model_id)
    yield SimpleNamespace(out=out, ref=ref, model=model, model_id=model_id)
    _wipe(deps, model_id)


@pytest.fixture(scope="module")
def ragged(deps, model_tok):
    """A left-padded batch of unequal-length prompts."""
    model, tok = model_tok
    inputs = tok(["The capital of France is", "Hi"], return_tensors="pt", padding=True).to("cuda")
    model_id = f"test_internal::ragged::{uuid.uuid4().hex}"[:120]
    out = _monitored(deps, model, inputs, model_id)
    yield SimpleNamespace(out=out, n_prompts=2, model_id=model_id)
    _wipe(deps, model_id)


# --- dict return shape ------------------------------------------------------

def test_returns_dict_output_with_dmi_internal(single):
    assert single.out.sequences.dim() == 2  # [batch, total_len]
    assert hasattr(single.out, "dmi_internal")


def test_output_matches_plain_hf(single):
    # Monitoring must not change what the model generates.
    assert torch.equal(single.out.sequences, single.ref)


# --- get_internal: structure ------------------------------------------------

def test_available_lists_captured_field(single, deps):
    internal = single.out.dmi_internal
    assert internal.available == ["hidden_states", "token_mask"]


def test_hidden_states_tuple_shape(single, deps):
    internal = single.out.dmi_internal
    hs = internal.hidden_states
    token_mask = internal.token_mask
    cfg = single.model.config
    assert len(hs) == cfg.num_hidden_layers          # one entry per block (resid_pre)
    assert all(t.dim() == 3 for t in hs)             # [batch, seq, hidden]
    assert hs[0].shape[0] == 1                        # batch
    assert hs[0].shape[2] == cfg.hidden_size         # hidden
    assert all(t.shape == hs[0].shape for t in hs)   # uniform across layers
    assert token_mask.shape == hs[0].shape[:2]
    assert token_mask.dtype == torch.bool


def test_token_count_drops_unforwarded_last(single, deps):
    # The final generated token is never fed through a forward, so it has no
    # captured hidden state: captured seq == sequence length - 1.
    hs = single.out.dmi_internal.hidden_states
    assert hs[0].shape[1] == single.out.sequences.shape[1] - 1


# --- get_internal: model_id + reader handling ------------------------------

def test_source_model_id_with_explicit_reader(single, deps):
    host, port = _db_host_port()
    reader = deps.CHClickhouseDriverReadOnly(host=host, port=port)
    from_out = single.out.dmi_internal.hidden_states
    from_id = deps.get_internal(single.model_id, reader).hidden_states
    assert len(from_out) == len(from_id)
    assert torch.equal(from_out[0], from_id[0])


def test_uncaptured_field_raises(single, deps):
    internal = single.out.dmi_internal
    with pytest.raises(AttributeError, match="not captured"):
        internal.attention


# --- get_internal: ragged batch left-pad -----------------------------------

def test_ragged_batch_left_pads(ragged, deps):
    hs = ragged.out.dmi_internal.hidden_states
    layer0 = hs[0]
    assert layer0.shape[0] == ragged.n_prompts

    nonzero = (layer0.abs().sum(dim=-1) > 0)          # [batch, seq] real-token mask
    counts = nonzero.sum(dim=1)
    assert counts.min() < counts.max()                # genuinely ragged -> padding added
    assert bool(nonzero[:, -1].all())                 # real tokens are right-aligned
    short = int(counts.argmin())
    assert not bool(nonzero[short, 0])                # shorter request padded at the front
