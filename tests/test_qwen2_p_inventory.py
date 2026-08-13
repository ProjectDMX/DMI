"""CPU checks for the monitored Qwen2 model's canonical hook inventory."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch.nn as nn

from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)


def _load_qwen2_p_class():
    try:
        from vllm.model_executor.models.qwen2_p import Qwen2PForCausalLM

        return Qwen2PForCausalLM
    except ModuleNotFoundError:
        path = (
            Path(__file__).resolve().parents[1]
            / "integration/vllm/vllm/model_executor/models/qwen2_p.py"
        )
        name = "vllm.model_executor.models.qwen2_p"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.Qwen2PForCausalLM


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()


class _FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hook_post = HookPoint()


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()
        self.mlp = _FakeMLP()
        for name in (
            "hook_resid_pre",
            "hook_ln1",
            "hook_attn_out",
            "hook_resid_mid",
            "hook_ln2",
            "hook_mlp_in",
            "hook_mlp_out",
        ):
            setattr(self, name, HookPoint())


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])
        self.start_layer = 0
        self.end_layer = len(self.layers)
        self.hook_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()


def _fake_qwen2_p():
    cls = _load_qwen2_p_class()
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.model = _FakeModel()
    model.hook_token_ids = HookPoint()
    model.hook_final_logits = HookPoint()
    return model


def test_qwen2_inventory_has_all_dense_hook_types_per_layer():
    model = _fake_qwen2_p()
    specs = model.get_hook_specs()
    per_layer = {
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
    }

    assert {s.hook_type for s in specs if s.layer_no == 0} == per_layer
    assert {s.hook_type for s in specs if s.layer_no == 1} == per_layer
    assert {s.hook_type for s in specs if s.layer_no == -1} == {
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    }
    assert all(s.module is not None for s in specs)


def test_qwen2_model_wide_inventory_is_module_free():
    specs = _fake_qwen2_p().get_hook_specs(model_wide=True)

    assert len(specs) == 29
    assert all(s.module is None for s in specs)
    assert all(
        s.dim0_is_actual_tokens
        for s in specs
        if s.hook_type != HOOK_TYPE_FINAL_LOGITS
    )
