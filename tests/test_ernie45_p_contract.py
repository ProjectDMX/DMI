"""CPU contracts for bounded ERNIE 4.5 dense monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from integration.vllm_adapter import _ARCH_REMAP
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
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)
from vllm.model_executor.models.ernie45 import Ernie4_5ForCausalLM
from vllm.model_executor.models.ernie45_compare import (
    Ernie4_5CompareForCausalLM,
)
from vllm.model_executor.models.ernie45_p import (
    Ernie4_5PForCausalLM,
    _apply_ernie45_attention_contract,
    _require_supported_ernie45_config,
)
from vllm.model_executor.models.llama_compare import LlamaCompareForCausalLM
from vllm.model_executor.models.llama_p import LlamaPForCausalLM


pytestmark = pytest.mark.framework_fork


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "ernie4_5",
        "hidden_act": "silu",
        "use_bias": False,
        "is_causal": True,
        "layer_types": None,
        "tie_word_embeddings": True,
        "logit_scale": None,
        "head_dim": 128,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 500_000.0,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(**overrides) -> SimpleNamespace:
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=_config(**overrides)))


def test_ernie45_preserves_upstream_llama_class_contracts() -> None:
    assert _ARCH_REMAP["Ernie4_5ForCausalLM"] == "Ernie4_5PForCausalLM"
    assert issubclass(Ernie4_5ForCausalLM, nn.Module)
    assert issubclass(Ernie4_5PForCausalLM, LlamaPForCausalLM)
    assert issubclass(Ernie4_5CompareForCausalLM, LlamaCompareForCausalLM)
    assert (
        Ernie4_5PForCausalLM.packed_modules_mapping
        == Ernie4_5ForCausalLM.packed_modules_mapping
    )
    assert (
        vars(Ernie4_5PForCausalLM.hf_to_vllm_mapper)
        == vars(Ernie4_5ForCausalLM.hf_to_vllm_mapper)
    )
    assert (
        Ernie4_5PForCausalLM.embedding_modules
        == Ernie4_5ForCausalLM.embedding_modules
    )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"model_type": "llama"}, "model_type"),
        ({"hidden_act": "gelu"}, "SiLU"),
        ({"use_bias": True}, "bias-free"),
        ({"is_causal": False}, "causal"),
        ({"layer_types": ["sliding_attention"]}, "schedules"),
        ({"tie_word_embeddings": False}, "tied"),
        ({"logit_scale": 0.5}, "logit scale"),
        ({"head_dim": None}, "head_dim"),
        ({"rope_parameters": None}, "rope_parameters"),
        (
            {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": 500_000.0,
                }
            },
            "default RoPE",
        ),
        (
            {
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                }
            },
            "theta 500000",
        ),
    ],
)
def test_ernie45_fails_closed_outside_the_audited_contract(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_ernie45_config(_vllm_config(**overrides))


def test_ernie45_accepts_the_official_03b_config_contract() -> None:
    _require_supported_ernie45_config(_vllm_config())


def test_ernie45_replays_non_neox_rope_and_bias_free_output_projection() -> None:
    rotary = SimpleNamespace(is_neox_style=True)
    output_projection = SimpleNamespace(
        bias=torch.tensor([1.0]),
        skip_bias_add=False,
    )
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    rotary_emb=rotary,
                    o_proj=output_projection,
                )
            )
        ]
    )

    _apply_ernie45_attention_contract(model)

    assert rotary.is_neox_style is False
    assert output_projection.bias is None
    assert output_projection.skip_bias_add is True


def test_ernie45_model_wide_manifest_has_203_truthful_families() -> None:
    subject = Ernie4_5PForCausalLM.__new__(Ernie4_5PForCausalLM)
    nn.Module.__init__(subject)
    subject.model = SimpleNamespace(layers=[None] * 18)
    specs = subject.get_hook_specs(model_wide=True)
    layer_types = [
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
        HOOK_TYPE_MLP_OUT,
    ]

    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(layer_types * 18),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert len(specs) == 203
    assert all(spec.module is None for spec in specs)


def test_ernie45_global_hook_points_keep_llama_dtypes_and_token_axes() -> None:
    subject = Ernie4_5PForCausalLM.__new__(Ernie4_5PForCausalLM)
    nn.Module.__init__(subject)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=0,
        layers=[],
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()

    specs = subject.get_hook_specs()

    assert specs[0].dtype == torch.int32
    assert specs[0].dim0_is_actual_tokens is True
    assert specs[1].dim0_is_actual_tokens is True
    assert specs[-3].dim0_is_actual_tokens is True
    assert specs[-2].dim0_is_actual_tokens is True
    assert specs[-1].dim0_is_actual_tokens is False
