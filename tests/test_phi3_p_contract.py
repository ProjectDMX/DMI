"""CPU contracts for the Phi-3 DMI thin variant."""

import pytest

from integration.vllm_adapter import (
    _ARCH_REMAP,
    _LLAMA_COMPAT_ARCHES,
)
from vllm.model_executor.models.llama_p import LlamaPForCausalLM
from vllm.model_executor.models.phi3 import Phi3ForCausalLM
from vllm.model_executor.models.phi3_compare import Phi3CompareForCausalLM
from vllm.model_executor.models.phi3_p import Phi3PForCausalLM


pytestmark = pytest.mark.framework_fork


def test_phi3_has_a_distinct_remap_for_its_fused_weight_packing() -> None:
    assert "Phi3ForCausalLM" not in _LLAMA_COMPAT_ARCHES
    assert _ARCH_REMAP["Phi3ForCausalLM"] == "Phi3PForCausalLM"


def test_phi3_variant_reuses_llama_math_but_preserves_phi3_packing() -> None:
    assert issubclass(Phi3PForCausalLM, LlamaPForCausalLM)
    assert (
        Phi3PForCausalLM.packed_modules_mapping
        == Phi3ForCausalLM.packed_modules_mapping
        == {
            "qkv_proj": ["qkv_proj"],
            "gate_up_proj": ["gate_up_proj"],
        }
    )
    assert (
        Phi3PForCausalLM.hf_to_vllm_mapper
        is LlamaPForCausalLM.hf_to_vllm_mapper
    )


def test_phi3_compare_variant_preserves_the_same_packing_contract() -> None:
    assert (
        Phi3CompareForCausalLM.packed_modules_mapping
        == Phi3ForCausalLM.packed_modules_mapping
    )
