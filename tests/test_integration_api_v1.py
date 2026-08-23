"""Contract tests for the additive DMI integration API v1 facade."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import get_type_hints

import pytest
import torch


pytestmark = pytest.mark.native_backend


@pytest.fixture(scope="module", autouse=True)
def _require_native_backend() -> None:
    try:
        from dmi.transport import native as _native_engine

        _native_engine._load_extension()
    except Exception as exc:
        pytest.skip(f"DMI native backend is unavailable: {exc}")


_HOOK_EXPORTS = {
    "HOOK_TYPE_RESID_PRE",
    "HOOK_TYPE_LN1",
    "HOOK_TYPE_ATTN_OUT",
    "HOOK_TYPE_RESID_MID",
    "HOOK_TYPE_ATTN_SCORES",
    "HOOK_TYPE_PATTERN",
    "HOOK_TYPE_Q",
    "HOOK_TYPE_K",
    "HOOK_TYPE_V",
    "HOOK_TYPE_Z",
    "HOOK_TYPE_LN2",
    "HOOK_TYPE_MLP_IN",
    "HOOK_TYPE_MLP_OUT",
    "HOOK_TYPE_MLP_POST",
    "HOOK_TYPE_RESID_FINAL",
    "HOOK_TYPE_EMBED",
    "HOOK_TYPE_POS_EMBED",
    "HOOK_TYPE_FINAL_LN",
    "HOOK_TYPE_TOKEN_IDS",
    "HOOK_TYPE_FINAL_LOGITS",
    "HOOK_TYPE_ROUTER_LOGITS",
    "HOOK_TYPE_TOPK_IDS",
    "HOOK_TYPE_TOPK_WEIGHTS",
}

_NON_HOOK_EXPORTS = {
    "DMI_INTEGRATION_API_VERSION",
    "BackendAdapter",
    "BackendAdaptor",
    "StepPlan",
    "StepReservation",
    "StepContext",
    "MonitoringEngine",
    "RingCapacities",
    "MonitoringConfig",
    "CaptureSchedule",
    "HostEngineConfig",
    "deactivate_ring_transport",
    "HookPoint",
    "HookSpec",
    "HookRowBasis",
    "ModelShapeConfig",
    "hook_row_basis",
    "compute_hook_shape",
    "align_up",
    "make_model_shape_from_hf_config",
    "install_ring_hooks",
    "configure_hook_padding_strip",
    "ALL_HOOK_TYPES",
    "ATTENTION_WEIGHT_HOOK_TYPES",
    "register_preset",
    "is_preset_registered",
    "select_hook_specs",
    "hook_belongs_to_pp_rank",
    "hook_belongs_to_tp_rank",
    "RingConfig",
    "ClickHouseClientConfig",
    "StageConfig",
    "QueueConfig",
    "EnqueuePolicy",
    "OnFullPolicy",
    "OnClosedPolicy",
    "DMXHostEngine",
    "ThreadFailure",
    "CHClickhouseDriverReadOnly",
    "InternalRequirement",
    "InternalRequirements",
    "IncompleteInternalError",
    "LazyInternal",
    "make_lazy_internal",
}


def test_v1_public_surface_is_exact() -> None:
    from dmi.api import v1

    expected = _NON_HOOK_EXPORTS | _HOOK_EXPORTS
    assert len(v1.__all__) == len(set(v1.__all__))
    assert set(v1.__all__) == expected
    assert v1.DMI_INTEGRATION_API_VERSION == 1
    assert not hasattr(v1, "RingTransport")
    assert not hasattr(v1, "RingEngine")


def test_v1_reexports_existing_objects_and_state() -> None:
    from dmi.transport import native as _native_engine
    from dmi.adapters import base as adapter_base
    from dmi.storage import clickhouse
    from dmi import config
    from dmi import engine
    from dmi.hooks import dispatch
    from dmi.hooks import point
    from dmi.hooks import specs
    from dmi.storage import internals
    from dmi.hooks import selection
    from dmi.adapters import types
    from dmi.api import v1

    assert v1.BackendAdapter is adapter_base.BackendAdapter
    assert v1.BackendAdaptor is v1.BackendAdapter
    assert v1.StepPlan is adapter_base.StepPlan
    assert v1.StepReservation is adapter_base.StepReservation
    assert v1.StepContext is types.StepContext
    assert v1.MonitoringEngine is engine.MonitoringEngine
    assert v1.RingCapacities is engine.RingCapacities
    assert v1.MonitoringConfig is config.MonitoringConfig
    assert v1.CaptureSchedule is config.CaptureSchedule
    assert v1.HostEngineConfig is engine.HostEngineConfig
    assert v1.HookPoint is point.HookPoint
    assert v1.HookSpec is specs.HookSpec
    assert v1.HookRowBasis is specs.HookRowBasis
    assert v1.ModelShapeConfig is specs.ModelShapeConfig
    assert v1.hook_row_basis is specs.hook_row_basis
    assert v1.install_ring_hooks is dispatch.install_ring_hooks
    assert v1.register_preset is selection.register_preset
    assert v1.select_hook_specs is selection.select_hook_specs
    assert v1.hook_belongs_to_pp_rank is selection.hook_belongs_to_pp_rank
    assert v1.hook_belongs_to_tp_rank is selection.hook_belongs_to_tp_rank
    assert v1.CHClickhouseDriverReadOnly is (
        clickhouse.CHClickhouseDriverReadOnly
    )
    assert v1.InternalRequirement is internals.InternalRequirement
    assert internals._Requirement is internals.InternalRequirement
    assert v1.InternalRequirement.__name__ == "InternalRequirement"
    assert v1.InternalRequirements is internals.InternalRequirements
    assert v1.IncompleteInternalError is internals.IncompleteInternalError
    assert v1.LazyInternal is internals.LazyInternal
    assert internals._LazyInternal is internals.LazyInternal
    assert v1.LazyInternal.__name__ == "LazyInternal"
    assert isinstance(v1.make_lazy_internal("model"), v1.LazyInternal)
    assert get_type_hints(v1.make_lazy_internal)["return"] is v1.LazyInternal
    assert get_type_hints(v1.InternalRequirements.requirement)["return"] == (
        v1.InternalRequirement | None
    )

    for name in _HOOK_EXPORTS:
        assert getattr(v1, name) == getattr(specs, name)

    native = _native_engine._load_extension()
    for name in {
        "RingConfig",
        "ClickHouseClientConfig",
        "StageConfig",
        "QueueConfig",
        "EnqueuePolicy",
        "OnFullPolicy",
        "OnClosedPolicy",
        "DMXHostEngine",
        "ThreadFailure",
    }:
        assert getattr(v1, name) is getattr(native, name)


def test_v1_public_surface_is_documented() -> None:
    from dmi.api import v1

    root = Path(__file__).resolve().parents[1]
    document = (root / "docs" / "integration-api-v1.md").read_text()
    missing = [name for name in v1.__all__ if f"`{name}`" not in document]
    assert missing == []
    member_names = {
        "active_hook_specs",
        "capture_enabled",
        "commit_step",
        "model_shape",
        "plan_step",
        "ring_capacities",
        "set_capture_enabled",
    }
    missing_members = [
        name
        for name in member_names
        if f"`{name}`" not in document
        and f"`{name}()`" not in document
    ]
    assert missing_members == []


def test_v1_public_names_preserve_existing_shape_and_selection_behavior() -> None:
    from dmi.hooks import selection
    from dmi.hooks import specs
    from dmi.api import v1

    cfg = v1.ModelShapeConfig(
        hidden_dim=64,
        num_heads=8,
        num_kv_heads=4,
        head_dim=8,
        dtype=torch.float16,
        vocab_size=128,
        intermediate_dim=256,
        tp_size=2,
    )
    args = (v1.HOOK_TYPE_Q, cfg, 0, 7, 11)
    assert v1.compute_hook_shape(*args) == specs.compute_hook_shape(*args)
    assert v1.align_up(33, 16) == specs.align_up_py(33, 16)
    assert v1.ALL_HOOK_TYPES == selection._ALL_HOOK_TYPES
    assert v1.ATTENTION_WEIGHT_HOOK_TYPES == specs._ATTN_WT_TYPES
    assert v1.is_preset_registered("full")
    assert not v1.is_preset_registered("not-a-real-preset")


def test_v1_padding_strip_configuration_preserves_existing_modes() -> None:
    from dmi.api import v1

    hook = v1.HookPoint()
    row_count = torch.tensor([7], dtype=torch.int64)

    v1.configure_hook_padding_strip(hook, row_count, row_bytes=128)
    assert hook._strip_tensor is row_count
    assert hook._strip_row_bytes == 128

    v1.configure_hook_padding_strip(hook, row_count)
    assert hook._strip_tensor is row_count
    assert hook._strip_row_bytes == 0

    v1.configure_hook_padding_strip(hook, None)
    assert hook._strip_tensor is None
    assert hook._strip_row_bytes == 0


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(
            model_type="gpt2",
            n_embd=64,
            n_head=8,
            n_inner=None,
            vocab_size=128,
            torch_dtype=torch.float32,
        ),
        SimpleNamespace(
            model_type="qwen2_moe",
            hidden_size=96,
            num_attention_heads=12,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=384,
            num_experts=16,
            num_experts_per_tok=2,
            vocab_size=256,
            torch_dtype=torch.bfloat16,
        ),
    ],
)
def test_v1_model_shape_helper_matches_existing_helper(config) -> None:
    from dmi.adapters.huggingface.model_shape import _make_model_shape_from_hf_config
    from dmi.api import v1

    assert v1.make_model_shape_from_hf_config(config) == (
        _make_model_shape_from_hf_config(config)
    )


def test_v1_import_has_no_framework_or_preset_side_effects() -> None:
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        from dmi.hooks import selection

        before = dict(selection._HOOK_SELECTIONS)

        class RejectFrameworkImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                blocked = (
                    fullname == "vllm"
                    or fullname.startswith("vllm.")
                    or fullname == "dmi.adapters.huggingface.adapter"
                )
                if blocked:
                    raise AssertionError(f"unexpected framework import: {fullname}")
                return None

        sys.meta_path.insert(0, RejectFrameworkImports())
        import dmi.api.v1 as api

        assert api.DMI_INTEGRATION_API_VERSION == 1
        assert selection._HOOK_SELECTIONS == before
        assert not any(name == "vllm" or name.startswith("vllm.")
                       for name in sys.modules)
        assert "dmi.adapters.huggingface.adapter" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
