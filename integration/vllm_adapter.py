"""vLLM integration: VLLMAdaptor + monitored GPU worker.

Phase 3a of the unified-adaptor refactor consolidates the vLLM-specific
orchestration that used to live in ``monitoring/vllm_integration.py``
into one file under ``integration/``.

Key pieces:

  * ``VLLMAdaptor`` -- concrete ``BackendAdaptor`` for vLLM models.
    Owns the framework-fragile pieces in localized methods:
      ``_record_real_layout`` (copies post-prepare packed request order);
      ``_preflight_force_eager`` (evaluates cached worst-role formulas);
      ``build_step_context`` (uses the real layout and batch descriptor);
      ``adapt_for_cpu_direct`` (swaps padded -> unpadded q_len when an
      oversize step forces eager dispatch + safety net for this batch);
      ``before_forward`` (commits one already-computed actual plan);
      ``_warn_once_capacity`` (per-(total_q, num_reqs) shape warn).
  * ``DMXGPUWorker`` -- vLLM ``Worker`` subclass that owns a
    ``VLLMAdaptor``, records the real input layout, and commits at real
    dispatch. Architecture remap stays here.
  * Module-level ``register_preset("vllm-full", ...)`` -- relocated
    from ``monitoring/selection.py``'s default ``_HOOK_SELECTIONS``
    (deferred from Phase 1.5 per the unified-adaptor plan).  Lands as
    a side-effect of importing this module.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import os
from pathlib import Path
import re
import time
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from vllm import LLM
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.model_executor.models import ModelRegistry
from vllm.v1.worker.gpu_worker import Worker

from monitoring import ring_transport
from monitoring.adaptor_base import BackendAdaptor
from monitoring.ring_transport import (
    HookRowBasis,
    HookSpec,
    ModelShapeConfig,
    _compute_hook_shape,
    align_up_py,
    hook_row_basis,
    install_ring_hooks,
)
from monitoring.selection import (
    hook_belongs_to_pp_rank,
    hook_belongs_to_tp_rank,
    register_preset,
    select_hook_specs,
    _HOOK_SELECTIONS,
    _ALL_HOOK_TYPES,
)
# Import the C++-mirror constant we need for the "vllm-full" preset
# (full minus attention-weight matrices).  Import-time only.
from monitoring.ring_transport import _ATTN_WT_TYPES
from monitoring.step_context import StepContext

from integration.model_shape import _make_model_shape_from_hf_config
from monitoring.clickhouse_reader import CHClickhouseDriverReadOnly
from monitoring.internal_mapper import make_lazy_internal


# ---------------------------------------------------------------------------
# vLLM-full preset registration (deferred from Phase 1.5).
#
# Moved out of monitoring/selection.py's default _HOOK_SELECTIONS so the
# core selection module is framework-neutral.  Registers when this
# module is imported -- which happens whenever DMXGPUWorker is loaded
# via worker_cls="integration.vllm_adapter.DMXGPUWorker".
#
# `register_preset` raises on duplicates, so re-import within the same
# process is a no-op (Python caches the module body).  Across separate
# processes (e.g. each TP rank in vLLM) each subprocess imports fresh
# and registers once.
# ---------------------------------------------------------------------------

if "vllm-full" not in _HOOK_SELECTIONS:
    register_preset("vllm-full", _ALL_HOOK_TYPES - _ATTN_WT_TYPES)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _cfg(ac: dict, key: str, env_key: str, default: Any) -> Any:
    """additional_config preferred, env var fallback, then default."""
    val = ac.get(key)
    if val is not None:
        return val
    env_val = os.environ.get(env_key)
    if env_val is not None:
        if isinstance(default, bool):
            return env_val not in ("0", "false", "False", "")
        if isinstance(default, int):
            return int(env_val)
        return env_val
    return default


_VLLM_REQ_ID_SUFFIX = re.compile(r"-[0-9a-f]{8}$")

# vLLM 0.27.1 owns the default TCPStore in global rank 0.  DMI's rank-local
# resource teardown can make rank 0 return from Worker.shutdown slightly before
# its peers, allowing the store to disappear while another rank's NCCL heartbeat
# thread is still stopping.  Keep the store owner alive briefly after its local
# CUDA/model teardown; WorkerProc destroys process groups immediately after this
# method returns, so non-owner ranks get a deterministic head start.
_STORE_OWNER_SHUTDOWN_GRACE_S = 0.5


def normalize_vllm_request_id(req_id: str) -> str:
    """Strip the UUID suffix that vLLM V1 appends to request IDs."""
    return _VLLM_REQ_ID_SUFFIX.sub("", req_id)


# Architecture remap so vLLM's registry resolves to the hooked variant.
#
# Keep the Llama aliases aligned with upstream registry entries that resolve
# directly to ``llama.LlamaForCausalLM``.  These architectures therefore use
# the same implementation and can safely share DMI's hooked Llama variant.
_LLAMA_COMPAT_ARCHES = {
    "AquilaModel",
    "AquilaForCausalLM",
    "CwmForCausalLM",
    "InternLMForCausalLM",
    "InternLM3ForCausalLM",
    "IQuestCoderForCausalLM",
    "LlamaForCausalLM",
    "LLaMAForCausalLM",
    "XverseForCausalLM",
}

_ARCH_REMAP = {
    "ApertusForCausalLM": "ApertusPForCausalLM",
    "Ernie4_5ForCausalLM": "Ernie4_5PForCausalLM",
    "FalconH1ForCausalLM": "FalconH1PForCausalLM",
    "Gemma3ForCausalLM": "Gemma3PForCausalLM",
    "GlmMoeDsaForCausalLM": "GlmMoeDsaPForCausalLM",
    "GPT2LMHeadModel": "GPT2PLMHeadModel",
    "GptOssForCausalLM": "GptOssPForCausalLM",
    "GraniteForCausalLM": "GranitePForCausalLM",
    "JambaForCausalLM": "JambaPForCausalLM",
    "Lfm2ForCausalLM": "Lfm2PForCausalLM",
    "Llama4ForConditionalGeneration": "Llama4PForConditionalGeneration",
    "MiniCPMForCausalLM": "MiniCPMPForCausalLM",
    "MiniMaxM2ForCausalLM": "MiniMaxM2PForCausalLM",
    "MistralForCausalLM": "MistralPForCausalLM",
    "Olmo3ForCausalLM": "Olmo3PForCausalLM",
    "Phi3ForCausalLM": "Phi3PForCausalLM",
    "Qwen2ForCausalLM": "Qwen2PForCausalLM",
    "Qwen2MoeForCausalLM": "Qwen2MoePForCausalLM",
    "Qwen3ForCausalLM": "Qwen3PForCausalLM",
    "Qwen3MoeForCausalLM": "Qwen3MoePForCausalLM",
    "Qwen3_5ForConditionalGeneration": "Qwen3_5PForConditionalGeneration",
    **{arch: "LlamaPForCausalLM" for arch in _LLAMA_COMPAT_ARCHES},
}


# DMI model variants can be consumed in two ways:
#
# 1. the integration/vllm fork registers them in its own registry.py; or
# 2. an official vLLM wheel imports this adapter, which exposes the submodule's
#    model directory as an out-of-tree model package and registers lazy class
#    references through vLLM's public ModelRegistry API.
#
# Lazy strings are important here: importing model implementations eagerly in
# the parent process can initialize CUDA before vLLM forks its workers.
_DMI_MODEL_VARIANTS = {
    "ApertusPForCausalLM": "apertus_p:ApertusPForCausalLM",
    "Ernie4_5PForCausalLM": "ernie45_p:Ernie4_5PForCausalLM",
    "FalconH1PForCausalLM": "falcon_h1_p:FalconH1PForCausalLM",
    "Gemma3PForCausalLM": "gemma3_p:Gemma3PForCausalLM",
    "GlmMoeDsaPForCausalLM": "glm_moe_dsa_p:GlmMoeDsaPForCausalLM",
    "GPT2PLMHeadModel": "gpt2_p:GPT2PLMHeadModel",
    "GptOssPForCausalLM": "gpt_oss_p:GptOssPForCausalLM",
    "GranitePForCausalLM": "granite_p:GranitePForCausalLM",
    "JambaPForCausalLM": "jamba_p:JambaPForCausalLM",
    "Lfm2PForCausalLM": "lfm2_p:Lfm2PForCausalLM",
    "Llama4PForConditionalGeneration": (
        "mllama4_p:Llama4PForConditionalGeneration"
    ),
    "MiniCPMPForCausalLM": "minicpm_p:MiniCPMPForCausalLM",
    "MiniMaxM2PForCausalLM": "minimax_m2_p:MiniMaxM2PForCausalLM",
    "Qwen2PForCausalLM": "qwen2_p:Qwen2PForCausalLM",
    "Qwen2MoePForCausalLM": "qwen2_moe_p:Qwen2MoePForCausalLM",
    "Qwen3PForCausalLM": "qwen3_p:Qwen3PForCausalLM",
    "Qwen3MoePForCausalLM": "qwen3_moe_p:Qwen3MoePForCausalLM",
    "Qwen3_5PForConditionalGeneration": (
        "qwen3_5_p:Qwen3_5PForConditionalGeneration"
    ),
    "LlamaPForCausalLM": "llama_p:LlamaPForCausalLM",
    "MistralPForCausalLM": "mistral_p:MistralPForCausalLM",
    "Olmo3PForCausalLM": "olmo3_p:Olmo3PForCausalLM",
    "Phi3PForCausalLM": "phi3_p:Phi3PForCausalLM",
}


def _register_dmi_model_variants() -> None:
    """Register bundled DMI models when running against an official wheel."""

    import vllm.model_executor.models as model_package

    model_dir = Path(__file__).resolve().parent / "vllm/vllm/model_executor/models"
    if not model_dir.is_dir():
        raise RuntimeError(f"DMI vLLM model directory is missing: {model_dir}")

    package_path = str(model_dir)
    if package_path not in model_package.__path__:
        model_package.__path__.append(package_path)

    registered = set(ModelRegistry.get_supported_archs())
    module_prefix = "vllm.model_executor.models."
    for architecture, target in _DMI_MODEL_VARIANTS.items():
        if architecture in registered:
            continue
        module_name, class_name = target.split(":", 1)
        ModelRegistry.register_model(
            architecture,
            f"{module_prefix}{module_name}:{class_name}",
        )


_register_dmi_model_variants()


# ---------------------------------------------------------------------------
# Per-call request-layout / dispatch state
# ---------------------------------------------------------------------------


class VLLMStepPhase(Enum):
    IDLE = auto()
    ARMED = auto()
    LAYOUT_READY = auto()
    COMMITTED = auto()


class VLLMValidationMode(Enum):
    OFF = auto()
    VERIFY = auto()


@dataclass(frozen=True)
class _VLLMRealLayout:
    raw_req_ids: Tuple[str, ...]
    req_ids: Tuple[str, ...]
    scheduled_counts: Tuple[int, ...]
    computed_counts: Tuple[int, ...]
    token_ranges: Tuple[Tuple[int, int], ...]
    dim0_offsets: Tuple[int, ...]
    total_rows: int


@dataclass
class _VLLMStepState:
    phase: VLLMStepPhase = VLLMStepPhase.IDLE
    scheduler_output: Any = None
    layout: Optional[_VLLMRealLayout] = None
    force_eager_latch: bool = False
    prelayout_dispatch_seen: bool = False
    capacity_candidate: Optional[Tuple[Any, Any]] = None
    expected_dispatch: Optional[Tuple[Any, Any]] = None
    execution_bound: Optional[int] = None
    real_rows_bound: Optional[int] = None
    request_rows_bound: Optional[int] = None
    real_cudagraph_mode: Any = None
    real_batch_descriptor: Any = None


@dataclass(frozen=True)
class _VLLMHookSelection:
    local_hooks: Tuple[HookSpec, ...]
    candidate_rank_hook_sets: Tuple[Tuple[HookSpec, ...], ...]

    @classmethod
    def from_model(
        cls,
        *,
        model: Any,
        local_hooks: Tuple[HookSpec, ...],
        hook_selection: str,
        cfg: ModelShapeConfig,
        parallel_config: Any,
        hf_config: Any,
    ) -> "_VLLMHookSelection":
        """Select local and model-wide rank-type hooks once at attachment."""
        hf_config = getattr(hf_config, "text_config", hf_config)
        tp_size = int(parallel_config.tensor_parallel_size)
        pp_size = int(parallel_config.pipeline_parallel_size)
        if tp_size < 1 or pp_size < 1:
            raise RuntimeError(
                f"Invalid vLLM parallel sizes: TP={tp_size}, PP={pp_size}"
            )

        num_layers = getattr(
            hf_config,
            "num_hidden_layers",
            getattr(hf_config, "n_layer", None),
        )
        if num_layers is None:
            raise RuntimeError("DMI vLLM model has no hidden-layer count")
        num_layers = int(num_layers)

        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        selection_model = (
            model.unwrap()
            if isinstance(model, CUDAGraphWrapper)
            else model
        )
        get_hook_specs = getattr(selection_model, "get_hook_specs", None)
        if get_hook_specs is None:
            raise RuntimeError(
                "DMI vLLM model does not provide get_hook_specs"
            )
        try:
            model_wide_specs = get_hook_specs(model_wide=True)
        except TypeError as exc:
            raise RuntimeError(
                "DMI vLLM model does not support "
                "get_hook_specs(model_wide=True)"
            ) from exc
        if any(spec.module is not None for spec in model_wide_specs):
            raise RuntimeError(
                "DMI vLLM model-wide HookSpecs must not bind live modules"
            )
        inventory_layers = {
            spec.layer_no for spec in model_wide_specs if spec.layer_no >= 0
        }
        if inventory_layers != set(range(num_layers)):
            raise RuntimeError(
                "DMI vLLM model-wide hook inventory does not cover every layer"
            )

        selected_model_wide = select_hook_specs(
            model_wide_specs, hook_selection, cfg
        )

        from vllm.distributed.utils import get_pp_indices

        pp_intervals = tuple(
            get_pp_indices(num_layers, pp_rank, pp_size)
            for pp_rank in range(pp_size)
        )
        covered_layers = [
            layer
            for start, end in pp_intervals
            for layer in range(start, end)
        ]
        if covered_layers != list(range(num_layers)):
            raise RuntimeError("vLLM PP layer partition is not an exact cover")

        tp_role_count = min(tp_size, 2)
        candidate_rank_hook_sets: List[Tuple[HookSpec, ...]] = []
        for pp_rank, (start_layer, end_layer) in enumerate(pp_intervals):
            pp_specs = tuple(
                spec
                for spec in selected_model_wide
                if (
                    (
                        spec.layer_no < 0
                        or start_layer <= spec.layer_no < end_layer
                    )
                    and hook_belongs_to_pp_rank(
                        spec,
                        is_first_rank=(pp_rank == 0),
                        is_last_rank=(pp_rank == pp_size - 1),
                    )
                )
            )
            for tp_role in range(tp_role_count):
                candidate_rank_hook_sets.append(
                    tuple(
                        spec
                        for spec in pp_specs
                        if hook_belongs_to_tp_rank(spec, tp_role)
                    )
                )

        return cls(
            local_hooks=local_hooks,
            candidate_rank_hook_sets=tuple(candidate_rank_hook_sets),
        )


@dataclass(frozen=True)
class _VLLMRoleFormula:
    real_terms: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    execution_terms: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    request_terms: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    hook_count: int = 0

    def bytes_for(
        self, execution_rows: int, real_rows: int, request_rows: int
    ) -> int:
        total = 0
        for row_bytes, multiplicity in self.real_terms:
            total += multiplicity * align_up_py(real_rows * row_bytes, 16)
        for row_bytes, multiplicity in self.execution_terms:
            total += multiplicity * align_up_py(execution_rows * row_bytes, 16)
        for row_bytes, multiplicity in self.request_terms:
            total += multiplicity * align_up_py(request_rows * row_bytes, 16)
        return total


# ---------------------------------------------------------------------------
# VLLMAdaptor
# ---------------------------------------------------------------------------


class VLLMAdaptor(BackendAdaptor):
    """``BackendAdaptor`` for vLLM models running under ``DMXGPUWorker``.

    vLLM's per-step protocol differs from HF's:
      * Tensors are packed/flattened (no batch dim; rows from each
        request are concatenated along dim 0).
      * Request IDs come from ``scheduler_output``, not auto-minted.
      * CUDA-graph dispatch may pad ``total_q`` up to the nearest
        capture size; meta shape must match the padded tensor.
      * Cached role formulas decide whether the current dispatch must be
        eager before vLLM selects its real batch descriptor.
      * Reservation and metadata use that real descriptor plus the packed
        request order recorded after vLLM prepares its inputs.
    """

    def __init__(
        self, engine: Any, model_id: str, vllm_config: Any,
        *, gpu_padding_strip: bool = True,
    ) -> None:
        super().__init__(engine, model_id)
        self.vllm_config = vllm_config
        self._debug_step: bool = bool(os.environ.get("RING_DEBUG_STEP"))
        # The per-call state carries a monotonic pre-dispatch eager latch.
        # transport.force_eager is assigned during the actual-layout commit
        # for the producer path of that same model step.
        # User opt-in; when True, ring transport stays in null mode after
        # warmup (kernels fire as no-ops, FIFO stays empty).
        self.user_wants_null_mode: bool = False
        # Stashed during build_step_context so adapt_for_cpu_direct can
        # restore the unpadded q_len without recomputing.
        self._last_total_q: int = 0
        # vLLM step counter (debug logging only).
        self._step_counter: int = 0
        # gpu_padding_strip mode: when True, the producer copies only
        # actual_q_len * row_bytes per eligible hook (instead of the
        # full padded captured tensor).  Default False = today's
        # behavior verbatim.  When True, attach_model allocates the
        # shared row_count tensor pair and wires each eligible
        # HookPoint's _strip_tensor + _strip_row_bytes; build_step_context
        # populates ctx.actual_q_len; before_forward does the per-step
        # 8-byte cudaMemcpyAsync of the new actual_q_len.
        self.gpu_padding_strip: bool = gpu_padding_strip
        self._row_count_dev: Optional[torch.Tensor] = None      # 1 int64 on GPU
        self._pinned_row_count: Optional[torch.Tensor] = None   # 1 int64 pinned host
        self._strip_eligible_hps: List[Any] = []                # for inspection
        self._step_state = _VLLMStepState()
        self._validation_mode = (
            VLLMValidationMode.VERIFY
            if self._debug_step
            else VLLMValidationMode.OFF
        )
        self._role_formulas: Tuple[_VLLMRoleFormula, ...] = ()
        self._local_role_formula: Optional[_VLLMRoleFormula] = None
        self._has_global_hooks = False
        self._byte_capacity = 0
        self._task_capacity = 0
        self._max_num_batched_tokens = 0
        self._max_num_seqs = 0
        self._capture_sizes: Tuple[int, ...] = ()
        self._max_capture_size = 0

    # ------- abstract overrides ---------------------------------------

    def detect_model_shape(self, model: Any) -> ModelShapeConfig:
        # Use the vllm_config dtype (authoritative); fall back to the
        # HF config's torch_dtype if vllm_config is missing.
        vllm_dtype = getattr(
            getattr(self.vllm_config, "model_config", None), "dtype", None
        )
        cfg = _make_model_shape_from_hf_config(
            self.vllm_config.model_config.hf_config, dtype=vllm_dtype
        )
        if cfg is None:
            raise RuntimeError(
                "VLLMAdaptor.detect_model_shape: vllm_config.model_config.hf_config "
                "is missing hidden_size / num_attention_heads."
            )
        # vLLM TP world size comes from its parallel-state group (NOT
        # torch.distributed; vLLM constructs its own TP/PP groups).
        # ``_compute_hook_shape`` divides head/intermediate dims by
        # ``cfg.tp_size`` for sharded hooks (q, k, v, z, mlp_post,
        # attn_scores, pattern) -- so getting this right is load-bearing
        # for TP > 1: the producer-side meta shape must match the actual
        # sharded tensor, otherwise the drain thread rejects the row.
        from vllm.distributed.parallel_state import get_tp_group
        cfg.tp_size = max(1, get_tp_group().world_size)
        return cfg

    def detect_parallel_ranks(self) -> Tuple[int, int, int, int]:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        tp_rank = get_tp_group().rank_in_group
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        ep_rank = 0  # vLLM does not expose EP groups today; placeholder.
        pp_rank = get_pp_group().rank_in_group
        return (tp_rank, dp_rank, ep_rank, pp_rank)

    def is_pp_first(self) -> bool:
        from vllm.distributed.parallel_state import get_pp_group
        return get_pp_group().is_first_rank

    def is_pp_last(self) -> bool:
        from vllm.distributed.parallel_state import get_pp_group
        return get_pp_group().is_last_rank

    def on_capacity_exceeded(self, ctx: StepContext) -> None:
        # No-op.  transport.force_eager is owned by adaptor_base
        # before_forward (`force_eager = (result == 2) or needs_eager`).
        # Kept as a framework hook for subclasses that want to react to
        # overflow events (telemetry, custom logging, etc.).
        return

    def adapt_for_cpu_direct(self, ctx: StepContext) -> StepContext:
        # When the worker forces eager for this batch, the actual
        # tensor will not be padded -- swap meta q_len from padded
        # back to the unpadded total_q stashed during build_step_context.
        import dataclasses
        if self._last_total_q and ctx.q_len != self._last_total_q:
            return dataclasses.replace(ctx, q_len=self._last_total_q)
        return ctx

    def _warn_once_capacity(
        self, ctx: StepContext, total_bytes: int, n_hooks: int
    ) -> None:
        # Per-(total_q, num_reqs) shape warn.  Stored on adapter (not
        # transport) so multiple adapters in the same process don't
        # share state.
        shape_key = (ctx.q_len, len(ctx.req_ids))
        if shape_key in self._warned_shapes:
            return
        self._warned_shapes.add(shape_key)
        re = self.ring_engine
        if re is None:
            return
        pcap = re.payload_cap()
        scap = re.staging_cap()
        if total_bytes > pcap and total_bytes > scap:
            reason = (
                f"exceeds both GPU ring ({pcap / 1e6:.0f} MB) "
                f"and pinned staging ({scap / 1e6:.0f} MB)"
            )
        elif total_bytes > pcap:
            reason = f"exceeds GPU ring ({pcap / 1e6:.0f} MB)"
        else:
            reason = f"exceeds pinned staging ({scap / 1e6:.0f} MB)"
        warnings.warn(
            f"[vllm_integration] Step data ({total_bytes / 1e6:.1f} MB) "
            f"{reason}. Falling back to eager dispatch + per-hook safety net "
            f"for {n_hooks} hooks.",
            stacklevel=2,
        )

    # ------- vLLM-specific helpers ------------------------------------

    def predict_padded_q_len(self, model_runner: Any, total_q: int) -> int:
        """Read vLLM's CUDA-graph capture size table and return the
        padded ``q_len`` the producer kernel will see.

        Risk surface: this reads ``cudagraph_dispatcher._bs_to_padded_graph_size``,
        a private list[int] indexed by token count.  On a vLLM upgrade,
        verify the attribute still exists and the per-step dispatch
        logic hasn't added new conditions that bypass CUDA graphs (LoRA,
        cascade, encoder, etc.).  Today the only per-step disabler is
        our own ``force_eager_next_batch``.
        """
        pad_table = getattr(
            getattr(model_runner, "cudagraph_dispatcher", None),
            "_bs_to_padded_graph_size", None,
        )
        if pad_table is not None and total_q < len(pad_table):
            return pad_table[total_q]
        return total_q

    def build_step_context(
        self, scheduler_output: Any, model_runner: Any
    ) -> Optional[StepContext]:
        """Build metadata from the real packed layout and real dispatch."""
        state = self._step_state
        layout = state.layout
        batch_descriptor = state.real_batch_descriptor
        if layout is None or batch_descriptor is None:
            raise RuntimeError(
                "DMI vLLM step context requested before real layout/dispatch"
            )
        if state.scheduler_output is not scheduler_output:
            raise RuntimeError("DMI vLLM scheduler output changed within one step")
        if layout.total_rows == 0:
            return None

        self._step_counter += 1
        _step = self._step_counter
        self._last_total_q = layout.total_rows

        if self._debug_step:
            for i, req_id in enumerate(layout.req_ids):
                start, end = layout.token_ranges[i]
                print(
                    f"[dmx_worker] step={_step} req[{i}] rid={req_id} "
                    f"offset={layout.dim0_offsets[i]} "
                    f"n={layout.scheduled_counts[i]} "
                    f"t_start={start} t_end={end} "
                    f"pre_computed={layout.computed_counts[i]} "
                    f"q_len={batch_descriptor.num_tokens}",
                    flush=True,
                )

        # Read input_ids dtype from model_runner buffer.  On non-first
        # PP ranks the buffer may not exist; pass None (the token_ids
        # hook is already filtered out by filter_by_pp_rank).
        ids_buf = getattr(model_runner, "input_ids", None)
        ids_dtype = ids_buf.gpu.dtype if ids_buf is not None else None

        tp_rank, dp_rank, ep_rank, pp_rank = self.detect_parallel_ranks()

        # logits_to_keep=num_reqs: vLLM's compute_logits returns one
        # logit per request shaped [num_reqs, vocab].  In flattened
        # mode _compute_hook_shape uses logits_to_keep directly as
        # dim0 (no batch dim), so the meta shape becomes
        # [num_reqs, vocab].  The p2p thread then slices row j for
        # request j and adjusts the DB token range to (end_token-1,
        # end_token) -- the single predicted position per request.
        return StepContext(
            model_id=str(self.model_id),
            flattened=True,
            req_ids=list(layout.req_ids),
            token_ranges=list(layout.token_ranges),
            dim0_offsets=list(layout.dim0_offsets),
            kv_offsets=[0] * len(layout.req_ids),
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            ep_rank=ep_rank,
            pp_rank=pp_rank,
            batch=0,
            q_len=int(batch_descriptor.num_tokens),
            kv_dim=0,
            logits_to_keep=len(layout.req_ids),
            token_ids_dtype=ids_dtype,
            actual_q_len=(
                layout.total_rows if self.gpu_padding_strip else None
            ),
        )

    # ----- gpu_padding_strip integration -----

    def attach_model(self, model: Any, hook_selection: str = "full") -> None:
        """Standard attach, plus gpu_padding_strip pool setup when enabled.

        When `gpu_padding_strip=True`, allocate one shared int64[1] device
        tensor (`_row_count_dev`) and one pinned-host counterpart.  For
        every active spec with `dim0_is_actual_tokens=True`, point the
        HookPoint's `_strip_tensor` at the shared tensor and bake its
        per-spec `_strip_row_bytes` (CPU-known constant).  Every step
        we update the shared tensor's value in place (see
        before_forward); each HookPoint's captured producer_prefix
        call reads the freshly-written value and multiplies by its
        baked row_bytes.
        """
        super().attach_model(model, hook_selection)
        if self.gpu_padding_strip:
            device = (
                next(model.parameters()).device
                if hasattr(model, "parameters")
                else "cuda"
            )
            self._row_count_dev = torch.empty(
                1, dtype=torch.int64, device=device
            )
            self._pinned_row_count = torch.empty(
                1, dtype=torch.int64, pin_memory=True
            )
            for spec in self.active_specs:
                if not spec.dim0_is_actual_tokens:
                    continue
                hp = spec.module
                rb = _row_bytes_for_spec(spec, self.model_cfg)
                if rb <= 0:
                    continue
                hp._strip_tensor = self._row_count_dev
                hp._strip_row_bytes = rb
                self._strip_eligible_hps.append(hp)

        if self.transport is not None and self.transport.null_offload:
            return
        cfg = self.model_cfg
        if cfg is None or self.ring_engine is None:
            raise RuntimeError("DMI vLLM role formulas require an attached ring")
        selection = _VLLMHookSelection.from_model(
            model=model,
            local_hooks=tuple(self.active_specs),
            hook_selection=hook_selection,
            cfg=cfg,
            parallel_config=self.vllm_config.parallel_config,
            hf_config=self.vllm_config.model_config.hf_config,
        )
        self._compile_role_formulas(selection)

    def _compile_role_formulas(
        self, selection: _VLLMHookSelection
    ) -> None:
        """Compile exact byte/hook formulas for distinct TP/PP rank types."""
        cfg = self.model_cfg
        if cfg is None or self.ring_engine is None:
            raise RuntimeError("DMI vLLM role formulas require an attached ring")

        parallel = self.vllm_config.parallel_config
        scheduler = self.vllm_config.scheduler_config
        tp_size = int(parallel.tensor_parallel_size)
        if parallel.data_parallel_size > 1 and parallel.use_ubatching:
            raise RuntimeError(
                "DMI vLLM does not support DP with DBO/ubatching"
            )

        try:
            self._byte_capacity = min(
                int(self.ring_engine.payload_cap()),
                int(self.ring_engine.staging_cap()),
            )
            self._task_capacity = int(self.ring_engine.task_cap())
            self._max_num_batched_tokens = int(
                scheduler.max_num_batched_tokens
            )
            self._max_num_seqs = int(scheduler.max_num_seqs)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "DMI vLLM could not cache finite scheduler/ring bounds"
            ) from exc
        if (
            self._byte_capacity <= 0
            or self._task_capacity <= 0
            or self._max_num_batched_tokens <= 0
            or self._max_num_seqs <= 0
        ):
            raise RuntimeError("DMI vLLM scheduler/ring bounds must be positive")

        compilation = self.vllm_config.compilation_config
        capture_sizes = compilation.cudagraph_capture_sizes or ()
        try:
            self._capture_sizes = tuple(
                sorted({int(size) for size in capture_sizes if int(size) > 0})
            )
            self._max_capture_size = int(
                compilation.max_cudagraph_capture_size
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "DMI vLLM could not cache CUDA-graph capture bounds"
            ) from exc
        if self._capture_sizes and (
            self._max_capture_size <= 0
            or self._capture_sizes[-1] > self._max_capture_size
        ):
            raise RuntimeError("Invalid vLLM CUDA-graph capture sizes")

        if (
            self.vllm_config.speculative_config is not None
            and any(
                hook_row_basis(spec.hook_type)
                is HookRowBasis.REQUEST_ROWS
                for specs in selection.candidate_rank_hook_sets
                for spec in specs
            )
        ):
            raise RuntimeError(
                "DMI vLLM request-row metadata does not support "
                "speculative decoding"
            )

        tp_role_count = min(tp_size, 2)

        def add_term(
            terms: dict[int, int], row_bytes: int, multiplicity: int
        ) -> None:
            terms[row_bytes] = terms.get(row_bytes, 0) + multiplicity

        def compile_formula(
            specs: Tuple[HookSpec, ...]
        ) -> _VLLMRoleFormula:
            real_terms: dict[int, int] = {}
            execution_terms: dict[int, int] = {}
            request_terms: dict[int, int] = {}
            hook_count = 0
            for spec in specs:
                row_bytes = _row_bytes_for_spec(spec, cfg)
                if row_bytes <= 0:
                    continue
                row_basis = hook_row_basis(spec.hook_type)
                if row_basis is HookRowBasis.REQUEST_ROWS:
                    add_term(request_terms, row_bytes, 1)
                elif self.gpu_padding_strip and spec.dim0_is_actual_tokens:
                    add_term(real_terms, row_bytes, 1)
                else:
                    add_term(execution_terms, row_bytes, 1)
                hook_count += 1
            return _VLLMRoleFormula(
                real_terms=tuple(sorted(real_terms.items())),
                execution_terms=tuple(sorted(execution_terms.items())),
                request_terms=tuple(sorted(request_terms.items())),
                hook_count=hook_count,
            )

        candidate_formulas = tuple(
            compile_formula(specs)
            for specs in selection.candidate_rank_hook_sets
        )

        tp_rank, _dp_rank, _ep_rank, pp_rank = self.detect_parallel_ranks()
        local_tp_role = 0 if tp_rank == 0 else 1
        local_candidate_index = pp_rank * tp_role_count + local_tp_role
        if not 0 <= local_candidate_index < len(candidate_formulas):
            raise RuntimeError("DMI vLLM could not identify its local role")
        local_formula = candidate_formulas[local_candidate_index]
        concrete_formula = compile_formula(selection.local_hooks)
        if local_formula != concrete_formula:
            raise RuntimeError(
                "DMI vLLM cached role formula does not match local hook layout: "
                f"cached={local_formula}, concrete={concrete_formula}"
            )

        self._role_formulas = tuple(dict.fromkeys(candidate_formulas))
        self._local_role_formula = local_formula
        self._has_global_hooks = any(
            formula.hook_count > 0 for formula in self._role_formulas
        )
        max_hooks = max(
            (formula.hook_count for formula in self._role_formulas),
            default=0,
        )
        if max_hooks > self._task_capacity:
            raise RuntimeError(
                "DMI vLLM selected hooks exceed task-ring capacity: "
                f"hooks={max_hooks}, capacity={self._task_capacity}"
            )

    def _record_real_layout(
        self,
        scheduler_output: Any,
        model_runner: Any,
        num_scheduled_tokens: Any,
    ) -> None:
        """Copy and validate the packed order produced by real vLLM prep."""
        state = self._step_state
        if state.phase is not VLLMStepPhase.ARMED:
            raise RuntimeError(
                "DMI vLLM observed duplicate or unarmed _prepare_inputs"
            )
        if state.scheduler_output is not scheduler_output:
            raise RuntimeError("DMI vLLM scheduler output changed during prep")

        num_reqs = int(model_runner.input_batch.num_reqs)
        raw_req_ids = tuple(model_runner.input_batch.req_ids[:num_reqs])
        scheduled_counts = tuple(
            int(value) for value in num_scheduled_tokens
        )
        computed_counts = tuple(
            int(value)
            for value in model_runner.input_batch.num_computed_tokens_cpu[
                :num_reqs
            ]
        )
        if (
            len(raw_req_ids) != num_reqs
            or len(scheduled_counts) != num_reqs
            or len(computed_counts) != num_reqs
        ):
            raise RuntimeError("DMI vLLM packed layout lengths disagree")
        if any(not isinstance(req_id, str) for req_id in raw_req_ids):
            raise RuntimeError("DMI vLLM request IDs must be strings")
        if len(set(raw_req_ids)) != num_reqs:
            raise RuntimeError("DMI vLLM packed request IDs are not unique")

        req_ids = tuple(
            normalize_vllm_request_id(req_id) for req_id in raw_req_ids
        )
        if len(set(req_ids)) != num_reqs:
            raise RuntimeError(
                "DMI vLLM normalized request IDs are not unique"
            )
        if any(value < 0 for value in scheduled_counts):
            raise RuntimeError("DMI vLLM scheduled token count is negative")
        if any(value < 0 for value in computed_counts):
            raise RuntimeError("DMI vLLM computed token count is negative")

        scheduler_counts = scheduler_output.num_scheduled_tokens
        if (
            len(scheduler_counts) != num_reqs
            or set(scheduler_counts) != set(raw_req_ids)
        ):
            raise RuntimeError(
                "DMI vLLM packed IDs do not match scheduler membership"
            )
        for req_id, count in zip(raw_req_ids, scheduled_counts):
            if int(scheduler_counts[req_id]) != count:
                raise RuntimeError(
                    f"DMI vLLM scheduled count mismatch for {req_id!r}"
                )

        total_rows = sum(scheduled_counts)
        if total_rows != int(scheduler_output.total_num_scheduled_tokens):
            raise RuntimeError(
                "DMI vLLM packed rows do not match scheduler total"
            )

        offsets: List[int] = []
        token_ranges: List[Tuple[int, int]] = []
        offset = 0
        for scheduled, computed in zip(
            scheduled_counts, computed_counts
        ):
            offsets.append(offset)
            token_ranges.append((computed, computed + scheduled))
            offset += scheduled
        if offset != total_rows:
            raise RuntimeError("DMI vLLM packed offsets do not cover all rows")

        state.layout = _VLLMRealLayout(
            raw_req_ids=raw_req_ids,
            req_ids=req_ids,
            scheduled_counts=scheduled_counts,
            computed_counts=computed_counts,
            token_ranges=tuple(token_ranges),
            dim0_offsets=tuple(offsets),
            total_rows=total_rows,
        )
        state.phase = VLLMStepPhase.LAYOUT_READY

    def _preview_exact_dispatch(
        self,
        model_runner: Any,
        *,
        num_tokens: int,
        num_reqs: int,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        force_eager: bool,
        force_uniform_decode: Optional[bool],
        force_has_lora: Optional[bool],
        force_num_active_loras: Optional[int],
        num_encoder_reqs: int,
    ) -> Tuple[Any, Any]:
        """Run only vLLM's read-only dispatcher calculation."""
        from vllm.config import CUDAGraphMode

        uniform_decode = model_runner._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=model_runner.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        has_encoder_output = (
            model_runner.model_config.is_encoder_decoder
            and num_encoder_reqs > 0
        )
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(model_runner.input_batch.lora_id_to_lora_request)
        )
        has_lora = (
            num_active_loras > 0
            if force_has_lora is None
            else force_has_lora
        )
        valid_modes = (
            {CUDAGraphMode.NONE}
            if force_eager
            else set(CUDAGraphMode.valid_runtime_modes())
        )
        invalid_modes = (
            {CUDAGraphMode.FULL}
            if use_cascade_attn or has_encoder_output
            else set()
        )
        return model_runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens,
            has_lora=has_lora,
            uniform_decode=uniform_decode,
            num_active_loras=num_active_loras,
            valid_modes=valid_modes,
            invalid_modes=invalid_modes,
        )

    def _conservative_execution_bound(
        self, num_tokens: int, num_reqs: int
    ) -> Tuple[int, int, int]:
        """Return bounded execution, real-row, and request-row counts."""
        parallel = self.vllm_config.parallel_config
        compilation = self.vllm_config.compilation_config
        if parallel.data_parallel_size > 1:
            real_rows = self._max_num_batched_tokens
            request_rows = self._max_num_seqs
        else:
            real_rows = int(num_tokens)
            request_rows = int(num_reqs)

        dispatch_rows = real_rows
        if compilation.pass_config.enable_sp:
            tp_size = int(parallel.tensor_parallel_size)
            dispatch_rows = (
                (dispatch_rows + tp_size - 1) // tp_size
            ) * tp_size

        execution_rows = dispatch_rows
        if (
            self._capture_sizes
            and dispatch_rows <= self._max_capture_size
        ):
            execution_rows = self._max_capture_size
        if execution_rows < dispatch_rows:
            raise RuntimeError(
                "DMI vLLM conservative graph bound is not an upper bound"
            )
        return execution_rows, real_rows, request_rows

    def _preflight_force_eager(
        self,
        model_runner: Any,
        *,
        num_tokens: int,
        num_reqs: int,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        caller_force_eager: bool,
        force_uniform_decode: Optional[bool],
        force_has_lora: Optional[bool],
        force_num_active_loras: Optional[int],
        num_encoder_reqs: int,
    ) -> bool:
        """Evaluate cached worst-role formulas with no transport effects."""
        state = self._step_state
        parallel = self.vllm_config.parallel_config
        compilation = self.vllm_config.compilation_config
        exact = (
            not compilation.pass_config.enable_sp
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
        )

        if exact:
            capacity_candidate = self._preview_exact_dispatch(
                model_runner,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                force_eager=(
                    caller_force_eager or state.force_eager_latch
                ),
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )
            state.capacity_candidate = capacity_candidate
            state.execution_bound = None
            state.real_rows_bound = None
            state.request_rows_bound = None
            execution_rows = int(capacity_candidate[1].num_tokens)
            real_rows = int(num_tokens)
            request_rows = int(num_reqs)
        else:
            (
                execution_rows,
                real_rows,
                request_rows,
            ) = self._conservative_execution_bound(
                num_tokens, num_reqs
            )
            state.capacity_candidate = None
            state.execution_bound = execution_rows
            state.real_rows_bound = real_rows
            state.request_rows_bound = request_rows

        worst_bytes = max(
            (
                formula.bytes_for(
                    execution_rows, real_rows, request_rows
                )
                for formula in self._role_formulas
            ),
            default=0,
        )
        force_eager = worst_bytes > self._byte_capacity

        if (
            exact
            and self._validation_mode is VLLMValidationMode.VERIFY
        ):
            state.expected_dispatch = self._preview_exact_dispatch(
                model_runner,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                force_eager=(
                    caller_force_eager
                    or state.force_eager_latch
                    or force_eager
                ),
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )
        else:
            state.expected_dispatch = None
        return force_eager

    def _commit_actual_dispatch(
        self,
        scheduler_output: Any,
        model_runner: Any,
        cudagraph_mode: Any,
        batch_descriptor: Any,
        combined_force_eager: bool,
    ) -> None:
        """Commit metadata/reservation from the real layout and descriptor."""
        state = self._step_state
        layout = state.layout
        if state.phase is not VLLMStepPhase.LAYOUT_READY or layout is None:
            raise RuntimeError(
                "DMI vLLM dispatch committed without one real layout"
            )
        if int(batch_descriptor.num_tokens) < layout.total_rows:
            raise RuntimeError(
                "DMI vLLM dispatch has fewer rows than the packed layout"
            )

        state.real_cudagraph_mode = cudagraph_mode
        state.real_batch_descriptor = batch_descriptor

        if self._validation_mode is VLLMValidationMode.VERIFY:
            if state.expected_dispatch is not None:
                if state.expected_dispatch != (
                    cudagraph_mode,
                    batch_descriptor,
                ):
                    raise RuntimeError(
                        "DMI vLLM exact dispatch preview disagrees with vLLM"
                    )
            elif state.execution_bound is not None:
                if int(batch_descriptor.num_tokens) > state.execution_bound:
                    raise RuntimeError(
                        "DMI vLLM execution rows exceed conservative bound"
                    )
                if (
                    state.real_rows_bound is None
                    or layout.total_rows > state.real_rows_bound
                    or state.request_rows_bound is None
                    or len(layout.req_ids) > state.request_rows_bound
                ):
                    raise RuntimeError(
                        "DMI vLLM real layout exceeds conservative bound"
                    )

        ctx = self.build_step_context(scheduler_output, model_runner)
        if ctx is None:
            raise RuntimeError("DMI vLLM committed an empty armed step")
        actual_plan = self._compute_step_plan(ctx)
        actual_bytes, actual_hooks, actual_needs_eager = actual_plan
        if actual_hooks > self._task_capacity:
            raise RuntimeError(
                "DMI vLLM actual hook count exceeds cached task capacity"
            )
        actual_requires_eager = (
            actual_needs_eager
            or actual_bytes > self._byte_capacity
        )
        if actual_requires_eager and not combined_force_eager:
            raise RuntimeError(
                "DMI vLLM eager preflight produced a false negative"
            )

        if self._validation_mode is VLLMValidationMode.VERIFY:
            formula = self._local_role_formula
            if formula is None:
                raise RuntimeError("DMI vLLM local role formula is missing")
            formula_plan = (
                formula.bytes_for(
                    int(batch_descriptor.num_tokens),
                    layout.total_rows,
                    len(layout.req_ids),
                ),
                formula.hook_count,
            )
            if formula_plan != actual_plan[:2] or actual_needs_eager:
                raise RuntimeError(
                    "DMI vLLM cached local formula disagrees with actual plan: "
                    f"cached={formula_plan}, actual={actual_plan}"
                )

        self.before_forward(ctx, actual_plan)
        state.phase = VLLMStepPhase.COMMITTED

    def before_forward(
        self, ctx: StepContext, step_plan: Tuple[int, int, bool]
    ) -> None:
        """Commit one precomputed plan, then update padding-strip rows."""
        if self.transport is None or self.transport.null_offload:
            return

        total_bytes, n_hooks, needs_eager = step_plan
        if n_hooks > 0:
            result = self.ring_engine.prepare_step(total_bytes, n_hooks)
            self.transport.force_eager = (result == 2) or needs_eager
            if result == 2:
                ctx = self.adapt_for_cpu_direct(ctx)
                self.on_capacity_exceeded(ctx)
                self._warn_once_capacity(ctx, total_bytes, n_hooks)
        else:
            self.transport.force_eager = False

        self.transport.set_step_context(**ctx.transport_kwargs())
        self.transport.pre_push_all_metas(
            batch=ctx.batch,
            q_len=ctx.q_len,
            kv_dim=ctx.kv_dim,
            logits_to_keep=ctx.logits_to_keep,
            token_ids_dtype=ctx.token_ids_dtype,
            actual_q_len=ctx.actual_q_len,
        )

        if (self.gpu_padding_strip
                and self._pinned_row_count is not None
                and self._row_count_dev is not None
                and self._last_total_q > 0):
            # CPU: write the current step's actual_q_len to pinned host.
            # GPU: enqueue an async copy to the shared device scalar on
            # the model stream (captureable; values propagate to replays).
            self._pinned_row_count[0] = self._last_total_q
            self._row_count_dev.copy_(self._pinned_row_count, non_blocking=True)


def _row_bytes_for_spec(spec, model_cfg) -> int:
    """Bytes-per-token for `spec`'s shape under vLLM flat layout.

    Computed as prod(shape_excluding_total_tokens) * elem_size where
    the shape comes from _compute_hook_shape with batch=0, q_len=1.
    The "1" stands in for "one token's worth"; the result is the
    per-token byte stride.  Returns 0 if the spec produces an empty
    shape (skip).
    """
    shape = _compute_hook_shape(
        spec.hook_type, model_cfg, batch=0, q_len=1, kv_dim=0,
        logits_to_keep=0,
    )
    if not shape:
        return 0
    # In flat mode, dim-0 of the spec's shape IS the token count.
    # shape[0] should be 1 here (since we passed q_len=1); the
    # remaining dims times elem_size = bytes per token.
    nelem = 1
    for d in shape[1:]:
        nelem *= d
    dtype = spec.dtype if spec.dtype is not None else model_cfg.dtype
    elem_size = torch._utils._element_size(dtype)
    return int(nelem) * int(elem_size)


# ---------------------------------------------------------------------------
# DMXGPUWorker
# ---------------------------------------------------------------------------


class DMXGPUWorker(Worker):
    """vLLM ``Worker`` subclass that owns a ``VLLMAdaptor`` and
    delegates per-step work to it.

    Lifecycle:
      ``init_device``           -- super + build engine + adaptor + null mode +
                                    wrap input preparation and dispatch.
      ``load_model``            -- arch remap + super + adaptor.attach_model.
      ``compile_or_warm_up_model`` -- super + clear null_mode (unless user
                                       opted-in via ``dmx_null_mode``).
      ``execute_model``         -- arm per-call state + super; the dispatch
                                    wrapper commits DMI metadata before forward.
      ``stop_monitoring``       -- adaptor.close (CUDA sync, ring stop,
                                    deactivate transport, host engine stop).
      ``shutdown``              -- best-effort stop_monitoring + super.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.adaptor: Optional[VLLMAdaptor] = None
        self._dmx_host_engine: Any = None
        self._dmx_hook_selection: str = "vllm-full"

    def init_device(self) -> None:
        if getattr(self, "use_v2_model_runner", False):
            raise RuntimeError(
                "DMI does not yet support vLLM's V2 model runner. "
                "Set VLLM_USE_V2_MODEL_RUNNER=0 before constructing the "
                "vLLM engine to select the supported V1 runner."
            )
        super().init_device()

        from monitoring import _native_engine as _ne
        from monitoring.engine import MonitoringEngine

        ac = self.vllm_config.additional_config
        if not isinstance(ac, dict):
            ac = {}

        self._dmx_hook_selection = _cfg(
            ac, "dmx_hook_selection", "DMX_HOOK_SELECTION", "vllm-full"
        )
        model_id = _cfg(ac, "dmx_model_id", "DMX_MODEL_ID", "")
        ring_payload_mb = _cfg(ac, "dmx_ring_payload_mb", "DMX_RING_PAYLOAD_MB", 4096)
        ring_pinned_mb = _cfg(ac, "dmx_ring_pinned_mb", "DMX_RING_PINNED_MB", 4096)
        ring_entries = _cfg(ac, "dmx_ring_task_entries", "DMX_RING_TASK_ENTRIES", 65536)
        null_mode = _cfg(ac, "dmx_null_mode", "DMX_NULL_MODE", False)
        db_host = _cfg(ac, "dmx_db_host", "DMX_DB_HOST", "")
        db_port = _cfg(ac, "dmx_db_port", "DMX_DB_PORT", 9000)
        db_database = _cfg(ac, "dmx_db_database", "DMX_DB_DATABASE", "default")
        db_table = _cfg(ac, "dmx_db_table", "DMX_DB_TABLE", "offload")
        ch_parallelism = int(
            _cfg(ac, "dmx_ch_parallelism", "DMX_CH_PARALLELISM", 10)
        )

        resolved_model_id = model_id or str(self.vllm_config.model_config.model)

        # Host engine (ClickHouse), optional.
        host_engine = None
        if db_host:
            ch_cfg = _ne.ClickHouseClientConfig()
            ch_cfg.host = db_host
            ch_cfg.port = db_port
            ch_cfg.database = db_database
            ch_cfg.table = db_table
            ch_cfg.create_database_if_missing = True
            stage_cfg = _ne.StageConfig.clickhouse_insert(
                ch_cfg, parallelism=ch_parallelism, name="clickhouse_insert"
            )
            q = stage_cfg.input_queue
            q.max_batch_items = int(_cfg(
                ac, "dmx_ch_max_batch_items", "DMX_CH_MAX_BATCH_ITEMS", 1024))
            q.high_watermark_items = q.max_batch_items
            q.max_batch_size = int(_cfg(
                ac, "dmx_ch_max_batch_bytes", "DMX_CH_MAX_BATCH_BYTES",
                2048 * 1024 * 1024))
            q.high_watermark_size = q.max_batch_size
            host_engine = _ne.DMXHostEngine(stage_cfg)
            host_engine.start()
            self._dmx_host_engine = host_engine

        ring_cfg = _ne.RingConfig()
        ring_cfg.payload_ring_bytes = ring_payload_mb * 1024 * 1024
        ring_cfg.pinned_staging_bytes = ring_pinned_mb * 1024 * 1024
        ring_cfg.task_ring_entries = ring_entries
        ring_cfg.insert_queue_max_bytes = int(_cfg(
            ac, "dmx_insert_queue_max_bytes", "DMX_INSERT_QUEUE_MAX_BYTES",
            4096 * 1024 * 1024))
        ring_cfg.insert_queue_max_items = int(_cfg(
            ac, "dmx_insert_queue_max_items", "DMX_INSERT_QUEUE_MAX_ITEMS",
            65536))
        ring_cfg.drain_flush_timeout_us = int(_cfg(
            ac, "dmx_drain_flush_timeout_us", "DMX_DRAIN_FLUSH_TIMEOUT_US",
            0))

        # MonitoringEngine + ring transport.
        engine = MonitoringEngine(
            config=None,
            model_id=resolved_model_id,
            host_engine=host_engine,
            ring_config=ring_cfg,
        )

        # Build the adaptor.  Hooks aren't installed yet -- that
        # happens in load_model after the model is materialized.
        # gpu_padding_strip is on by default; flip it off via
        # additional_config["dmx_gpu_padding_strip"]=False or
        # DMX_GPU_PADDING_STRIP=0 if needed for debugging.
        gpu_padding_strip = _cfg(
            ac, "dmx_gpu_padding_strip", "DMX_GPU_PADDING_STRIP", True)
        self.adaptor = VLLMAdaptor(
            engine, resolved_model_id, self.vllm_config,
            gpu_padding_strip=bool(gpu_padding_strip))
        self.adaptor.user_wants_null_mode = bool(null_mode)
        if null_mode:
            self.adaptor.transport.null_offload = True

        # Enable null mode for warmup so producer kernels fire (CUDA
        # graph capture needs them) but no-op on the data path.  Cleared
        # after warmup in compile_or_warm_up_model unless the user
        # explicitly opted in to permanent null mode.
        self.adaptor.ring_engine.set_null_mode(True)

        # DMI's transport and native active-engine slots are process/device
        # global. This integration therefore assumes one active monitored
        # engine/forward owner per process.
        adaptor = self.adaptor
        model_runner = self.model_runner
        orig_prepare = model_runner._prepare_inputs
        orig_fn = self.model_runner._determine_batch_execution_and_padding

        def _wrapped_prepare(
            scheduler_output: Any,
            num_scheduled_tokens: np.ndarray,
        ) -> Any:
            result = orig_prepare(
                scheduler_output,
                num_scheduled_tokens,
            )
            phase = adaptor._step_state.phase
            if phase is VLLMStepPhase.ARMED:
                adaptor._record_real_layout(
                    scheduler_output,
                    model_runner,
                    num_scheduled_tokens,
                )
            elif phase is not VLLMStepPhase.IDLE:
                raise RuntimeError(
                    "DMI vLLM observed a second _prepare_inputs in one step"
                )
            return result

        def _wrapped_determine(
            num_tokens: int,
            num_reqs: int,
            num_scheduled_tokens_np: np.ndarray,
            max_num_scheduled_tokens: int,
            use_cascade_attn: bool,
            allow_microbatching: bool = True,
            force_eager: bool = False,
            force_uniform_decode: Optional[bool] = None,
            force_has_lora: Optional[bool] = None,
            force_num_active_loras: Optional[int] = None,
            num_encoder_reqs: int = 0,
        ) -> Any:
            state = adaptor._step_state
            if state.phase is VLLMStepPhase.IDLE:
                return orig_fn(
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=use_cascade_attn,
                    allow_microbatching=allow_microbatching,
                    force_eager=force_eager,
                    force_uniform_decode=force_uniform_decode,
                    force_has_lora=force_has_lora,
                    force_num_active_loras=force_num_active_loras,
                    num_encoder_reqs=num_encoder_reqs,
                )
            if state.phase is VLLMStepPhase.COMMITTED:
                raise RuntimeError(
                    "DMI vLLM observed dispatch after metadata commit"
                )
            if state.phase not in {
                VLLMStepPhase.ARMED,
                VLLMStepPhase.LAYOUT_READY,
            }:
                raise RuntimeError(
                    f"DMI vLLM invalid dispatch phase: {state.phase}"
                )

            state.force_eager_latch |= adaptor._preflight_force_eager(
                model_runner,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                caller_force_eager=force_eager,
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )
            combined_force_eager = force_eager or state.force_eager_latch
            result = orig_fn(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=use_cascade_attn,
                allow_microbatching=allow_microbatching,
                force_eager=combined_force_eager,
                force_uniform_decode=force_uniform_decode,
                force_has_lora=force_has_lora,
                force_num_active_loras=force_num_active_loras,
                num_encoder_reqs=num_encoder_reqs,
            )

            if state.phase is VLLMStepPhase.ARMED:
                parallel = adaptor.vllm_config.parallel_config
                compilation = adaptor.vllm_config.compilation_config
                if (
                    parallel.pipeline_parallel_size <= 1
                    or not compilation.pass_config.enable_sp
                    or state.prelayout_dispatch_seen
                ):
                    raise RuntimeError(
                        "DMI vLLM observed unexpected pre-layout dispatch"
                    )
                state.prelayout_dispatch_seen = True
                if (
                    adaptor._validation_mode
                    is VLLMValidationMode.VERIFY
                    and state.execution_bound is not None
                    and int(result[1].num_tokens) > state.execution_bound
                ):
                    raise RuntimeError(
                        "DMI vLLM early dispatch exceeds conservative bound"
                    )
                return result

            adaptor._commit_actual_dispatch(
                state.scheduler_output,
                model_runner,
                result[0],
                result[1],
                combined_force_eager,
            )
            return result

        model_runner._prepare_inputs = _wrapped_prepare
        model_runner._determine_batch_execution_and_padding = _wrapped_determine

        # Backwards-compat attributes for external subclasses that read
        # the pre-refactor names (e.g. tests/compare_worker.py reads
        # `self._dmx_tp_rank` + `self._dmx_tp_size` to format per-rank
        # filenames).
        from vllm.distributed.parallel_state import get_tp_group
        tp_rank, dp_rank, ep_rank, pp_rank = self.adaptor.detect_parallel_ranks()
        self._dmx_tp_rank = tp_rank
        self._dmx_dp_rank = dp_rank
        self._dmx_ep_rank = ep_rank
        self._dmx_pp_rank = pp_rank
        self._dmx_tp_size = get_tp_group().world_size

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        # Remap the architecture string in the HF config so vLLM's
        # registry resolves to the hooked variant (Qwen3PForCausalLM
        # etc.).  Mutates vllm_config in place.
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs

        # vLLM 0.19 added the keyword-only ``load_dummy_weights`` argument;
        # forwarding the call unchanged keeps this worker compatible with
        # both the older no-argument API and the newer one.
        super().load_model(*args, **kwargs)

        # Now the model is materialized; install hooks via the adapter.
        if self.adaptor is None:
            return
        self.adaptor.attach_model(
            self.model_runner.model, hook_selection=self._dmx_hook_selection
        )

    def compile_or_warm_up_model(self) -> Any:
        # Warmup runs with null_mode=True (set in init_device).
        # Producer kernels fire but are no-ops -- ring stays clean.
        result = super().compile_or_warm_up_model()

        # Warmup done.  Turn off null mode unless the user explicitly
        # asked for permanent null mode.  set_null_mode does
        # cudaDeviceSynchronize internally so it's safe to call here.
        if (
            self.adaptor is not None
            and not self.adaptor.user_wants_null_mode
            and self.adaptor.ring_engine is not None
        ):
            self.adaptor.ring_engine.set_null_mode(False)

        return result

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        adaptor = self.adaptor
        if (
            adaptor is None
            or scheduler_output.total_num_scheduled_tokens <= 0
            or adaptor.transport is None
            or adaptor.transport.null_offload
            or not adaptor._has_global_hooks
            # EC producers return after state update without preparing or
            # executing a model batch, so there is no DMI step to commit.
            or (has_ec_transfer() and get_ec_transfer().is_producer)
        ):
            return super().execute_model(scheduler_output)
        if adaptor._step_state.phase is not VLLMStepPhase.IDLE:
            raise RuntimeError("DMI vLLM execute_model calls overlap")

        adaptor._step_state = _VLLMStepState(
            phase=VLLMStepPhase.ARMED,
            scheduler_output=scheduler_output,
        )
        state = adaptor._step_state
        try:
            result = super().execute_model(scheduler_output)
            if state.phase is not VLLMStepPhase.COMMITTED:
                raise RuntimeError(
                    "DMI vLLM forward completed without metadata commit"
                )
            return result
        finally:
            adaptor._step_state = _VLLMStepState()

    def stop_monitoring(self) -> None:
        """Flush and stop DMI, surfacing incomplete-shutdown failures.

        Cleanup remains exhaustive and reentrant: every component gets a stop
        attempt even if an earlier one fails, then the explicit caller receives
        one aggregated error. ``shutdown()`` keeps its best-effort semantics.
        """
        errors: list[tuple[str, Exception]] = []
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                errors.append(("torch.cuda.synchronize", exc))

        if self.adaptor is not None:
            engine = self.adaptor.engine
            ring_engine = self.adaptor.ring_engine
            if ring_engine is not None:
                try:
                    ring_engine.stop()
                except Exception as exc:
                    errors.append(("ring_engine.stop", exc))
            try:
                ring_transport.deactivate()
            except Exception as exc:
                errors.append(("ring_transport.deactivate", exc))
            try:
                engine.close()
            except Exception as exc:
                errors.append(("monitoring_engine.close", exc))
            self.adaptor = None

        if self._dmx_host_engine is not None:
            try:
                self._dmx_host_engine.stop()
            except Exception as exc:
                errors.append(("host_engine.stop", exc))
            self._dmx_host_engine = None

        if errors:
            summary = "; ".join(
                f"{operation}: {error}" for operation, error in errors
            )
            raise RuntimeError(
                f"DMI monitoring shutdown was incomplete: {summary}"
            ) from errors[0][1]

    def shutdown(self) -> None:
        import logging
        if self.adaptor is not None:
            logging.getLogger(__name__).warning(
                "DMI engine not explicitly stopped before shutdown. "
                "Data may be incomplete. Call stop_monitoring() first."
            )
        # Best-effort flush (may be killed by vLLM's 8 s deadline). Explicit
        # collective_rpc("stop_monitoring") calls still receive failures.
        try:
            self.stop_monitoring()
        except Exception:
            logging.getLogger(__name__).exception(
                "DMI monitoring shutdown was incomplete during worker teardown"
            )
        super().shutdown()
        vllm_config = getattr(self, "vllm_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        world_size = int(getattr(parallel_config, "world_size", 1))
        if world_size > 1 and int(getattr(self, "rank", -1)) == 0:
            time.sleep(_STORE_OWNER_SHUTDOWN_GRACE_S)


def _attach_dmi_internal(outputs, model_id, reader=None):
    """Tag each vLLM ``RequestOutput`` with a lazy ``.dmi_internal`` keyed by
    its (normalized) request_id, so the captured internals read back per
    request -- the same object HF's ``out.dmi_internal`` gives you."""
    for r in outputs:
        rid = normalize_vllm_request_id(r.request_id)
        prompt_token_ids = getattr(r, "prompt_token_ids", None) or ()
        candidates = getattr(r, "outputs", None) or ()
        generated_token_ids = (
            getattr(candidates[0], "token_ids", None) or ()
            if candidates
            else ()
        )
        # The last sampled token is returned but is never fed through another
        # model forward, so no activation exists for it. Describe coverage as
        # one logical interval; the reader coalesces vLLM's physical prefill /
        # decode segments before validating completeness.
        forwarded_tokens = max(
            0, len(prompt_token_ids) + len(generated_token_ids) - 1
        )
        token_ranges = (
            {rid: ((0, forwarded_tokens),)} if forwarded_tokens else None
        )
        r.dmi_internal = make_lazy_internal(
            model_id,
            reader=reader,
            request_ids=[rid],
            token_ranges=token_ranges,
        )
    return outputs


class DMILLM(LLM):
    """Drop-in replacement for vLLM's ``LLM`` that captures model internals.

    Injects the DMI worker (so ``worker_cls`` need not be passed) and tags every
    ``RequestOutput`` from ``generate`` with a lazy ``.dmi_internal``. Pass DMI
    settings through ``additional_config`` exactly as with plain ``LLM``
    (``dmx_model_id``, ``dmx_db_host``, ``dmx_db_port``, ``dmx_hook_selection``, ...).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("worker_cls", "integration.vllm_adapter.DMXGPUWorker")
        ac = kwargs.get("additional_config") or {}
        self._dmx_model_id = ac.get("dmx_model_id") or str(
            kwargs.get("model") or (args[0] if args else ""))
        self._dmx_db_host = ac.get("dmx_db_host", "localhost")
        self._dmx_db_port = int(ac.get("dmx_db_port", 9000))
        super().__init__(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        outputs = super().generate(*args, **kwargs)
        reader = CHClickhouseDriverReadOnly(host=self._dmx_db_host, port=self._dmx_db_port)
        return _attach_dmi_internal(outputs, self._dmx_model_id, reader)


__all__ = [
    "VLLMAdaptor",
    "DMXGPUWorker",
    "DMILLM",
    "normalize_vllm_request_id",
]
