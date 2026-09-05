"""Payload size estimation for a configuration, before anything runs.

The configurator's job is to let someone choose observations without
discovering afterwards that the choice was unaffordable. This module answers
"how many bytes will that cost?" using the *same* arithmetic the runtime uses
to reserve ring space, so the number the UI shows and the number
``prepare_step`` enforces cannot drift:

    compute_hook_shape(...) -> elem_size * prod(shape) -> align_up(_, 16)

That is ``BackendAdapter.plan_step`` (``src/dmi/adapters/base.py``) reduced to
one spec. This module walks a synthesized spec set instead of a live one,
because the configurator runs where no model is loaded.

Three properties of DMI's transport shape the output, and each is a thing a
naive per-model estimate gets wrong:

**The binding constraint is one step, not one second.** ``prepare_step``
compares a single step's total against ``min(payload_cap, staging_cap)``. Over
that, capture does not fail -- it silently falls back to eager CPU-direct
dispatch, which is a large performance cliff. Prefill sets that peak, because
``q_len`` is at its largest.

**Load is not spread evenly across ranks.** ``compute_hook_shape`` divides
tensor-parallel-sharded hooks by ``tp_size``, and ``filter_by_tp_rank`` keeps
*unsharded* hooks only on rank 0. Aggregate bytes are therefore roughly
TP-invariant while per-rank pressure is not. Under pipeline parallelism the
``PP_FIRST``/``PP_LAST`` hooks land on specific stages, and ``final_logits``
carries a whole vocabulary. Ring capacity has to be judged against the worst
rank, so that is what this module reports.

**Estimates are estimates.** Every result carries its assumptions and any
warnings, and nothing here claims to predict serving overhead -- that needs
measurement on the real model, GPU, and batch distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..hooks.catalog import HOOK_DEFS
from .schema import DMIConfig, LayerSelection, ModelDescriptor, ModelTopology

# Mirrors ring::PAYLOAD_ALIGN. Every reservation is rounded up to this, so an
# estimate that ignored it would under-count small tensors.
PAYLOAD_ALIGN = 16

SECONDS_PER_DAY = 86_400

# Hooks whose payload is attention weights: shaped [heads, q_len, kv_dim], so
# they scale with the square of sequence length and are only meaningful in the
# batched (non-packed) convention.
_ATTN_WEIGHT_HOOKS = frozenset({"attn_scores", "pattern"})

# Catalog projections, by short name.
_PER_LAYER: dict[str, bool] = {}
_HOOK_ID: dict[str, int] = {}
_PP_STAGE: dict[str, int] = {}
_TP_SHARDED: dict[str, bool] = {}
for _id, _act, _short, _pl, _grp, _tp, _sc, _pp in HOOK_DEFS:
    _PER_LAYER[_short] = _pl
    _HOOK_ID[_short] = _id
    _PP_STAGE[_short] = _pp
    _TP_SHARDED[_short] = _tp

_PP_ANY, _PP_FIRST, _PP_LAST = 0, 1, 2


@dataclass(frozen=True)
class Workload:
    """The traffic an estimate is computed against.

    None of this belongs in a ``DMIConfig``: it describes the *serving* the
    capture rides along with, not the capture itself. Two identical
    configurations cost different amounts under different traffic.

    ``packed`` selects the tensor convention. vLLM concatenates every active
    request's rows into dim 0 with no batch dimension (packed); Hugging Face
    ``generate()`` keeps a leading batch dimension (batched). The distinction
    changes both the shapes and which hooks are meaningful.
    """

    batch_size: int = 8
    prompt_tokens: int = 2048
    decode_tokens: int = 256
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    packed: bool = True
    dtype: str = "float16"
    decode_steps_per_second: float = 0.0
    # Physical KV-cache width when the backend preallocates one (HF
    # StaticCache: max_cache_len). The runtime sizes hooks off the cache it
    # is handed, so a StaticCache prefill attention payload is
    # [H, prompt, cache_max_len], NOT [H, prompt, prompt] -- with prompt 128
    # and cache 2176 that is 17x this figure without the override. None
    # means a dynamically grown cache (kv = current context length), which
    # is the smaller shape and is called out in the warnings.
    cache_max_len: Optional[int] = None

    def __post_init__(self) -> None:
        # Exact types first, mirroring ModelTopology: a JSON `8192.0` passes
        # every magnitude comparison here and then explodes with TypeError
        # inside the byte arithmetic (float & int), and `True` reads as 1.
        for name in (
            "batch_size",
            "prompt_tokens",
            "decode_tokens",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{name} must be an integer, got "
                    f"{type(value).__name__} ({value!r})."
                )
        if self.dtype is not None and not isinstance(self.dtype, str):
            raise ValueError(f"dtype must be a string, got {type(self.dtype).__name__}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.prompt_tokens < 1:
            raise ValueError(
                f"prompt_tokens must be >= 1, got {self.prompt_tokens}."
            )
        if self.decode_tokens < 0:
            raise ValueError(
                f"decode_tokens must be >= 0, got {self.decode_tokens}."
            )
        if self.tensor_parallel_size < 1:
            raise ValueError(
                "tensor_parallel_size must be >= 1, got "
                f"{self.tensor_parallel_size}."
            )
        if self.pipeline_parallel_size < 1:
            raise ValueError(
                "pipeline_parallel_size must be >= 1, got "
                f"{self.pipeline_parallel_size}."
            )
        import math

        if self.decode_steps_per_second < 0 or not math.isfinite(
            self.decode_steps_per_second
        ):
            raise ValueError(
                "decode_steps_per_second must be a finite number >= 0, got "
                f"{self.decode_steps_per_second!r}."
            )
        if self.cache_max_len is not None and self.cache_max_len < 1:
            raise ValueError(
                f"cache_max_len must be >= 1, got {self.cache_max_len}."
            )


@dataclass(frozen=True)
class RankLoad:
    """What one (pipeline stage, tensor-parallel rank) pair carries."""

    label: str
    pp_stage: int
    tp_rank: int
    prefill_step_bytes: int
    decode_step_bytes: int
    prefill_hooks: int
    decode_hooks: int


@dataclass(frozen=True)
class Estimate:
    """Byte estimates for one configuration under one workload."""

    peak_step_bytes: int
    peak_step_rank: str
    decode_step_bytes: int
    bytes_per_request: int
    # Sum of every rank's peak step (prefill or decode, whichever is enabled
    # and larger) -- the cluster-wide burst, not a prefill-only figure.
    aggregate_peak_step_bytes: int
    sustained_bytes_per_second: Optional[float]
    bytes_per_day: Optional[float]
    ranks: tuple[RankLoad, ...] = ()
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RingFit:
    """How an estimate sits against a ring configuration."""

    effective_bytes: int
    peak_step_bytes: int
    fits: bool
    occupancy_percent: float
    detail: str


def _resolve_dtype(dtype_name: str):
    """Look up a torch dtype by name, raising ``ValueError`` on a bad name.

    Torch itself raises ``AttributeError`` for an unknown attribute, which
    would surface as an internal error rather than the configuration error it
    actually is. Imported lazily to keep torch off the descriptor and
    validation paths.
    """
    import torch

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown dtype: {dtype_name!r}")
    return dtype


def _element_size(dtype_name: str) -> int:
    import torch

    return torch._utils._element_size(_resolve_dtype(dtype_name))


def _stage_layers(num_layers: int, pp_size: int, stage: int) -> range:
    """Contiguous split of layers across pipeline stages.

    Matches vLLM's ``get_pp_indices``: the remainder layers go to the FIRST
    stages (8 layers / 3 stages -> [3, 3, 2], not [2, 3, 3]). Per-rank peaks
    move with the split, so this has to agree with the serving backend.
    """
    base = num_layers // pp_size
    extra = num_layers % pp_size
    start = stage * base + min(stage, extra)
    end = start + base + (1 if stage < extra else 0)
    return range(start, end)


def _hook_on_stage(short: str, stage: int, pp_size: int) -> bool:
    placement = _PP_STAGE.get(short, _PP_ANY)
    if placement == _PP_FIRST:
        return stage == 0
    if placement == _PP_LAST:
        return stage == pp_size - 1
    return True


def _hook_on_tp_rank(short: str, tp_rank: int) -> bool:
    """Mirror ``filter_by_tp_rank``: rank 0 keeps all, others keep sharded."""
    return tp_rank == 0 or _TP_SHARDED.get(short, False)


def _selected_layers(
    layers: Optional[LayerSelection], topology: ModelTopology
) -> set[int]:
    if layers is None:
        return set(range(topology.num_layers))
    return {
        layer
        for layer in range(topology.num_layers)
        if layers.contains(layer)
    }


def _step_bytes(
    hooks: list[str],
    layer_set: set[int],
    topology: ModelTopology,
    workload: Workload,
    *,
    stage: int,
    tp_rank: int,
    q_len: int,
    batch: int,
    kv_dim: int,
    logits_to_keep: int,
) -> tuple[int, int]:
    """Total aligned bytes and firing-hook count for one step on one rank.

    Deliberately identical in structure to ``BackendAdapter.plan_step``: a
    hook whose ``compute_hook_shape`` comes back empty does not fire and does
    not count, which is how unavailable observations (``mlp_post`` with no
    ``intermediate_size``, MoE hooks on a dense model) drop out for free.
    """
    from ..hooks.specs import align_up_py, compute_hook_shape

    from .manifest import to_model_shape_config

    cfg = to_model_shape_config(
        topology,
        dtype=_resolve_dtype(workload.dtype),
        tp_size=workload.tensor_parallel_size,
        tp_rank=tp_rank,
    )
    model_elem = _element_size(workload.dtype)
    # The runtime reserves per the dtype each framework actually writes: HF
    # input_ids are int64; the pinned vLLM integration carries int32 token
    # and top-k ids and float32 top-k weights. Everything else rides the
    # model dtype in both conventions.
    if workload.packed:
        _dtype_overrides = {
            "token_ids": "int32",
            "topk_ids": "int32",
            "topk_weights": "float32",
        }
    else:
        _dtype_overrides = {"token_ids": "int64"}

    stage_layers = set(_stage_layers(
        topology.num_layers, workload.pipeline_parallel_size, stage
    ))

    total = 0
    count = 0
    for short in hooks:
        hook_type = _HOOK_ID.get(short)
        if hook_type is None:
            continue
        if not _hook_on_stage(short, stage, workload.pipeline_parallel_size):
            continue
        if not _hook_on_tp_rank(short, tp_rank):
            continue

        if _PER_LAYER.get(short, False):
            instances = len(layer_set & stage_layers)
        else:
            instances = 1
        if instances == 0:
            continue

        shape = compute_hook_shape(
            hook_type, cfg, batch, q_len, kv_dim,
            logits_to_keep=logits_to_keep,
        )
        if not shape:
            continue

        elem = _element_size(_dtype_overrides.get(short, workload.dtype))
        nbytes = elem
        for dim in shape:
            nbytes *= dim
        total += align_up_py(nbytes, PAYLOAD_ALIGN) * instances
        count += instances

    return total, count


def estimate_config(
    config: DMIConfig,
    descriptor: ModelDescriptor,
    workload: Optional[Workload] = None,
) -> Estimate:
    """Estimate the payload cost of ``config`` on ``descriptor``'s model.

    Reports the worst rank, because that is the one whose ring fills first.
    """
    workload = workload or Workload()
    topology = descriptor.topology
    hooks = list(dict.fromkeys(config.observations.hooks))
    layer_set = _selected_layers(config.observations.layers, topology)

    assumptions: list[str] = []
    warnings: list[str] = []

    # A range outside the model resolves to nothing, and every per-layer hook
    # then contributes zero bytes. Zero reads as "free" rather than "selects
    # nothing", so say which it is. Validation reports the range as an error
    # separately; this is what the figures themselves have to explain.
    requested = config.observations.layers
    if requested is not None:
        if not layer_set:
            warnings.append(
                f"Layer range {requested.start}-{requested.end} selects no "
                f"layer of this {topology.num_layers}-layer model, so "
                "per-layer observations contribute nothing to these figures."
            )
        elif len(layer_set) < requested.count:
            warnings.append(
                f"Layer range {requested.start}-{requested.end} was clipped "
                f"to the model: {len(layer_set)} of {requested.count} "
                f"requested layers exist (0-{topology.num_layers - 1})."
            )

    convention = "packed" if workload.packed else "batched"
    assumptions.append(
        f"{convention} tensor convention "
        f"({'vLLM' if workload.packed else 'Hugging Face generate()'})"
    )

    if workload.packed:
        # No batch dimension; every active request's rows share dim 0.
        prefill_batch, decode_batch = 0, 0
        prefill_q = workload.batch_size * workload.prompt_tokens
        decode_q = workload.batch_size
        # One logit row per request per step.
        prefill_logits = workload.batch_size
        decode_logits = workload.batch_size
        if config.observations.layers is not None:
            # The pinned vLLM integration's attach_model has no `layers`
            # keyword, so attach_config refuses a ranged configuration on
            # that backend at launch rather than dropping the range. The
            # estimate itself is still meaningful (it is what the range
            # *would* cost), so this is a warning, said here because the
            # estimator is where the UI learns which backend is targeted.
            warnings.append(
                "The pinned vLLM integration does not accept a layer range "
                "yet: attach_config will refuse this configuration on "
                "Packed (vLLM) until third_party/vllm-integration is "
                "updated (https://github.com/ProjectDMX/"
                "DMI-vLLM-Integration/issues/20). Clear the layer range or "
                "target Batched (Hugging Face) to run it today."
            )
    else:
        prefill_batch = decode_batch = workload.batch_size
        prefill_q = workload.prompt_tokens
        decode_q = 1
        # HF generate() materializes ONLY the last token's logits per request
        # (logits_to_keep=1) for every model that supports it, so that is the
        # shape the runtime actually reserves; counting every prefill row
        # over-counted the prefill peak by vocab x (prompt-1) bytes -- the
        # largest single tensor on a large-vocab model, ~1000x the reality.
        prefill_logits = 1
        decode_logits = 1

    # Attention weights span the whole KV window. Decode peaks at the end of
    # generation, which is the step that has to fit. A preallocated cache
    # (StaticCache) is physically wide from the first prefill step, so its
    # prefill attention shapes use the full cache length.
    prefill_kv = (
        workload.cache_max_len
        if workload.cache_max_len is not None
        else workload.prompt_tokens
    )
    decode_kv = (
        workload.cache_max_len
        if workload.cache_max_len is not None
        else workload.prompt_tokens + workload.decode_tokens
    )
    if workload.cache_max_len is None and set(hooks) & _ATTN_WEIGHT_HOOKS:
        warnings.append(
            "Prefill attention shapes assume a dynamically grown cache. With "
            "a preallocated cache (HF StaticCache) the KV window is "
            "max_cache_len from the first step: set workload.cache_max_len "
            f"to about {workload.prompt_tokens + workload.decode_tokens} "
            "(prompt + decode) for that case, or these prefill figures are "
            "low."
        )

    selected_attn_weights = sorted(set(hooks) & _ATTN_WEIGHT_HOOKS)
    if selected_attn_weights and workload.packed:
        warnings.append(
            f"{', '.join(selected_attn_weights)} "
            "cannot be shaped in the packed convention -- attention weights "
            "are per-request. Serving backends exclude them (the vllm-full "
            "preset is full minus attn_scores/pattern); this estimate treats "
            "them as not firing."
        )
        hooks = [h for h in hooks if h not in _ATTN_WEIGHT_HOOKS]

    pp_size = workload.pipeline_parallel_size
    tp_size = workload.tensor_parallel_size

    # tp ranks are symmetric apart from rank 0, which additionally carries
    # every unsharded hook. Probing rank 0 and one non-zero rank is enough.
    tp_probes = [0] if tp_size == 1 else [0, 1]

    ranks: list[RankLoad] = []
    for stage in range(pp_size):
        for tp_rank in tp_probes:
            prefill_bytes, prefill_hooks = _step_bytes(
                hooks, layer_set, topology, workload,
                stage=stage, tp_rank=tp_rank,
                q_len=prefill_q, batch=prefill_batch,
                kv_dim=prefill_kv, logits_to_keep=prefill_logits,
            )
            decode_bytes, decode_hooks = _step_bytes(
                hooks, layer_set, topology, workload,
                stage=stage, tp_rank=tp_rank,
                q_len=decode_q, batch=decode_batch,
                kv_dim=decode_kv, logits_to_keep=decode_logits,
            )
            ranks.append(RankLoad(
                label=f"pp{stage}/tp{tp_rank}",
                pp_stage=stage,
                tp_rank=tp_rank,
                prefill_step_bytes=prefill_bytes,
                decode_step_bytes=decode_bytes,
                prefill_hooks=prefill_hooks,
                decode_hooks=decode_hooks,
            ))

    schedule = config.schedule
    capture_prefill = schedule.capture_prefill
    capture_decode = schedule.capture_decode

    def _peak(load: RankLoad) -> int:
        candidates = []
        if capture_prefill:
            candidates.append(load.prefill_step_bytes)
        if capture_decode:
            candidates.append(load.decode_step_bytes)
        return max(candidates) if candidates else 0

    worst = max(ranks, key=_peak) if ranks else None
    peak_step_bytes = _peak(worst) if worst else 0
    # The worst rank's decode step, not the maximum across ranks. Under
    # pipeline parallelism the stage with the largest prefill step need not be
    # the one with the largest decode step -- a first stage carrying many
    # layers versus a last stage carrying final_logits -- so taking a max here
    # would leave decode_step_bytes describing a different rank than
    # peak_step_rank names and bytes_per_request is computed from.
    decode_step_bytes = (
        worst.decode_step_bytes if (worst and capture_decode) else 0
    )

    # Aggregate across every real rank: probes stand in for the tp ranks they
    # represent, so a non-zero probe counts (tp_size - 1) times.
    aggregate = 0
    for load in ranks:
        multiplier = 1 if load.tp_rank == 0 else tp_size - 1
        aggregate += _peak(load) * multiplier
    # VOLUME figures are sums over every real rank: every PP stage and every
    # TP shard emits its own records, and the storage they land in sees the
    # total. The peak-pressure rank decides ring capacity only -- TP/PP
    # splitting divides per-rank pressure, not what is captured overall.
    # Probes stand in for the tp ranks they represent, so a non-zero probe
    # counts (tp_size - 1) times, exactly as the aggregate does.
    captured_decode_steps = workload.decode_tokens if capture_decode else 0

    def _rank_volume(load: RankLoad) -> int:
        volume = 0
        if capture_prefill:
            volume += load.prefill_step_bytes
        if capture_decode:
            volume += load.decode_step_bytes * captured_decode_steps
        return volume

    total_volume = 0
    for load in ranks:
        multiplier = 1 if load.tp_rank == 0 else tp_size - 1
        total_volume += _rank_volume(load) * multiplier

    # Step/request sampling IS enforced (the adapter driver gates on
    # should_capture_step / should_capture_request), so the captured volume
    # divides by both strides. The enforcement is conditional on what the
    # adapter reports, and the figures say so when it matters: an adapter
    # that reports no phase keeps both phase flags off the gate, and one
    # whose scheduler ids have no numeric prefix passes every request.
    sampling_divisor = max(1, schedule.step_stride) * max(1, schedule.request_stride)
    if sampling_divisor > 1 or not (capture_prefill and capture_decode):
        assumptions.append(
            "assumes the serving adapter reports a phase and numeric request "
            "ids; an adapter that reports neither applies only the stride "
            "and warmup parts of this schedule"
        )

    per_request = total_volume // sampling_divisor
    # A step covers the whole batch in either convention: packed rows share
    # dim 0, batched shapes carry the leading batch dimension.
    if workload.batch_size > 1:
        per_request = per_request // workload.batch_size
        assumptions.append(
            "per-request figures divide the whole-batch step totals by "
            f"batch_size={workload.batch_size}"
        )

    sustained: Optional[float] = None
    per_day: Optional[float] = None
    if workload.decode_steps_per_second > 0 and capture_decode:
        # Sustained decode volume across all ranks, with both strides
        # applied: the driver's request gate drops whole requests too, so
        # request_stride thins the arriving stream exactly like step_stride.
        decode_volume = sum(
            load.decode_step_bytes * (1 if load.tp_rank == 0 else tp_size - 1)
            for load in ranks
        )
        effective_rate = (
            workload.decode_steps_per_second
            / (max(1, schedule.step_stride) * max(1, schedule.request_stride))
        )
        sustained = decode_volume * effective_rate
        per_day = sustained * SECONDS_PER_DAY
        assumptions.append(
            "sustained rate covers decode only; prefill arrives in bursts "
            "that the peak-step figure bounds"
        )
    elif workload.decode_steps_per_second > 0:
        assumptions.append(
            "sustained rate is zero because decode capture is disabled"
        )
    else:
        warnings.append(
            "No sustained rate: set decode_steps_per_second to estimate "
            "bandwidth and per-day volume."
        )

    if not capture_prefill and not capture_decode:
        warnings.append(
            "Both prefill and decode capture are disabled -- this "
            "configuration captures nothing."
        )
    if not hooks:
        warnings.append("No observations selected.")

    if tp_size > 1:
        assumptions.append(
            f"tensor_parallel_size={tp_size}: sharded hooks are divided "
            "across ranks, unsharded hooks are captured on rank 0 only, so "
            "rank 0 is the busiest"
        )
    if pp_size > 1:
        assumptions.append(
            f"pipeline_parallel_size={pp_size}: layers split evenly and "
            "contiguously across stages"
        )

    return Estimate(
        peak_step_bytes=peak_step_bytes,
        peak_step_rank=worst.label if worst else "",
        decode_step_bytes=decode_step_bytes,
        bytes_per_request=per_request,
        aggregate_peak_step_bytes=aggregate,
        sustained_bytes_per_second=sustained,
        bytes_per_day=per_day,
        ranks=tuple(ranks),
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
    )


def check_ring_fit(
    estimate: Estimate,
    payload_bytes: int,
    pinned_bytes: int = 0,
    task_entries: Optional[int] = None,
) -> RingFit:
    """Judge a peak step against ring capacity.

    ``prepare_step`` compares a step against ``min(payload_cap, staging_cap)``
    AND against the task ring's entry count, so the verdict here mirrors
    both: pass the ring's ``task_ring_entries`` as ``task_entries`` and a
    step whose firing-hook count exceeds it reports OVERSIZED in the runtime
    no matter how many bytes were free.
    """
    if payload_bytes < 1:
        raise ValueError(f"payload_bytes must be >= 1, got {payload_bytes}.")
    effective = min(payload_bytes, pinned_bytes) if pinned_bytes else payload_bytes
    peak = estimate.peak_step_bytes
    fits = peak <= effective
    occupancy = (peak / effective * 100.0) if effective else 0.0

    # prepare_step refuses on the TASK ring too: more firing hooks in one
    # step than task_ring_entries returns OVERSIZED regardless of bytes.
    # The hook counts are already on every rank; the cap is the caller's
    # ring configuration, so it arrives as an argument.
    over_task_cap = 0
    if task_entries is not None and task_entries >= 1:
        for load in estimate.ranks:
            worst_hooks = max(load.prefill_hooks, load.decode_hooks)
            if worst_hooks > over_task_cap:
                over_task_cap = worst_hooks
        if over_task_cap > task_entries:
            fits = False

    if fits:
        detail = (
            f"Peak step uses {occupancy:.1f}% of the {_mib(effective)} "
            "effective ring."
        )
    elif over_task_cap:
        detail = (
            f"The busiest rank fires {over_task_cap} hooks in one step, "
            f"above the {task_entries} task entries the ring was configured "
            "with. prepare_step returns STEP_OVERSIZED regardless of bytes "
            "and the adapter falls back to eager CPU-direct dispatch -- "
            "capture keeps working, but the serving path pays for it. "
            "Raise ring task entries, narrow the layer range, or deselect "
            "observations."
        )
    else:
        detail = (
            f"Peak step ({_mib(peak)}) exceeds the {_mib(effective)} "
            "effective ring. prepare_step will return STEP_OVERSIZED and the "
            "adapter falls back to eager CPU-direct dispatch -- capture keeps "
            "working, but the serving path pays for it."
        )
    if pinned_bytes and pinned_bytes < payload_bytes:
        detail += (
            f" Pinned staging ({_mib(pinned_bytes)}) is the binding limit, "
            f"not the payload ring ({_mib(payload_bytes)})."
        )

    return RingFit(
        effective_bytes=effective,
        peak_step_bytes=peak,
        fits=fits,
        occupancy_percent=occupancy,
        detail=detail,
    )


def _mib(nbytes: int) -> str:
    return f"{nbytes / (1024 * 1024):.1f} MiB"


def estimate_payload(estimate: Estimate) -> dict:
    """Render an estimate as JSON-safe data for the configurator API."""
    return {
        "peak_step_bytes": estimate.peak_step_bytes,
        "peak_step_rank": estimate.peak_step_rank,
        "decode_step_bytes": estimate.decode_step_bytes,
        "bytes_per_request": estimate.bytes_per_request,
        "aggregate_peak_step_bytes": estimate.aggregate_peak_step_bytes,
        "sustained_bytes_per_second": estimate.sustained_bytes_per_second,
        "bytes_per_day": estimate.bytes_per_day,
        "ranks": [
            {
                "label": load.label,
                "pp_stage": load.pp_stage,
                "tp_rank": load.tp_rank,
                "prefill_step_bytes": load.prefill_step_bytes,
                "decode_step_bytes": load.decode_step_bytes,
                "prefill_hooks": load.prefill_hooks,
                "decode_hooks": load.decode_hooks,
            }
            for load in estimate.ranks
        ],
        "assumptions": list(estimate.assumptions),
        "warnings": list(estimate.warnings),
    }


__all__ = [
    "PAYLOAD_ALIGN",
    "Workload",
    "RankLoad",
    "Estimate",
    "RingFit",
    "estimate_config",
    "check_ring_fit",
    "estimate_payload",
]
