"""Focused CPU tests for the vLLM real-layout/eager-preflight fix."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import vllm
try:
    import vllm._C as _vllm_native_extension
except ModuleNotFoundError:
    import vllm._C_stable_libtorch as _vllm_native_extension
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from benchmark import bench_vllm_transport
from integration.vllm_adapter import (
    DMXGPUWorker,
    VLLMAdaptor,
    VLLMStepPhase,
    VLLMValidationMode,
    _VLLMHookSelection,
    _VLLMRealLayout,
    _VLLMRoleFormula,
    _VLLMStepState,
)
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
    HOOK_TYPE_POS_EMBED,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    SHAPE_LOGITS,
    HookRowBasis,
    HookSpec,
    ModelShapeConfig,
    PP_LAST_ONLY,
    PP_FIRST_ONLY,
    _HOOK_DEFS,
    hook_row_basis,
)
from monitoring.selection import (
    filter_by_pp_rank,
    hook_belongs_to_pp_rank,
    hook_belongs_to_tp_rank,
    select_hook_specs,
)
from tests.compare_worker import CompareWorker
from tests.ref_disk_worker import RefDiskWorker


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_VLLM_ROOT = (REPO_ROOT / "integration" / "vllm").resolve()


def test_runtime_uses_the_pinned_source_and_extension():
    source = Path(vllm.__file__).resolve()
    extension = Path(_vllm_native_extension.__file__).resolve()

    assert source.is_relative_to(PINNED_VLLM_ROOT)
    assert extension.is_relative_to(PINNED_VLLM_ROOT)


def test_worker_load_model_forwards_v027_keyword(monkeypatch):
    calls = []

    def fake_load_model(self, *, load_dummy_weights=False):
        calls.append(load_dummy_weights)

    monkeypatch.setattr(Worker, "load_model", fake_load_model)
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"]
            )
        )
    )
    worker.adaptor = None

    worker.load_model(load_dummy_weights=True)

    assert calls == [True]
    assert worker.vllm_config.model_config.hf_config.architectures == [
        "LlamaPForCausalLM"
    ]


def test_v2_model_runner_fails_before_device_initialization(monkeypatch):
    initialized = []
    monkeypatch.setattr(
        Worker,
        "init_device",
        lambda _self: initialized.append(True),
    )
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.use_v2_model_runner = True

    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        worker.init_device()

    assert initialized == []


def _scheduler(order=("A", "B"), counts=(2, 3), total=None):
    mapping = dict(zip(order, counts))
    return SimpleNamespace(
        num_scheduled_tokens=mapping,
        total_num_scheduled_tokens=(
            sum(counts) if total is None else total
        ),
    )


def _runner(order=("B", "A"), computed=(7, 11)):
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=len(order),
            req_ids=list(order),
            num_computed_tokens_cpu=list(computed),
            lora_id_to_lora_request={},
        ),
        input_ids=None,
    )


def _armed_adaptor(scheduler_output):
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )
    return adaptor


def test_records_real_packed_order_not_scheduler_order():
    scheduler_output = _scheduler()
    adaptor = _armed_adaptor(scheduler_output)

    adaptor._record_real_layout(
        scheduler_output,
        _runner(),
        [3, 2],
    )

    layout = adaptor._step_state.layout
    assert layout is not None
    assert layout.raw_req_ids == ("B", "A")
    assert layout.req_ids == ("B", "A")
    assert layout.scheduled_counts == (3, 2)
    assert layout.computed_counts == (7, 11)
    assert layout.token_ranges == ((7, 10), (11, 13))
    assert layout.dim0_offsets == (0, 3)
    assert layout.total_rows == 5
    assert adaptor._step_state.phase is VLLMStepPhase.LAYOUT_READY


@pytest.mark.parametrize(
    ("scheduler_output", "runner", "ordered_counts", "match"),
    [
        (
            _scheduler(),
            _runner(order=("A", "A")),
            [2, 3],
            "not unique",
        ),
        (
            _scheduler(),
            _runner(order=("A", 7)),
            [2, 3],
            "must be strings",
        ),
        (
            _scheduler(
                order=("A-12345678", "A-abcdef12"),
                counts=(2, 3),
            ),
            _runner(order=("A-12345678", "A-abcdef12")),
            [2, 3],
            "normalized request IDs",
        ),
        (
            _scheduler(),
            _runner(order=("B", "C")),
            [3, 2],
            "scheduler membership",
        ),
        (
            _scheduler(),
            _runner(),
            [2, 3],
            "count mismatch",
        ),
        (
            _scheduler(counts=(2, -1), total=1),
            _runner(order=("A", "B")),
            [2, -1],
            "negative",
        ),
        (
            _scheduler(),
            _runner(computed=(7, -1)),
            [3, 2],
            "computed token count is negative",
        ),
        (
            _scheduler(),
            _runner(),
            [3],
            "lengths disagree",
        ),
        (
            _scheduler(total=99),
            _runner(),
            [3, 2],
            "scheduler total",
        ),
    ],
)
def test_real_layout_validation_failures(
    scheduler_output, runner, ordered_counts, match
):
    adaptor = _armed_adaptor(scheduler_output)
    with pytest.raises(RuntimeError, match=match):
        adaptor._record_real_layout(
            scheduler_output,
            runner,
            ordered_counts,
        )


def test_duplicate_prepare_observation_fails():
    scheduler_output = _scheduler()
    adaptor = _armed_adaptor(scheduler_output)
    adaptor._record_real_layout(scheduler_output, _runner(), [3, 2])

    with pytest.raises(RuntimeError, match="duplicate or unarmed"):
        adaptor._record_real_layout(scheduler_output, _runner(), [3, 2])


def test_real_layout_keeps_immutable_copies():
    scheduler_output = _scheduler()
    runner = _runner()
    ordered = np.array([3, 2], dtype=np.int32)
    adaptor = _armed_adaptor(scheduler_output)

    adaptor._record_real_layout(scheduler_output, runner, ordered)
    runner.input_batch.req_ids[:] = ["changed", "changed-again"]
    runner.input_batch.num_computed_tokens_cpu[:] = [99, 100]
    ordered[:] = [0, 0]

    layout = adaptor._step_state.layout
    assert layout is not None
    assert layout.raw_req_ids == ("B", "A")
    assert layout.computed_counts == (7, 11)
    assert layout.scheduled_counts == (3, 2)


def test_ref_disk_oracle_slices_the_real_packed_order(tmp_path):
    raw_a = "cmpl-a-0-aaaaaaaa"
    raw_b = "cmpl-b-0-bbbbbbbb"
    scheduler_output = SimpleNamespace(
        # Deliberately opposite the real packed order below.
        num_scheduled_tokens={raw_a: 2, raw_b: 1},
        total_num_scheduled_tokens=3,
    )

    class _RefModel:
        @staticmethod
        def get_ref_buffers():
            return {
                "token_ids": torch.tensor([201, 101, 102], dtype=torch.int32),
                "final_logits": torch.tensor([[20, 21], [10, 11]]),
            }

    worker = RefDiskWorker.__new__(RefDiskWorker)
    worker.model_runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=[raw_b, raw_a],
            num_computed_tokens_cpu=np.array([7, 3]),
        ),
        model=_RefModel(),
    )
    worker._output_dir = str(tmp_path)
    worker._tp_rank = 0
    worker._tp_size = 1
    worker._pp_is_first = True
    worker._pp_is_last = True

    layout = worker._capture_packed_layout(
        scheduler_output,
        np.array([1, 2]),
    )
    worker._save_step(
        layout.request_ids,
        layout.scheduled_counts,
        layout.computed_counts,
    )

    assert layout.request_ids == ("cmpl-b-0", "cmpl-a-0")
    assert torch.equal(
        torch.load(tmp_path / "cmpl-b-0/token_ids_T7_8.pt", weights_only=True),
        torch.tensor([201], dtype=torch.int32),
    )
    assert torch.equal(
        torch.load(tmp_path / "cmpl-a-0/token_ids_T3_5.pt", weights_only=True),
        torch.tensor([101, 102], dtype=torch.int32),
    )
    assert torch.equal(
        torch.load(
            tmp_path / "cmpl-b-0/final_logits_T7_8.pt", weights_only=True
        ),
        torch.tensor([[20, 21]]),
    )
    assert torch.equal(
        torch.load(
            tmp_path / "cmpl-a-0/final_logits_T4_5.pt", weights_only=True
        ),
        torch.tensor([[10, 11]]),
    )


@pytest.mark.parametrize(
    ("is_first", "is_last", "expected"),
    [
        (True, False, {"token_ids", "embed", "resid_pre"}),
        (False, True, {"resid_pre", "resid_final", "final_ln", "final_logits"}),
    ],
)
def test_ref_disk_oracle_filters_global_hooks_by_pp_stage(
    tmp_path, is_first, is_last, expected
):
    buffers = {
        "token_ids": torch.tensor([1], dtype=torch.int32),
        "embed": torch.tensor([[2.0]]),
        "resid_pre_L0": torch.tensor([[3.0]]),
        "resid_final": torch.tensor([[4.0]]),
        "final_ln": torch.tensor([[5.0]]),
        "final_logits": torch.tensor([[6.0]]),
    }

    worker = RefDiskWorker.__new__(RefDiskWorker)
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(get_ref_buffers=lambda: buffers)
    )
    worker._output_dir = str(tmp_path)
    worker._tp_rank = 0
    worker._tp_size = 1
    worker._pp_is_first = is_first
    worker._pp_is_last = is_last

    worker._save_step(("request",), (1,), (0,))

    saved = {
        path.name.split("_L", 1)[0].split("_T", 1)[0]
        for path in (tmp_path / "request").glob("*.pt")
    }
    assert saved == expected


def test_compare_worker_saves_committed_real_layout(
    monkeypatch, tmp_path
):
    transport = SimpleNamespace(
        _current_req_ids=["stale"],
        _current_token_ranges=[(0, 1)],
        _current_dim0_offsets=[0],
        _current_flattened=True,
    )
    adaptor = SimpleNamespace(transport=transport, _step_counter=4)
    buffers = {
        "resid_pre_L0": torch.tensor(
            [[100], [101], [102], [200], [201]],
            dtype=torch.int32,
        ),
        "final_logits": torch.tensor(
            [[300], [400]],
            dtype=torch.int32,
        ),
    }
    model = SimpleNamespace(get_ref_buffers=lambda: buffers)

    worker = CompareWorker.__new__(CompareWorker)
    worker.adaptor = adaptor
    worker.model_runner = SimpleNamespace(model=model)
    worker._compare_output_dir = str(tmp_path)
    worker._compare_step = 0
    worker._dmx_tp_rank = 0
    worker._dmx_tp_size = 1

    def monitored_forward(self, scheduler_output):
        assert list(scheduler_output.num_scheduled_tokens) == ["A", "B"]
        self.adaptor.transport._current_req_ids = ["B", "A"]
        self.adaptor.transport._current_token_ranges = [(24, 27), (10, 12)]
        self.adaptor.transport._current_dim0_offsets = [0, 3]
        self.adaptor.transport._current_flattened = True
        self.adaptor._step_counter += 1
        return "ok"

    monkeypatch.setattr(
        DMXGPUWorker,
        "execute_model",
        monitored_forward,
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"A": 2, "B": 3},
        total_num_scheduled_tokens=5,
    )

    assert worker.execute_model(scheduler_output) == "ok"
    assert worker._compare_step == 1
    assert torch.equal(
        torch.load(
            tmp_path / "B" / "resid_pre_L0_T24_27.pt",
            weights_only=True,
        ),
        buffers["resid_pre_L0"][:3],
    )
    assert torch.equal(
        torch.load(
            tmp_path / "A" / "resid_pre_L0_T10_12.pt",
            weights_only=True,
        ),
        buffers["resid_pre_L0"][3:],
    )
    assert torch.equal(
        torch.load(
            tmp_path / "B" / "final_logits_T26_27.pt",
            weights_only=True,
        ),
        buffers["final_logits"][:1],
    )
    assert torch.equal(
        torch.load(
            tmp_path / "A" / "final_logits_T11_12.pt",
            weights_only=True,
        ),
        buffers["final_logits"][1:2],
    )


@pytest.mark.parametrize(
    ("step_increment", "flattened", "offsets", "total", "match"),
    [
        (0, True, [0, 3], 5, "exactly one committed"),
        (1, False, [0, 3], 5, "invalid committed DMI layout"),
        (1, True, [0, 2], 5, "invalid committed DMI range"),
        (1, True, [0, 3], 6, "do not match scheduler total"),
    ],
)
def test_compare_worker_rejects_invalid_committed_layout(
    monkeypatch,
    tmp_path,
    step_increment,
    flattened,
    offsets,
    total,
    match,
):
    transport = SimpleNamespace(
        _current_req_ids=["B", "A"],
        _current_token_ranges=[(24, 27), (10, 12)],
        _current_dim0_offsets=offsets,
        _current_flattened=flattened,
    )
    adaptor = SimpleNamespace(transport=transport, _step_counter=4)
    worker = CompareWorker.__new__(CompareWorker)
    worker.adaptor = adaptor
    worker._compare_output_dir = str(tmp_path)
    worker._compare_step = 0

    def monitored_forward(self, scheduler_output):
        self.adaptor._step_counter += step_increment
        return "ok"

    monkeypatch.setattr(
        DMXGPUWorker,
        "execute_model",
        monitored_forward,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=total,
    )

    with pytest.raises(RuntimeError, match=match):
        worker.execute_model(scheduler_output)
    assert worker._compare_step == 0


def test_build_context_uses_real_layout_and_real_descriptor():
    scheduler_output = _scheduler()
    adaptor = _armed_adaptor(scheduler_output)
    adaptor._record_real_layout(scheduler_output, _runner(), [3, 2])
    adaptor._step_state.real_batch_descriptor = BatchDescriptor(
        num_tokens=8
    )
    adaptor.model_id = "model"
    adaptor.gpu_padding_strip = True
    adaptor._step_counter = 0
    adaptor._debug_step = False
    adaptor.detect_parallel_ranks = lambda: (1, 0, 0, 1)

    ctx = adaptor.build_step_context(scheduler_output, _runner())

    assert ctx is not None
    assert ctx.req_ids == ["B", "A"]
    assert ctx.token_ranges == [(7, 10), (11, 13)]
    assert ctx.dim0_offsets == [0, 3]
    assert ctx.q_len == 8
    assert ctx.actual_q_len == 5
    assert ctx.logits_to_keep == 2
    assert (ctx.tp_rank, ctx.pp_rank) == (1, 1)


def test_role_formula_alignment_and_row_sources():
    formula = _VLLMRoleFormula(
        real_terms=((10, 2),),
        execution_terms=((7, 1),),
        request_terms=((13, 1),),
        hook_count=4,
    )

    assert formula.bytes_for(
        execution_rows=5,
        real_rows=3,
        request_rows=2,
    ) == 2 * 32 + 48 + 32


class _Dispatcher:
    def __init__(self, padded_tokens=8):
        self.padded_tokens = padded_tokens
        self.calls = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["valid_modes"] == {CUDAGraphMode.NONE}:
            return (
                CUDAGraphMode.NONE,
                BatchDescriptor(num_tokens=kwargs["num_tokens"]),
            )
        return (
            CUDAGraphMode.FULL,
            BatchDescriptor(
                num_tokens=self.padded_tokens,
                num_reqs=kwargs["num_tokens"],
            ),
        )


def _preflight_adaptor(
    *,
    byte_capacity=128,
    validation=VLLMValidationMode.OFF,
    enable_sp=False,
    dp_size=1,
    enable_ep=False,
):
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED
    )
    adaptor._validation_mode = validation
    adaptor._byte_capacity = byte_capacity
    adaptor._role_formulas = (
        _VLLMRoleFormula(
            execution_terms=((16, 1),),
            hook_count=1,
        ),
    )
    adaptor._max_num_batched_tokens = 32
    adaptor._max_num_seqs = 8
    adaptor._capture_sizes = (1, 2, 4, 8, 16, 32)
    adaptor._max_capture_size = 32
    adaptor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=dp_size,
            enable_expert_parallel=enable_ep,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp),
        ),
    )
    dispatcher = _Dispatcher()
    runner = SimpleNamespace(
        uniform_decode_query_len=1,
        _is_uniform_decode=lambda **kwargs: False,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        cudagraph_dispatcher=dispatcher,
    )
    return adaptor, runner, dispatcher


def _call_preflight(adaptor, runner):
    return adaptor._preflight_force_eager(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_num_scheduled_tokens=3,
        use_cascade_attn=False,
        caller_force_eager=False,
        force_uniform_decode=None,
        force_has_lora=None,
        force_num_active_loras=None,
        num_encoder_reqs=0,
    )


def test_exact_preflight_is_one_read_only_dispatch_when_validation_off():
    adaptor, runner, dispatcher = _preflight_adaptor(
        byte_capacity=127
    )
    adaptor.ring_engine = SimpleNamespace(
        prepare_step=lambda *_: pytest.fail("preflight touched ring")
    )

    assert _call_preflight(adaptor, runner) is True
    assert len(dispatcher.calls) == 1
    assert adaptor._step_state.capacity_candidate == (
        CUDAGraphMode.FULL,
        BatchDescriptor(num_tokens=8, num_reqs=5),
    )
    assert adaptor._step_state.expected_dispatch is None


def test_exact_preflight_verify_predicts_final_eager_dispatch():
    adaptor, runner, dispatcher = _preflight_adaptor(
        byte_capacity=127,
        validation=VLLMValidationMode.VERIFY,
    )

    assert _call_preflight(adaptor, runner) is True
    assert len(dispatcher.calls) == 2
    assert adaptor._step_state.expected_dispatch == (
        CUDAGraphMode.NONE,
        BatchDescriptor(num_tokens=5),
    )


def test_exact_preflight_capacity_boundary():
    fit, runner, _ = _preflight_adaptor(byte_capacity=128)
    over, runner2, _ = _preflight_adaptor(byte_capacity=127)

    assert _call_preflight(fit, runner) is False
    assert _call_preflight(over, runner2) is True


def _real_dispatch_runner(mode):
    dispatcher = CudagraphDispatcher.__new__(CudagraphDispatcher)
    dispatcher.compilation_config = SimpleNamespace(
        max_cudagraph_capture_size=8,
    )
    dispatcher.vllm_config = SimpleNamespace(
        lora_config=SimpleNamespace(max_loras=4),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    dispatcher.uniform_decode_query_len = 1
    dispatcher.keys_initialized = True
    dispatcher.specialize_lora_count = False
    dispatcher.captured_lora_counts = []
    dispatcher.cudagraph_mode = mode
    dispatcher._bs_to_padded_graph_size = [0, 1, 2, 4, 4, 8, 8, 8, 8]
    dispatcher.cudagraph_keys = {
        CUDAGraphMode.PIECEWISE: set(),
        CUDAGraphMode.FULL: set(),
    }
    if mode is not CUDAGraphMode.NONE:
        for size in (1, 2, 4, 8):
            for has_lora, lora_count in ((False, 0), (True, 5)):
                descriptor = dispatcher._create_padded_batch_descriptor(
                    size,
                    uniform_decode=False,
                    has_lora=has_lora,
                    num_active_loras=lora_count,
                )
                if mode is CUDAGraphMode.FULL:
                    dispatcher.cudagraph_keys[CUDAGraphMode.FULL].add(
                        descriptor
                    )
                else:
                    dispatcher.cudagraph_keys[
                        CUDAGraphMode.PIECEWISE
                    ].add(
                        BatchDescriptor(
                            num_tokens=descriptor.num_tokens,
                            num_reqs=None,
                            uniform=False,
                            has_lora=descriptor.has_lora,
                            num_active_loras=descriptor.num_active_loras,
                        )
                    )

    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
    )
    compilation = SimpleNamespace(
        pass_config=SimpleNamespace(enable_sp=False),
    )
    runner = SimpleNamespace(
        uniform_decode_query_len=1,
        _is_uniform_decode=GPUModelRunner._is_uniform_decode,
        _pad_for_sequence_parallelism=lambda value: value,
        model_config=SimpleNamespace(is_encoder_decoder=True),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        cudagraph_dispatcher=dispatcher,
        compilation_config=compilation,
        parallel_config=parallel,
        vllm_config=SimpleNamespace(
            parallel_config=parallel,
            observability_config=SimpleNamespace(
                cudagraph_metrics=False
            ),
        ),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor.vllm_config = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=compilation,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    return adaptor, runner


def _real_dispatch(
    runner,
    *,
    num_tokens=3,
    num_reqs=2,
    max_num_scheduled_tokens=2,
    use_cascade_attn=False,
    force_eager=False,
    force_uniform_decode=None,
    force_has_lora=None,
    force_num_active_loras=None,
    num_encoder_reqs=0,
):
    kwargs = dict(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_scheduled_tokens_np=np.array([2, 1], dtype=np.int32),
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        use_cascade_attn=use_cascade_attn,
        force_eager=force_eager,
        force_uniform_decode=force_uniform_decode,
        force_has_lora=force_has_lora,
        force_num_active_loras=force_num_active_loras,
        num_encoder_reqs=num_encoder_reqs,
    )
    return GPUModelRunner._determine_batch_execution_and_padding(
        runner, **kwargs
    )[:2]


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.NONE,
    ],
)
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"force_uniform_decode": True},
        {"use_cascade_attn": True},
        {"num_encoder_reqs": 1},
        {"force_has_lora": True, "force_num_active_loras": 1},
        {"force_eager": True},
    ],
)
def test_exact_preview_matches_pinned_determine(mode, overrides):
    adaptor, runner = _real_dispatch_runner(mode)
    args = dict(
        num_tokens=3,
        num_reqs=2,
        max_num_scheduled_tokens=2,
        use_cascade_attn=False,
        force_eager=False,
        force_uniform_decode=None,
        force_has_lora=None,
        force_num_active_loras=None,
        num_encoder_reqs=0,
    )
    args.update(overrides)

    preview = adaptor._preview_exact_dispatch(runner, **args)
    actual = _real_dispatch(runner, **args)

    assert preview == actual


def test_exact_preview_distinguishes_equal_rows_with_different_modes():
    adaptor, runner = _real_dispatch_runner(CUDAGraphMode.FULL)
    graph = adaptor._preview_exact_dispatch(
        runner,
        num_tokens=4,
        num_reqs=2,
        max_num_scheduled_tokens=2,
        use_cascade_attn=False,
        force_eager=False,
        force_uniform_decode=None,
        force_has_lora=None,
        force_num_active_loras=None,
        num_encoder_reqs=0,
    )
    eager = _real_dispatch(
        runner,
        num_tokens=4,
        num_reqs=2,
        force_eager=True,
    )

    assert graph[0] is CUDAGraphMode.FULL
    assert eager[0] is CUDAGraphMode.NONE
    assert graph[1].num_tokens == eager[1].num_tokens == 4
    assert graph != eager


def test_preflight_has_no_transport_tensor_or_collective_side_effects(
    monkeypatch,
):
    adaptor, runner = _real_dispatch_runner(CUDAGraphMode.FULL)
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED
    )
    adaptor._validation_mode = VLLMValidationMode.OFF
    adaptor._byte_capacity = 10_000
    adaptor._role_formulas = (
        _VLLMRoleFormula(
            execution_terms=((16, 1),),
            hook_count=1,
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden preflight side effect")

    adaptor.ring_engine = SimpleNamespace(
        prepare_step=forbidden,
        payload_cap=forbidden,
        staging_cap=forbidden,
        task_cap=forbidden,
        reserve=forbidden,
        flush=forbidden,
    )
    adaptor.transport = SimpleNamespace(
        set_step_context=forbidden,
        pre_push_all_metas=forbidden,
        flush=forbidden,
    )

    class ForbiddenSpecs:
        def __iter__(self):
            forbidden()

    adaptor.active_specs = ForbiddenSpecs()
    monkeypatch.setattr(
        "integration.vllm_adapter._compute_hook_shape",
        forbidden,
    )
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "as_tensor", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.distributed, "all_reduce", forbidden)
    monkeypatch.setattr(torch.distributed, "barrier", forbidden)

    assert (
        adaptor._preflight_force_eager(
            runner,
            num_tokens=3,
            num_reqs=2,
            max_num_scheduled_tokens=2,
            use_cascade_attn=False,
            caller_force_eager=False,
            force_uniform_decode=None,
            force_has_lora=None,
            force_num_active_loras=None,
            num_encoder_reqs=0,
        )
        is False
    )


def test_nonlocal_worst_role_forces_the_same_decision_on_every_rank():
    adaptor, runner, _dispatcher = _preflight_adaptor(
        byte_capacity=128
    )
    adaptor._role_formulas = (
        _VLLMRoleFormula(
            execution_terms=((8, 1),),
            hook_count=1,
        ),
        _VLLMRoleFormula(
            execution_terms=((32, 1),),
            hook_count=1,
        ),
    )

    decisions = []
    for _simulated_tp_rank in (0, 1):
        for _simulated_pp_rank in (0, 1):
            adaptor._step_state = _VLLMStepState(
                phase=VLLMStepPhase.ARMED
            )
            decisions.append(_call_preflight(adaptor, runner))

    assert decisions == [True, True, True, True]


def test_conservative_bound_uses_dp_max_and_sp_rounding():
    adaptor, _runner_obj, _dispatcher_obj = _preflight_adaptor(
        enable_sp=True,
        dp_size=2,
        enable_ep=True,
    )

    assert adaptor._conservative_execution_bound(5, 2) == (
        32,
        32,
        8,
    )


def test_conservative_bound_uses_graph_ceiling_not_mutable_size_table():
    adaptor, _runner_obj, _dispatcher_obj = _preflight_adaptor(
        enable_sp=True,
        dp_size=1,
        enable_ep=True,
    )
    adaptor._capture_sizes = (1, 4, 8)
    adaptor._max_capture_size = 32

    assert adaptor._conservative_execution_bound(5, 2) == (
        32,
        5,
        2,
    )


@pytest.mark.parametrize("local_tokens", [0, 1, 2, 3, 7, 8, 15, 31])
def test_dp_conservative_bounds_are_rank_independent(local_tokens):
    adaptor, _runner_obj, _dispatcher_obj = _preflight_adaptor(
        enable_sp=True,
        dp_size=4,
        enable_ep=True,
    )
    adaptor._max_num_batched_tokens = 31
    adaptor._max_num_seqs = 7
    adaptor._max_capture_size = 32

    assert adaptor._conservative_execution_bound(
        local_tokens, max(1, local_tokens // 2)
    ) == (32, 31, 7)


@pytest.mark.parametrize("dp_size", [1, 4])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("enable_ep", [False, True])
def test_conservative_bound_exhausts_graph_sp_and_dp_boundaries(
    dp_size,
    enable_sp,
    enable_ep,
):
    capture_cases = (
        ((), 0),
        ((1, 2, 4, 8), 8),
        ((1, 4, 7, 16), 16),
    )
    for capture_sizes, max_capture_size in capture_cases:
        dispatcher = CudagraphDispatcher.__new__(CudagraphDispatcher)
        dispatcher.compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=max_capture_size,
            cudagraph_capture_sizes=list(capture_sizes),
            compile_sizes=[],
        )
        dispatcher.cudagraph_mode = (
            CUDAGraphMode.FULL
            if capture_sizes
            else CUDAGraphMode.NONE
        )
        if capture_sizes:
            dispatcher._compute_bs_to_padded_graph_size()
        dispatcher.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=9),
        )
        dispatcher.uniform_decode_query_len = 1

        for tp_size in (1, 2, 4, 8):
            for max_tokens in (1, 7, 8, 9, 15, 16, 17, 31, 32, 33):
                adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
                adaptor._max_num_batched_tokens = max_tokens
                adaptor._max_num_seqs = 9
                adaptor._capture_sizes = capture_sizes
                adaptor._max_capture_size = max_capture_size
                adaptor.vllm_config = SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        tensor_parallel_size=tp_size,
                        data_parallel_size=dp_size,
                        enable_expert_parallel=enable_ep,
                    ),
                    compilation_config=SimpleNamespace(
                        pass_config=SimpleNamespace(
                            enable_sp=enable_sp
                        ),
                    ),
                )
                roles = (
                    _VLLMRoleFormula(
                        real_terms=((6, 2),),
                        execution_terms=((10, 1),),
                        request_terms=((14, 1),),
                        hook_count=4,
                    ),
                    _VLLMRoleFormula(
                        real_terms=((18, 1),),
                        execution_terms=((4, 3),),
                        request_terms=((8, 2),),
                        hook_count=6,
                    ),
                )
                boundary_tokens = {0, 1, max_tokens - 1, max_tokens}
                for size in capture_sizes:
                    boundary_tokens.update((size - 1, size, size + 1))
                for multiple in range(0, max_tokens + tp_size, tp_size):
                    boundary_tokens.update(
                        (multiple - 1, multiple, multiple + 1)
                    )
                boundary_tokens = {
                    value
                    for value in boundary_tokens
                    if 0 <= value <= max_tokens
                }

                dp_decisions = []
                for local_tokens in sorted(boundary_tokens):
                    local_reqs = min(9, max(0, local_tokens))
                    (
                        execution_bound,
                        real_bound,
                        request_bound,
                    ) = adaptor._conservative_execution_bound(
                        local_tokens,
                        local_reqs,
                    )
                    bounded_role_bytes = max(
                        role.bytes_for(
                            execution_bound,
                            real_bound,
                            request_bound,
                        )
                        for role in roles
                    )
                    bounded_hook_count = max(
                        role.hook_count for role in roles
                    )
                    dp_decisions.append(bounded_role_bytes > 4096)

                    actual_real_candidates = (
                        range(max_tokens + 1)
                        if dp_size > 1
                        else (local_tokens,)
                    )
                    actual_request_candidates = (
                        range(10)
                        if dp_size > 1
                        else (local_reqs,)
                    )
                    for actual_real_rows in actual_real_candidates:
                        dispatch_rows = actual_real_rows
                        if enable_sp:
                            dispatch_rows = (
                                (
                                    dispatch_rows
                                    + tp_size
                                    - 1
                                )
                                // tp_size
                            ) * tp_size
                        execution_candidates = [dispatch_rows]
                        if (
                            capture_sizes
                            and dispatch_rows <= max_capture_size
                        ):
                            execution_candidates.append(
                                dispatcher._create_padded_batch_descriptor(
                                    dispatch_rows,
                                    uniform_decode=False,
                                    has_lora=False,
                                ).num_tokens
                            )

                        for actual_execution_rows in execution_candidates:
                            assert (
                                actual_execution_rows
                                <= execution_bound
                            )
                            assert actual_real_rows <= real_bound
                            for actual_request_rows in (
                                actual_request_candidates
                            ):
                                assert (
                                    actual_request_rows
                                    <= request_bound
                                )
                                for role in roles:
                                    assert role.bytes_for(
                                        actual_execution_rows,
                                        actual_real_rows,
                                        actual_request_rows,
                                    ) <= role.bytes_for(
                                        execution_bound,
                                        real_bound,
                                        request_bound,
                                    )
                                    assert (
                                        role.hook_count
                                        <= bounded_hook_count
                                    )

                if dp_size > 1:
                    assert len(set(dp_decisions)) == 1


class _Ring:
    def __init__(self, result=0):
        self.result = result
        self.prepare_calls = []

    def prepare_step(self, total_bytes, n_hooks):
        self.prepare_calls.append((total_bytes, n_hooks))
        return self.result


class _Transport:
    def __init__(self):
        self.null_offload = False
        self.force_eager = False
        self.contexts = []
        self.metas = []

    def set_step_context(self, **kwargs):
        self.contexts.append(kwargs)

    def pre_push_all_metas(self, **kwargs):
        self.metas.append(kwargs)


def _commit_adaptor(byte_capacity=80, task_capacity=8):
    scheduler_output = _scheduler()
    layout = _VLLMRealLayout(
        raw_req_ids=("B", "A"),
        req_ids=("B", "A"),
        scheduled_counts=(3, 2),
        computed_counts=(7, 11),
        token_ranges=((7, 10), (11, 13)),
        dim0_offsets=(0, 3),
        total_rows=5,
    )
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.LAYOUT_READY,
        scheduler_output=scheduler_output,
        layout=layout,
    )
    adaptor.model_id = "model"
    adaptor.gpu_padding_strip = True
    adaptor._step_counter = 0
    adaptor._debug_step = False
    adaptor._byte_capacity = byte_capacity
    adaptor._task_capacity = task_capacity
    adaptor._validation_mode = VLLMValidationMode.VERIFY
    adaptor._local_role_formula = _VLLMRoleFormula(
        real_terms=((16, 1),),
        hook_count=1,
    )
    adaptor.model_cfg = ModelShapeConfig(
        hidden_dim=4,
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
    )
    adaptor.active_specs = [
        HookSpec(
            HOOK_TYPE_RESID_PRE,
            nn.Identity(),
            layer_no=0,
            dim0_is_actual_tokens=True,
        )
    ]
    adaptor.transport = _Transport()
    adaptor.ring_engine = _Ring()
    adaptor._row_count_dev = None
    adaptor._pinned_row_count = None
    adaptor._last_total_q = 0
    adaptor._warned_shapes = set()
    adaptor.detect_parallel_ranks = lambda: (0, 0, 0, 0)
    return adaptor, scheduler_output


def test_actual_commit_reserves_and_pushes_real_order_once():
    adaptor, scheduler_output = _commit_adaptor()
    runner = _runner()
    adaptor._step_state.capacity_candidate = (
        CUDAGraphMode.FULL,
        BatchDescriptor(num_tokens=999),
    )
    original_plan = adaptor._compute_step_plan
    plan_calls = 0

    def counted_plan(ctx):
        nonlocal plan_calls
        plan_calls += 1
        return original_plan(ctx)

    adaptor._compute_step_plan = counted_plan

    adaptor._commit_actual_dispatch(
        scheduler_output,
        runner,
        CUDAGraphMode.FULL,
        BatchDescriptor(num_tokens=8),
        combined_force_eager=False,
    )

    assert adaptor._step_state.phase is VLLMStepPhase.COMMITTED
    assert adaptor.ring_engine.prepare_calls == [(80, 1)]
    assert adaptor.transport.contexts[0]["req_ids"] == ["B", "A"]
    assert adaptor.transport.metas[0]["q_len"] == 8
    assert adaptor.transport.metas[0]["actual_q_len"] == 5
    assert plan_calls == 1


def test_false_negative_fails_before_any_ring_operation():
    adaptor, scheduler_output = _commit_adaptor(byte_capacity=79)

    with pytest.raises(RuntimeError, match="false negative"):
        adaptor._commit_actual_dispatch(
            scheduler_output,
            _runner(),
            CUDAGraphMode.FULL,
            BatchDescriptor(num_tokens=8),
            combined_force_eager=False,
        )

    assert adaptor.ring_engine.prepare_calls == []
    assert adaptor.transport.contexts == []
    assert adaptor.transport.metas == []


def test_actual_task_overflow_and_short_descriptor_fail_before_ring():
    task_overflow, scheduler_output = _commit_adaptor(
        task_capacity=0
    )
    with pytest.raises(RuntimeError, match="hook count"):
        task_overflow._commit_actual_dispatch(
            scheduler_output,
            _runner(),
            CUDAGraphMode.FULL,
            BatchDescriptor(num_tokens=8),
            combined_force_eager=True,
        )
    assert task_overflow.ring_engine.prepare_calls == []

    too_short, scheduler_output = _commit_adaptor()
    with pytest.raises(RuntimeError, match="fewer rows"):
        too_short._commit_actual_dispatch(
            scheduler_output,
            _runner(),
            CUDAGraphMode.FULL,
            BatchDescriptor(num_tokens=4),
            combined_force_eager=True,
        )
    assert too_short.ring_engine.prepare_calls == []


def test_worker_arms_and_resets_per_call(monkeypatch):
    adaptor = SimpleNamespace(
        transport=SimpleNamespace(null_offload=False),
        _has_global_hooks=True,
        _step_state=_VLLMStepState(),
    )
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.adaptor = adaptor
    scheduler_output = _scheduler()
    monkeypatch.setattr(
        "integration.vllm_adapter.has_ec_transfer",
        lambda: False,
    )

    def fake_execute(_self, received):
        assert received is scheduler_output
        assert adaptor._step_state.phase is VLLMStepPhase.ARMED
        assert adaptor._step_state.scheduler_output is scheduler_output
        adaptor._step_state.phase = VLLMStepPhase.COMMITTED
        return "ok"

    monkeypatch.setattr(Worker, "execute_model", fake_execute)

    assert worker.execute_model(scheduler_output) == "ok"
    assert adaptor._step_state.phase is VLLMStepPhase.IDLE


class _FakeNativeRingConfig:
    pass


class _FakeAdaptor:
    def __init__(self, events, preflight_eager=False):
        self.events = events
        self.preflight_eager = preflight_eager
        self._step_state = _VLLMStepState()
        self.transport = SimpleNamespace(null_offload=False)
        self.ring_engine = SimpleNamespace(
            set_null_mode=lambda value: events.append(
                ("null", value)
            )
        )
        self._validation_mode = VLLMValidationMode.OFF
        self.user_wants_null_mode = False

    def detect_parallel_ranks(self):
        return (0, 0, 0, 0)

    def _record_real_layout(
        self, scheduler_output, model_runner, ordered_counts
    ):
        self.events.append(
            (
                "record",
                tuple(model_runner.input_batch.req_ids),
                tuple(int(value) for value in ordered_counts),
            )
        )
        self._step_state.layout = object()
        self._step_state.phase = VLLMStepPhase.LAYOUT_READY

    def _preflight_force_eager(self, _model_runner, **kwargs):
        self.events.append(("preflight", kwargs))
        return self.preflight_eager

    def _commit_actual_dispatch(
        self,
        scheduler_output,
        _model_runner,
        mode,
        descriptor,
        combined_force_eager,
    ):
        self.events.append(
            (
                "commit",
                scheduler_output,
                mode,
                descriptor,
                combined_force_eager,
            )
        )
        self._step_state.phase = VLLMStepPhase.COMMITTED


def _install_worker_wrappers(
    monkeypatch,
    *,
    pipeline_parallel_size=1,
    enable_sp=False,
    preflight_eager=False,
):
    import integration.vllm_adapter as adapter_module
    import monitoring.engine as engine_module
    import monitoring._native_engine as native_module
    import vllm.distributed.parallel_state as parallel_state

    events = []
    scheduler_output = _scheduler()
    model_runner = _runner()

    def original_prepare(received, ordered_counts):
        events.append(("prepare", received, tuple(ordered_counts)))
        model_runner.input_batch.req_ids[:] = ["B", "A"]
        return "prepared"

    def original_determine(**kwargs):
        events.append(("determine", kwargs))
        descriptor = BatchDescriptor(num_tokens=kwargs["num_tokens"])
        return CUDAGraphMode.NONE, descriptor, False, None, None

    model_runner._prepare_inputs = original_prepare
    model_runner._determine_batch_execution_and_padding = (
        original_determine
    )

    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=1,
        enable_expert_parallel=False,
    )
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.model_runner = model_runner
    worker.adaptor = None
    worker.vllm_config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(model="model"),
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp)
        ),
    )

    fake_adaptor = _FakeAdaptor(events, preflight_eager)
    fake_adaptor.vllm_config = worker.vllm_config
    monkeypatch.setattr(Worker, "init_device", lambda _self: None)
    monkeypatch.setattr(native_module, "RingConfig", _FakeNativeRingConfig)
    monkeypatch.setattr(engine_module, "MonitoringEngine", lambda **_kw: object())
    monkeypatch.setattr(
        adapter_module,
        "VLLMAdaptor",
        lambda *_args, **_kwargs: fake_adaptor,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    worker.init_device()
    return worker, fake_adaptor, model_runner, scheduler_output, events


def _determine_kwargs(force_eager=False):
    return dict(
        num_tokens=5,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([3, 2], dtype=np.int32),
        max_num_scheduled_tokens=3,
        use_cascade_attn=False,
        allow_microbatching=False,
        force_eager=force_eager,
        force_uniform_decode=False,
        force_has_lora=False,
        force_num_active_loras=0,
        num_encoder_reqs=0,
    )


def test_wrappers_record_after_real_prepare_then_commit_real_dispatch(
    monkeypatch,
):
    (
        _worker,
        adaptor,
        runner,
        scheduler_output,
        events,
    ) = _install_worker_wrappers(monkeypatch, preflight_eager=True)
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )

    assert runner._prepare_inputs(
        scheduler_output,
        np.array([3, 2], dtype=np.int32),
    ) == "prepared"
    result = runner._determine_batch_execution_and_padding(
        **_determine_kwargs()
    )

    names = [event[0] for event in events if event[0] != "null"]
    assert names == [
        "prepare",
        "record",
        "preflight",
        "determine",
        "commit",
    ]
    determine_call = next(
        event[1] for event in events if event[0] == "determine"
    )
    expected = _determine_kwargs()
    expected["force_eager"] = True
    np.testing.assert_array_equal(
        determine_call.pop("num_scheduled_tokens_np"),
        expected.pop("num_scheduled_tokens_np"),
    )
    assert determine_call == expected
    assert result[0] is CUDAGraphMode.NONE
    assert adaptor._step_state.phase is VLLMStepPhase.COMMITTED
    commit = next(event for event in events if event[0] == "commit")
    assert commit[4] is True
    assert commit[3] is result[1]


def test_wrappers_preserve_pinned_parameter_contract(monkeypatch):
    _, _adaptor, runner, _scheduler_output, _events = (
        _install_worker_wrappers(monkeypatch)
    )

    wrapped_prepare = list(
        inspect.signature(runner._prepare_inputs).parameters.values()
    )
    pinned_prepare = list(
        inspect.signature(GPUModelRunner._prepare_inputs).parameters.values()
    )[1:]
    wrapped_determine = list(
        inspect.signature(
            runner._determine_batch_execution_and_padding
        ).parameters.values()
    )
    pinned_determine = list(
        inspect.signature(
            GPUModelRunner._determine_batch_execution_and_padding
        ).parameters.values()
    )[1:]

    assert [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in wrapped_prepare
    ] == [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in pinned_prepare
    ]
    assert [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in wrapped_determine
    ] == [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in pinned_determine
    ]


def test_wrapper_preserves_caller_eager_and_passes_idle_unchanged(
    monkeypatch,
):
    _, adaptor, runner, scheduler_output, events = _install_worker_wrappers(
        monkeypatch
    )

    runner._determine_batch_execution_and_padding(
        **_determine_kwargs(force_eager=True)
    )
    assert [event[0] for event in events].count("preflight") == 0
    idle_call = next(
        event[1] for event in events if event[0] == "determine"
    )
    assert idle_call["force_eager"] is True

    events.clear()
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )
    runner._prepare_inputs(
        scheduler_output,
        np.array([3, 2], dtype=np.int32),
    )
    runner._determine_batch_execution_and_padding(
        **_determine_kwargs(force_eager=True)
    )
    active_call = next(
        event[1] for event in events if event[0] == "determine"
    )
    assert active_call["force_eager"] is True


def test_early_pp_sp_dispatch_latches_without_commit(monkeypatch):
    _, adaptor, runner, scheduler_output, events = _install_worker_wrappers(
        monkeypatch,
        pipeline_parallel_size=2,
        enable_sp=True,
        preflight_eager=True,
    )
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )

    runner._determine_batch_execution_and_padding(
        **_determine_kwargs()
    )

    assert adaptor._step_state.phase is VLLMStepPhase.ARMED
    assert adaptor._step_state.force_eager_latch is True
    assert adaptor._step_state.prelayout_dispatch_seen is True
    assert not any(event[0] == "commit" for event in events)


def test_wrapper_rejects_duplicate_prepare_and_invalid_dispatch_phases(
    monkeypatch,
):
    _, adaptor, runner, scheduler_output, _events = (
        _install_worker_wrappers(monkeypatch)
    )
    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )
    runner._prepare_inputs(
        scheduler_output,
        np.array([3, 2], dtype=np.int32),
    )
    with pytest.raises(RuntimeError, match="second _prepare_inputs"):
        runner._prepare_inputs(
            scheduler_output,
            np.array([3, 2], dtype=np.int32),
        )

    adaptor._step_state.phase = VLLMStepPhase.COMMITTED
    with pytest.raises(RuntimeError, match="after metadata commit"):
        runner._determine_batch_execution_and_padding(
            **_determine_kwargs()
        )

    adaptor._step_state = _VLLMStepState(
        phase=VLLMStepPhase.ARMED,
        scheduler_output=scheduler_output,
    )
    with pytest.raises(RuntimeError, match="unexpected pre-layout"):
        runner._determine_batch_execution_and_padding(
            **_determine_kwargs()
        )


@pytest.mark.parametrize(
    ("total_tokens", "null_offload", "has_hooks"),
    [
        (0, False, True),
        (5, True, True),
        (5, False, False),
    ],
)
def test_worker_passthrough_paths_never_arm(
    monkeypatch, total_tokens, null_offload, has_hooks
):
    adaptor = SimpleNamespace(
        transport=SimpleNamespace(null_offload=null_offload),
        _has_global_hooks=has_hooks,
        _step_state=_VLLMStepState(),
    )
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.adaptor = adaptor
    scheduler_output = _scheduler(total=total_tokens)
    monkeypatch.setattr(
        "integration.vllm_adapter.has_ec_transfer",
        lambda: False,
    )
    monkeypatch.setattr(
        Worker,
        "execute_model",
        lambda _self, _output: "passthrough",
    )

    assert worker.execute_model(scheduler_output) == "passthrough"
    assert adaptor._step_state.phase is VLLMStepPhase.IDLE


def test_public_null_attach_skips_active_formula_compilation(monkeypatch):
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor.gpu_padding_strip = False
    adaptor.transport = SimpleNamespace(null_offload=True)
    monkeypatch.setattr(
        "monitoring.adaptor_base.BackendAdaptor.attach_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adaptor,
        "_compile_role_formulas",
        lambda *_args, **_kwargs: pytest.fail(
            "public null mode compiled active formulas"
        ),
    )

    adaptor.attach_model(SimpleNamespace(), "vllm-full")


def test_worker_resets_state_when_upstream_raises(monkeypatch):
    adaptor = SimpleNamespace(
        transport=SimpleNamespace(null_offload=False),
        _has_global_hooks=True,
        _step_state=_VLLMStepState(),
    )
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.adaptor = adaptor
    monkeypatch.setattr(
        "integration.vllm_adapter.has_ec_transfer",
        lambda: False,
    )

    def fail(_self, _output):
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(Worker, "execute_model", fail)
    with pytest.raises(RuntimeError, match="upstream failure"):
        worker.execute_model(_scheduler())
    assert adaptor._step_state.phase is VLLMStepPhase.IDLE


def test_ec_transfer_producer_passthrough_never_arms(monkeypatch):
    adaptor = SimpleNamespace(
        transport=SimpleNamespace(null_offload=False),
        _has_global_hooks=True,
        _step_state=_VLLMStepState(),
    )
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.adaptor = adaptor
    monkeypatch.setattr(
        "integration.vllm_adapter.has_ec_transfer",
        lambda: True,
    )
    monkeypatch.setattr(
        "integration.vllm_adapter.get_ec_transfer",
        lambda: SimpleNamespace(is_producer=True),
    )
    monkeypatch.setattr(
        Worker,
        "execute_model",
        lambda _self, _output: "ec-producer",
    )

    assert worker.execute_model(_scheduler()) == "ec-producer"
    assert adaptor._step_state.phase is VLLMStepPhase.IDLE


class _FakeOutput:
    def __init__(self, token_count=2):
        self.outputs = [
            SimpleNamespace(token_ids=list(range(token_count)))
        ]


class _FakeLLM:
    instances = []
    shutdown_error = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.rpc_calls = []
        self.__class__.instances.append(self)

    def generate(self, prompts, params):
        return [_FakeOutput() for _ in prompts]

    def collective_rpc(self, method):
        self.rpc_calls.append(method)
        if self.shutdown_error is not None:
            raise self.shutdown_error


@pytest.fixture
def fake_benchmark(monkeypatch):
    import vllm

    _FakeLLM.instances.clear()
    _FakeLLM.shutdown_error = None
    monkeypatch.setattr(vllm, "LLM", _FakeLLM)
    monkeypatch.setattr(
        vllm,
        "SamplingParams",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return _FakeLLM


def _bench_config():
    return bench_vllm_transport.BenchConfig(
        warmup=0,
        iters=1,
        num_prompts=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
    )


def test_benchmark_baseline_is_monitoring_off(fake_benchmark):
    bench_vllm_transport._run_mode("baseline", _bench_config())
    instance = fake_benchmark.instances[-1]

    assert instance.kwargs["tensor_parallel_size"] == 2
    assert instance.kwargs["pipeline_parallel_size"] == 3
    assert "worker_cls" not in instance.kwargs
    assert "additional_config" not in instance.kwargs
    assert instance.rpc_calls == []


def test_benchmark_ring_active_is_on_without_database(fake_benchmark):
    bench_vllm_transport._run_mode("ring_active", _bench_config())
    instance = fake_benchmark.instances[-1]

    assert (
        instance.kwargs["worker_cls"]
        == "integration.vllm_adapter.DMXGPUWorker"
    )
    assert instance.kwargs["additional_config"]["dmx_null_mode"] is False
    assert instance.kwargs["additional_config"]["dmx_db_host"] == ""
    assert instance.rpc_calls == ["stop_monitoring"]


def test_benchmark_public_null_skips_monitoring_shutdown(fake_benchmark):
    bench_vllm_transport._run_mode("ring_null", _bench_config())
    instance = fake_benchmark.instances[-1]

    assert instance.kwargs["additional_config"]["dmx_null_mode"] is True
    assert instance.rpc_calls == []


def test_benchmark_active_shutdown_failure_propagates(fake_benchmark):
    fake_benchmark.shutdown_error = RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        bench_vllm_transport._run_mode(
            "ring_active",
            _bench_config(),
        )


def test_benchmark_ring_db_uses_checked_shutdown(fake_benchmark):
    bench_vllm_transport._run_mode("ring_db", _bench_config())
    instance = fake_benchmark.instances[-1]

    assert instance.rpc_calls == ["stop_monitoring"]


def test_benchmark_cli_keeps_defaults_and_parses_tp_pp(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--tensor-parallel-size",
            "2",
            "--pipeline-parallel-size",
            "3",
        ],
    )

    cfg = bench_vllm_transport._parse_args()

    assert cfg.modes == ["baseline", "ring_null"]
    assert cfg.tensor_parallel_size == 2
    assert cfg.pipeline_parallel_size == 3
    assert cfg.warmup == 2
    assert cfg.iters == 5


def test_hook_row_basis_uses_native_shape_classes():
    for hook_type, _act, _short, _pl, _grp, _tp, shape_class, _pp in _HOOK_DEFS:
        expected = (
            HookRowBasis.REQUEST_ROWS
            if shape_class == SHAPE_LOGITS
            else HookRowBasis.TOKEN_ROWS
        )
        assert hook_row_basis(hook_type) is expected

    unknown_hook_type = max(row[0] for row in _HOOK_DEFS) + 1
    with pytest.raises(ValueError, match="Unknown hook type"):
        hook_row_basis(unknown_hook_type)


def test_resid_final_is_last_stage_only():
    assert HOOK_TYPE_RESID_FINAL in PP_LAST_ONLY
    assert HOOK_TYPE_RESID_FINAL not in PP_FIRST_ONLY
    first = HookSpec(HOOK_TYPE_RESID_FINAL, nn.Identity())
    last = HookSpec(HOOK_TYPE_RESID_FINAL, nn.Identity())

    assert filter_by_pp_rank([first], True, False) == []
    assert filter_by_pp_rank([last], False, True) == [last]


class _ModelWideHookInventory:
    def __init__(self, specs):
        self.specs = tuple(specs)

    def get_hook_specs(self, *, model_wide=False):
        if not model_wide:
            raise AssertionError("formula compilation requested local specs")
        return list(self.specs)


def _model_wide_specs(num_layers=4):
    specs = [
        HookSpec(
            HOOK_TYPE_TOKEN_IDS,
            None,
            dtype=torch.int32,
            dim0_is_actual_tokens=True,
        ),
        HookSpec(
            HOOK_TYPE_EMBED,
            None,
            dim0_is_actual_tokens=True,
        ),
    ]
    for layer_no in range(num_layers):
        specs.extend([
            HookSpec(
                HOOK_TYPE_RESID_PRE,
                None,
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_Q,
                None,
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
        ])
        if layer_no % 2 == 0:
            specs.append(
                HookSpec(
                    HOOK_TYPE_MLP_POST,
                    None,
                    layer_no=layer_no,
                    dim0_is_actual_tokens=True,
                )
            )
        else:
            specs.extend([
                HookSpec(
                    HOOK_TYPE_ROUTER_LOGITS,
                    None,
                    layer_no=layer_no,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_TOPK_IDS,
                    None,
                    layer_no=layer_no,
                    dtype=torch.int32,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_TOPK_WEIGHTS,
                    None,
                    layer_no=layer_no,
                    dtype=torch.float32,
                    dim0_is_actual_tokens=True,
                ),
            ])
        specs.append(
            HookSpec(
                HOOK_TYPE_MLP_OUT,
                None,
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            )
        )
    specs.extend([
        HookSpec(
            HOOK_TYPE_RESID_FINAL,
            None,
            dim0_is_actual_tokens=True,
        ),
        HookSpec(
            HOOK_TYPE_FINAL_LN,
            None,
            dim0_is_actual_tokens=True,
        ),
        HookSpec(HOOK_TYPE_FINAL_LOGITS, None),
    ])
    return specs


def _bind_spec(spec):
    return HookSpec(
        spec.hook_type,
        nn.Identity(),
        layer_no=spec.layer_no,
        dtype=spec.dtype,
        allow_token_cnt_mismatch=spec.allow_token_cnt_mismatch,
        dim0_is_actual_tokens=spec.dim0_is_actual_tokens,
    )


def _formula_adaptor(
    *,
    tp_size,
    tp_rank,
    pp_rank,
    gpu_padding_strip,
    task_capacity=10_000,
    use_ubatching=False,
    data_parallel_size=1,
    speculative_config=None,
    hook_selection="vllm-full",
):
    from vllm.distributed.utils import get_pp_indices

    num_layers = 4
    pp_size = 2
    cfg = ModelShapeConfig(
        hidden_dim=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float16,
        vocab_size=16,
        intermediate_dim=12,
        num_experts=4,
        top_k=2,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    hf_config = SimpleNamespace(num_hidden_layers=num_layers)
    parallel = SimpleNamespace(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        data_parallel_size=data_parallel_size,
        use_ubatching=use_ubatching,
        enable_expert_parallel=False,
    )
    adaptor = VLLMAdaptor.__new__(VLLMAdaptor)
    adaptor.model_cfg = cfg
    adaptor.vllm_config = SimpleNamespace(
        parallel_config=parallel,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=32,
            max_num_seqs=8,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
            max_cudagraph_capture_size=32,
        ),
        speculative_config=speculative_config,
        model_config=SimpleNamespace(hf_config=hf_config),
    )
    adaptor.ring_engine = SimpleNamespace(
        payload_cap=lambda: 1_000_000,
        staging_cap=lambda: 1_000_000,
        task_cap=lambda: task_capacity,
    )
    adaptor.gpu_padding_strip = gpu_padding_strip
    adaptor.detect_parallel_ranks = lambda: (
        tp_rank,
        0,
        0,
        pp_rank,
    )

    model = _ModelWideHookInventory(_model_wide_specs(num_layers))
    selected = select_hook_specs(
        model.get_hook_specs(model_wide=True),
        hook_selection,
        cfg,
    )
    start, end = get_pp_indices(num_layers, pp_rank, pp_size)
    adaptor.active_specs = [
        _bind_spec(spec)
        for spec in selected
        if (
            (spec.layer_no < 0 or start <= spec.layer_no < end)
            and hook_belongs_to_pp_rank(
                spec,
                is_first_rank=(pp_rank == 0),
                is_last_rank=(pp_rank == pp_size - 1),
            )
            and hook_belongs_to_tp_rank(spec, tp_rank)
        )
    ]
    return adaptor, model


def _compile_formula_adaptor(adaptor, model, hook_selection):
    selection = _VLLMHookSelection.from_model(
        model=model,
        local_hooks=tuple(adaptor.active_specs),
        hook_selection=hook_selection,
        cfg=adaptor.model_cfg,
        parallel_config=adaptor.vllm_config.parallel_config,
        hf_config=adaptor.vllm_config.model_config.hf_config,
    )
    adaptor._compile_role_formulas(selection)


@pytest.mark.parametrize(
    "hook_selection",
    ["vllm-full", "resid_pre,final_logits"],
)
@pytest.mark.parametrize("gpu_padding_strip", [False, True])
@pytest.mark.parametrize(("tp_size", "tp_rank"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("pp_rank", [0, 1])
def test_compiled_role_formula_matches_concrete_actual_plan(
    hook_selection,
    gpu_padding_strip,
    tp_size,
    tp_rank,
    pp_rank,
):
    adaptor, model = _formula_adaptor(
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        gpu_padding_strip=gpu_padding_strip,
        hook_selection=hook_selection,
    )

    _compile_formula_adaptor(adaptor, model, hook_selection)

    ctx = SimpleNamespace(
        batch=0,
        q_len=8,
        actual_q_len=5 if gpu_padding_strip else None,
        kv_dim=0,
        logits_to_keep=2,
    )
    actual = adaptor._compute_step_plan(ctx)
    formula = adaptor._local_role_formula
    assert formula is not None
    assert (
        formula.bytes_for(8, 5, 2),
        formula.hook_count,
        False,
    ) == actual
    assert len(adaptor._role_formulas) == (2 if tp_size == 1 else 3)
    assert len(adaptor._role_formulas) == len(
        set(adaptor._role_formulas)
    )


def test_compiled_role_formula_unwraps_vllm_cudagraph_wrapper():
    plain, plain_model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
    )
    wrapped, wrapped_model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
    )
    wrapper = object.__new__(CUDAGraphWrapper)
    wrapper.runnable = wrapped_model

    _compile_formula_adaptor(plain, plain_model, "vllm-full")
    _compile_formula_adaptor(wrapped, wrapper, "vllm-full")

    assert wrapped._role_formulas == plain._role_formulas
    assert wrapped._local_role_formula == plain._local_role_formula
    assert wrapped._has_global_hooks == plain._has_global_hooks


def test_task_capacity_exact_fit_and_one_extra_hook_fail_attachment():
    adaptor, model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
    )
    _compile_formula_adaptor(adaptor, model, "vllm-full")
    maximum = max(
        formula.hook_count for formula in adaptor._role_formulas
    )

    exact, exact_model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
        task_capacity=maximum,
    )
    _compile_formula_adaptor(exact, exact_model, "vllm-full")

    overflow, overflow_model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
        task_capacity=maximum - 1,
    )
    with pytest.raises(RuntimeError, match="task-ring capacity"):
        _compile_formula_adaptor(overflow, overflow_model, "vllm-full")


def test_compile_rejects_dp_ubatching_and_all_speculative_decoding():
    dbo, dbo_model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
        data_parallel_size=2,
        use_ubatching=True,
    )
    with pytest.raises(RuntimeError, match="DBO"):
        _compile_formula_adaptor(dbo, dbo_model, "vllm-full")

    speculative, speculative_model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
        speculative_config=object(),
        hook_selection="resid_pre",
    )
    with pytest.raises(RuntimeError, match="speculative"):
        _compile_formula_adaptor(
            speculative, speculative_model, "resid_pre"
        )


def test_absent_model_wide_hook_selection_produces_zero_plan():
    adaptor, model = _formula_adaptor(
        tp_size=2,
        tp_rank=1,
        pp_rank=1,
        gpu_padding_strip=True,
        hook_selection="pos_embed",
    )

    _compile_formula_adaptor(adaptor, model, "pos_embed")

    assert adaptor.active_specs == []
    assert adaptor._role_formulas == (_VLLMRoleFormula(),)
    assert adaptor._local_role_formula == _VLLMRoleFormula()
    assert adaptor._has_global_hooks is False


def test_selection_requires_model_wide_inventory_api():
    adaptor, _model = _formula_adaptor(
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
        gpu_padding_strip=True,
    )

    with pytest.raises(RuntimeError, match="get_hook_specs"):
        _compile_formula_adaptor(
            adaptor, SimpleNamespace(), "vllm-full"
        )


def test_qwen2_moe_rejects_zero_sparse_step():
    from vllm.model_executor.models.qwen2_moe_p import (
        _is_sparse_moe_layer,
    )

    config = SimpleNamespace(
        decoder_sparse_step=0,
        mlp_only_layers=[],
        num_experts=4,
    )
    with pytest.raises(RuntimeError, match="decoder_sparse_step"):
        _is_sparse_moe_layer(config, 0)
