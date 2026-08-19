"""Focused GPU ground truth for the vLLM request-order fix.

Run once per topology:
  python tests/vllm_request_order_fix_gpu.py --tp 1 --pp 1 --evidence-dir /tmp/...
  python tests/vllm_request_order_fix_gpu.py --tp 2 --pp 1 --evidence-dir /tmp/...
  python tests/vllm_request_order_fix_gpu.py --tp 1 --pp 2 --evidence-dir /tmp/...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch
from vllm import LLM, SamplingParams

from integration.vllm_adapter import (
    DMXGPUWorker,
    VLLMStepPhase,
    _attach_dmi_internal,
    normalize_vllm_request_id,
)
from monitoring import _native_engine
from monitoring.ring_transport import (
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TO_SHORT_NAME,
    PP_LAST_ONLY,
)


def _evidence_path(tp_rank: int, pp_rank: int) -> Path:
    root = Path(os.environ["DMI_ORDER_EVIDENCE_DIR"])
    return root / f"tp{tp_rank}-pp{pp_rank}.jsonl"


class ProbeWorker(DMXGPUWorker):
    """Test-only worker that observes the two already-wrapped boundaries."""

    def _emit(self, payload: dict[str, Any]) -> None:
        payload.update(tp_rank=self._dmx_tp_rank, pp_rank=self._dmx_pp_rank)
        with _evidence_path(
            self._dmx_tp_rank, self._dmx_pp_rank
        ).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def init_device(self) -> None:
        super().init_device()
        self._probe_step = 0
        self._probe_plan_calls = 0
        self._probe_last_plan = None
        runner = self.model_runner
        original_prepare = runner._prepare_inputs
        original_determine = (
            runner._determine_batch_execution_and_padding
        )

        def observed_prepare(scheduler_output, ordered_counts):
            result = original_prepare(scheduler_output, ordered_counts)
            state = self.adaptor._step_state
            layout = state.layout
            if (
                state.phase is not VLLMStepPhase.LAYOUT_READY
                or layout is None
            ):
                return result

            self._probe_step += 1
            total = layout.total_rows
            cpu_rows = runner.input_ids.cpu[:total].tolist()
            gpu_rows = runner.input_ids.gpu[:total].cpu().tolist()
            expected_by_req = {}
            for index, req_id in enumerate(layout.raw_req_ids):
                start = layout.computed_counts[index]
                end = start + layout.scheduled_counts[index]
                expected_by_req[req_id] = (
                    runner.input_batch.token_ids_cpu[
                        index, start:end
                    ].tolist()
                )
            real_flat = [
                token
                for req_id in layout.raw_req_ids
                for token in expected_by_req[req_id]
            ]
            scheduler_order = list(
                scheduler_output.num_scheduled_tokens
            )
            old_flat = [
                token
                for req_id in scheduler_order
                for token in expected_by_req[req_id]
            ]
            self._emit(
                {
                    "event": "layout",
                    "step": self._probe_step,
                    "scheduler_order": scheduler_order,
                    "raw_req_ids": list(layout.raw_req_ids),
                    "req_ids": list(layout.req_ids),
                    "scheduled_counts": list(layout.scheduled_counts),
                    "computed_counts": list(layout.computed_counts),
                    "token_ranges": list(layout.token_ranges),
                    "dim0_offsets": list(layout.dim0_offsets),
                    "cpu_token_rows": cpu_rows,
                    "gpu_token_rows": gpu_rows,
                    "expected_by_req": expected_by_req,
                    "expected_real_rows": real_flat,
                    "old_scheduler_rows": old_flat,
                }
            )
            return result

        def observed_determine(*args, **kwargs):
            result = original_determine(*args, **kwargs)
            state = self.adaptor._step_state
            layout = state.layout
            if state.phase is VLLMStepPhase.COMMITTED and layout is not None:
                formula = self.adaptor._local_role_formula
                descriptor = state.real_batch_descriptor
                formula_plan = [
                    formula.bytes_for(
                        descriptor.num_tokens,
                        layout.total_rows,
                        len(layout.req_ids),
                    ),
                    formula.hook_count,
                    False,
                ]
                self._emit(
                    {
                        "event": "dispatch",
                        "step": self._probe_step,
                        "mode": str(result[0]),
                        "execution_rows": descriptor.num_tokens,
                        "dmi_force_eager": state.force_eager_latch,
                        "actual_plan": self._probe_last_plan,
                        "formula_plan": formula_plan,
                        "plan_calls": self._probe_plan_calls,
                    }
                )
            return result

        runner._prepare_inputs = observed_prepare
        runner._determine_batch_execution_and_padding = observed_determine

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.distributed.utils import get_pp_indices

        adaptor = self.adaptor
        original_plan = adaptor._compute_step_plan

        def observed_plan(ctx):
            result = original_plan(ctx)
            self._probe_plan_calls += 1
            self._probe_last_plan = list(result)
            return result

        adaptor._compute_step_plan = observed_plan
        if os.environ.get("DMI_PREFIX_CACHE_PROBE") == "1":
            for spec in adaptor.active_specs:
                if spec.hook_type != HOOK_TYPE_RESID_PRE:
                    continue

                def observe_capture(tensor, *, hook):
                    state = adaptor._step_state
                    layout = state.layout
                    if (
                        state.phase is VLLMStepPhase.COMMITTED
                        and layout is not None
                    ):
                        self._emit(
                            {
                                "event": "capture",
                                "step": self._probe_step,
                                "raw_req_ids": list(layout.raw_req_ids),
                                "token_ranges": list(layout.token_ranges),
                                "rows": int(tensor.shape[0]),
                            }
                        )

                spec.module.add_hook(observe_capture, is_permanent=True)
                break
            else:
                raise RuntimeError(
                    "prefix-cache probe requires a resid_pre HookPoint"
                )

        hf_config = self.vllm_config.model_config.hf_config
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = hf_config.n_layer
        num_layers = int(num_layers)
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        interval = get_pp_indices(
            num_layers, self._dmx_pp_rank, pp_size
        )
        self._emit(
            {
                "event": "model",
                "pp_size": pp_size,
                "num_layers": num_layers,
                "layer_interval": list(interval),
                "hooks": [
                    HOOK_TYPE_TO_SHORT_NAME[spec.hook_type]
                    for spec in adaptor.active_specs
                ],
                "resid_final_pp_last": (
                    HOOK_TYPE_RESID_FINAL in PP_LAST_ONLY
                ),
            }
        )


def _outputs(outputs) -> list[list[int]]:
    return [list(output.outputs[0].token_ids) for output in outputs]


def _load_evidence(root: Path) -> dict[str, list[dict[str, Any]]]:
    records = {}
    for path in sorted(root.glob("tp*-pp*.jsonl")):
        records[path.stem] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    return records


def _validate(
    records: dict[str, list[dict[str, Any]]],
    *,
    tp: int,
    pp: int,
) -> dict[str, Any]:
    assert len(records) == tp * pp, records.keys()
    models = {
        rank: next(row for row in rows if row["event"] == "model")
        for rank, rows in records.items()
    }
    layouts = {
        rank: [row for row in rows if row["event"] == "layout"]
        for rank, rows in records.items()
    }
    dispatches = {
        rank: [row for row in rows if row["event"] == "dispatch"]
        for rank, rows in records.items()
    }
    assert all(layouts.values())
    assert all(len(layouts[r]) == len(dispatches[r]) for r in records)

    reference_rank = next(iter(records))
    reference_layouts = layouts[reference_rank]
    for rank in records:
        assert [
            (
                row["raw_req_ids"],
                row["scheduled_counts"],
                row["token_ranges"],
                row["dim0_offsets"],
            )
            for row in layouts[rank]
        ] == [
            (
                row["raw_req_ids"],
                row["scheduled_counts"],
                row["token_ranges"],
                row["dim0_offsets"],
            )
            for row in reference_layouts
        ]
        assert [row["dmi_force_eager"] for row in dispatches[rank]] == [
            row["dmi_force_eager"]
            for row in dispatches[reference_rank]
        ]
        for layout in layouts[rank]:
            assert layout["gpu_token_rows"] == layout["cpu_token_rows"]
            assert layout["gpu_token_rows"] == layout["expected_real_rows"]
            assert sum(layout["scheduled_counts"]) == len(
                layout["gpu_token_rows"]
            )
            reconstructed_rows = []
            for req_id, computed, count, token_range, offset in zip(
                layout["raw_req_ids"],
                layout["computed_counts"],
                layout["scheduled_counts"],
                layout["token_ranges"],
                layout["dim0_offsets"],
            ):
                expected_rows = layout["expected_by_req"][req_id]
                assert token_range == [computed, computed + count]
                assert offset == len(reconstructed_rows)
                assert len(expected_rows) == count
                assert (
                    layout["gpu_token_rows"][offset : offset + count]
                    == expected_rows
                )
                reconstructed_rows.extend(expected_rows)
            assert reconstructed_rows == layout["gpu_token_rows"]
        for dispatch in dispatches[rank]:
            assert dispatch["actual_plan"] == dispatch["formula_plan"]
            assert dispatch["plan_calls"] == dispatch["step"]
        assert any(
            not dispatch["mode"].endswith("NONE")
            for dispatch in dispatches[rank]
        ), f"{rank} never used a CUDA graph"

    divergences = [
        row
        for row in reference_layouts
        if row["scheduler_order"] != row["raw_req_ids"]
    ]
    if divergences:
        assert any(
            row["old_scheduler_rows"] != row["gpu_token_rows"]
            for row in divergences
        ), "old scheduler order was not a negative control"
    else:
        assert pp > 1, "workload did not expose scheduler/packed divergence"

    if pp == 1:
        assert any(
            set(previous["raw_req_ids"]) & set(current["raw_req_ids"])
            and set(current["raw_req_ids"]) - set(previous["raw_req_ids"])
            for previous, current in zip(
                reference_layouts, reference_layouts[1:]
            )
        ), "workload did not retain one request while admitting another"
        assert any(
            any(
                index < len(previous["raw_req_ids"])
                and previous["raw_req_ids"][index] != req_id
                for index, req_id in enumerate(current["raw_req_ids"])
                if req_id not in previous["raw_req_ids"]
            )
            for previous, current in zip(
                reference_layouts, reference_layouts[1:]
            )
        ), "workload did not reuse a packed request slot"

    if pp == 2:
        assert any(
            len(layout["raw_req_ids"]) > 1 for layout in reference_layouts
        ), "PP workload did not execute a multi-request batch"
        pp_models = {
            model["pp_rank"]: model for model in models.values()
        }
        assert set(pp_models) == {0, 1}
        intervals = [
            tuple(pp_models[rank]["layer_interval"]) for rank in (0, 1)
        ]
        assert intervals[0][0] == 0
        assert intervals[0][1] == intervals[1][0]
        assert intervals[1][1] == pp_models[0]["num_layers"]
        first_hooks = pp_models[0]["hooks"]
        last_hooks = pp_models[1]["hooks"]
        assert "resid_final" not in first_hooks
        assert "final_ln" not in first_hooks
        assert "final_logits" not in first_hooks
        assert last_hooks.count("resid_final") == 1
        assert last_hooks.count("final_ln") == 1
        assert last_hooks.count("final_logits") == 1
        assert "token_ids" not in last_hooks
        assert "embed" not in last_hooks
        assert "pos_embed" not in last_hooks
        assert all(
            model["resid_final_pp_last"] for model in pp_models.values()
        )

    return {
        "ranks": sorted(records),
        "steps": len(reference_layouts),
        "divergences": len(divergences),
    }


def _run_prefix_cache_probe(model: str, evidence_dir: Path) -> dict[str, Any]:
    """Prove a cache-hit request records only the suffix it executes."""

    evidence_dir.mkdir()
    os.environ["DMI_ORDER_EVIDENCE_DIR"] = str(evidence_dir)
    os.environ["DMI_PREFIX_CACHE_PROBE"] = "1"
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        max_model_len=128,
        block_size=16,
        enable_prefix_caching=True,
        enforce_eager=True,
        async_scheduling=False,
        gpu_memory_utilization=0.35,
        worker_cls="tests.vllm_request_order_fix_gpu.ProbeWorker",
        additional_config={
            "dmx_hook_selection": "resid_pre",
            "dmx_ring_payload_mb": 256,
            "dmx_ring_pinned_mb": 256,
            "dmx_ring_task_entries": 4096,
            "dmx_null_mode": False,
            "dmx_db_host": "",
        },
    )
    prompt_token_ids = [100 + index for index in range(80)]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )
    cold = llm.generate(
        [{"prompt_token_ids": list(prompt_token_ids)}],
        params,
        use_tqdm=False,
    )[0]
    warm = llm.generate(
        [{"prompt_token_ids": list(prompt_token_ids)}],
        params,
        use_tqdm=False,
    )[0]

    _attach_dmi_internal([warm], "prefix-cache-probe")
    assert not warm.dmi_internal._token_ranges

    llm.collective_rpc("stop_monitoring")
    del llm
    torch.cuda.empty_cache()

    assert cold.num_cached_tokens == 0
    cached = int(warm.num_cached_tokens or 0)
    assert cached > 0, "second identical request did not hit prefix cache"

    records = _load_evidence(evidence_dir)
    assert len(records) == 1, records.keys()
    rows = next(iter(records.values()))
    request_id = normalize_vllm_request_id(warm.request_id)

    def request_index(record: dict[str, Any]) -> int | None:
        normalized = [
            normalize_vllm_request_id(value)
            for value in record.get("raw_req_ids", ())
        ]
        return (
            normalized.index(request_id)
            if request_id in normalized
            else None
        )

    matched_layouts = [
        (row, index)
        for row in rows
        if row.get("event") == "layout"
        if (index := request_index(row)) is not None
    ]
    assert len(matched_layouts) == 1, matched_layouts
    layout, index = matched_layouts[0]
    scheduled = int(layout["scheduled_counts"][index])
    assert layout["computed_counts"][index] == cached
    assert layout["token_ranges"][index] == [cached, cached + scheduled]
    actual_rows = layout["expected_by_req"][layout["raw_req_ids"][index]]
    assert len(actual_rows) == scheduled

    matched_captures = [
        row
        for row in rows
        if row.get("event") == "capture"
        and request_index(row) is not None
    ]
    assert len(matched_captures) == 1, matched_captures
    assert matched_captures[0]["rows"] == scheduled

    return {
        "request_id": request_id,
        "cached_tokens": cached,
        "executed_suffix_tokens": scheduled,
        "token_range": layout["token_ranges"][index],
        "captured_resid_pre_rows": matched_captures[0]["rows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()
    if args.tp * args.pp > 2:
        raise ValueError("focused validation uses at most two GPUs")

    repo_root = Path(__file__).resolve().parents[1]
    pinned_root = (repo_root / "integration" / "vllm").resolve()
    import vllm
    try:
        import vllm._C as vllm_native_extension
    except ModuleNotFoundError:
        import vllm._C_stable_libtorch as vllm_native_extension

    assert Path(vllm.__file__).resolve().is_relative_to(pinned_root)
    assert Path(vllm_native_extension.__file__).resolve().is_relative_to(
        pinned_root
    )
    native = _native_engine._load_extension()
    assert Path(native.__file__).resolve().is_relative_to(repo_root)
    assert HOOK_TYPE_RESID_FINAL in PP_LAST_ONLY

    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    os.environ["DMI_ORDER_EVIDENCE_DIR"] = str(args.evidence_dir)
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    prompts = [
        "Once upon a",
        "One two three four",
        "Red green blue yellow orange",
        "The capital of France is",
    ]
    params = [
        SamplingParams(
            temperature=0.0, max_tokens=count, ignore_eos=True
        )
        for count in (1, 2, 5, 4)
    ]
    common = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        max_num_seqs=4 if args.pp > 1 else 2,
        max_model_len=64,
        enforce_eager=False,
        async_scheduling=False,
        gpu_memory_utilization=0.35,
    )

    reference_llm = LLM(**common)
    reference = _outputs(
        reference_llm.generate(prompts, params, use_tqdm=False)
    )
    del reference_llm
    torch.cuda.empty_cache()

    active_llm = LLM(
        **common,
        worker_cls="tests.vllm_request_order_fix_gpu.ProbeWorker",
        additional_config={
            "dmx_hook_selection": "vllm-full",
            "dmx_ring_payload_mb": 256,
            "dmx_ring_pinned_mb": 256,
            "dmx_ring_task_entries": 4096,
            "dmx_null_mode": False,
            "dmx_db_host": "",
        },
    )
    active = _outputs(
        active_llm.generate(prompts, params, use_tqdm=False)
    )
    active_llm.collective_rpc("stop_monitoring")
    del active_llm
    torch.cuda.empty_cache()

    assert active == reference, {
        "active": active,
        "reference": reference,
    }
    summary = _validate(
        _load_evidence(args.evidence_dir),
        tp=args.tp,
        pp=args.pp,
    )
    prefix_cache = None
    if args.tp == 1 and args.pp == 1:
        prefix_cache = _run_prefix_cache_probe(
            args.model,
            args.evidence_dir / "prefix-cache",
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "tp": args.tp,
                "pp": args.pp,
                "outputs": active,
                "prefix_cache": prefix_cache,
                **summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
