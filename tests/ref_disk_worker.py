"""RefDiskWorker: vLLM Worker that runs the ref model, saves captured
tensors to disk after each forward pass.

Post-forward: slices GPU buffers per request using the real packed order
captured from ``input_batch`` after vLLM prepares the inputs, then saves them
as .pt files.
"""
from dataclasses import dataclass
import json
import os
import re
from typing import Any

import torch

from vllm.v1.worker.gpu_worker import Worker


_ARCH_REMAP = {
    "GPT2LMHeadModel": "GPT2RefLMHeadModel",
    "Qwen3ForCausalLM": "Qwen3RefForCausalLM",
    "Qwen2MoeForCausalLM": "Qwen2MoeRefForCausalLM",
}


# Hook names that are TP-sharded (output of ColumnParallel, before RowParallel).
# Unsharded hooks are identical across TP ranks — only rank 0 saves them.
_TP_SHARDED_HOOKS = {"q", "k", "v", "z", "mlp_post"}
_PP_FIRST_HOOKS = {"token_ids", "embed", "pos_embed"}
_PP_LAST_HOOKS = {"resid_final", "final_ln", "final_logits"}
_VLLM_REQ_ID_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


@dataclass(frozen=True)
class _PackedLayout:
    scheduler_output: Any
    request_ids: tuple[str, ...]
    scheduled_counts: tuple[int, ...]
    computed_counts: tuple[int, ...]
    total_rows: int


class RefDiskWorker(Worker):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ref_config: dict | None = None
        self._output_dir: str = ""
        self._step: int = 0
        self._tp_rank: int = 0
        self._tp_size: int = 1
        self._pp_is_first: bool = True
        self._pp_is_last: bool = True
        self._active_scheduler_output: Any = None
        self._packed_layout: _PackedLayout | None = None

    def init_device(self) -> None:
        super().init_device()

        model_runner = self.model_runner
        original_prepare = model_runner._prepare_inputs

        def _wrapped_prepare(
            scheduler_output: Any,
            num_scheduled_tokens: Any,
        ) -> Any:
            result = original_prepare(
                scheduler_output,
                num_scheduled_tokens,
            )
            active = self._active_scheduler_output
            if active is None:
                return result
            if active is not scheduler_output:
                raise RuntimeError(
                    "RefDiskWorker scheduler output changed during input prep"
                )
            if self._packed_layout is not None:
                raise RuntimeError(
                    "RefDiskWorker observed a second _prepare_inputs in one step"
                )
            self._packed_layout = self._capture_packed_layout(
                scheduler_output,
                num_scheduled_tokens,
            )
            return result

        model_runner._prepare_inputs = _wrapped_prepare

    def _capture_packed_layout(
        self,
        scheduler_output: Any,
        num_scheduled_tokens: Any,
    ) -> _PackedLayout:
        input_batch = self.model_runner.input_batch
        num_reqs = int(input_batch.num_reqs)
        raw_req_ids = tuple(input_batch.req_ids[:num_reqs])
        scheduled_counts = tuple(int(value) for value in num_scheduled_tokens)
        computed_counts = tuple(
            int(value)
            for value in input_batch.num_computed_tokens_cpu[:num_reqs]
        )

        if (
            len(raw_req_ids) != num_reqs
            or len(scheduled_counts) != num_reqs
            or len(computed_counts) != num_reqs
        ):
            raise RuntimeError("RefDiskWorker packed layout lengths disagree")
        if any(not isinstance(req_id, str) for req_id in raw_req_ids):
            raise RuntimeError("RefDiskWorker request IDs must be strings")
        if len(set(raw_req_ids)) != num_reqs:
            raise RuntimeError("RefDiskWorker packed request IDs are not unique")

        request_ids = tuple(
            _VLLM_REQ_ID_SUFFIX.sub("", req_id) for req_id in raw_req_ids
        )
        if len(set(request_ids)) != num_reqs:
            raise RuntimeError(
                "RefDiskWorker normalized request IDs are not unique"
            )
        if any(value < 0 for value in scheduled_counts):
            raise RuntimeError("RefDiskWorker scheduled token count is negative")
        if any(value < 0 for value in computed_counts):
            raise RuntimeError("RefDiskWorker computed token count is negative")

        scheduler_counts = scheduler_output.num_scheduled_tokens
        if (
            len(scheduler_counts) != num_reqs
            or set(scheduler_counts) != set(raw_req_ids)
        ):
            raise RuntimeError(
                "RefDiskWorker packed IDs do not match scheduler membership"
            )
        for req_id, count in zip(raw_req_ids, scheduled_counts):
            if int(scheduler_counts[req_id]) != count:
                raise RuntimeError(
                    f"RefDiskWorker scheduled count mismatch for {req_id!r}"
                )

        total_rows = sum(scheduled_counts)
        if total_rows != int(scheduler_output.total_num_scheduled_tokens):
            raise RuntimeError(
                "RefDiskWorker packed rows do not match scheduler total"
            )

        return _PackedLayout(
            scheduler_output=scheduler_output,
            request_ids=request_ids,
            scheduled_counts=scheduled_counts,
            computed_counts=computed_counts,
            total_rows=total_rows,
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # Remap to ref variant
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs

        super().load_model(load_dummy_weights=load_dummy_weights)

        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        self._tp_rank = get_tp_group().rank_in_group
        self._tp_size = get_tp_group().world_size
        pp_group = get_pp_group()
        self._pp_is_first = pp_group.is_first_rank
        self._pp_is_last = pp_group.is_last_rank

        cfg_path = os.environ.get("REF_CONFIG")
        if cfg_path:
            with open(cfg_path) as f:
                self._ref_config = json.load(f)
            self._output_dir = self._ref_config["output_dir"]
            os.makedirs(self._output_dir, exist_ok=True)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        total_tokens = int(scheduler_output.total_num_scheduled_tokens)
        if self._ref_config is None or total_tokens == 0:
            return super().execute_model(scheduler_output)
        if (
            self._active_scheduler_output is not None
            or self._packed_layout is not None
        ):
            raise RuntimeError("RefDiskWorker execute_model calls overlap")

        self._active_scheduler_output = scheduler_output
        try:
            result = super().execute_model(scheduler_output)
            layout = self._packed_layout
            if layout is None:
                raise RuntimeError(
                    "RefDiskWorker forward completed without a packed layout"
                )
            if layout.scheduler_output is not scheduler_output:
                raise RuntimeError(
                    "RefDiskWorker scheduler output changed during execution"
                )
            if layout.total_rows != total_tokens:
                raise RuntimeError(
                    "RefDiskWorker captured rows changed during execution"
                )

            self._save_step(
                layout.request_ids,
                layout.scheduled_counts,
                layout.computed_counts,
            )
            self._step += 1
            return result
        finally:
            self._active_scheduler_output = None
            self._packed_layout = None

    def _save_step(
        self,
        request_ids: tuple[str, ...],
        scheduled_counts: tuple[int, ...],
        computed_counts: tuple[int, ...],
    ) -> None:
        model = self.model_runner.model
        if not hasattr(model, "get_ref_buffers"):
            return

        bufs = model.get_ref_buffers()
        if not bufs:
            return

        offset = 0
        for i, (request_id, n, pre_computed) in enumerate(
            zip(request_ids, scheduled_counts, computed_counts)
        ):
            t_start = pre_computed
            t_end = pre_computed + n

            req_dir = os.path.join(self._output_dir, request_id)
            os.makedirs(req_dir, exist_ok=True)

            for name, buf in bufs.items():
                # Parse hook name: "resid_pre_L0" → hook=resid_pre
                # or "embed" → hook=embed
                if "_L" in name:
                    hook_name = name.rsplit("_L", 1)[0]
                else:
                    hook_name = name

                # Global hooks exist in each PP worker's ref model, but only
                # the stage that executes the corresponding operation owns
                # their captured value.
                if hook_name in _PP_FIRST_HOOKS and not self._pp_is_first:
                    continue
                if hook_name in _PP_LAST_HOOKS and not self._pp_is_last:
                    continue

                # TP: non-zero ranks skip unsharded hooks (identical to rank 0)
                is_sharded = hook_name in _TP_SHARDED_HOOKS
                if self._tp_rank != 0 and not is_sharded:
                    continue

                # Shard rank suffix (only when TP > 1)
                sr = f"_SR{self._tp_rank}" if self._tp_size > 1 else ""

                # final_logits: dim0 = num_reqs (one per request), not total_tokens.
                # Slice by request index, save as single-token range (last predicted).
                if name == "final_logits":
                    chunk = buf[i:i + 1].cpu().clone()
                    fl_start = t_end - 1
                    fl_end = t_end
                    fname = f"final_logits_T{fl_start}_{fl_end}{sr}.pt"
                    torch.save(chunk, os.path.join(req_dir, fname))
                    continue

                chunk = buf[offset:offset + n].cpu().clone()
                if "_L" in name:
                    parts = name.rsplit("_L", 1)
                    layer = int(parts[1])
                    fname = f"{hook_name}_L{layer}_T{t_start}_{t_end}{sr}.pt"
                else:
                    fname = f"{hook_name}_T{t_start}_{t_end}{sr}.pt"

                torch.save(chunk, os.path.join(req_dir, fname))

            offset += n
