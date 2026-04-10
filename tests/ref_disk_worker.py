"""RefDiskWorker: vLLM Worker that runs the ref model, saves captured
tensors to disk after each forward pass.

Post-forward: slices GPU buffers per-request using offsets computed from
scheduler_output, saves as .pt files.
"""
import json
import os
from typing import Any

import torch

from vllm.v1.worker.gpu_worker import Worker


_ARCH_REMAP = {
    "GPT2LMHeadModel": "GPT2RefLMHeadModel",
    "Qwen3ForCausalLM": "Qwen3RefForCausalLM",
}


# Hook names that are TP-sharded (output of ColumnParallel, before RowParallel).
# Unsharded hooks are identical across TP ranks — only rank 0 saves them.
_TP_SHARDED_HOOKS = {"q", "k", "v", "z", "mlp_post"}


class RefDiskWorker(Worker):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ref_config: dict | None = None
        self._output_dir: str = ""
        self._step: int = 0
        self._tp_rank: int = 0
        self._tp_size: int = 1

    def load_model(self) -> None:
        # Remap to ref variant
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs

        super().load_model()

        from vllm.distributed.parallel_state import get_tp_group
        self._tp_rank = get_tp_group().rank_in_group
        self._tp_size = get_tp_group().world_size

        cfg_path = os.environ.get("REF_CONFIG")
        if cfg_path:
            with open(cfg_path) as f:
                self._ref_config = json.load(f)
            self._output_dir = self._ref_config["output_dir"]
            os.makedirs(self._output_dir, exist_ok=True)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        # Collect per-request metadata BEFORE forward
        total_tokens = scheduler_output.total_num_scheduled_tokens
        num_scheduled = scheduler_output.num_scheduled_tokens  # dict[req_id, int]
        req_ids = list(num_scheduled.keys())
        num_per_req = list(num_scheduled.values())

        # Compute per-request token ranges
        computed_map: dict = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            computed_map[new_req.req_id] = new_req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            computed_map[rid] = cached.num_computed_tokens[i]

        # Run forward
        result = super().execute_model(scheduler_output)

        if self._ref_config is None or total_tokens == 0:
            return result

        # Post-forward: slice buffers and save
        self._save_step(req_ids, num_per_req, computed_map)
        self._step += 1
        return result

    def _save_step(
        self,
        req_ids: list[str],
        num_per_req: list[int],
        computed_map: dict[str, int],
    ) -> None:
        model = self.model_runner.model
        if not hasattr(model, "get_ref_buffers"):
            return

        bufs = model.get_ref_buffers()
        if not bufs:
            return

        # Normalize request IDs (strip vLLM UUID suffix)
        import re
        _suffix_re = re.compile(r"-[0-9a-f]{8}$")

        offset = 0
        for i, rid in enumerate(req_ids):
            n = num_per_req[i]
            pre_computed = computed_map.get(rid, 0)
            t_start = pre_computed
            t_end = pre_computed + n
            norm_id = _suffix_re.sub("", rid)

            req_dir = os.path.join(self._output_dir, norm_id)
            os.makedirs(req_dir, exist_ok=True)

            for name, buf in bufs.items():
                # Parse hook name: "resid_pre_L0" → hook=resid_pre
                # or "embed" → hook=embed
                if "_L" in name:
                    hook_name = name.rsplit("_L", 1)[0]
                else:
                    hook_name = name

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
