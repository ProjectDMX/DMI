"""CompareWorker: DMXGPUWorker that also saves D2D capture buffers to disk.

Runs the compare model (qwen3_compare / gpt2_compare) which has both
HookPoints (ring::producer) and .copy_() capture in the same compiled graph.
After each forward, saves the .copy_() buffers to disk. The ring transport
writes to ClickHouse. Compare disk vs ClickHouse for transport correctness.
"""
import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from integration.vllm_adapter import DMXGPUWorker
from monitoring.ring_transport import HOOK_TYPE_TO_SHORT_NAME, HookSpec


_COMPARE_MODEL_VARIANTS = {
    "ApertusCompareForCausalLM": (
        "apertus_compare:ApertusCompareForCausalLM"
    ),
    "Ernie4_5CompareForCausalLM": (
        "ernie45_compare:Ernie4_5CompareForCausalLM"
    ),
    "FalconH1CompareForCausalLM": (
        "falcon_h1_compare:FalconH1CompareForCausalLM"
    ),
    "Gemma3CompareForCausalLM": (
        "gemma3_compare:Gemma3CompareForCausalLM"
    ),
    "GPT2CompareForCausalLM": "gpt2_compare:GPT2CompareForCausalLM",
    "GptOssCompareForCausalLM": (
        "gpt_oss_compare:GptOssCompareForCausalLM"
    ),
    "GlmMoeDsaCompareForCausalLM": (
        "glm_moe_dsa_compare:GlmMoeDsaCompareForCausalLM"
    ),
    "GraniteCompareForCausalLM": (
        "granite_compare:GraniteCompareForCausalLM"
    ),
    "JambaCompareForCausalLM": (
        "jamba_compare:JambaCompareForCausalLM"
    ),
    "Lfm2CompareForCausalLM": (
        "lfm2_compare:Lfm2CompareForCausalLM"
    ),
    "Llama4CompareForConditionalGeneration": (
        "mllama4_compare:Llama4CompareForConditionalGeneration"
    ),
    "MiniCPMCompareForCausalLM": (
        "minicpm_compare:MiniCPMCompareForCausalLM"
    ),
    "MiniMaxM2CompareForCausalLM": (
        "minimax_m2_compare:MiniMaxM2CompareForCausalLM"
    ),
    "Qwen2MoeCompareForCausalLM": (
        "qwen2_moe_compare:Qwen2MoeCompareForCausalLM"
    ),
    "Qwen3CompareForCausalLM": "qwen3_compare:Qwen3CompareForCausalLM",
    "Qwen3MoeCompareForCausalLM": (
        "qwen3_moe_compare:Qwen3MoeCompareForCausalLM"
    ),
    "Qwen3_5CompareForConditionalGeneration": (
        "qwen3_5_compare:Qwen3_5CompareForConditionalGeneration"
    ),
    "LlamaCompareForCausalLM": "llama_compare:LlamaCompareForCausalLM",
    "MistralCompareForCausalLM": (
        "mistral_compare:MistralCompareForCausalLM"
    ),
    "Olmo3CompareForCausalLM": (
        "olmo3_compare:Olmo3CompareForCausalLM"
    ),
    "Phi3CompareForCausalLM": (
        "phi3_compare:Phi3CompareForCausalLM"
    ),
}


def _register_compare_model_variants() -> None:
    """Lazily register test-only compare models with an official wheel."""

    from vllm.model_executor.models import ModelRegistry

    registered = set(ModelRegistry.get_supported_archs())
    module_prefix = "vllm.model_executor.models."
    for architecture, target in _COMPARE_MODEL_VARIANTS.items():
        if architecture in registered:
            continue
        module_name, class_name = target.split(":", 1)
        ModelRegistry.register_model(
            architecture,
            f"{module_prefix}{module_name}:{class_name}",
        )


_register_compare_model_variants()


_ARCH_REMAP = {
    "ApertusForCausalLM": "ApertusCompareForCausalLM",
    "Ernie4_5ForCausalLM": "Ernie4_5CompareForCausalLM",
    "FalconH1ForCausalLM": "FalconH1CompareForCausalLM",
    "Gemma3ForCausalLM": "Gemma3CompareForCausalLM",
    "GPT2LMHeadModel": "GPT2CompareForCausalLM",
    "GptOssForCausalLM": "GptOssCompareForCausalLM",
    "GlmMoeDsaForCausalLM": "GlmMoeDsaCompareForCausalLM",
    "GraniteForCausalLM": "GraniteCompareForCausalLM",
    "JambaForCausalLM": "JambaCompareForCausalLM",
    "Lfm2ForCausalLM": "Lfm2CompareForCausalLM",
    "Llama4ForConditionalGeneration": "Llama4CompareForConditionalGeneration",
    "MiniCPMForCausalLM": "MiniCPMCompareForCausalLM",
    "MiniMaxM2ForCausalLM": "MiniMaxM2CompareForCausalLM",
    "Qwen2MoeForCausalLM": "Qwen2MoeCompareForCausalLM",
    "Qwen3ForCausalLM": "Qwen3CompareForCausalLM",
    "Qwen3MoeForCausalLM": "Qwen3MoeCompareForCausalLM",
    "Qwen3_5ForConditionalGeneration": "Qwen3_5CompareForConditionalGeneration",
    "LlamaForCausalLM": "LlamaCompareForCausalLM",
    "MistralForCausalLM": "MistralCompareForCausalLM",
    "Olmo3ForCausalLM": "Olmo3CompareForCausalLM",
    "Phi3ForCausalLM": "Phi3CompareForCausalLM",
}

# Hook names that are TP-sharded
_TP_SHARDED_HOOKS = {"q", "k", "v", "z", "mlp_post"}

_LAYER_BUFFER_SUFFIX = re.compile(r"_L\d+$")


def _buffer_hook_name(buffer_name: str) -> str:
    """Return the canonical hook short name for one compare buffer."""

    return _LAYER_BUFFER_SUFFIX.sub("", buffer_name)


def _filter_ref_buffers(
    buffers: Mapping[str, torch.Tensor],
    active_specs: Iterable[HookSpec],
) -> dict[str, torch.Tensor]:
    """Keep exactly the buffers promised by the active DMI hook contract.

    Compare models allocate copies for every hook they can expose.  A matrix
    cell may intentionally select a smaller set, so the reference oracle must
    be derived from the adapter's bound ``active_specs`` rather than from that
    larger capability inventory.
    """

    selected_names: set[str] = set()
    for spec in active_specs:
        try:
            selected_names.add(HOOK_TYPE_TO_SHORT_NAME[spec.hook_type])
        except KeyError as exc:
            raise RuntimeError(
                f"CompareWorker received unknown active hook type {spec.hook_type}"
            ) from exc

    available_names = {_buffer_hook_name(name) for name in buffers}
    unknown_names = available_names - set(HOOK_TYPE_TO_SHORT_NAME.values())
    if unknown_names:
        raise RuntimeError(
            "CompareWorker received non-canonical reference buffers: "
            f"{sorted(unknown_names)}"
        )

    missing_names = selected_names - available_names
    if missing_names:
        raise RuntimeError(
            "CompareWorker has no reference buffers for active hooks: "
            f"{sorted(missing_names)}"
        )

    return {
        name: buffer
        for name, buffer in buffers.items()
        if _buffer_hook_name(name) in selected_names
    }


class CompareWorker(DMXGPUWorker):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compare_output_dir: str = ""
        self._compare_step: int = 0

    def load_model(self, *args, **kwargs) -> None:
        # Remap to compare variant
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs

        super().load_model(*args, **kwargs)

        # Allocate compare buffers. Use max_num_batched_tokens (not
        # E2E_REF_MAX_LEN) because profiling runs the model with that many
        # tokens and the .copy_() calls are unconditional.
        max_len = self.vllm_config.scheduler_config.max_num_batched_tokens
        model = self.model_runner.model
        if hasattr(model, "allocate_compare_buffers"):
            model.allocate_compare_buffers(max_len, self.vllm_config)

        self._compare_output_dir = os.environ.get("COMPARE_OUTPUT_DIR", "")
        if self._compare_output_dir:
            os.makedirs(self._compare_output_dir, exist_ok=True)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        total_tokens = scheduler_output.total_num_scheduled_tokens
        should_save = bool(self._compare_output_dir and total_tokens > 0)
        adaptor = self.adaptor
        if should_save:
            if adaptor is None or adaptor.transport is None:
                raise RuntimeError(
                    "CompareWorker requires active DMI transport"
                )
            step_before = adaptor._step_counter

        # Run forward (DMXGPUWorker.execute_model handles ring transport)
        result = super().execute_model(scheduler_output)

        if should_save:
            if adaptor._step_counter != step_before + 1:
                raise RuntimeError(
                    "CompareWorker did not observe exactly one committed DMI step"
                )

            transport = adaptor.transport
            req_ids = list(transport._current_req_ids or ())
            token_ranges = list(transport._current_token_ranges or ())
            dim0_offsets = list(transport._current_dim0_offsets or ())
            if (
                not transport._current_flattened
                or not req_ids
                or len(req_ids) != len(token_ranges)
                or len(req_ids) != len(dim0_offsets)
                or len(set(req_ids)) != len(req_ids)
            ):
                raise RuntimeError(
                    "CompareWorker received an invalid committed DMI layout"
                )

            num_per_req: list[int] = []
            computed_map: dict[str, int] = {}
            expected_offset = 0
            for req_id, token_range, offset in zip(
                req_ids, token_ranges, dim0_offsets
            ):
                start, end = (int(value) for value in token_range)
                if (
                    not isinstance(req_id, str)
                    or start < 0
                    or end <= start
                    or int(offset) != expected_offset
                ):
                    raise RuntimeError(
                        "CompareWorker received an invalid committed DMI range"
                    )
                count = end - start
                num_per_req.append(count)
                computed_map[req_id] = start
                expected_offset += count

            if expected_offset != int(total_tokens):
                raise RuntimeError(
                    "CompareWorker committed DMI rows do not match scheduler total"
                )

            self._save_compare_step(req_ids, num_per_req, computed_map)
            self._compare_step += 1

        return result

    def _save_compare_step(
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
        adaptor = self.adaptor
        if adaptor is None:
            raise RuntimeError("CompareWorker requires an active DMI adapter")
        bufs = _filter_ref_buffers(bufs, adaptor.active_specs)
        if not bufs:
            # A TP/PP rank can legitimately own no hooks after placement
            # filtering (for example rank 1 with an unsharded-only selection).
            return

        _suffix_re = re.compile(r"-[0-9a-f]{8}$")
        tp_rank = self._dmx_tp_rank
        tp_size = getattr(self, '_dmx_tp_size', 1)
        # Get tp_size from the group if available
        try:
            from vllm.distributed.parallel_state import get_tp_group
            tp_size = get_tp_group().world_size
        except Exception:
            pass

        sr = f"_SR{tp_rank}" if tp_size > 1 else ""

        offset = 0
        for i, rid in enumerate(req_ids):
            n = num_per_req[i]
            pre_computed = computed_map.get(rid, 0)
            t_start = pre_computed
            t_end = pre_computed + n
            norm_id = _suffix_re.sub("", rid)

            req_dir = os.path.join(self._compare_output_dir, norm_id)
            os.makedirs(req_dir, exist_ok=True)

            for name, buf in bufs.items():
                if "_L" in name:
                    hook_name = name.rsplit("_L", 1)[0]
                else:
                    hook_name = name

                is_sharded = hook_name in _TP_SHARDED_HOOKS
                if tp_rank != 0 and not is_sharded:
                    continue

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
