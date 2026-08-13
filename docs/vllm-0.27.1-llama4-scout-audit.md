# Llama 4 Scout decoder lite support audit for vLLM 0.27.1

This record covers the local decoder-only implementation gate for Llama 4
Scout. It does not claim runtime support until the pinned 4×H100 matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | public wrapper `vllm/model_executor/models/mllama4.py`; text decoder `llama4.py` |
| DMI integration | branch `dmi-v0.27.1-llama4-decoder`; commit `aab8586c8abe06d4506387a7534a6ae516bf776f` |
| Production checkpoint | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, revision `c2b440bc2b8c784ad310291d035b8550a771f24f` |
| Weight contract | gated BF16 checkpoint, about 109B total parameters / 17B active; TP4 is the minimum declared H100 cell |
| Decoder contract | 48 all-MoE layers; 16 routed experts plus shared expert; top-1 sigmoid routing; 3 chunked local RoPE layers followed by 1 global NoPE layer; Q/K norm and NoPE temperature tuning |
| Multimodal tier | public text and image inputs remain owned by upstream; DMI exports the fused language-decoder inputs and layers only |
| Explicitly excluded from DMI | vision encoder, pixel-shuffle path, multimodal projector, image patch tensors, encoder attention, attention weights, KV cache, and expert-local dispatch/combine tensors |
| Explicitly untested | authorized production weight loading, public text/image parity, storage values, PP/DP/EP/SP/EPLB, serving, speculative/Eagle, LoRA, quantization, and Maverick/other Llama 4 variants |

The gated production tree exposes the pinned revision but not `config.json` to
the local unauthenticated environment. The CPU metadata contract was also
loaded through the public RedHatAI Scout mirror; the exact gated config and
weights remain an H100 prerequisite, not assumed evidence.

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | The monitored public class subclasses `Llama4ForConditionalGeneration` and inherits its processor, encoder, projector, multimodal mapping, forward, and complete streaming weight loader. Only the already-constructed native language model tree is instrumented. |
| M04 | source verified; runtime pending | Disabled hooks delegate to upstream attention, MoE, and decoder-layer forwards. Enabled hooks preserve local-RoPE/NoPE attention, Q/K norm, temperature tuning, fused residual norms, shared+routed expert output, and logits. Public text and deterministic-image H100 comparison remains required. |
| M05-M07 | preserved, unclaimed | Outer multimodal and inner Eagle/PP contracts remain upstream-owned. The lite config requires TP4 because BF16 weights do not fit one H100, and rejects PP/DP/EP/SP/EPLB rather than inferring them. |
| M08 | adapted, CPU verified | All 48 MoE layers expose replicated router logits and the authoritative custom-router top-1 IDs/weights. The shared expert remains inside upstream `FusedMoE` and is not falsely presented as a routed expert-local tensor. |
| M09-M10 | unchanged, runtime pending | Public wrapper weight renaming, streaming language-model load, fused expert mapping, QKV packing, and vision/projector loading are inherited unchanged. The gated shards require authorized H100 evidence. |
| M11-M13 | adapted, CPU verified | The decoder manifest contains token IDs, fused decoder embeddings, 14 per-layer families, final residual/norm, and logits: 677 families. Compare buffers cover every family at TP4 shapes. Q/K/V are raw pre-RoPE observations and Z is pre-output projection. |
| M14-M15 | bounded | The adapter accepts the exact Scout geometry/schedule and BF16 TP4 cell. Vision/projector tensors and all non-Scout Llama 4 variants fail closed or remain outside the manifest. Nested `text_config` now drives DMI model shape and PP inventory for multimodal wrappers. |
| R01-R03 | verified | vLLM maps the public architecture to `mllama4:Llama4ForConditionalGeneration`; DMI remaps it to `mllama4_p:Llama4PForConditionalGeneration`. Separate decoder-aware compare registration is present. |
| R04-R07 | CPU verified | Official-wheel lazy targets resolve without eager model import. The wrapper delegates `get_hook_specs(model_wide=True)` only to the language model; encoder/projector objects expose no DMI inventory. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `31 passed` in `tests/test_llama4_p_contract.py` |
| Full cumulative lite gate | GPT-OSS, Qwen3-MoE, Llama 4, version, matrix, compare-runner, and black-box contracts: `166 passed` |
| General multimodal black-box contracts | deterministic public image input is schema-validated and materialized only as `LLM.generate` public input |
| Pinned metadata | public Scout mirror resolves to `Llama4ForConditionalGeneration`, 48 layers, 16 experts and passes the normalized config contract |
| Lint/compile | new decoder, wrapper, compare model, and focused test pass ruff and Python compilation |
| H100 public case | generated as `public-llama4_scout-tp4-eager-graph`, including a deterministic image input |
| H100 storage cases | generated as `storage-llama4_scout-{eager,cudagraph}-tp4` with per-rank 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 TP4 pending`. Runtime support requires the
authorized pinned checkpoint to pass public text/image baseline comparison,
eager/graph decoder transport, ClickHouse exact-value/exact-once/tail checks,
and clean four-rank shutdown.
