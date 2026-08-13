# vLLM model coverage roadmap

This document defines the model backlog for the vLLM 0.27.1 port. It is a
versioned discovery snapshot, not a support claim.

## Snapshot and scope

| Field | Value |
| --- | --- |
| Snapshot time | 2026-08-13 03:32 UTC |
| Current upstream release | [`v0.27.1`](https://github.com/vllm-project/vllm/releases/tag/v0.27.1) |
| Target upstream commit | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Current DMI supported port | root `ed43791eedac99c6fb18e24af8253e780bd56a54` / integration `fdfe631884ae318050ce371e472c1135f317cfa2` / upstream vLLM `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Registry source | [`v0.27.1` model registry](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/registry.py) |
| Runtime registry evidence | [`vllm-0.27.1-runtime-registry.json`](vllm-0.27.1-runtime-registry.json), official wheel, 41/41 lazy classes resolved |
| Upstream test exemplars | [`v0.27.1` test registry](https://github.com/vllm-project/vllm/blob/v0.27.1/tests/models/registry.py) |
| Popularity/config source | timestamped [Hugging Face model API](https://huggingface.co/docs/hub/api) lookups of official publisher checkpoints |

"Mainstream" here means a curated general-purpose generation family from an
official publisher that has meaningful ecosystem adoption or represents a major
current open-model release. Download counts are only a moving prioritization
signal. Community quantizations, duplicate sizes, pure embedding/reward models,
audio-only models, and draft-only architectures are excluded. "Open model" is
not a license conclusion; checkpoint access and deployment terms must be checked
per family.

The inventory was generated with the `dmi-port-vllm` skill's
`catalog_vllm_models.py`. It resolves each checkpoint's declared architecture
against the exact target registry and separately inspects DMI's static remap.
The source catalog was also replayed through the official `vllm==0.27.1` wheel
in a CPU-only process with Python 3.12.8 and PyTorch 2.13.0+cu130. The
repository verifier resolved all 41 lazy classes, including Kimi-K3 and
DeepSeek-V4 under `vllm.models.*`. This is stronger import evidence than parsing
the registry source, but remains upstream-static evidence rather than DMI
runtime support. Across the 41 representative checkpoints below:

- all 41 resolve in vLLM 0.27.1;
- 14 architectures have a DMI model remap on the current stacked expansion
  branch;
- 27 are unmapped by DMI;
- registry presence is upstream static evidence, not DMI runtime support.

Between vLLM 0.25.1 and 0.27.1, the combined text/multimodal registry adds eight
architectures, removes eight, and remaps two. In particular, Kimi-K3 first
appears in the newer registry, Qwen3.5 gains text-only architecture entries, and
OLMo3 no longer resolves through the OLMo2 implementation.

## Phase 0: validate and re-port existing DMI variants

These models gate all expansion work. The 0.25.1 status comes from
[`vllm-0.25.1-compatibility-audit.md`](vllm-0.25.1-compatibility-audit.md). Every
row must be re-audited after the worker/runner port to 0.27.1.

| Representative checkpoint | architecture | vLLM 0.27.1 implementation | DMI 0.25.1 evidence | DMI 0.27.1 status |
| --- | --- | --- | --- | --- |
| `openai-community/gpt2` | `GPT2LMHeadModel` | `gpt2:GPT2LMHeadModel` | supported, eager+graph+storage | supported; TP1 public and TP1/TP2 storage value cells passed the final matrix |
| `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen2ForCausalLM` | `qwen2:Qwen2ForCausalLM` | supported, eager+graph | supported; TP1 public eager+graph passed the final matrix |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` | `Qwen2MoeForCausalLM` | `qwen2_moe:Qwen2MoeForCausalLM` | supported, TP2 eager+graph+storage | supported; TP2 public/storage value cells passed; EP remains excluded |
| `Qwen/Qwen3-8B` | `Qwen3ForCausalLM` | `qwen3:Qwen3ForCausalLM` | supported via 0.6B checkpoint, eager+graph+storage | supported; TP1 public and TP1/TP2 storage value cells passed the final matrix |
| `meta-llama/Llama-3.2-1B-Instruct` | `LlamaForCausalLM` | `llama:LlamaForCausalLM` | supported via 3.1-8B checkpoint, TP2 eager+graph+storage | supported; Llama-3.1-8B TP2 public/storage value cells passed the final matrix |

## Phase 1: bounded single-GPU families

Start with small dense and hybrid checkpoints that fit a 24 GiB accelerator.
This phase deliberately includes different implementation contracts instead of
adding only Llama-like aliases.

| Order | Representative checkpoint | downloads snapshot | architecture | vLLM 0.27.1 implementation | contract class |
| ---: | --- | ---: | --- | --- | --- |
| 1 | `google/gemma-3-1b-it` | 5,088,478 | `Gemma3ForCausalLM` | `gemma3:Gemma3ForCausalLM` | dense, Gemma norm/attention |
| 2 | `microsoft/Phi-3.5-mini-instruct` | 1,021,348 | `Phi3ForCausalLM` | `phi3:Phi3ForCausalLM` | Llama subclass with distinct packing |
| 3 | `mistralai/Mistral-7B-Instruct-v0.2` | 1,169,540 | `MistralForCausalLM` | `mistral:MistralForCausalLM` | Llama-derived dense, sliding/scale branches |
| 4 | `tiiuae/Falcon-H1-0.5B-Instruct` | 23,911 | `FalconH1ForCausalLM` | `falcon_h1:FalconH1ForCausalLM` | attention/Mamba hybrid |
| 5 | `LiquidAI/LFM2.5-1.2B-Instruct` | 510,295 | `Lfm2ForCausalLM` | `lfm2:Lfm2ForCausalLM` | attention/short-convolution hybrid |
| 6 | `ai21labs/AI21-Jamba2-3B` | 26,935 | `JambaForCausalLM` | `jamba:JambaForCausalLM` | dense attention/Mamba hybrid; MoE configs excluded |
| 7 | `ibm-granite/granite-4.1-8b` | 4,155,476 | `GraniteForCausalLM` | `granite:GraniteForCausalLM` | dense with four model-specific scalar operations; runtime-qualified on the 3B sibling |
| 8 | `allenai/Olmo-3-7B-Instruct` | 425,421 | `Olmo3ForCausalLM` | `olmo3:Olmo3ForCausalLM` | dense; implementation changed since 0.25.1 |
| 9 | `swiss-ai/Apertus-8B-Instruct-2509` | 472,853 | `ApertusForCausalLM` | `apertus:ApertusForCausalLM` | dense with model-specific attention/norm |
| 10 | `baidu/ERNIE-4.5-0.3B-PT` | 22,897 | `Ernie4_5ForCausalLM` | `ernie45:Ernie4_5ForCausalLM` | modified Llama subclass |
| 11 | `openbmb/MiniCPM4.1-8B` | 48,765 | `MiniCPMForCausalLM` | `minicpm:MiniCPMForCausalLM` | dense, remote-code/version constraints |

Current expansion status: Gemma 3 is `experimental` for the TP1 V1 offline
eager/default-graph cells at integration commit `b71a55ae4227`. The
[`Gemma 3 audit`](vllm-0.27.1-gemma3-audit.md) records strict public parity,
complete hook storage, and byte-identical eager/graph transport on a six-layer
tiny fixture. The official `google/gemma-3-1b-it` checkpoint was gated for the
available account, so the real-checkpoint completion gate remains open and the
row is not yet `supported`.

Phi-3 is `supported` for the bounded TP1 BF16 V1 offline eager/default-graph
cell at integration commit `b4965771d8a4`. The
[`Phi-3 audit`](vllm-0.27.1-phi3-audit.md) records strict public tests on the
official `microsoft/Phi-3.5-mini-instruct` checkpoint plus byte-identical
eager/graph full-hook transport on a tiny fixture. TP>1, PP, quantization,
serving, speculative, and non-text task variants remain excluded.

Mistral is `supported` for the bounded TP1 BF16 V1 offline eager/default-graph
cell at integration commit `8c89ec33bbc5`. The
[`Mistral audit`](vllm-0.27.1-mistral-audit.md) records strict public parity on
the official `mistralai/Mistral-7B-Instruct-v0.2` checkpoint and byte-identical
eager/graph full-hook transport on a qualified small fixture. Configurations
that enable Llama-4 attention scaling or adaptive conditional RMS normalization,
alternate loaders, TP>1, quantization, serving, speculative, and non-text
variants remain excluded.

Falcon-H1 is `supported` for the bounded TP1 BF16 V1 offline
eager/default-graph cell at integration commit `e69be3978552`. The
[`Falcon-H1 audit`](vllm-0.27.1-falcon-h1-audit.md) records public validation on
the official 0.5B checkpoint, byte-identical eager/graph full-hook transport on
the official Tiny checkpoint, and the bounded SSM branch capability manifest.
Internal Mamba state, TP>1, prefix caching, quantization, serving, speculative,
and non-text variants remain excluded.

LFM2 is `supported` for the bounded TP1 BF16 V1 offline eager/default-graph
cell at integration commit `229014dd9692`. The
[`LFM2 audit`](vllm-0.27.1-lfm2-audit.md) records strict public parity on the
official `LiquidAI/LFM2.5-1.2B-Instruct` checkpoint and byte-identical
eager/graph full-hook transport on a qualified two-layer hybrid fixture. The
manifest distinguishes ten production convolution layers from six attention
layers. Internal convolution/cache state, LFM2-MoE/VL, TP>1, prefix caching,
quantization, serving, and speculative modes remain excluded.

Jamba is `supported` only for the dense official `ai21labs/AI21-Jamba2-3B`
configuration in the bounded TP1 BF16 V1 offline eager/default-graph cell at
integration commit `b4556982f63d`. The
[`Jamba audit`](vllm-0.27.1-jamba-audit.md) records public API parity and
byte-identical eager/graph storage on the same production checkpoint. Its
truthful 263-family manifest distinguishes 26 Mamba1 layers from attention
layers 7 and 21. The loader fails closed if any layer has more than one expert;
Jamba-MoE, other Jamba schedules, internal recurrent/cache state, TP>1, prefix
caching, quantization, serving, and speculative modes remain excluded.

Granite is `supported` for the official
`ibm-granite/granite-4.1-3b` checkpoint in the bounded TP1 BF16 V1 offline
eager/default-graph cell at integration commit `dca631b19ec7`. The
[`Granite audit`](vllm-0.27.1-granite-audit.md) records public API parity and
byte-identical eager/graph storage on that same production checkpoint. Its
485-family manifest preserves Granite's attention, embedding, residual, and
logits scaling and adds the exact post-activation MLP boundary. Granite 4.1 8B
and 30B share the statically audited dense path but remain runtime-unqualified;
Granite MoE/hybrid/multimodal models, TP>1, prefix caching, quantization,
serving, and speculative modes remain excluded.

OLMo 3 is `supported` for the official `allenai/Olmo-3-7B-Instruct`
checkpoint in the bounded TP1 BF16 V1 offline eager/default-graph cell at
integration commit `9f6a5c762bc5`. The
[`OLMo 3 audit`](vllm-0.27.1-olmo3-audit.md) records public API parity and
byte-identical eager/graph storage on that same production checkpoint. Its
389-family manifest follows OLMo 3's post-norm residual order and observes Q/K
after Q/K normalization but before the layer-type-specific RoPE. Available tiny
fixtures do not independently cover the production sliding/full schedule plus
both default and YaRN RoPE branches, so they are not used to generalize the
runtime verdict. Other checkpoints, TP>1, prefix caching, quantization, serving,
speculative, and other OLMo-family architectures remain excluded.

Apertus is `supported` for the official
`swiss-ai/Apertus-8B-Instruct-2509` checkpoint in the bounded TP1 BF16 V1
offline eager/default-graph cell at integration commit `3d5e5fb6b87f`. The
[`Apertus audit`](vllm-0.27.1-apertus-audit.md) records strict public API parity
and byte-identical eager/graph storage on that same production checkpoint. Its
389-family manifest follows fused pre-norm residual arithmetic, observes Q/K
after per-head normalization and before RoPE, and exposes the xIELU
post-activation boundary. The verdict covers vLLM's Python xIELU fallback;
Apertus v1.1/70B and other config branches, the optional fused xIELU extension,
TP>1, prefix caching, quantization, serving, and speculative modes remain
excluded.

For each row, use an upstream tiny/random fixture for fast focused tests when
available, then require one real-checkpoint baseline/monitored run before moving
beyond `static-only`.

## Phase 2: MoE, MLA, large, and quantization-dependent text models

These need two or three local GPUs, a quantized production checkpoint, an
upstream tiny fixture, or a larger external test cell. They must not inherit a
dense-model verdict.

| Representative checkpoint | downloads snapshot | architecture | vLLM 0.27.1 implementation | primary new contract |
| --- | ---: | --- | --- | --- |
| `Qwen/Qwen3-30B-A3B` | 2,481,310 | `Qwen3MoeForCausalLM` | `qwen3_moe:Qwen3MoeForCausalLM` | fused MoE, EP/TP routing |
| `deepseek-ai/DeepSeek-R1` | 8,523,653 | `DeepseekV3ForCausalLM` | `deepseek_v2:DeepseekV3ForCausalLM` | MLA, fused MoE, shared experts |
| `deepseek-ai/DeepSeek-V3.2` | 1,093,344 | `DeepseekV32ForCausalLM` | `deepseek_v2:DeepseekV3ForCausalLM` | same module, distinct architecture/config branches |
| `openai/gpt-oss-20b` | 7,892,919 | `GptOssForCausalLM` | `gpt_oss:GptOssForCausalLM` | MoE plus MXFP4 production weights |
| `zai-org/GLM-4.7-Flash` | 1,865,359 | `Glm4MoeLiteForCausalLM` | `glm4_moe_lite:Glm4MoeLiteForCausalLM` | MLA/MoE routing |
| `zai-org/GLM-5.2` | 2,517,575 | `GlmMoeDsaForCausalLM` | `deepseek_v2:GlmMoeDsaForCausalLM` | DeepSeek-derived DSA/MLA branch |
| `MiniMaxAI/MiniMax-M2.7` | 861,627 | `MiniMaxM2ForCausalLM` | `minimax_m2:MiniMaxM2ForCausalLM` | MoE, large expert set |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 701,704 | `MixtralForCausalLM` | `mixtral:MixtralForCausalLM` | MoE; upstream tiny fixture exists |
| `LiquidAI/LFM2.5-8B-A1B` | 169,971 | `Lfm2MoeForCausalLM` | `lfm2_moe:Lfm2MoeForCausalLM` | hybrid short-convolution plus MoE |
| `tencent/Hunyuan-A13B-Instruct` | 41,334 | `HunYuanMoEV1ForCausalLM` | `hunyuan_v1:HunYuanMoEV1ForCausalLM` | MoE, remote-code checkpoint |
| `baidu/ERNIE-4.5-21B-A3B-PT` | 40,021 | `Ernie4_5_MoeForCausalLM` | `ernie45_moe:Ernie4_5_MoeForCausalLM` | MoE counterpart to dense ERNIE |
| `XiaomiMiMo/MiMo-V2-Flash` | 89,530 | `MiMoV2FlashForCausalLM` | `mimo_v2:MiMoV2FlashForCausalLM` | MoE and model-specific attention |
| `ByteDance-Seed/Seed-OSS-36B-Instruct` | 41,334 | `SeedOssForCausalLM` | `seed_oss:SeedOssForCausalLM` | large dense/parallel ownership |
| `meituan-longcat/LongCat-Flash-Chat` | 52,519 | `LongcatFlashForCausalLM` | `longcat_flash:LongcatFlashForCausalLM` | MLA/MoE and remote code |
| `stepfun-ai/Step-3.5-Flash` | 128,850 | `Step3p5ForCausalLM` | `step3p5:Step3p5ForCausalLM` | large MoE and specialized routing |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 2,588,012 | `NemotronHForCausalLM` | `nemotron_h:NemotronHForCausalLM` | Mamba/MoE hybrid plus NVFP4 |
| `LGAI-EXAONE/EXAONE-4.5-33B` | 174,939 | `Exaone4_5_ForConditionalGeneration` | `exaone4_5:Exaone4_5_ForConditionalGeneration` | conditional-generation text model |

## Phase 3: multimodal and external-plugin implementations

Text-backbone testing is insufficient for these rows. Their support cells must
also cover processor inputs, multimodal batching, public outputs, graph behavior,
and modality-specific lifecycle paths. Implementations under `vllm.models.*`
also require exact plugin package and lazy-resolution evidence.

| Representative checkpoint | downloads snapshot | architecture | vLLM 0.27.1 implementation | reason for later phase |
| --- | ---: | --- | --- | --- |
| `Qwen/Qwen3.6-27B` | 6,577,318 | `Qwen3_5ForConditionalGeneration` | `qwen3_5:Qwen3_5ForConditionalGeneration` | current official checkpoint is conditional/multimodal |
| `Qwen/Qwen3-VL-8B-Instruct` | 4,538,226 | `Qwen3VLForConditionalGeneration` | `qwen3_vl:Qwen3VLForConditionalGeneration` | vision-language API |
| `google/gemma-4-E2B-it` | 3,780,950 | `Gemma4ForConditionalGeneration` | `gemma4_mm:Gemma4ForConditionalGeneration` | multimodal/MoE family |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 460,443 | `Llama4ForConditionalGeneration` | `mllama4:Llama4ForConditionalGeneration` | multimodal/MoE family |
| Ministral 3 3B Instruct (2512) | 322,387 | Mistral 3 conditional generation | `mistral3` module | multimodal conditional wrapper |
| `moonshotai/Kimi-K2.5` | 855,995 | `KimiK25ForConditionalGeneration` | `kimi_k25:KimiK25ForConditionalGeneration` | multimodal and very large |
| `moonshotai/Kimi-K3` | 1,565,484 | `KimiK3ForConditionalGeneration` | `vllm.models.kimi_k3:KimiK3ForConditionalGeneration` | added after 0.25.1; external plugin, multimodal |
| `deepseek-ai/DeepSeek-V4-Flash` | 2,283,943 | `DeepseekV4ForCausalLM` | `vllm.models.deepseek_v4:DeepseekV4ForCausalLM` | external plugin and very large |

## Per-model completion gate

A row moves to `supported` only after all applicable evidence exists:

1. M01-M14 and R01-R07 agent audit against the exact target implementation;
2. lazy import and official-wheel registry resolution;
3. constructor, forward, loader, hook-inventory, shape, and placement regressions;
4. separate-process baseline/monitored public-output parity on a tiny fixture and
   a real checkpoint;
5. eager and every claimed CUDA-graph mode;
6. transport and ClickHouse value/exact-once/tail/isolation checks;
7. every claimed TP/PP/EP topology, including rank ownership;
8. bounded capacity, failure, cancellation, and repeated-stop paths;
9. multimodal/serve/quant/speculative gates when those dimensions are claimed.

`unmapped`, `static-only`, `experimental`, and `untested` rows keep the roadmap
open. A registry gap or infeasible/inaccessible checkpoint must be recorded as an
explicit `unsupported` decision rather than silently removed.

## GPU scheduling rule

The two-GPU regression starts only when at least two physical GPUs satisfy all
of the following for three consecutive samples:

- no compute process is attached;
- less than 1 GiB device memory is in use;
- utilization remains below 10 percent.

Unrelated processes are never terminated. Once two cards qualify, pin the exact
indices with `CUDA_VISIBLE_DEVICES`/`E2E_GPUS`, run the 0.27.1 focused and public
black-box gates, then run the scoped ClickHouse TP=1/TP=2 transport matrix. Use a
unique capture ID and delete only rows belonging to that capture.
