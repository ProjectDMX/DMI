# Mistral model support audit for vLLM 0.27.1

This agent-authored record supports the text-only `MistralForCausalLM`
architecture in a bounded TP1 BF16 V1 offline cell. It does not extend the
verdict to other Mistral architectures, alternate loaders, parallel modes, or
unexercised configuration branches.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/mistral.py`, `MistralForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-mistral`, commit `8c89ec33bbc5f08027fb1aab6b55c1287cf45f89` |
| Production checkpoint | `mistralai/Mistral-7B-Instruct-v0.2`, revision `63a8b081895390a26e140280378bc85ec8bce07a` |
| Focused fixture | `openaccess-ai-collective/tiny-mistral`, revision `2a2db3d57a46dd87d62c12b50bb8c34cc1afe126` |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, full canonical DMI hook inventory |
| Excluded | TP>1, PP/DP/EP/SP, V2, serve/async, speculative/Eagle, LoRA, quantization, `--load-format mistral`, embedding/classification tasks, remote-code variants, Mistral 3 and multimodal wrappers |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01 | verified | The exact target file defines independent Mistral MLP, attention, decoder, model, and causal-LM classes. DMI compares against that file rather than inferring compatibility from the family name. |
| M02-M04 | bounded and adapted-verified | For the named checkpoints, Mistral attention is numerically Llama attention because `llama_4_scaling` is absent, and the decoder is numerically the Llama decoder because `ada_rms_norm_t_cond` is absent. DMI reuses the audited hooked Llama math and fails closed if either branch is enabled. Tiny and production eager/graph public differential tests verify the accepted path. |
| M05-M09 | excluded where applicable | Pipeline, speculative, TP>1, EP/MoE, and quantized paths were not run and are not claimed. |
| M10 | adapted-verified for standard HF weights | Mistral preserves Llama split-weight packing but changes LoRA embedding declarations and defines a Mistral/consolidated name remapper with Q/K permutation. The DMI subclass keeps `embedding_modules = {}`, the upstream `mistral_mapping`, and the exact remapper. The official three-shard, 14.48 GB HF checkpoint loaded in both eager and graph tests. The alternate Mistral load format remains excluded because it was not run. |
| M11-M13 | verified for TP1 | The accepted math path inherits the audited Llama hook boundaries: eleven canonical families per layer and five model-wide families. The eight-layer fixture therefore exposes 93 families. Independent compare buffers and ClickHouse transport were byte-identical in eager and graph executions. Model-wide/module-free semantics are inherited from the exact Llama DMI implementation and locked by a method-identity contract. |
| M14 | bounded | Only text `MistralForCausalLM` is remapped. `Ministral3ForCausalLM`, Mistral Large 3, Mistral 3 conditional generation, Mixtral, Pixtral, embedding/task heads, remote-code, quantized, and alternate-loader implementations do not inherit this verdict. |
| M15 | verified for the claimed layer kind | Both exercised checkpoints contain homogeneous dense attention/MLP layers; no MoE, SSM, multimodal, or heterogeneous hook family is fabricated. |
| R01-R03 | verified | Upstream resolves `MistralForCausalLM` to `mistral:MistralForCausalLM`; DMI resolves it one-to-one to `mistral_p:MistralPForCausalLM`. It is intentionally not added to the broad Llama alias set because its loader and embedding declarations differ. |
| R04-R07 | verified | The 0.27.1 out-of-tree lazy-registration surface resolves the exported DMI and compare classes without changing the public architecture. Registry, remap, class-attribute, fail-closed, and release-matrix contracts pass in a CPU process. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Focused model and matrix contracts | 20/20 selected tests passed; the final expansion focused sweep passed 246/246. Tests reject broad Llama aliasing, loss of the Mistral mapping/remapper, changed hook inventory, acceptance of unaudited math branches, and unsafe fixture/memory settings. |
| Tiny public eager+graph | 2/2 full 12-case API-only tests passed in 85.82 s with strict baseline/monitored equality and no ambiguity or stability exemption. |
| Tiny eager full-hook transport | 14,880/14,880 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Tiny CUDA-graph full-hook transport | 14,880/14,880 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Production eager+graph | 2/2 full 12-case API-only tests passed in 124.35 s with strict baseline/monitored equality. |
| Lifecycle | No worker exception, force-kill, teardown warning, residual vLLM process, residual capture, or GPU allocation remained after the accepted runs. |

The storage oracle compares values generated inside the same execution; it does
not infer correctness from generation success or compare unrelated row counts.
The production public oracle separately proves real checkpoint loading and
observable transparency.

## Fixture qualification evidence

The first candidate,
`sanchit-gandhi/tiny-random-MistralForCausalLM-1-layer`, was rejected before
release use. Its four-dimensional attention heads are below the target
FlexAttention graph compiler's minimum embedding dimension of 16. In eager
mode, the tiny model also caused vLLM to derive enough KV blocks for a 53.17 GiB
physical-to-logical metadata allocation. Those are upstream fixture/runtime
prerequisite failures, not DMI mismatches.

The accepted fixture has 32-dimensional heads, eight layers, BF16 weights, and
the intended `MistralForCausalLM` architecture. Its release cells pin
`gpu_memory_utilization=0.2`, `max_model_len=128`, and
`max_num_batched_tokens=128`; CPU contracts prevent those bounds or the fixture
identity from drifting silently.

## Support decision

`MistralForCausalLM` is supported for the named TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact commits above. A checkpoint that enables
Llama-4 attention scaling, adaptive conditional RMS normalization, an alternate
loader, quantization, remote code, or a different architecture requires its own
audit and runtime cell. All excluded modes remain untested rather than
implicitly supported.
