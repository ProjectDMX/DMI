# MiniCPM 4.1 model support audit for vLLM 0.27.1

This agent-authored record supports the dense text-only
`MiniCPMForCausalLM` architecture in a bounded TP1 BF16 V1 offline cell. The
runtime-qualified checkpoint is OpenBMB's official MiniCPM 4.1 8B model. Its
configuration requires pinned repository code, but vLLM executes its native
`minicpm.py` implementation rather than the Hugging Face model implementation.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/minicpm.py`, `MiniCPMForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-minicpm`, commit `db9e7cfccc0bab6e52b311f6e7259d74af5feaa7` |
| Production and storage checkpoint | `openbmb/MiniCPM4.1-8B`, revision `3a8dfed9c79a45e07dbff95bcd49d792343fa1a3` |
| Focused fixture | `tiny-random/minicpm4.1`, revision `7f2c67170aba7da9673b66efd2e332a52f41d907` |
| Production shape | 8,185,253,888 BF16 parameters; hidden 4,096; intermediate 16,384; 32 layers; 32 query heads; 2 KV heads; head dimension 128; vocabulary 73,448; untied embeddings |
| Block contract | dense bias-free causal attention and SiLU gated MLP; embeddings scaled by 12; both attention and MLP residual branches scaled by `1.4 / sqrt(32)`; final normalized width divided by `4096 / 256` before logits |
| RoPE contract | LongRoPE, theta 10,000, original and maximum context 65,536, equal positive short/long factor vectors of length 64 |
| Remote-code boundary | `trust_remote_code=True` loads the frozen MiniCPM configuration under Transformers 5.15.0; runtime model execution remains vLLM's native implementation |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, all 389 declared hook families, bounded 128-token runtime |
| Excluded | Earlier MiniCPM/MiniCPM3 families, MoE, FatReLU and sparse-attention configurations, MiniCPM-V/O and other multimodal variants, Hugging Face remote model execution, TP>1, PP/DP/EP/SP, V2, serve/async, speculative, Eagle, LoRA, quantization, pooling/task heads, attention weights/cache internals, and contexts beyond the bounded runtime cell |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | The monitored implementation instruments the exact native MiniCPM module tree. Disabled hooks delegate to upstream forwards; enabled paths preserve scaled embeddings, per-layer normalization and residual order, LongRoPE placement, MLP arithmetic, final normalization, and pre-logit width scaling. Separate upstream and monitored eager/graph executions close the runtime path. |
| M05-M09 | preserved but excluded from the support claim | DMI preserves upstream packed modules, embedding interfaces, loader ownership, pipeline interfaces, compilation behavior, and inherited LoRA/Eagle declarations. Those broader modes remain outside this verdict. |
| M10 | adapted-verified for official HF weights | All four official safetensors shards loaded in unmodified upstream, public baseline/monitored, and storage runs. The untied production output head and tied tiny-fixture branch both loaded through the unchanged upstream loader. |
| M11-M13 | verified for TP1 | Each layer exposes completed residual input, normalized attention input, pre-RoPE Q/K/V, pre-output-projection Z, raw attention output, completed mid-residual, normalized MLP input, post-activation MLP width, and raw MLP output. Five global families expose token IDs, scaled embeddings, final residual, final norm, and logits. Thirty-two layers therefore expose 389 truthful families. |
| M14 | fail-closed and bounded | Construction requires `model_type=minicpm`, dense SiLU blocks, no sparse configuration, explicit tied/untied embedding ownership, positive embedding/depth/width scales, a valid head layout, context 65,536, and the audited LongRoPE contract. MoE, FatReLU, sparse, and incompatible RoPE/context branches fail before model construction. |
| M15 | verified with an ordered dense manifest | Hook specs follow actual firing order and do not claim attention weights, KV-cache state, MoE routes, sparse state, or other unavailable internals. |
| R01-R03 | verified | Upstream resolves `MiniCPMForCausalLM` to `minicpm:MiniCPMForCausalLM`; DMI remaps it one-to-one to `minicpm_p:MiniCPMPForCausalLM`. `MiniCPMCompareForCausalLM` has a separate test-only registry entry. |
| R04-R07 | verified | Runtime remap, lazy registration, compare-worker registration, remote-code propagation, config rejection, hook inventory, public aliases, and bounded release cells are locked by focused contracts. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream production qualification | The unmodified official 8B checkpoint loaded all four shards and generated successfully in eager and default CUDA-graph modes before DMI was enabled. |
| Upstream fixture qualification | The frozen two-layer random fixture loaded and generated identical eager/graph tokens under the same native vLLM implementation. |
| Focused model and release contracts | The MiniCPM-focused selection passed 51/51. The final cross-model focused sweep passed 384/384 in the pinned vLLM 0.27.1 environment. It covers remap/registry behavior, fail-closed configuration branches, exact scale and hook boundaries, manifest inventory/order, aliases, trust propagation, storage runners, and release-matrix bounds. |
| Fixture public eager+graph | 2/2 full API-only tests passed in 89.84 s with zero strict and zero ambiguity mismatches. |
| Production public eager+graph | 2/2 full API-only tests passed in 120.78 s. Separate baseline and monitored processes had zero strict and zero ambiguity mismatches across tokens, text, finish reasons, request attribution, reverse-batch relations, generated metamorphic cases, and public decision logprobs. |
| Fixture reduced eager+graph transport | 232/232 independent same-graph D2D reference rows were byte-identical to ClickHouse rows in each mode. |
| Production reduced eager+graph transport | 3,112/3,112 independent same-graph D2D reference rows were byte-identical to ClickHouse rows in each mode. |
| Production eager full-hook transport | 62,240/62,240 independent same-graph D2D reference rows were byte-identical to ClickHouse rows. Eight requests each generated the configured 20 tokens. |
| Production CUDA-graph full-hook transport | 62,240/62,240 independent same-graph D2D reference rows were byte-identical to ClickHouse rows. Eight requests each generated the configured 20 tokens. |
| Lifecycle | Accepted runs flushed monitoring, shut down vLLM cleanly, removed only capture-scoped ClickHouse rows, and left no residual GPU allocation. |

The public oracle uses only vLLM's offline API and compares separate baseline
and monitored processes. The storage oracle is independent: one execution
contains both DMI ring producers and preallocated same-graph D2D copies, then
compares every retained tensor and request/token range against ClickHouse byte
for byte.

## Implementation decision

The DMI variants subclass the native vLLM MiniCPM implementation and upgrade
the exact tree it constructs instead of translating Hugging Face remote model
code. Disabled layer, attention, and MLP hook paths delegate to upstream. The
enabled and compare paths replay upstream expression order, including the two
depth-scaled residual additions. The scaled embedding is observed after
`scale_emb`; Q/K/V are observed before LongRoPE; Z is observed before the output
projection; and logits are observed only after the native top-level forward has
applied `hidden_size / dim_model_base` width scaling.

The remote repository is trusted only to parse the frozen configuration. That
decision is explicit in both public and storage matrix cells, and the installed
Transformers version is part of the bounded support cell.

## Support decision

`MiniCPMForCausalLM` is supported for the named
`openbmb/MiniCPM4.1-8B` TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact integration commit above. Every earlier,
MoE, sparse, FatReLU, multimodal, distributed, quantized, serving, speculative,
alternate-task, remote-model-execution, or different configuration path remains
untested rather than implicitly supported.
