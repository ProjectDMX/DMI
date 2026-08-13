# ERNIE 4.5 dense model support audit for vLLM 0.27.1

This agent-authored record supports the text-only dense
`Ernie4_5ForCausalLM` architecture in a bounded TP1 BF16 V1 offline cell. The
runtime-qualified checkpoint is Baidu's official 0.3B pretraining model. The
upstream implementation is a deliberately thin Llama subclass, but its
post-construction rotary and output-projection changes are part of the model
contract and are replayed explicitly by both DMI variants.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/ernie45.py`, `Ernie4_5ForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-ernie45`, commit `6f2cf03f68e411e89732f74ec8ef3d588735d2a7` |
| Production and storage checkpoint | `baidu/ERNIE-4.5-0.3B-PT`, revision `b565cf6caebdb7a1eadf00100857b1ed5e044f12` |
| Production shape | 360,748,032 BF16 parameters; hidden 1,024; intermediate 3,072; 18 layers; 16 query heads; 2 KV heads; head dimension 128; vocabulary 103,424; tied embeddings |
| Block contract | bias-free causal Llama block with SiLU gated MLP; output projection has no bias and returns its bias separately; non-NeoX rotary layout |
| RoPE contract | default RoPE, theta 500,000 |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, all 203 declared hook families, bounded 128-token runtime |
| Excluded | Other ERNIE sizes, ERNIE MoE/VL architectures, biased or untied configurations, alternate activation/layer schedules/RoPE, TP>1, PP/DP/EP/SP, V2, serve/async, speculative, Eagle, LoRA, quantization, pooling/task heads, attention weights/cache internals, and contexts beyond the bounded runtime cell |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | DMI reuses the audited hooked Llama computation while preserving ERNIE's explicit head dimension, non-NeoX rotary layout, bias-free output projection, causal attention, SiLU MLP, and residual order. Separate upstream and monitored eager/graph executions on the frozen checkpoint close the runtime path. |
| M05-M09 | preserved but excluded from the support claim | The monitored class preserves the upstream Llama loader, packed-module declarations, mapper, embedding interface, compilation dynamic dimensions, and inherited intermediate/parallel/speculative interfaces. Those broader execution modes remain outside this verdict. |
| M10 | adapted-verified for official HF weights | The official safetensors checkpoint loaded in upstream qualification, public baseline/monitored, and storage runs. Tied embeddings and the inherited Llama weight mapper/loader are unchanged. |
| M11-M13 | verified for TP1 | Each layer exposes completed residual input, normalized attention input, pre-RoPE Q/K/V, pre-output-projection Z, raw attention output, completed mid-residual, normalized MLP input, post-activation MLP width, and raw MLP output. Five global families expose token IDs, embeddings, final residual, final norm, and logits. Eighteen layers therefore expose 203 truthful families. |
| M14 | fail-closed and bounded | Construction requires `model_type=ernie4_5`, SiLU, causal bias-free blocks, tied embeddings, no mixed/sliding layer schedule, unit logits scaling, an explicit positive head dimension, and exact default RoPE theta 500,000. Semantically different known branches fail before model construction. |
| M15 | verified with an ordered dense manifest | Hook specs retain the inherited Llama firing order and do not claim attention weights, KV-cache state, gates, experts, recurrent state, or other unavailable internals. |
| R01-R03 | verified | Upstream resolves `Ernie4_5ForCausalLM` to `ernie45:Ernie4_5ForCausalLM`; DMI remaps it one-to-one to `ernie45_p:Ernie4_5PForCausalLM`. `Ernie4_5CompareForCausalLM` has a separate test-only registry entry. |
| R04-R07 | verified | Runtime remap, lazy registration, compare-worker registration, config rejection, upstream post-init replay, hook inventory, public aliases, and bounded release cells are locked by focused contracts. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream production qualification | The unmodified official 0.3B checkpoint loaded and generated successfully in eager and default CUDA-graph modes before DMI was enabled. |
| Focused model and release contracts | The initial ERNIE-focused selection passed 46/46. The final cross-model focused sweep passed 363/363 in the pinned vLLM 0.27.1 environment. It covers remap/registry behavior, fail-closed configuration branches, exact upstream post-init replay, manifest inventory/order, aliases, storage runners, and release-matrix bounds. |
| Production public eager+graph | 2/2 full API-only tests passed in 84.06 s. Separate baseline and monitored processes had zero strict and zero ambiguity mismatches across tokens, text, finish reasons, request attribution, reverse-batch relations, generated metamorphic cases, and public decision logprobs. |
| Reduced eager full-hook transport | 1,624/1,624 independent same-graph D2D reference rows were byte-identical to ClickHouse rows. |
| Reduced CUDA-graph full-hook transport | 1,624/1,624 independent same-graph D2D reference rows were byte-identical to ClickHouse rows. |
| Production eager full-hook transport | 16,240/16,240 independent same-graph D2D reference rows were byte-identical to ClickHouse rows. The eight requests reached EOS after nine generated tokens each, below the configured 20-token ceiling. |
| Production CUDA-graph full-hook transport | 16,240/16,240 independent same-graph D2D reference rows were byte-identical to ClickHouse rows, with the same deterministic EOS behavior. |
| Lifecycle | Accepted runs flushed monitoring, shut down vLLM cleanly, removed only capture-scoped ClickHouse rows, and left no residual GPU allocation. |

The public oracle uses only vLLM's offline API and compares separate baseline
and monitored processes. The storage oracle is independent: one execution
contains both DMI ring producers and preallocated same-graph D2D copies, then
compares every retained tensor and request/token range against ClickHouse byte
for byte.

## Implementation decision

Both DMI classes inherit the existing Llama monitoring implementation because
the upstream ERNIE class inherits the same Llama implementation without a new
forward. Inheritance alone is not the compatibility argument: after base model
construction, DMI explicitly repeats upstream's `is_neox_style=False`, removes
the attention output-projection bias, and enables `skip_bias_add`. Focused tests
lock those assignments so a future upstream subclass change cannot silently
fall through the alias.

## Support decision

`Ernie4_5ForCausalLM` is supported for the named
`baidu/ERNIE-4.5-0.3B-PT` TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact integration commit above. Other ERNIE
checkpoints and every MoE, multimodal, parallel, quantized, serving,
speculative, alternate-task, or different configuration path remain untested
rather than implicitly supported.
