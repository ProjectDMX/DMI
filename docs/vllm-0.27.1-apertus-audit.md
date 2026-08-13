# Apertus model support audit for vLLM 0.27.1

This agent-authored record supports the text-only dense
`ApertusForCausalLM` architecture in a bounded TP1 BF16 V1 offline cell. The
runtime-qualified checkpoint is the official Apertus 8B Instruct model. The
implementation is treated as its own contract rather than a Llama alias:
Apertus uses xIELU, per-head Q/K RMSNorm, fused pre-norm residual arithmetic,
and Llama-3-scaled RoPE with a model-specific theta.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/apertus.py`, `ApertusForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-apertus`, commit `3d5e5fb6b87febb0ed0b18ba58f2a417916ce186` |
| Production and storage checkpoint | `swiss-ai/Apertus-8B-Instruct-2509`, revision `b946d40447b2b597999b9c86d44bee0b452c919f` |
| Production shape | 8,053,338,240 BF16 parameters; hidden 4,096; intermediate 21,504; 32 layers; 32 query heads; 8 KV heads; vocabulary 131,072; untied embeddings |
| Block contract | bias-free fused pre-norm residual blocks; per-head Q/K RMSNorm; xIELU MLP with a single up projection |
| RoPE contract | Llama-3 scaling factor 8, high-frequency factor 4, low-frequency factor 1, original context 8,192, theta 12,000,000 |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, Python xIELU fallback, all 389 declared hook families, bounded 128-token runtime |
| Excluded | Apertus v1.1 and 70B configurations, optional fused xIELU extension, tied-embedding/default-RoPE branches, TP>1, PP/DP/EP/SP, V2, serve/async, speculative, Eagle, LoRA, quantization, prefix caching, pooling/task heads, attention weights/cache internals, and contexts beyond the bounded runtime cell |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | DMI follows the exact upstream xIELU, Q/K normalization, RoPE, fused add-RMSNorm, attention, and residual order. Q/K are observed after per-head RMSNorm and before RoPE; V is raw; Z precedes the output projection. Upstream and monitored eager/graph execution on the frozen checkpoint close the path. |
| M05-M09 | preserved but excluded from the support claim | The monitored class preserves the upstream loader, mapper, packed-module declarations, embedding interface, intermediate-tensor behavior, and inherited LoRA/Eagle/PP interfaces. The accepted cell deliberately excludes parallel, speculative, adapter, quantized, prefix-cache, serving, and alternate-task paths. |
| M10 | adapted-verified for official HF weights | The monitored wrapper preserves upstream `WeightsMapper`, packed QKV declarations, and `AutoWeightsLoader`. The official four-shard checkpoint loaded in upstream qualification, public baseline/monitored, and storage runs. The checkpoint's untied `lm_head` path is preserved. |
| M11-M13 | verified for TP1 | Every layer exposes completed residual input, normalized attention input, post-QK-norm/pre-RoPE Q/K, raw V, pre-output-projection Z, raw attention output, completed mid-residual, normalized MLP input, post-xIELU MLP width, and raw MLP output. Five global families expose token IDs, embeddings, final residual, final norm, and logits. Thirty-two layers therefore expose 389 truthful families. |
| M14 | fail-closed and bounded | Construction requires `model_type=apertus`, xIELU, pre-norm, Q/K normalization, causal bias-free blocks, untied embeddings, no mixed/sliding layer schedule, unit logits scaling, and the exact audited Llama-3 RoPE fields. Semantically different known configuration branches fail before model construction; same-contract size variants remain outside the runtime verdict. |
| M15 | verified with an ordered dense manifest | Hook specs follow the fused pre-norm firing order. The manifest does not invent attention weights, KV-cache state, gates, experts, recurrent state, or other families absent from this implementation. |
| R01-R03 | verified | Upstream resolves `ApertusForCausalLM` to `apertus:ApertusForCausalLM`; DMI remaps it one-to-one to `apertus_p:ApertusPForCausalLM`. `ApertusCompareForCausalLM` has a separate test-only registry entry. |
| R04-R07 | verified | Runtime remap, public aliases, compare-worker registration, fail-closed config checks, exact hook placement, manifest order, concrete compiled backbone construction, and matrix resource bounds are locked by focused contracts. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream production qualification | The unmodified official 8B checkpoint loaded and generated successfully in eager and default CUDA-graph modes before DMI was enabled. |
| Focused model and release contracts | Apertus contract plus release-matrix selections passed 45/45; the final cross-model focused sweep passed 346/346 in the pinned vLLM 0.27.1 / PyTorch 2.13.0+cu130 environment. They cover remap/registry, config rejection, fused pre-norm order, Q/K normalization and RoPE boundaries, xIELU placement, manifest inventory/order, model-wide specs, concrete compare construction, request/lifecycle behavior, and bounded release cells. |
| Production public eager+graph | 2/2 full 12-case API-only tests passed in 112.33 s. Separate baseline and monitored processes agreed exactly on tokens, text, finish reasons, request attribution, reverse-batch relations, generated metamorphic cases, and public decision logprobs in both modes. |
| Public-oracle mutation found during qualification | The first CUDA-graph attempt correctly rejected token, finish-reason, and decision-logprob drift. A second independent upstream baseline was stable. The cause was an extra `hidden_states + residual` node inserted only to observe pre-norm state; capture now reads the completed residual returned by fused add-RMSNorm. The final full eager/graph rerun has zero strict and zero ambiguity mismatches. |
| Production eager full-hook transport | 62,240/62,240 independent same-graph D2D reference rows were byte-identical to ClickHouse rows. Eight requests generated 20 tokens each; 160 generated tokens multiplied by 389 families gives the exact retained row count. |
| Production CUDA-graph full-hook transport | 62,240/62,240 independent same-graph D2D reference rows were byte-identical to ClickHouse rows, including the corrected fused residual capture. |
| Reduced storage regression | Eager and CUDA graph each passed 3,112/3,112 after the fused-residual correction. |
| Lifecycle | Accepted runs flushed monitoring, shut down vLLM cleanly, removed only capture-scoped ClickHouse rows, and left no residual GPU allocation. |

The public oracle uses only vLLM's offline API and compares separate baseline
and monitored processes. The storage oracle is independent: one execution
contains both DMI ring producers and preallocated same-graph D2D copies, then
compares every retained tensor and request/token range against ClickHouse byte
for byte.

## Fixture and implementation decisions

An available tiny Apertus fixture follows the original 8B-style Llama-3 RoPE
branch, but it is not used to generalize production storage. All public and
storage evidence comes from the frozen official 8B checkpoint. Newer Apertus
v1.1 small configs use materially different tied-embedding and RoPE branches,
so they are explicitly outside this verdict rather than inherited by name.

The optional CUDA-fused xIELU package was not installed. This cell therefore
qualifies vLLM's official Python xIELU fallback in both eager and default graph
modes. Disabled numerical-block hooks delegate to the exact upstream forward.
When enabled, `resid_pre` and `resid_final` read the completed residuals returned
by their fused add-RMSNorm operations instead of adding second CUDA nodes; the
compare oracle copies the same authoritative tensors. The compare model is constructed as a concrete
backbone so the upstream compilation decorator cannot retain an older
model-wide forward.

## Support decision

`ApertusForCausalLM` is supported for the named
`swiss-ai/Apertus-8B-Instruct-2509` TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact integration commit above. Other Apertus
checkpoints and every parallel, quantized, serving, speculative, optional
fused-xIELU, alternate-task, or different configuration path remain untested
rather than implicitly supported.
