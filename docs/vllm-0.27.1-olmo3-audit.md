# OLMo 3 model support audit for vLLM 0.27.1

This agent-authored record supports the text-only dense
`Olmo3ForCausalLM` architecture in a bounded TP1 BF16 V1 offline cell. The
runtime-qualified checkpoint is the official OLMo 3 7B Instruct model. OLMo 3
is not treated as an OLMo 2 alias: vLLM 0.27.1 has a distinct post-norm
implementation with Q/K normalization, a periodic sliding/full-attention
schedule, and different RoPE behavior for the two attention kinds.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/olmo3.py`, `Olmo3ForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-olmo3`, commit `9f6a5c762bc5337aa08f38b31d70eb0020d60816` |
| Production and storage checkpoint | `allenai/Olmo-3-7B-Instruct`, revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |
| Production shape | 7,298,011,136 BF16 parameters; hidden 4,096; intermediate 11,008; 32 layers; 32 query and KV heads; vocabulary 100,278; untied embeddings |
| Attention schedule | Three 4,096-token sliding-attention layers followed by one full-attention layer, repeated eight times |
| RoPE contract | sliding attention uses default RoPE at theta 500,000; full attention uses YaRN factor 8 with original context 8,192 |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, all 389 declared hook families |
| Excluded | other OLMo/OLMoE/hybrid architectures, TP>1, PP/DP/EP/SP, V2, serve/async, speculative, LoRA, quantization, prefix caching, pooling/task heads, attention weights/cache internals, and contexts beyond the bounded 128-token runtime cell |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | DMI follows the exact OLMo 3 post-norm arithmetic rather than a Llama/OLMo 2 template. Attention output is normalized before the first residual addition; MLP output is normalized before the second. Q/K hooks observe values after OLMo 3 Q/K RMSNorm and before type-specific RoPE. Upstream and monitored eager/graph execution on the official checkpoint close this path. |
| M05-M09 | preserved but excluded from the support claim | The DMI class inherits OLMo 3's LoRA/PP interfaces, packed-module declarations, mapper, and intermediate-tensor behavior. The accepted runtime cell is deliberately limited to TP1 standard-HF BF16 offline generation; parallel, quantized, prefix-cache, speculative, serving, and alternate-task paths remain unclaimed. |
| M10 | adapted-verified for official HF weights | The DMI wrapper preserves the upstream stacked QKV/gate-up mapper and `AutoWeightsLoader` method. The official three-shard checkpoint loaded in both upstream qualification modes, both public DMI modes, and four storage qualification runs. Untied `lm_head` construction is preserved. |
| M11-M13 | verified for TP1 | Every layer exposes residual input, post-QK-norm/pre-RoPE Q/K, raw V, pre-output-projection Z, raw attention output, post-attention norm, completed mid-residual, MLP input, post-activation MLP width, raw MLP output, and post-feedforward norm. Five global families expose token IDs, embeddings, final residual, final norm, and logits. Thirty-two layers therefore expose 389 truthful families. |
| M14 | fail-closed and bounded | DMI requires `model_type=olmo3`, SiLU, no attention bias, complete known layer types, a positive sliding window, and per-attention-type runtime RoPE parameters. Unknown layer types, missing schedule entries, and missing RoPE branches fail before model construction. |
| M15 | verified with an ordered dense manifest | Hook specs follow the actual post-norm firing order. The manifest does not fabricate attention weights, KV-cache state, MoE routing, SSM state, or other OLMo-family internals absent from this implementation. |
| R01-R03 | verified | Upstream resolves `Olmo3ForCausalLM` to `olmo3:Olmo3ForCausalLM`; DMI resolves it one-to-one to `olmo3_p:Olmo3PForCausalLM`. `Olmo3CompareForCausalLM` has a separate test-only registry entry. |
| R04-R07 | verified | Official-wheel lazy registration, runtime remap, aliases, compare-worker registration, fail-closed schedule/RoPE checks, exact-tree instrumentation, manifest ordering, and concrete compare-backbone construction are locked by CPU contracts. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream production qualification | The unmodified official 7B checkpoint loaded and generated successfully in eager and default CUDA-graph modes before DMI was enabled. |
| Focused model and release contracts | The pre-public focused selections passed 26/26; the final aggregate focused sweep passed 321/321 in the pinned vLLM 0.27.1 / PyTorch 2.13.0+cu130 environment. They cover config rejection, disabled-forward delegation, post-norm ordering, Q/K normalization and RoPE boundaries, MLP post-activation placement, exact in-place identity, manifest inventory/order, concrete compare-backbone construction, registry, black-box, comparator, and matrix bounds. |
| Production public eager+graph | 2/2 full 12-case API-only tests passed in 98.91 s. Baseline and monitored public outputs, request attribution, reverse-batch relations, generated metamorphic cases, and decision logprobs agreed in both modes. |
| Production eager full-hook transport | 61,073/61,073 independent D2D reference rows were byte-identical to ClickHouse rows. One request stopped normally at EOS after 16 tokens, so 157 generated tokens multiplied by 389 families gives the exact retained row count. |
| Production CUDA-graph full-hook transport | 61,073/61,073 independent D2D reference rows were byte-identical to ClickHouse rows, with the same deterministic EOS behavior. |
| Reduced storage regression | Eager and CUDA graph each passed 3,112/3,112 before the full production cells were run. |
| Lifecycle | Accepted runs flushed monitoring, shut down vLLM cleanly, removed only their capture-scoped ClickHouse rows, and left no residual GPU allocations. |

The public oracle uses only vLLM's offline API and compares separate baseline
and monitored processes. The storage oracle is independent: one execution
contains both DMI ring producers and preallocated same-graph D2D copies, then
compares every retained tensor and request/token range against ClickHouse byte
for byte.

## Fixture and implementation decisions

Available tiny OLMo 3 fixtures were inspected but not used to qualify the
production storage cell. One two-layer fixture covers sliding then full
attention but does not preserve the production YaRN branch, while another
contains YaRN metadata but only sliding-attention layers. Neither independently
kills errors in the production combination of periodic schedule and
attention-type-specific RoPE. The official 7B checkpoint therefore supplies
both public and storage evidence; focused synthetic contracts separately reject
missing schedule and RoPE branches.

Disabled numerical-block hooks delegate directly to the exact upstream
forward. When enabled, Q/K are observed after `_apply_qk_norm` but before RoPE;
V is observed before attention, Z before the output projection, and MLP post
after SiLU-and-multiply but before the down projection. The compare oracle
constructs `Olmo3CompareModel` as the concrete backbone from initialization,
which prevents the upstream compilation decorator from retaining an older
model-wide forward.

## Support decision

`Olmo3ForCausalLM` is supported for the named
`allenai/Olmo-3-7B-Instruct` TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact integration commit above. Other
checkpoints that declare the same architecture remain runtime-unqualified, and
every parallel, quantized, serving, speculative, alternate-task, or different
OLMo-family path remains untested rather than implicitly supported.
