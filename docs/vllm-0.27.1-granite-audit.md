# Granite model support audit for vLLM 0.27.1

This agent-authored record supports the text-only dense
`GraniteForCausalLM` architecture in a bounded TP1 BF16 V1 offline cell. The
runtime-qualified checkpoint is the official Granite 4.1 3B instruct model.
Granite 4.1 8B and 30B declare the same implementation and scalar contract,
but those larger checkpoints were inspected rather than executed and are not
part of the runtime-qualified cell. Granite MoE, shared-MoE, hybrid Mamba, and
vision architectures are separate implementations and are excluded.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/granite.py`, `GraniteForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-granite`, commit `dca631b19ec7b24816dd50578c05ee532f5769b8` |
| Production and storage checkpoint | `ibm-granite/granite-4.1-3b`, revision `c0650403e44e78ec0262dab1c90914c65b196c4e` |
| Production shape | 3,402,836,480 BF16 parameters; hidden 2,560; intermediate 8,192; 40 layers; 40 query heads; 8 KV heads; vocabulary 100,352; tied embeddings |
| Granite arithmetic | attention multiplier 0.015625; embedding multiplier 12.0; residual multiplier 0.22; logits divisor 10.0; SiLU gated MLP |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, all 485 declared hook families |
| Excluded | Granite 4.1 8B/30B runtime verdicts, `GraniteMoeForCausalLM`, `GraniteMoeSharedForCausalLM`, `GraniteMoeHybridForCausalLM`, Granite vision/speech, TP>1, PP/DP/EP/SP, V2, serve/async, speculative, LoRA, quantization, prefix caching, pooling/task heads, and attention/cache internals |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | Granite has model-specific attention scaling, embedding scaling, two scaled residual additions per layer, and logits scaling. DMI constructs the upstream Granite modules and weight tree, then exposes these exact operations rather than reusing Llama math. Unmodified-upstream qualification and DMI public eager/graph tests close the accepted path. |
| M05-M09 | preserved but excluded from the support claim | The DMI class inherits Granite's LoRA, PP, quantization, mapper, intermediate-tensor, and task interfaces. The accepted runtime cell exercises only TP1 standard-HF BF16 offline generation; parallel, quantized, prefix-cache, speculative, serving, and alternate-task paths remain unclaimed. |
| M10 | adapted-verified for standard HF weights | The subclass inherits the upstream stacked QKV/gate-up mapper and `AutoWeightsLoader` behavior. The official two-shard checkpoint loaded in upstream qualification, public DMI, and both production storage cells. Tied embedding/lm-head identity and the upstream logits divisor are preserved. |
| M11-M13 | verified for TP1 | Every layer exposes residual input, first norm, pre-RoPE Q/K/V by head, pre-output-projection Z, attention output, scaled mid-residual, second norm, MLP input, post-activation MLP width, and raw MLP output. Five global families expose token IDs, embeddings after Granite's multiplier, final residual, final norm, and scaled logits. Forty layers therefore expose 485 truthful families. |
| M14 | fail-closed and bounded | DMI requires `model_type=granite`, SiLU, and explicit attention, embedding, residual, and logits scalar fields. Granite MoE/hybrid architectures have no remap and cannot inherit this dense verdict. Missing scalar semantics or a non-Granite model type raise `NotImplementedError`. |
| M15 | verified with an ordered dense manifest | Per-layer specs follow forward order and include the Granite-specific post-activation width. The manifest does not fabricate MoE, SSM, recurrent, attention-weight, or cache-state activations. |
| R01-R03 | verified | Upstream resolves `GraniteForCausalLM` to `granite:GraniteForCausalLM`; DMI resolves it one-to-one to `granite_p:GranitePForCausalLM`. `GraniteCompareForCausalLM` has a separate test-only registry entry. |
| R04-R07 | verified | Official-wheel lazy registration, runtime remap, aliases, release cells, compare worker, scalar rejection, exact-tree instrumentation, manifest ordering, and concrete compare-backbone construction are locked by CPU contracts. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream production qualification | The unmodified official 3B checkpoint loaded and generated successfully in eager and default CUDA-graph modes before DMI was enabled. |
| Focused model and release contracts | The aggregate focused sweep passed 301/301 in the pinned vLLM 0.27.1 / PyTorch 2.13.0+cu130 environment. It covers Granite scalar semantics, disabled-forward delegation, exact in-place identity, manifest inventory/order, attention and MLP boundaries, concrete compare-backbone construction, registry, black-box, comparator, and matrix bounds. |
| Production public eager+graph | 2/2 full 12-case API-only tests passed in 97.31 s. Baseline and monitored public outputs, request attribution, reverse-batch relations, generated metamorphic cases, and decision logprobs agreed in both modes. |
| Production eager full-hook transport | 47,045/47,045 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Production CUDA-graph full-hook transport | 41,225/41,225 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Reduced storage regression | Eager and CUDA graph each passed 3,880/3,880 before the full production cells were run. |
| Lifecycle | Accepted runs shut down monitoring and vLLM cleanly without residual workers or GPU allocations. |

The public oracle uses only vLLM's offline API and compares separate baseline
and monitored processes. The storage oracle is intentionally different: one
execution contains both DMI ring producers and independent preallocated D2D
copies, then compares every retained tensor and request/token range against
ClickHouse byte for byte. Both storage modes use the official 3B production
checkpoint because no smaller fixture is needed to cover this bounded dense
contract.

## Granite-specific implementation decisions

The hook named `embed` observes the scaled embedding that actually enters the
first decoder layer, after multiplication by `embedding_multiplier`. Per-layer
`attn_out` and `mlp_out` observe the raw branch values before multiplication by
`residual_multiplier`; `resid_mid` observes the first completed scaled residual
addition. Q/K/V are captured before rotary embedding and Z before the output
projection. `final_logits` observes the output after Granite's logits divisor.

Disabled numerical-block hooks delegate directly to the exact upstream
forward. The compare oracle constructs `GraniteCompareModel` as the concrete
backbone from initialization, then upgrades the exact upstream-created layer
tree in place. This prevents a compilation decorator from retaining an older
model-wide forward and is locked by constructor and identity contracts.

## Support decision

`GraniteForCausalLM` is supported for the named
`ibm-granite/granite-4.1-3b` TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact commits above. Granite 4.1 8B and 30B
share the audited dense implementation and scalar schema but remain
runtime-unqualified because their real weights were not executed. Every MoE,
hybrid, multimodal, parallel, quantized, serving, and speculative path remains
untested rather than implicitly supported.
