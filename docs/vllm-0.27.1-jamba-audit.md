# Jamba model support audit for vLLM 0.27.1

This agent-authored record supports the text-only `JambaForCausalLM`
architecture in a bounded TP1 BF16 V1 offline cell. The qualified
`AI21-Jamba2-3B` configuration alternates 26 Mamba1 layers with two full
attention layers and uses a dense MLP in every layer. The verdict deliberately
does not extend to Jamba configurations containing MoE layers.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/jamba.py`, `JambaForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-jamba`, commit `b4556982f63db7a5ee786738b62154d4c0c898a4` |
| Production and storage checkpoint | `ai21labs/AI21-Jamba2-3B`, revision `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9` |
| Production schedule | 28 layers: attention at layers 7 and 21, Mamba1 at the other 26 layers, one dense expert in every layer |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, canonical residual/MLP hooks plus layer-kind-specific attention or Mamba boundaries |
| Excluded | Any configuration with an expert count other than one, Jamba 1.5/Mini/Large, TP>1, PP/DP/EP/SP, V2, serve/async, speculative, LoRA, quantization, prefix caching, pooling/task heads, and internal recurrent/convolution/cache state |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | Jamba has distinct attention, Mamba1, decoder-layer, model, and causal-LM paths. DMI instruments the exact upstream production tree in place and delegates a disabled numerical block to the exact upstream forward. Independent upstream qualification and DMI public eager/graph tests close the accepted numerical path. |
| M05-M09 | preserved but excluded from the support claim | The DMI causal-LM class inherits Jamba's hybrid/state interfaces, PP intermediate contract, packed-module behavior, and Mamba state dtype/shape/copy calculators. Parallel, prefix-cache, speculative, quantized, and alternate runtime paths were not run and remain unclaimed. |
| M10 | adapted-verified for standard HF weights | The subclass inherits the upstream QKV/gate-up weight mapper and `AutoWeightsLoader`. The official two-shard 3B BF16 checkpoint loaded in upstream qualification, public DMI, and storage-oracle cells. |
| M11-M13 | verified for TP1 | Every layer exposes seven common residual/MLP families. A Mamba layer adds hidden-width `ssm_in` and `ssm_out`; an attention layer instead exposes Q/K/V by head, pre-output-projection Z, and attention output. Five model-wide families cover token IDs, embeddings, final residual, final norm, and logits. The production schedule therefore exposes 263 truthful families. |
| M14 | fail-closed and bounded | DMI validates `layers_block_type` and `layers_num_experts` before construction. Unknown layer kinds and every layer with an expert count other than one raise `NotImplementedError`; no dense verdict is inherited by Jamba-MoE checkpoints. |
| M15 | verified with an explicit heterogeneous manifest | The manifest follows the declared layer schedule and forward firing order: residual/norm prefix, attention or SSM boundaries, then residual/norm/MLP suffix. It does not fabricate attention hooks on Mamba layers or expose internal recurrent state as an activation contract. |
| R01-R03 | verified | Upstream resolves `JambaForCausalLM` to `jamba:JambaForCausalLM`; DMI resolves it one-to-one to `jamba_p:JambaPForCausalLM`. The compare architecture has a separate test-only registry entry. |
| R04-R07 | verified | The official-wheel lazy-registration path, runtime remap, model aliases, release cells, compare worker, manifest, exact-tree instrumentation, and dense-only gate are locked by CPU contracts. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream production qualification | The unmodified official checkpoint loaded and generated successfully in eager and default CUDA-graph modes before DMI was enabled. |
| Focused model and release contracts | The final focused sweep passed 284/284 in the pinned vLLM 0.27.1 / PyTorch 2.13.0+cu130 environment. It covers loader/state inheritance, fail-closed MoE handling, disabled-forward delegation, exact in-place identity, manifest inventory/order, attention and SSM boundaries, concrete compare-backbone construction, registry, black-box, comparator, and matrix bounds. |
| Production public eager+graph | 2/2 full 12-case API-only tests passed in 94.13 s. Baseline and monitored public outputs, request attribution, metamorphic relations, and decision logprobs agreed in both modes. |
| Production eager full-hook transport | 37,083/37,083 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Production CUDA-graph full-hook transport | 36,557/36,557 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Lifecycle | Accepted runs shut down monitoring and vLLM cleanly without residual workers or GPU allocations. |

The storage oracle uses the official production checkpoint because the
available upstream tiny Jamba fixtures contain MoE layers and therefore cannot
qualify this bounded dense contract. It compares independently copied and
transported tensors from the same execution, including request/token ranges
and all 263 model-specific hook families. The public oracle is a separate
baseline-versus-monitored comparison that only uses vLLM's offline API.

## Compiler regression found by the oracle

The upstream `JambaModel` compilation decorator binds the concrete backbone
forward method during construction. An initial compare implementation first
constructed upstream Jamba and changed the model's Python class afterward.
Layer hooks executed because the layer objects were upgraded in place, but the
compiled backbone retained the old model-wide forward and omitted
`embed`, `resid_final`, and `final_ln`. Eager storage passed while CUDA-graph
metadata shifted onto the wrong tensors.

The accepted compare oracle constructs `JambaCompareModel` as the concrete
backbone from the start, then upgrades the exact upstream-created layer tree in
place. A focused constructor contract locks this requirement. The reduced
graph regression passed 263/263 before the full 36,557-row graph cell was
rerun. This failure mode would not have been detected by generation-only or
eager-only testing.

## Support decision

`JambaForCausalLM` is supported only for the named dense
`ai21labs/AI21-Jamba2-3B` TP1 BF16 V1 offline standard-HF eager/default-graph
cell at the exact commits above. The supported observation surface is the
canonical common inventory plus attention boundaries on the two attention
layers and hidden-width Mamba input/output boundaries on the other 26 layers.
Jamba-MoE, internal recurrent/cache state, and every excluded runtime mode
remain untested rather than implicitly supported.
