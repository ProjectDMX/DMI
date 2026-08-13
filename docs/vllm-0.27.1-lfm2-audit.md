# LFM2 model support audit for vLLM 0.27.1

This agent-authored record supports the text-only `Lfm2ForCausalLM`
architecture in a bounded TP1 BF16 V1 offline cell. LFM2 is a heterogeneous
hybrid: its decoder schedule alternates stateful short-convolution layers with
full-attention layers, while both layer kinds share the residual and MLP
contract. The verdict does not extend to LFM2-MoE, LFM2-VL, ColBERT, or an
untested runtime cell.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/lfm2.py`, `Lfm2ForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-lfm2`, commit `229014dd9692d3ad839af3ce533748c46a0c0772` |
| Production checkpoint | `LiquidAI/LFM2.5-1.2B-Instruct`, revision `df58c174f05ff733f83f8cae10ea9298224c8006` |
| Focused fixture | `tiny-random/lfm2`, revision `b0eb9390dd8310b5fed484e80e04508ca1ec74d3` |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, canonical residual/MLP hooks plus layer-kind-specific attention or short-convolution boundaries |
| Excluded | TP>1, PP/DP/EP/SP, V2, serve/async, speculative, LoRA, quantization, prefix caching, alternate layer schedules, LFM2-MoE, LFM2-VL, pooling/task heads, and internal short-convolution projections/cache state |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | The target implementation has distinct attention, short-convolution, two decoder-layer, model, and causal-LM classes. DMI instruments the exact upstream model tree in place, preserving module and parameter identity. A disabled hook path delegates to the exact upstream forward. Production eager/graph public differential tests close the accepted numerical path. |
| M05-M09 | preserved but excluded from the support claim | The DMI causal-LM class inherits the upstream hybrid cache interfaces, Mamba state shape/copy calculators, PP intermediate contract, and compilation decoration. Parallel, prefix-cache, speculative, and quantized paths were not run and remain unclaimed. |
| M10 | adapted-verified for standard HF weights | The subclass inherits LFM2's packed-module mapping, weight loader, embedding/lm-head behavior, and HF-to-vLLM mapper. Both the two-layer fixture and official 1.2B checkpoint loaded. The production checkpoint also closes the real 16-layer schedule, 64-dimensional heads, grouped KV heads, convolution cache width, and effective 8192-wide aligned MLP branch. |
| M11-M13 | verified for TP1 | Every layer exposes seven common residual/MLP families. A convolution layer additionally exposes typed hidden-width `conv_in`/`conv_out`; an attention layer instead exposes normalized pre-RoPE Q/K, V, pre-output-projection Z, and attention output. Five model-wide families cover token IDs, embeddings, final residual, final norm, and logits. The production schedule yields 167 truthful families. The two-layer fixture's 26-family manifest and independent compare buffers were byte-identical to ClickHouse in eager and graph runs. |
| M14 | bounded | Only text `Lfm2ForCausalLM` is remapped. `Lfm2MoeForCausalLM`, LFM2-VL, ColBERT/pooling, task heads, and unrelated convolution or SSM models do not inherit this verdict. |
| M15 | verified with an explicit heterogeneous manifest | The production checkpoint has ten convolution layers and six full-attention layers in the declared `layer_types` order. DMI chooses hook families per layer kind and orders the manifest exactly as forward fires: residual/norm prefix, operator-specific hooks, then residual/norm/MLP suffix. It does not mislabel short convolution as SSM or fabricate attention hooks on convolution layers. |
| R01-R03 | verified | Upstream resolves `Lfm2ForCausalLM` to `lfm2:Lfm2ForCausalLM`; DMI resolves it one-to-one to `lfm2_p:Lfm2PForCausalLM`. The compare architecture has a separate test-only registry entry. |
| R04-R07 | verified | The 0.27.1 out-of-tree lazy-registration surface resolves the exported DMI class without eagerly importing a model implementation in the parent process. Registry, remap, prefix construction, in-place instrumentation, hook inventory/order, model-shape, and release-matrix contracts are locked by CPU tests. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream fixture qualification | The unmodified fixture loaded and generated in eager and default CUDA-graph modes before DMI was enabled. Its 32-dimensional heads satisfy the graph backend, and its two layers exercise one convolution plus one attention layer. |
| Focused model and release contracts | Final focused sweep passed 273/273 in the pinned vLLM 0.27.1 / PyTorch 2.13.0+cu130 environment. The suite includes native convolution ABI/shape/selection/storage names, heterogeneous inventory and firing order, exact upstream delegation, in-place module identity, aligned MLP width, registry, black-box, comparator, and matrix bounds. |
| Production eager+graph | 2/2 full 12-case API-only tests passed in 89.42 s. Token IDs, public outputs, request attribution, reverse-batch metamorphic relations, and retained decision logprobs matched the independent baseline processes exactly. |
| Fixture eager full-hook transport | 4,160/4,160 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Fixture CUDA-graph full-hook transport | 4,160/4,160 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Lifecycle | Accepted runs stopped monitoring explicitly and exited without residual vLLM workers, captures, or GPU allocations. |

The storage oracle compares independently copied and transported tensors from
the same execution, including request/token ranges and the layer-kind-specific
manifest. The public oracle is a separate-process baseline-versus-monitored
comparison using only vLLM's offline API.

## Fixture qualification and port regressions

The accepted fixture has hidden size 64, two 32-dimensional attention heads,
one KV head, BF16 weights, convolution cache width 3, and exactly one layer of
each supported kind. Its release cells pin `gpu_memory_utilization=0.2`,
`max_model_len=128`, and `max_num_batched_tokens=128`.

Two port defects were exposed by value-level testing rather than generation
smoke tests. First, LFM2 derives its post-activation MLP width by applying the
upstream two-thirds adjustment, multiplier, and alignment; the fixture's
configured width 128 therefore becomes an effective tensor width 256. Second,
the heterogeneous hook manifest must interleave operator hooks between the
common residual/norm prefix and suffix. Using a uniform common-then-operator
order shifted metadata onto the wrong tensors. Both rules now have CPU
contracts and byte-exact eager/graph storage evidence.

The public implementation is built by running the exact upstream constructor
and changing only the existing model modules' Python classes in place before
adding HookPoints. When no hook inside a numerical block is enabled, its
forward delegates directly to upstream. This preserves the original module
tree, loader identity, and arithmetic path and prevents a wrapper-only tiny
fixture from concealing production-checkpoint drift.

## Support decision

`Lfm2ForCausalLM` is supported for the named TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact commits above. The supported observation
surface is the canonical common inventory plus attention boundaries on
full-attention layers and hidden-width input/output boundaries on
short-convolution layers. Internal convolution/cache state and every excluded
execution mode remain untested rather than implicitly supported.
