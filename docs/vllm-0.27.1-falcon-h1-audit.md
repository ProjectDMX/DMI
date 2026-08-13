# Falcon-H1 model support audit for vLLM 0.27.1

This agent-authored record supports the text-only `FalconH1ForCausalLM`
architecture in a bounded TP1 BF16 V1 offline cell. Falcon-H1 is a parallel
hybrid architecture: every decoder layer evaluates both attention and Mamba2,
merges the scaled branch outputs with the residual, and then evaluates its MLP.
The verdict does not extend to other Falcon architectures or untested runtime
cells.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/falcon_h1.py`, `FalconH1ForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-falcon-h1`, commit `e69be397855221bdfa395f51394a1f7ebeec68cb` |
| Production checkpoint | `tiiuae/Falcon-H1-0.5B-Instruct`, revision `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368` |
| Focused fixture | `tiiuae/Falcon-H1-Tiny-90M-Instruct`, revision `e6389502a0b12cd8da894b395ba5bf7436873b16` |
| Claimed cell | TP1, BF16, V1 offline `LLM`, standard Hugging Face weights, eager and default CUDA graph, canonical attention/MLP/residual hooks plus SSM branch input/output hooks |
| Excluded | TP>1, PP/DP/EP/SP, V2, serve/async, speculative, LoRA, quantization, prefix caching, remote-code variants, and internal Mamba convolution/state/B/C/dt tensors |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01-M04 | independently adapted and runtime-verified | The target implementation has distinct attention, Mamba2, parallel-hybrid, MLP, model, and causal-LM classes. DMI preserves the exact branch order, residual order, and upstream arithmetic expression order while adding bounded observations. Representative eager/graph public differential tests close the numerical path. |
| M05-M09 | preserved but excluded from the support claim | The DMI causal-LM class inherits upstream hybrid/state interfaces and state dtype, shape, and copy calculators. PP, TP>1, prefix caching, speculative execution, and quantized paths were not run and remain unclaimed. |
| M10 | adapted-verified for standard HF weights | The DMI subclass inherits Falcon-H1's packed QKV/gate-up mapping, HF-to-vLLM name mapper, embedding declarations, and weight loader. Both the official Tiny checkpoint and the official 0.5B checkpoint loaded. The 0.5B fixture closes untied embeddings and non-unit embedding, key, attention, SSM, MLP, and LM-head scaling branches that Tiny does not exercise. |
| M11-M13 | verified for TP1 | Each parallel-hybrid layer exposes `resid_pre`, `ln1`, Q/K/V/Z, scaled `attn_out`, scaled `ssm_in`, scaled `ssm_out`, `resid_mid`, `ln2`, `mlp_in`, `mlp_post`, and `mlp_out`. Five model-wide hooks cover token IDs, scaled embeddings, final residual, final norm, and logits. The 24-layer fixture exposes 341 hook families. Independent compare buffers and ClickHouse transport were byte-identical in eager and graph runs. |
| M14 | bounded | Only `FalconH1ForCausalLM` is remapped. `FalconForCausalLM`, Falcon Mamba, task heads, remote-code models, and unrelated hybrid implementations do not inherit this verdict. |
| M15 | verified with an explicit hybrid manifest | Every exercised layer contains both attention and Mamba2; it is not an alternating-layer model. Attention hooks describe only the attention branch. `ssm_in` and `ssm_out` describe hidden-width Mamba branch boundaries. DMI deliberately does not claim internal recurrent state, convolution, z/x/B/C/dt, or cache tensors. |
| R01-R03 | verified | Upstream resolves `FalconH1ForCausalLM` to `falcon_h1:FalconH1ForCausalLM`; DMI resolves it one-to-one to `falcon_h1_p:FalconH1PForCausalLM`. The compare architecture has its own test-only registry entry. |
| R04-R07 | verified | The 0.27.1 out-of-tree lazy-registration surface loads the DMI variant without eagerly loading CUDA. Constructor prefix keywords, registry remap, hybrid/state inheritance, hook inventory, and release-matrix bounds are locked by CPU contracts. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Upstream fixture qualification | Unmodified Tiny eager and default CUDA graph both loaded and generated successfully before DMI was enabled. |
| Focused model and release contracts | Final focused sweep passed 257/257. The suite includes non-unit branch-scale semantics, exact `make_layers(prefix=...)` construction, native SSM ABI/shape/selection, storage-name derivation, registry, black-box, and release-matrix contracts. |
| Tiny public eager+graph | 2/2 full 12-case API-only tests passed in 159.40 s. |
| Tiny eager full-hook transport | 54,560/54,560 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Tiny CUDA-graph full-hook transport | 54,560/54,560 independent D2D reference rows were byte-identical to ClickHouse rows. |
| Production eager+graph | 2/2 full 12-case API-only tests passed in 92.05 s. The public decision-logprob oracle accepted only bounded low-margin branch changes; there was no unexplained decision drift. |
| Upstream graph stability | Two independent production baseline graph runs produced identical normalized public JSON hashes. A token-only DMI graph run also matched that hash exactly after upstream arithmetic expression order was restored. |
| Lifecycle | Accepted runs exited cleanly without residual vLLM workers, captures, or GPU allocations. |

The storage oracle compares independently copied and transported values from the
same execution, including exact request/token ranges. The public oracle is a
separate baseline-versus-monitored process comparison. The comparator's short
hook names are derived from the native hook-definition table, so the new SSM
family cannot silently diverge between reference files and ClickHouse names.

## Fixture qualification and compiler regression

The official Tiny checkpoint is an appropriate runtime fixture: it uses the
target architecture, 24 parallel hybrid layers, 64-dimensional attention heads,
Mamba2 state and convolution, BF16 weights, and a bounded 182 MB checkpoint.
Its release cells pin `gpu_memory_utilization=0.2`, `max_model_len=128`, and
`max_num_batched_tokens=128`.

Tiny alone was insufficient for the final numerical verdict because its branch
and MLP multipliers are all `1.0`. The production checkpoint uses non-unit
embedding, key, attention-output, SSM-input/output, MLP, and LM-head scales.
An early DMI implementation materialized scaled branch values before the merge;
that was eager-equivalent but changed BF16 compiler fusion and failed the
production CUDA-graph public oracle. The accepted implementation computes hook
observations separately when enabled and preserves upstream expression order for
the model output. A non-unit focused contract and production graph cell prevent
an all-one tiny fixture from masking this class of regression again.

## Support decision

`FalconH1ForCausalLM` is supported for the named TP1 BF16 V1 offline standard-HF
eager/default-graph cell at the exact commits above. The supported capability is
the canonical attention/MLP/residual inventory plus hidden-width SSM branch
input/output boundaries. Internal Mamba state and every excluded execution mode
remain untested rather than implicitly supported.
