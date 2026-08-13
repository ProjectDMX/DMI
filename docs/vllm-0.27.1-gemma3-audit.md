# Gemma 3 model-expansion audit for vLLM 0.27.1

This is the agent-authored compatibility record for adding the text-only
`Gemma3ForCausalLM` architecture to DMI's vLLM 0.27.1 integration. The current
verdict is **experimental TP1 on the named tiny fixture**, not production
checkpoint support. The official checkpoint remains an unmet release gate.

## Frozen identity and support cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/gemma3.py`, `Gemma3ForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-model-expansion`, commit `b71a55ae4227edf80576be202768a317eeaded1c` |
| DMI classes | `Gemma3PForCausalLM`; test-only `Gemma3CompareForCausalLM` |
| Runtime package | official `vllm==0.27.1` wheel plus DMI out-of-tree lazy model registration |
| Runtime | Python 3.12.8, PyTorch 2.13.0+cu130, BF16, V1 offline API |
| Fixture | `shibatch/tinygemma3-2m`, subfolder `hf`; 6 layers, hidden size 128, 4 Q heads, 1 KV head, head size 32, local/global attention pattern |
| Claimed cells | TP1 eager and default CUDA-graph; public offline API; all 77 canonical hook families; ClickHouse transport values |
| Excluded | production checkpoints, TP>1, PP/DP/EP/SP, V2, serve/async, speculative, LoRA, quantization, multimodal `Gemma3ForConditionalGeneration` |

The official `google/gemma-3-1b-it` repository returned an access-denied gated
checkpoint response for the available Hugging Face account on 2026-08-13. A
tiny fixture proves constructor, loader, forward, graph, hook, and transport
contracts, but does not prove production weight loading. This missing gate is
why the roadmap status remains `experimental`.

## Model and registry checklist

| IDs | Verdict | Evidence and scope |
| --- | --- | --- |
| M01 | verified | The DMI class derives from the exact upstream class at the frozen vLLM commit, not from Llama or another similar family. |
| M02-M03 | verified for TP1 V1 | Constructors preserve cache, quantization object, prefix, embedding, four-norm decoder, intermediate-tensor keys, and arbitrary target-forward keywords. A focused regression exercises vLLM's keyword `prefix=` layer factory. |
| M04 | verified for the fixture | Embedding scaling, Q/K RMS normalization before RoPE, GQA, local/global attention selection, four RMSNorms, fused residual flow, tied embeddings, and final-logit soft-cap match upstream. Separate-process public outputs are strict-equal in eager and graph modes. |
| M05 | excluded | PP stage ownership and `IntermediateTensors` runtime paths were retained in source but not run. No PP claim is made. |
| M06 | excluded | No speculative/Eagle cell is claimed. |
| M07 | verified only for TP1 | Per-head shapes are checked at TP1. TP sharding and replicated-KV ownership remain untested. |
| M08 | N/A for this class | The audited text model is dense and has no expert routing. |
| M09 | excluded | Quantized weights and quantized activation boundaries were not run. |
| M10 | verified for fixture; production pending | Upstream HF-to-vLLM stacked mappings, packed module map, tied embeddings, skip-prefix policy, and loader method are inherited. The fixture weights loaded in every runtime cell; the official 1B weights were inaccessible. |
| M11-M13 | verified for TP1 | CPU semantic tests prove normalized per-head Q/K/V and pre-output-projection Z placement. Inventory tests cover 12 per-layer plus five top-level specs; runtime storage resolves all 77 families. |
| M14 | bounded | Only text-only `Gemma3ForCausalLM` is remapped. Gemma 3 conditional/multimodal, Gemma 3n, remote-code variants, LoRA, quantized, and production checkpoint branches do not inherit the verdict. |
| M15 | verified for fixture layer kinds | All six layers are dense MLP plus attention and expose the same canonical family set. Both sliding and global attention occur in the fixture; neither invents an unsupported SSM/MoE hook. |
| R01-R03 | verified | Upstream maps `Gemma3ForCausalLM` to `gemma3:Gemma3ForCausalLM`; DMI maps it one-to-one to `gemma3_p:Gemma3PForCausalLM`. Monitored runtime storage proves the hooked class executed. |
| R04-R05 | verified | Lazy resolution works against the official wheel and retains the already-audited 0.27.1 registry API shape. |
| R06 | verified | A fresh CPU process sees the architecture registered while the Gemma DMI module remains unloaded and CUDA remains uninitialized. |
| R07 | verified | Fork registry and official-wheel lazy targets resolve to the exact exported classes. |

## Hook capability manifest

Every fixture layer exposes the same canonical hooks. Q and K are observed
after Gemma's per-head RMS normalization and before RoPE; V is per KV head; Z is
the flattened attention-kernel output before `o_proj`. `attn_out` is after
Gemma's post-attention norm because that normalized branch is what enters the
next fused residual/norm operation. `mlp_out` is likewise after the
post-feedforward norm.

| Scope | Hook families | TP1 shape tail |
| --- | --- | --- |
| Per layer, layers 0-5 | `resid_pre`, `ln1`, `attn_out`, `resid_mid`, `ln2`, `mlp_in`, `mlp_out` | `[128]` |
| Per layer, layers 0-5 | `q` | `[4, 32]` |
| Per layer, layers 0-5 | `k`, `v` | `[1, 32]` |
| Per layer, layers 0-5 | `z` | `[128]` |
| Per layer, layers 0-5 | `mlp_post` | `[512]` |
| Model-wide | `token_ids`, `embed`, `resid_final`, `final_ln`, `final_logits` | `[]`, `[128]`, `[128]`, `[128]`, `[1024]` |

The independent contract is
[`gemma3_2m_storage_contract.json`](../tests/fixtures/vllm/gemma3_2m_storage_contract.json).
It is not generated from `get_hook_specs`; therefore an implementation that
silently drops a hook cannot make the expected set drop with it.

## Black-box and transport evidence

| Gate | Checklist IDs | Result |
| --- | --- | --- |
| Focused model/registry/storage contracts | M02-M04, M10-M13, R02-R07, N02-N04, N13, L10 | 45/45 selected CPU tests passed; the final expansion focused sweep passed 236/236. Negative storage cases reject duplicate identity, missing family, dtype/shape drift, and token-range gaps. |
| Full public corpus, eager+graph | S02-S04, G01-G04, G07-G09, P01-P08 | 2/2 tests passed in 69.55 s. Twelve cases include text and token-ID inputs, Unicode, a one-token tail, shared prefixes, ragged batching, and six fixed-seed generated prompts. Baseline and monitored outputs were strict-equal; no ambiguity or baseline-envelope fallback was used. |
| Full-hook eager value comparison | N02-N08, N13, L02, L05-L10, E06 | 11,935/11,935 independent D2D reference tensors were byte-identical to scoped ClickHouse rows. |
| Full-hook CUDA-graph value comparison | G04, G08, P08, N02-N08, N13, L02, L05-L10, E05-E06 | 11,550/11,550 independent D2D reference tensors were byte-identical to scoped ClickHouse rows. |
| Minimal external storage contract, eager+graph | M12-M13, N04, N07, L10 | Each mode produced 616 unique rows and all 77 expected families with contiguous per-request token coverage and exact dtype/shape tails. |
| Lifecycle | L02-L03 | Explicit monitoring flush and bounded EngineCore shutdown completed without worker error, force-kill, TCPStore warning, residual process, or residual GPU allocation. Capture cleanup was scoped to the unique model ID. |

The eager and graph row totals differ because the scheduler may partition the
same public workload differently. Each oracle reconstructs the exact token
ranges and compares every row against the same-execution reference; it does not
incorrectly require insertion order or cross-mode row-count equality.

## Fixture decisions and remaining gate

- `ccmodular/tiny-random-Gemma3ForCausalLM` was rejected as a regression fixture:
  its head size selects a costly FlexAttention/JIT path on this runtime.
- `optimum-intel-internal-testing/tiny-random-gemma3-text` has a portable root
  config but its 262,144-token vocabulary made two identical baseline/monitored
  runs spend minutes in sampling/JIT. It was terminated as a fixture-quality
  failure, not recorded as a DMI failure.
- `shibatch/tinygemma3-2m` uses a small 1,024-token vocabulary and exercises six
  layers, GQA, Q/K normalization, and both sliding/global attention. The release
  runner resolves its `hf` subfolder from the repository ID without embedding a
  machine-specific cache hash.

To move this row from `experimental` to `supported`, obtain authorized access to
an official Gemma 3 text checkpoint and rerun public eager/graph plus at least
the claimed full-hook storage cell. TP, PP, quantized, LoRA, serving, and
multimodal cells remain separate claims even after that checkpoint gate closes.
