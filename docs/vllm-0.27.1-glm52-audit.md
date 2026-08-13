# GLM-5.2 decoder lite support audit for vLLM 0.27.1

This record covers the local implementation gate for the GLM-5.2 text decoder.
It does not claim runtime support until the pinned TP32 H100 matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/deepseek_v2.py`, `GlmMoeDsaForCausalLM` over the DeepSeek-V2 MLA/DSA implementation |
| DMI integration | branch `dmi-v0.27.1-glm52`; commit `0c83d591245042928ebd4265fb3ab02e13afa8e9` |
| Production checkpoint | `zai-org/GLM-5.2`, revision `b4734de4facf877f85769a911abafc5283eab3d9` |
| Weight contract | unquantized BF16, 282 safetensor shards, 1,506,667,387,408 bytes (about 1.403 TiB); inherited MLA, indexer, dense, shared/routed expert, and output loaders |
| Decoder contract | 78 MLA+DSA layers; dense MLP in layers 0-2, then 75 MoE layers with 256 routed plus one shared expert and sigmoid normalized top-8 routing; FP32 router logits |
| Sparse-attention contract | 2048-token DSA index, 32 index heads of width 128, interleaved RoPE, first three full indexers followed by a repeated one-full/three-shared schedule |
| Lite cell | TP32/PP1/DP1, no EP/SP/EPLB/context parallelism, V1 offline API, BF16, public and ClickHouse cases generated for H100 |
| Capacity basis | TP16 cannot hold 1.403 TiB of raw weights on 16×80 GiB H100; TP32 is the smallest audited head-divisible H100 cell with credible weight and runtime headroom |
| Explicitly excluded from DMI | MLA latent Q/K/V representations, ordinary Q/K/V/Z families, DSA index scores/IDs, indexer FP8 cache, MLA KV cache, attention weights, expert-local activations, and dispatch/combine tensors |
| Explicitly untested | production shard loading, public output parity, storage values, multi-node launch, PP/DP/EP/SP/EPLB/context parallelism, serving, speculative/MTP/Eagle, LoRA, quantization, prefix caching, alternate GLM/DeepSeek configurations |

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | The monitored class subclasses the exact `GlmMoeDsaForCausalLM` registry target, retains its public forward/logits/load interfaces, and instruments the native model tree after upstream construction. |
| M04 | source verified; runtime pending | Disabled dense-MLP, MoE, and decoder hooks delegate to upstream. Enabled decoder hooks preserve the upstream fused residual norms, opaque `DeepseekV2MLAAttention` call, dense/MoE call, and output order. No replacement attention math is introduced. H100 public comparison remains required. |
| M05-M07 | preserved, unclaimed | PP, auxiliary hidden-state, and speculative interfaces remain inherited, but the named lite cell rejects PP, speculative/MTP, sequence/context parallelism, and non-TP32 execution. The current release runner requires 32 locally visible GPUs; multi-node launching is not yet evidence. |
| M08 | adapted, CPU verified | Each of the 75 sparse layers exposes FP32 gate logits plus the authoritative router's top-8 IDs/weights. Both external- and internal-router inputs are preserved; EPLB and SP paths fail closed. The shared expert remains upstream-owned and is not reported as a routed expert-local tensor. |
| M09-M10 | unchanged, runtime pending | GLM-specific DSA indexer FP8 dequantization/fusion, MLA A-projection fusion, dense packing, expert mapping, shared-expert handling, and streaming shard load remain entirely upstream. Loading all 1.403 TiB is an explicit runtime prerequisite. |
| M11-M13 | adapted, CPU verified | All 78 layers expose seven residual/norm/attention-output/MLP boundary families. The three dense layers additionally expose MLP post-activation; the 75 sparse layers expose three routing families. Together with five global families, the truthful reduced manifest has 779 families, with matching compare buffers. Ordinary Q/K/V/Z are absent because MLA/DSA does not provide that contract at the public model boundary. |
| M14-M15 | bounded | The validator freezes the exact geometry, dense/MoE schedule, router behavior, DSA index-sharing pattern, interleaved RoPE, BF16 dtype, MLA mode, and TP32 topology. Every opaque attention, index/cache, and expert-local omission is explicit. |
| R01-R03 | verified | vLLM maps `GlmMoeDsaForCausalLM` to `deepseek_v2:GlmMoeDsaForCausalLM`; DMI remaps it one-to-one to `glm_moe_dsa_p:GlmMoeDsaPForCausalLM`. A separate compare target is registered. |
| R04-R07 | CPU verified | Official-wheel lazy targets resolve without eager model construction. Model-shape derivation now recognizes DeepSeek-style `n_routed_experts`, and the model-wide inventory retains all 78 heterogeneous layer identities. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `46 passed` in `tests/test_glm_moe_dsa_p_contract.py` |
| Full cumulative lite gate | GPT-OSS, Qwen3-MoE, Llama 4, Qwen3.6, GLM-5.2, version, matrix, comparator, and black-box contracts: `266 passed` |
| Pinned official config | exact revision resolves to `GlmMoeDsaForCausalLM` and directly passes the BF16/MLA/TP32 validator |
| Pinned weight metadata | 282 weight files total 1,506,667,387,408 bytes; metadata only, no local shard download |
| Lint/compile | new monitored model, compare model, and focused test pass ruff and Python compilation |
| H100 public case | generated as `public-glm52-tp32-eager-graph` with two deterministic generated cases |
| H100 storage cases | generated as `storage-glm52-{eager,cudagraph}-tp32` with per-rank 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 TP32 pending`. Runtime support requires a
pre-cached pinned checkpoint and 32-H100 evidence for public baseline/monitored
parity, eager/default-graph decoder transport, ClickHouse exact-value,
exact-once and tail checks, and clean shutdown. A multi-node run also requires
an auditable launcher extension; a generated but unrunnable case is not a pass.
