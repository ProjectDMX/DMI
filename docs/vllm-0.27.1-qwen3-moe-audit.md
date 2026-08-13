# Qwen3-30B-A3B lite support audit for vLLM 0.27.1

This record covers the local implementation gate for Qwen3-30B-A3B. It does
not claim runtime support until the pinned H100 matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/qwen3_moe.py`, `Qwen3MoeForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-qwen3-moe`; commit `56218183a2f7fab9b6f89915575e3f5584d33388` |
| Production checkpoint | `Qwen/Qwen3-30B-A3B`, revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Weight contract | unquantized BF16 checkpoint; inherited Qwen3-MoE stacked-QKV, dense-MLP, and MoE loaders |
| Decoder contract | 48 all-MoE layers; bias-free GQA with post-projection Q/K RMSNorm; RoPE theta 1,000,000; 128 routed experts; normalized top-8 routing; no shared expert |
| Lite cell | TP1, V1 offline API, BF16, public eager/default graph and ClickHouse cases generated for H100 |
| Explicitly untested | runtime weight loading, public output parity, storage values, TP/PP/DP/EP/SP/EPLB, serving, speculative/Eagle, LoRA, quantization, shared experts, dense fallback layers, alternate Qwen3-MoE checkpoints |

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | The monitored class subclasses `Qwen3MoeForCausalLM`, retains its public signature and loader metadata, and instruments the exact native model tree after the upstream constructor runs. |
| M04 | source verified; runtime pending | Disabled hooks delegate to upstream attention, MoE, and decoder-layer forwards. Enabled hooks preserve upstream QKV, Q/K norm, RoPE, attention, fused residual norms, router, experts, and logits expression order. H100 public comparison remains required. |
| M05-M07 | preserved, unclaimed | Upstream PP intermediate tensors and Eagle auxiliary returns are retained. The lite config rejects non-TP1 topology, EP, sequence-parallel MoE, and EPLB rather than inferring them from single-GPU behavior. |
| M08 | adapted, CPU verified | Every layer exposes gate logits and `FusedMoE.router.select_experts` top-k IDs/weights. The internal-router call still receives hidden states exactly as upstream requires. CPU contracts verify both external and internal router paths. |
| M09-M10 | unchanged, runtime pending | The inherited HF mapper, QKV packing, expert weight loader, MoE metadata, and BF16 weight contract are unchanged. The pinned production shards still require H100 loading evidence. |
| M11-M13 | adapted, CPU verified | The ordered manifest contains token IDs, embeddings, 14 per-layer families, final residual/norm, and logits: 677 families total. Compare buffers cover every active family. Q/K observations are post-QK-norm and pre-RoPE; V is pre-attention; Z is pre-output projection. |
| M14-M15 | bounded | The adapter accepts only the audited all-MoE, no-shared-expert, normalized-top-k, unquantized Qwen3-30B-A3B contract. Expert-local activations, attention weights, KV cache, and dispatch/combine tensors are explicitly absent. |
| R01-R03 | verified | vLLM maps `Qwen3MoeForCausalLM` to `qwen3_moe:Qwen3MoeForCausalLM`; DMI remaps it one-to-one to `qwen3_moe_p:Qwen3MoePForCausalLM`. The fork registry contains separate monitored and compare targets. |
| R04-R07 | CPU verified | The official 0.27.1 wheel lazily resolves the bundled monitored and compare classes without importing model code during registration. Runtime construction remains part of the H100 matrix. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `31 passed` in `tests/test_qwen3_moe_p_contract.py` |
| Full local lite gate | GPT-OSS, Qwen3-MoE, version compatibility, release-matrix, and compare-runner contracts: `93 passed` |
| Pinned official config | exact revision resolves to `Qwen3MoeForCausalLM`, 48 layers, 128 experts, top-8 and passes the normalized-RoPE contract |
| Lint/compile | new model, compare model, and focused test pass ruff and Python compilation |
| H100 public case | generated as `public-qwen3_moe-tp1-eager-graph` |
| H100 storage cases | generated as `storage-qwen3_moe-{eager,cudagraph}-tp1` with 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 pending`. The architecture must not be listed
as supported until the pinned production checkpoint passes upstream loading,
separate-process public baseline/monitored comparison, eager/graph transport,
ClickHouse exact-value/exact-once/tail checks, and clean shutdown.
