# MiniMax-M2.7 lite support audit for vLLM 0.27.1

This record covers the local implementation gate for MiniMax-M2.7. It does not
claim runtime support until the pinned TP4 H100 matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/minimax_m2.py`, `MiniMaxM2ForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-sota-minimax-m27`; commit `3eaa295ad0ebb08fa8126b9f69bdc745e12d7bc5` |
| Production checkpoint | `MiniMaxAI/MiniMax-M2.7`, revision `d494266a4affc0d2995ba1fa35c8481cbd84294b` |
| Weight contract | native block-FP8 with dynamic activations and 128×128 blocks; gate, routing correction bias, and LM head excluded from FP8; 125 weight files total 230,134,260,592 bytes (about 214.329 GiB) |
| Decoder contract | 62 homogeneous full-attention MoE layers; 48 Q heads, 8 KV heads, 128 head dimension, per-layer Q/K norm, half-head RoPE; 256 experts, sigmoid normalized top-8 routing with FP32 gate and correction bias; no shared expert |
| Checkpoint MTP contract | three appended MTP modules are present in metadata; the inherited main-model loader skips them for ordinary decoding and the lite cell rejects speculative/MTP execution |
| Lite cell | TP4/PP1/DP1, no EP/SP/EPLB/context parallelism, V1 offline API, FP8 weights with BF16 activations; public and ClickHouse cells are deferred to the shared H100 qualification PR |
| Explicitly excluded from DMI | attention weights, KV cache, expert-local activations, correction-bias internals, dispatch/combine tensors, MTP layers, and quantizer/kernel internal tensors |
| Explicitly untested | production weight loading, public parity, storage values, PP/DP/EP/SP/EPLB/context parallelism, serving, speculative/MTP/Eagle, LoRA, alternate quantization, prefix caching, and other MiniMax versions |

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | The monitored class subclasses the exact `MiniMaxM2ForCausalLM` registry target, retains its public forward/logits/load contracts and mapper metadata, and instruments the model tree after upstream construction. |
| M04 | source verified; runtime pending | Disabled attention, MoE, and decoder hooks delegate to upstream. Enabled attention preserves fused QKV projection, authoritative MiniMax Q/K norm, partial RoPE, attention, and output projection order. Enabled decoder hooks preserve fused residual norms and the upstream MoE call. H100 public comparison remains required. |
| M05-M07 | preserved, unclaimed | PP and Eagle interfaces remain inherited, while the lite validator rejects PP, speculative/MTP, sequence/context parallelism, and non-TP4 execution. Ordinary main-decoder loading retains the upstream MTP-skip behavior. |
| M08 | adapted, CPU verified | Every layer exposes the FP32 gate logits and authoritative router top-8 IDs/weights. Routing correction bias, normalized sigmoid selection, and expert execution remain upstream-owned; EP/EPLB paths fail closed. |
| M09-M10 | unchanged, runtime pending | The QKV mapper, block-FP8 loader, FP32 routing exclusions, fused experts, correction bias, LM head, and appended-MTP skip logic are inherited unchanged. The pinned 214.329 GiB weight tree still requires TP4 H100 loading evidence. |
| M11-M13 | adapted, CPU verified | The ordered manifest contains token IDs, embedding, 14 families for each of 62 layers, final residual/norm, and logits: 873 families total. Compare buffers cover every family at TP4 shapes. Q/K are post-QK-norm and pre-RoPE, V is pre-attention, and Z is pre-output projection. |
| M14-M15 | bounded | The validator freezes the exact M2.7 geometry, all-full-attention schedule, MoE routing, partial RoPE, native block-FP8 config, BF16 runtime dtype, and TP4 topology. Cache, expert-local, MTP, and quantizer-internal omissions are explicit. |
| R01-R03 | verified | vLLM maps `MiniMaxM2ForCausalLM` to `minimax_m2:MiniMaxM2ForCausalLM`; DMI remaps it one-to-one to `minimax_m2_p:MiniMaxM2PForCausalLM`. A separate compare target is registered. |
| R04-R07 | CPU verified | Official-wheel lazy targets resolve without eager model construction. Model-wide inventory and compare buffers retain all 62 layer identities and TP4 head/expert shapes. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `41 passed` in `tests/test_minimax_m2_p_contract.py` |
| Pre-reorganization local gate | MiniMax-M2.7 and shared compatibility contracts passed on the cumulative development branch; this PR reruns the standalone model contract |
| Pinned official config | exact revision resolves to `MiniMaxM2ForCausalLM` and directly passes the block-FP8/BF16/TP4 validator |
| Pinned weight metadata | 125 weight files total 230,134,260,592 bytes; metadata only, no local shard download |
| Lint/compile | new monitored model, compare model, and focused test pass ruff and Python compilation |
| H100 public case | deferred to the shared qualification PR as `public-minimax_m27-tp4-eager-graph` with two deterministic cases |
| H100 storage cases | deferred to the shared qualification PR as `storage-minimax_m27-{eager,cudagraph}-tp4` with per-rank 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 TP4 pending`. Runtime support requires the
pinned checkpoint to pass FP8 weight loading, public baseline/monitored parity,
eager/default-graph decoder transport, ClickHouse exact-value, exact-once and
tail checks, and clean four-rank shutdown.
