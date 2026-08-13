# DeepSeek V4 Flash lite support audit for vLLM 0.27.1

This record covers the local implementation gate for the NVIDIA DeepSeek V4
Flash plugin model. It does not claim runtime support until the pinned TP4 H100
matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | hardware-isolated registry target `vllm.models.deepseek_v4:DeepseekV4ForCausalLM`; qualified implementation `vllm/models/deepseek_v4/nvidia/model.py` |
| DMI integration | branch `dmi-v0.27.1-deepseek-v4-flash`; commit `65280f85f` |
| Production checkpoint | `deepseek-ai/DeepSeek-V4-Flash`, revision `60d8d70770c6776ff598c94bb586a859a38244f1` |
| Weight contract | 46 safetensor files totaling 159,617,149,040 bytes (about 148.655 GiB); block-FP8 linear/attention weights, MXFP4 experts, dynamic BF16 activations, 128×128 blocks |
| Decoder contract | 43 MHC layers with four residual streams; hidden size 4096; sparse MLA/CSA schedule with two sliding layers followed by alternating 4×/128× compression and one extra MTP ratio; 256 routed experts plus one shared expert, top-6 sqrt-softplus routing; first three layers use token-hash routing |
| Lite cell | NVIDIA H100, TP4/PP1/DP1, no EP/SP/EPLB/ubatching/context parallelism, ordinary V1 offline decoding, native `deepseek_v4_fp8` quantizer, no SM100 MegaMoE and no speculative/MTP |
| Explicitly excluded from DMI | four-dimensional MHC residual streams, Q/K/V/Z, MLA/CSA compressor/indexer/cache tensors, router logits/top-k decisions, expert-local activations, MLP post-activation, attention weights, KV cache, and MTP layer |
| Explicitly untested | production loading, public parity, storage values, CUDA graph compatibility, PP/DP/EP/SP/EPLB/ubatching/context parallelism, serving/tool/reasoning parser paths, speculative/MTP/DSpark, LoRA, alternative MoE/attention backends, ROCm/XPU/SM100, and other DeepSeek V4 checkpoints |

Per-layer MHC residual state has shape `[tokens, 4, 4096]`, which the current
DMI hidden-state contract cannot represent. Sparse MLA/CSA and hash routing are
also backend-owned and do not implement the standard QKV/router hook contract.
The reduced manifest therefore exposes only uniform, authoritative 2D decoder
boundaries rather than publishing incorrect shapes or meanings.

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | DMI subclasses the exact NVIDIA class exported by the hardware-isolated plugin and mutates the native model/layer instances only after upstream construction. The forward signature, logits path, loader, mapper, expert mapping, and MoE protocol stay upstream-owned. |
| M04 | source verified; runtime pending | Disabled hooks delegate to upstream. Enabled hooks preserve both fused MHC post/pre calls, attention, MoE, final MHC reconstruction/collapse, and final RMSNorm order. H100 public parity remains required. |
| M05-M07 | preserved, unclaimed | PP, Eagle/MTP, sequence parallel, and backend interfaces remain inherited, while the lite validator rejects PP, speculative/MTP, SP, EP, and unverified runtime cells. Ordinary loading keeps the upstream `mtp.` skip. |
| M08 | intentionally omitted | The first three layers route by token-hash table; later layers use sqrt-softplus/noaux correction bias, and some backends own routing internally. No single public hook point is authoritative across the pinned native path, so router/top-k families are not advertised. |
| M09-M10 | unchanged, runtime pending | Native FP8/MXFP4 configuration, weight mapper, expert mapping, fused shared expert, MHC broadcast finalization, and MTP skip remain inherited. The pinned 148.655 GiB tree still requires TP4 H100 loading evidence. |
| M11-M13 | adapted, CPU verified | The ordered manifest contains token IDs, embedding, five uniform 2D boundaries for each of 43 layers, collapsed final residual, final norm, and logits: 220 families. Same-graph compare buffers cover all advertised families. |
| M14-M15 | bounded | The validator freezes MHC, sparse-attention compression, hash/MoE routing, YaRN, native FP8/FP4, BF16, TP4, and NVIDIA non-MegaMoE assumptions. Opaque and non-2D omissions are explicit. |
| R01-R03 | verified | vLLM maps `DeepseekV4ForCausalLM` to the hardware plugin; DMI remaps it to `deepseek_v4_p:DeepseekV4PForCausalLM`, which imports the NVIDIA implementation explicitly. A separate compare target is registered. |
| R04-R07 | CPU verified | Official-wheel lazy targets resolve, vLLM's pinned custom `DeepseekV4Config` passes the validator, and the model-wide/compare inventories retain all 43 layers and only the 220 advertised families. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `38 passed` in `tests/test_deepseek_v4_p_contract.py` |
| Bounded combined gate | DeepSeek V4 model, version, matrix, comparator, and black-box contracts: `122 passed` |
| Pinned official config | revision metadata instantiated with vLLM's `DeepseekV4Config` and passed the native FP8/FP4, BF16, and TP4 validator |
| Pinned weight metadata | 46 weight files totaling 159,617,149,040 bytes; metadata only, no shard download |
| Lint/compile | new monitored model, compare model, and focused test pass `ruff`; Python compilation passes |
| H100 public case | generated as `public-deepseek_v4_flash-tp4-eager-graph` with two deterministic generated cases |
| H100 storage cases | generated as `storage-deepseek_v4_flash-{eager,cudagraph}-tp4` with per-rank 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 TP4 pending`. Runtime support requires the
pinned checkpoint to pass native FP8/MXFP4 loading, baseline/monitored public
parity, eager/default-graph reduced decoder transport, ClickHouse exact-value,
exact-once and tail checks, and clean four-rank shutdown. MTP, DSpark, parser,
and non-NVIDIA platform paths remain outside this qualification cell.
