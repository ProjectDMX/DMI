# Kimi K3 lite support audit for vLLM 0.27.1

This record covers the local implementation gate for the NVIDIA Kimi K3
multimodal plugin. It does not claim runtime support until the pinned TP32 H100
matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | hardware-isolated registry target `vllm.models.kimi_k3:KimiK3ForConditionalGeneration`; qualified implementation `vllm/models/kimi_k3/nvidia/model.py` |
| DMI integration | branch `dmi-v0.27.1-kimi-k3`; commit `879791976efe0fea01334b9386fb44b3b220e4d1` |
| Production checkpoint | `moonshotai/Kimi-K3`, revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| Weight contract | 96 safetensor files totaling 1,560,936,091,448 bytes (about 1.420 TiB); compressed-tensors `mxfp4-pack-quantized` routed-expert weights with group size 32 and BF16 runtime tensors |
| Decoder contract | 93 layers, hidden size 7168, 69 KDA layers and 24 MLA layers; attention-residual block size 12; one dense layer followed by 92 MoE layers; 896 routed experts plus two shared experts, top-16 sigmoid/noaux routing |
| Multimodal contract | Kimi K2.5-derived 27-layer vision tower, patch size 14, 2×2 merge, `patchmergerv2` projector, public placeholder `<\|kimi_image_placeholder\|>` |
| Lite cell | NVIDIA H100, TP32/PP1/DP1, ordinary V1 offline text/image decoding, BF16, native compressed-tensors MXFP4 path, no EP/SP/EPLB/ubatching/context parallelism, no SM100 MegaMoE, and no speculative/MTP |
| Explicitly excluded from DMI | vision encoder and projector activations; three-dimensional attention-residual block state; KDA state/conv/gates; MLA Q/K/V/output gates and cache tensors; router logits/top-k decisions; expert-local tensors; MLP post-activation; attention weights; KV/KDA caches |
| Explicitly untested | production loading, public parity, image preprocessing/parity, storage values, CUDA graph compatibility, PP/DP/EP/SP/EPLB/ubatching/context parallelism, multi-node launch, serving/tool/reasoning parser paths, speculative/MTP, LoRA, alternative attention/MoE backends, ROCm/XPU/SM100, video, and other Kimi checkpoints |

Kimi K3's attention-residual state has shape `[tokens, 8, 7168]` for the
pinned PP1 cell, which the current DMI hidden-state contract cannot represent.
The KDA and MLA layers also have different internal attention meanings, and the
first decoder layer is dense while the remaining layers are MoE. The reduced
manifest therefore exports only stable two-dimensional boundaries common to
every layer. It does not label heterogeneous internal tensors as conventional
QKV, routing, or MLP-post families.

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | DMI subclasses the exact NVIDIA public wrapper and instruments the native `KimiLinearForCausalLM`, model, and decoder-layer instances only after upstream construction. Public forward signatures, multimodal protocols, loaders, mapper, hybrid-state interfaces, and quantization ownership remain upstream-owned. |
| M04 | source and CPU verified; runtime pending | Disabled layer hooks delegate directly to upstream. Enabled hooks preserve pre-attention normalization, KDA/MLA dispatch, SP branches, post-attention normalization, dense/MoE execution, final attention-residual collapse, and the intentionally deferred final RMSNorm order. H100 public parity remains required. |
| M05-M07 | preserved, unclaimed | PP, hybrid KDA cache, Eagle/MTP, encoder CUDA graphs, and sequence-parallel declarations remain inherited, while the lite validator rejects PP, speculative/MTP, SP, EP, and other unverified runtime cells. Native weight loading and multimodal mapping are unchanged. |
| M08 | intentionally omitted | KDA/MLA internals and latent MoE routing do not provide one authoritative conventional QKV/router boundary across all 93 layers. No QKV, router-logit, top-k, or expert-local hook family is advertised. |
| M09-M10 | unchanged, runtime pending | The official compressed-tensors MXFP4 config, ignored unquantized modules, BF16 dtype, shared/dense paths, and checkpoint mapper stay upstream-owned. The pinned 1.420 TiB tree still requires TP32 H100 loading evidence. |
| M11-M13 | adapted, CPU verified | The ordered manifest contains token IDs, embedding, five uniform boundaries for each of 93 layers, collapsed final residual, final norm, and logits: 470 families. Same-graph compare buffers cover exactly those advertised families. |
| M14-M15 | bounded | The validator freezes the public/text/vision configs, 69-KDA/24-MLA schedule, attention-residual layout, MoE geometry, MXFP4 format, BF16, and TP32 NVIDIA assumptions. Every nonuniform or opaque omission is explicit. |
| R01-R03 | verified | vLLM maps `KimiK3ForConditionalGeneration` to the hardware plugin; DMI remaps it one-to-one to `kimi_k3_p:KimiK3PForConditionalGeneration`. A separate `kimi_k3_compare` target is registered for storage validation. |
| R04-R07 | CPU verified | Official-wheel lazy monitored and compare targets resolve; vLLM's `KimiK3Config` normalization passes the pinned validator; the model-wide and compare inventories retain all 93 layers and only the 470 advertised families. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `38 passed` in `tests/test_kimi_k3_p_contract.py` |
| Bounded combined gate | Kimi K3 model, version, matrix, comparator, black-box, and artifact contracts: `128 passed` |
| Final cumulative focused gate | all vLLM 0.27.1 lite contracts, including the frozen nine-model scope: `722 passed` |
| Pinned official config | revision `config.json` instantiated with vLLM's `KimiK3Config` and passed the normalized public/text/vision, MXFP4, BF16, and TP32 validator |
| Pinned weight metadata | 96 weight files totaling 1,560,936,091,448 bytes; metadata only, no shard download |
| Lint/compile | new monitored model, compare model, and focused test pass `ruff`; Python compilation passes |
| H100 public case | generated as `public-kimi_k3-tp32-eager-graph` with deterministic text cases plus one image case using `<\|kimi_image_placeholder\|>` |
| H100 storage cases | generated as `storage-kimi_k3-{eager,cudagraph}-tp32` with per-rank 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 TP32 pending`. Runtime support requires the
pinned checkpoint to pass native MXFP4 loading, baseline/monitored public
text-and-image parity, eager/default-graph reduced decoder transport,
ClickHouse exact-value, exact-once and tail checks, and clean 32-rank shutdown.
The current release runner requires 32 GPUs visible on one host; a multi-node
run does not count until a launcher extension preserves the same artifact and
oracle contract.
