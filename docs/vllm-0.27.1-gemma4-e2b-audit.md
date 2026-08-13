# Gemma 4 E2B decoder lite support audit for vLLM 0.27.1

This record covers the local decoder-only implementation gate for the public
Gemma 4 E2B multimodal wrapper. It does not claim runtime support until the
pinned H100 matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/gemma4.py`, `Gemma4ForCausalLM`; `gemma4_mm.py`, `Gemma4ForConditionalGeneration` |
| DMI integration | branch `dmi-v0.27.1-gemma4-e2b`; commit `a2b095e24c262017e6a1c9c2990b5f9ef399edcb` |
| Production checkpoint | `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Weight contract | one BF16 weight file, 10,246,621,918 bytes (about 9.543 GiB); no quantization config |
| Decoder contract | 35 dense layers, hidden size 1536, 8 Q heads and 1 KV head; four sliding layers followed by one full-attention layer; sliding head dimension 256 and full head dimension 512; the final 20 layers use a double-wide MLP; PLE is enabled for every layer |
| Public multimodal contract | upstream vision, audio, video, embedding merge, PLE preparation, and logits suppression remain owned by `Gemma4ForConditionalGeneration`; DMI observes the fused language-decoder input and decoder boundaries only |
| Lite cell | TP1/PP1/DP1, BF16, no quantization, speculative execution, KV-sharing fast prefill, EP/SP/EPLB, or context parallelism; public text/image and ClickHouse cases generated for H100 |
| Explicitly excluded from DMI | vision/audio encoders, multimodal projectors and merge internals, Q/K/V/Z, `mlp_post`, attention weights, KV cache, PLE gates/projections, and logits-suppression internals |
| Explicitly untested | production weight loading, public text/image parity, storage values, audio/video requests, PP/DP/TP>1, serving, speculative/Eagle, LoRA, quantization, KV-sharing fast prefill, alternate Gemma 4 checkpoints, and encoder CUDA graphs |

Q/K/V/Z are omitted because the current DMI shape contract has one global
`head_dim`, while this checkpoint uses 256 and 512 in different layers.
`mlp_post` is omitted for the same reason: the current contract has one global
intermediate width, while the checkpoint changes from 6144 to 12288. Exposing
either family would produce incorrect metadata even if the tensor write itself
succeeded.

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | The monitored class subclasses the exact public conditional wrapper. The inner causal model, model tree, multimodal mapping, loader, packed-module metadata, and public input/output types stay upstream-owned. |
| M04 | source verified; runtime pending | Disabled decoder/model hooks delegate to upstream. Enabled hooks preserve Gemma 4's four norms, two residual additions, PLE contribution, per-layer scalar, final norm, and wrapper-level token/logit order. H100 public comparison remains required. |
| M05-M07 | preserved, unclaimed | PP, Eagle, YOCO, and fast-prefill interfaces remain inherited, but the lite validator fails closed for PP, speculative/Eagle, and KV-sharing fast prefill. The H100 cell covers only ordinary TP1 offline decoding. |
| M08 | not applicable | E2B has `enable_moe_block=false`; no routing families are advertised. |
| M09-M10 | unchanged, runtime pending | The conditional wrapper and inner causal loader are inherited without modification. The pinned BF16 file and tied embedding/LM-head path still require H100 loading evidence. |
| M11-M13 | adapted, CPU verified | The ordered manifest contains token IDs, embedding, seven uniform hidden-size boundaries for each of 35 layers, final residual/norm, and logits: 250 families total. Same-graph compare buffers cover all 250 families. Heterogeneous attention and MLP-post families are intentionally absent. |
| M14-M15 | bounded | The validator freezes the official text, vision, and audio configuration, heterogeneous head schedule, double-wide MLP schedule, PLE geometry, BF16 dtype, and TP1 topology. Encoder/projector and audio/video runtime omissions are explicit. |
| R01-R03 | verified | vLLM maps `Gemma4ForConditionalGeneration` to `gemma4_mm:Gemma4ForConditionalGeneration`; DMI remaps it one-to-one to `gemma4_p:Gemma4PForConditionalGeneration`. A separate compare target is registered. |
| R04-R07 | CPU verified | Official-wheel lazy targets resolve, the pinned real `Gemma4Config` passes the validator, and model-wide inventory/compare contracts retain all 35 layer identities without pretending that heterogeneous shapes are global. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `34 passed` in `tests/test_gemma4_p_contract.py` |
| Full focused lite gate | version, all model contracts through Gemma 4 E2B, storage, comparator, black-box, matrix, and lifecycle contracts: `646 passed` |
| Pinned official config | cached metadata-only revision instantiates as `Gemma4Config` / `Gemma4TextConfig` and passes the BF16/TP1 validator, including the heterogeneous per-layer view |
| Pinned weight metadata | one weight file totaling 10,246,621,918 bytes; metadata only, no local weight download |
| Lint/compile | new monitored model, compare model, and focused test pass `ruff`; Python compilation passes |
| H100 public case | generated as `public-gemma4_e2b-tp1-eager-graph` with deterministic text cases plus one 32×32 RGB image case using `<|image|>` |
| H100 storage cases | generated as `storage-gemma4_e2b-{eager,cudagraph}-tp1` with a 2 GiB ring buffer |

## Support decision

Status is `lite implemented; H100 TP1 pending`. Runtime support requires the
pinned checkpoint to pass BF16 loading, baseline/monitored public text and
image parity, eager/default-graph decoder transport, ClickHouse exact-value,
exact-once and tail checks, and clean shutdown. Audio and video remain outside
this qualification cell even if the image cell passes.
