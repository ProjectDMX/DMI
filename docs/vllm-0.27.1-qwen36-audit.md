# Qwen3.6-27B decoder lite support audit for vLLM 0.27.1

This record covers the local decoder-only implementation gate for Qwen3.6-27B.
It does not claim runtime support until the pinned H100 matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | public wrapper and hybrid decoder in `vllm/model_executor/models/qwen3_5.py`; full-attention primitive in `qwen3_next.py` |
| DMI integration | branch `dmi-v0.27.1-qwen3-6-decoder`; commit `63e1e541f7719b23b02e5bf071f3eb64a1ec47b8` |
| Production checkpoint | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Weight contract | unquantized BF16; inherited public-wrapper loader, QKV packing, dense MLP packing, and Qwen3.5 GDN projection mapper |
| Decoder contract | 64 dense layers: 48 Gated DeltaNet linear-attention layers and 16 full-attention layers in a repeated 3:1 schedule; 24 Q heads, 4 KV heads, 256 head dimension, gated full-attention output, interleaved MRoPE |
| Multimodal tier | public text and image inputs remain upstream-owned; DMI exports fused language-decoder inputs and layer boundaries only |
| Lite cell | TP1/PP1/DP1, V1 offline API, BF16, public eager/default graph and ClickHouse cases generated for H100 |
| Explicitly excluded from DMI | vision encoder, image/video preprocessing, visual patch tensors, multimodal merge/projector tensors, GDN convolution/recurrent state, attention weights, KV cache, and internal kernel temporaries |
| Explicitly untested | production weight loading, public text/image parity, storage values, video inputs, TP/PP/DP/EP/SP, serving, speculative/Eagle, LoRA, quantization, prefix-cache variants, alternate Qwen3.5/Qwen3.6 checkpoints |

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | The monitored public class subclasses `Qwen3_5ForConditionalGeneration` and inherits its processor, visual tower, multimodal mapping, forward contract, and complete loader. The already-constructed native language model is instrumented in place. |
| M04 | source verified; runtime pending | Disabled full-attention, MLP, and decoder hooks delegate to upstream. GDN layers continue to call the untouched upstream `linear_attn`. Enabled full attention reuses upstream `_project_qkv_gate`, attention, sigmoid output gate, and output projection order. Public H100 differential comparison remains required. |
| M05-M07 | preserved, unclaimed | Hybrid-state declarations and public multimodal behavior remain upstream-owned. The lite validator rejects non-TP1 topology and sequence-parallel execution rather than extrapolating from a single GPU. GDN state tensors are intentionally absent from DMI. |
| M08 | not applicable | The named 27B checkpoint is dense, not MoE; no router or expert tensor family is fabricated. |
| M09-M10 | unchanged, runtime pending | Public wrapper loading and the Qwen3.5 stacked mappings for QKV, GDN QKVZ/BA, and dense gate/up projections are inherited unchanged. The pinned BF16 shards still require H100 loading evidence. |
| M11-M13 | adapted, CPU verified | All 64 layers expose common residual, norm, attention-output, and MLP boundaries including post-activation. Only the 16 real full-attention layers expose Q/K/V/Z. With five global families the heterogeneous manifest contains 581 families, and compare buffers cover the same set. Q/K are post-norm and post-MRoPE, V is pre-attention, and Z is post-output-gate/pre-output-projection. |
| M14-M15 | bounded | The adapter accepts the exact 27B geometry, 3-GDN/1-full schedule, interleaved MRoPE, gated attention, vision-tower identity, unquantized BF16, and TP1 cell. Every omitted encoder, GDN-state, cache, and kernel-internal family is explicit. |
| R01-R03 | verified | vLLM maps `Qwen3_5ForConditionalGeneration` to `qwen3_5:Qwen3_5ForConditionalGeneration`; DMI remaps it one-to-one to `qwen3_5_p:Qwen3_5PForConditionalGeneration`. Separate monitored language-model and conditional compare targets are registered. |
| R04-R07 | CPU verified | Official-wheel lazy targets resolve without eager model construction. The wrapper delegates `get_hook_specs(model_wide=True)` only to its language decoder; the visual tower exposes no DMI inventory. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `46 passed` in `tests/test_qwen3_5_p_contract.py` |
| Full cumulative lite gate | GPT-OSS, Qwen3-MoE, Llama 4, Qwen3.6, version, matrix, comparator, and black-box contracts: `220 passed` |
| Pinned official config | exact revision resolves to `Qwen3_5ForConditionalGeneration` with the audited 64-layer 3:1 schedule and passes the validator directly |
| Model-specific public image input | deterministic RGB input uses Qwen's public `<|vision_start|><|image_pad|><|vision_end|>` placeholder rather than a family-agnostic token |
| Lint/compile | new model, compare model, and focused test pass ruff and Python compilation |
| H100 public case | generated as `public-qwen36-tp1-eager-graph`, including deterministic text and image inputs |
| H100 storage cases | generated as `storage-qwen36-{eager,cudagraph}-tp1` with 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 pending`. Runtime support requires the pinned
checkpoint to pass public text/image baseline comparison, eager/graph decoder
transport, ClickHouse exact-value/exact-once/tail checks, and clean shutdown.
