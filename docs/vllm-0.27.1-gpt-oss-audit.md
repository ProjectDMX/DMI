# GPT-OSS 20B lite support audit for vLLM 0.27.1

This record covers the local implementation gate for OpenAI's GPT-OSS 20B
decoder. It does not claim runtime support until the pinned H100 matrix passes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/gpt_oss.py`, `GptOssForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-sota-gpt-oss`; commit `a08206f1dce85cc9211a1cabb4fb02015eaa7a09` |
| Production checkpoint | `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee` |
| Weight contract | native MXFP4 experts; unquantized attention, router, embeddings, and LM head |
| Decoder contract | 24 layers alternating sliding/full attention; attention sinks; YaRN; biased QKV/O projections; 32 local experts; top-4 `swigluoai` routing |
| Lite cell | TP1, V1 offline API, native MXFP4; public eager/default graph and ClickHouse cells are deferred to the shared H100 qualification PR |
| Explicitly untested | runtime weight loading, public output parity, storage values, TP/PP/DP/EP/SP, serving, speculative/Eagle, LoRA, Quark/other quantization, alternate GPT-OSS derivatives |

## M/R checklist

| IDs | Lite verdict | Evidence and residual gate |
| --- | --- | --- |
| M01-M03 | adapted, CPU verified | The monitored class subclasses `GptOssForCausalLM`, retains its public signature, and instruments the exact native model tree after the upstream constructor runs. |
| M04 | source verified; runtime pending | Disabled hooks delegate to upstream attention, MoE, and block forwards. Enabled hooks retain upstream QKV/RoPE/attention, fused residual norms, router, experts, and logits expression order. H100 public comparison remains required. |
| M05-M07 | preserved, unclaimed | Upstream PP intermediate tensors and Eagle auxiliary returns are retained. PP, speculative, TP, and sequence-parallel execution are not inferred from TP1; sequence-parallel routing fails closed in the lite cell. |
| M08 | adapted, CPU verified | Every layer exposes router logits and the authoritative `FusedMoE.router.select_experts` top-k IDs/weights. CPU contracts verify router/select/expert ordering and dtypes. EP rank ownership remains an H100 gate. |
| M09-M10 | unchanged, runtime pending | The inherited 3-D expert declaration, complete HF mapper, native MXFP4 loader, bias handling, sinks, and stacked QKV mapping are unchanged. The pinned production shards still require H100 loading evidence. |
| M11-M13 | adapted, CPU verified | The ordered manifest contains token IDs, embeddings, 14 per-layer families, final residual/norm, and logits: 341 families total. Compare buffers cover every active family. |
| M14-M15 | bounded | The adapter accepts only the audited GPT-OSS alternating-attention, YaRN, biased-attention, untied-embedding, top-k MoE contract and BF16/native MXFP4 weights. Missing expert-local activations, attention weights, KV cache, and sequence-parallel routing are explicit. |
| R01-R03 | verified | vLLM maps `GptOssForCausalLM` to `gpt_oss:GptOssForCausalLM`; DMI remaps it one-to-one to `gpt_oss_p:GptOssPForCausalLM`. The fork registry contains separate monitored and compare targets. |
| R04-R07 | CPU verified | The official 0.27.1 wheel lazily resolves the bundled monitored and compare classes without importing model code during registration. Runtime construction remains part of the H100 matrix. |

## Local evidence

| Gate | Result |
| --- | --- |
| New focused contracts | `28 passed` in `tests/test_gpt_oss_p_contract.py` |
| Registry and official-wheel contracts | combined GPT-OSS plus version selection: `36 passed` |
| Pre-reorganization local gate | GPT-OSS and shared compatibility contracts passed on the cumulative development branch; this PR reruns the standalone model contract |
| Lint/compile | new model, compare model, and focused test pass ruff and Python compilation |
| H100 public case | deferred to the shared qualification PR as `public-gpt_oss-tp1-eager-graph` |
| H100 storage cases | deferred to the shared qualification PR as `storage-gpt_oss-{eager,cudagraph}-tp1` with 2 GiB ring buffers |

## Support decision

Status is `lite implemented; H100 pending`. The architecture must not be listed
as supported until the pinned production checkpoint passes upstream loading,
separate-process public baseline/monitored comparison, eager/graph transport,
ClickHouse exact-value/exact-once/tail checks, and clean shutdown.
