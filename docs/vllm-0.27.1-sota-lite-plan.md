# vLLM 0.27.1 SOTA model expansion plan

This file freezes the nine-model expansion scope requested for the DMI vLLM
0.27.1 port. It separates inexpensive local implementation evidence from the
H100 evidence required before a model is called supported.

## Frozen scope and order

| Order | Checkpoint | Architecture | vLLM implementation | DMI tier | Status |
| ---: | --- | --- | --- | --- | --- |
| 1 | `openai/gpt-oss-20b` | `GptOssForCausalLM` | `gpt_oss:GptOssForCausalLM` | full text decoder | lite implemented; H100 pending |
| 2 | `Qwen/Qwen3-30B-A3B` | `Qwen3MoeForCausalLM` | `qwen3_moe:Qwen3MoeForCausalLM` | full text decoder | lite implemented; H100 pending |
| 3 | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `Llama4ForConditionalGeneration` | `mllama4:Llama4ForConditionalGeneration` | multimodal decoder | lite implemented; H100 TP4 pending |
| 4 | `Qwen/Qwen3.6-27B` | `Qwen3_5ForConditionalGeneration` | `qwen3_5:Qwen3_5ForConditionalGeneration` | multimodal decoder | queued |
| 5 | `zai-org/GLM-5.2` | `GlmMoeDsaForCausalLM` | `deepseek_v2:GlmMoeDsaForCausalLM` | full text decoder | queued |
| 6 | `MiniMaxAI/MiniMax-M2.7` | `MiniMaxM2ForCausalLM` | `minimax_m2:MiniMaxM2ForCausalLM` | full text decoder | queued |
| 7 | `google/gemma-4-E2B-it` | `Gemma4ForConditionalGeneration` | `gemma4_mm:Gemma4ForConditionalGeneration` | multimodal decoder | queued |
| 8 | `deepseek-ai/DeepSeek-V4-Flash` | `DeepseekV4ForCausalLM` | `vllm.models.deepseek_v4:DeepseekV4ForCausalLM` | full text decoder/plugin | queued |
| 9 | `moonshotai/Kimi-K3` | `KimiK3ForConditionalGeneration` | `vllm.models.kimi_k3:KimiK3ForConditionalGeneration` | multimodal decoder/plugin | queued |

`multimodal decoder` means the public model still accepts the named text/image
inputs and produces ordinary vLLM outputs, while DMI exports only the fused
language-decoder tensors. Encoder, projector, audio, and video tensors are not
silently included in that tier. Each model audit must list its exact omissions.

## Lite gate

The local agent gate for each row is intentionally bounded:

1. freeze the exact checkpoint architecture and vLLM 0.27.1 registry target;
2. complete M01-M15 and R01-R07 source discovery for the named tier;
3. preserve the upstream constructor, forward, loader, quantization, and
   parallel declarations while adding the smallest hook boundary;
4. pass lazy official-wheel resolution, configuration, hook-inventory, shape,
   routing, and compare-buffer CPU contracts;
5. add pinned public and storage cases to the `sota` H100 matrix.

Lite completion is `lite implemented; H100 pending`, not `supported`.

## H100 completion gate

The H100 runner must retain separate baseline and monitored artifacts, execute
eager and default CUDA graph, compare the complete public output contract, and
compare active DMI tensors with same-graph D2D references through ClickHouse.
TP/EP cells are added only when their row ownership has a truthful oracle.
Skipped multimodal, plugin, kernel, topology, or quantization prerequisites stay
explicitly untested.

Run the currently implemented SOTA cells with:

```bash
MKL_THREADING_LAYER=GNU .venv/bin/python \
  tests/tools/run_vllm_release_matrix.py \
  --phase sota --gpus 0,1,2,3 --artifact-dir /path/to/sota-artifacts
```

The matrix writes bounded per-case logs and a structured manifest; only failing
log excerpts should be returned to the implementation agent.
