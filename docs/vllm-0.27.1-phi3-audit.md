# Phi-3 model support audit for vLLM 0.27.1

This agent-authored record supports the text-only `Phi3ForCausalLM`
architecture in a bounded TP1 BF16 V1 offline cell. It does not extend the
verdict to untested parallel, quantized, serving, or speculative modes.

## Frozen identity and cell

| Field | Value |
| --- | --- |
| Upstream vLLM | `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Upstream implementation | `vllm/model_executor/models/phi3.py`, `Phi3ForCausalLM` |
| DMI integration | branch `dmi-v0.27.1-phi3`, commit `b4965771d8a4cc25afad81732d86180ab105796d` |
| Production checkpoint | `microsoft/Phi-3.5-mini-instruct`, revision `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| Focused fixture | `optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM`, revision `e2c91eceaf7aa2b6e7ff09b89cd4f8e79eb8131c` |
| Claimed cell | TP1, BF16, V1 offline `LLM`, eager and default CUDA graph, full canonical DMI hook inventory |
| Excluded | TP>1, PP/DP/EP/SP, V2, serve/async, speculative, LoRA, quantization, embedding/classification tasks, remote-code variants |

## M/R checklist

| IDs | Verdict | Evidence and rationale |
| --- | --- | --- |
| M01 | verified | The target `phi3.py` is an eight-line thin subclass of the exact target Llama implementation. |
| M02-M04 | verified for the claimed cell | Upstream Phi-3 changes no constructor or numerical forward path. DMI therefore reuses the already-audited hooked Llama model rather than duplicating it. Production eager/graph strict public parity verifies the long-RoPE configuration path. |
| M05-M09 | excluded where applicable | PP, speculative, distributed ownership, MoE, and quantized paths were not run and are not claimed. |
| M10 | adapted-verified | Phi-3's only upstream class delta is `packed_modules_mapping = {qkv_proj: [qkv_proj], gate_up_proj: [gate_up_proj]}` for already-fused checkpoint tensors. DMI uses a distinct Phi-3 subclass that copies this map; it is deliberately not placed in the generic Llama alias set. Both official production shards loaded successfully. |
| M11-M13 | verified for TP1 | The inherited Llama hook boundaries and module-free inventory are unchanged. The tiny fixture's full-hook compare model produced byte-identical eager/graph ClickHouse tensors. |
| M14 | bounded | Only `Phi3ForCausalLM` text generation is remapped. Phi-3 Vision, Phi-3 Small, embedding/classification, remote-code, LoRA, and quantized implementations do not inherit this verdict. |
| M15 | verified for the dense fixture | The audited implementation contains only Llama-style dense attention/MLP layers; no heterogeneous hook family is fabricated. |
| R01-R03 | verified | Upstream resolves `Phi3ForCausalLM` to `phi3:Phi3ForCausalLM`; DMI resolves it one-to-one to `phi3_p:Phi3PForCausalLM`. The remap is distinct from Llama specifically to preserve packing. |
| R04-R07 | verified | Official-wheel lazy registration resolves the exported class without eager model import or parent CUDA initialization, through the already-audited 0.27.1 registry surface. |

## Runtime evidence

| Gate | Result |
| --- | --- |
| Focused packing/registry contracts | 29/29 selected tests passed; the final expansion focused sweep passed 240/240. The contract fails if Phi-3 is added to generic Llama aliases or if ordinary split Q/K/V and gate/up packing returns. |
| Tiny public eager+graph | 2/2 full 12-case API-only tests passed in 82.29 s with strict baseline/monitored equality. |
| Tiny eager full-hook transport | 4,320/4,320 independent D2D references were byte-identical to ClickHouse rows. |
| Tiny CUDA-graph full-hook transport | 4,320/4,320 independent D2D references were byte-identical to ClickHouse rows. |
| Production eager | 1/1 public test passed in 32.78 s. |
| Production eager+graph | 2/2 public tests passed in 107.00 s. The checkpoint's two 7.6 GB total fused-weight shards and long-RoPE config loaded successfully. |
| Lifecycle | No worker exception, force-kill, TCPStore warning, residual vLLM process, residual capture, or GPU allocation remained. |

The tiny fixture establishes hook values and transport correctness cheaply; the
official checkpoint separately closes the real fused-weight loader and
representative public-output gate. Storage row count is not compared across
unrelated executions: the byte oracle reconstructs exact per-request token
ranges within the same execution.

## Support decision

`Phi3ForCausalLM` is supported for the named TP1 BF16 V1 offline eager/default-
graph cell at the exact commits above. The supported architecture includes the
official Phi-3.5 Mini checkpoint exercised here, but a checkpoint with materially
different remote code, quantization, task wrapper, or runner branch requires its
own cell. All excluded modes remain untested rather than implicitly supported.
