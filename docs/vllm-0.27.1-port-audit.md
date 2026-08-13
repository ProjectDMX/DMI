# DMI-vLLM 0.27.1 pre-port audit

This is the agent-authored discovery record for the next version port. It is
not a support claim. Final checklist verdicts require a dedicated 0.27.1
worktree, a native rebuild, focused regressions, and accelerator evidence.

## Audit header

| Field | Value |
| --- | --- |
| DMI discovery commit | `dc9b642267366f242537338ef70e308b47732716` |
| Previous supported basis | vLLM `v0.25.1` / `752a3a504485790a2e8491cacbb35c137339ad34` |
| Previous vLLM integration | `2228df2b07ebcdb68dcf836dc46f1587fec2cdd1` |
| Target | vLLM `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Integration shape | root adapter plus versioned vLLM patch branch and official-wheel lazy registration |
| Target runtime inspected | official `vllm==0.27.1` PyPI wheel |
| Target environment | Python 3.12.8, PyTorch 2.13.0+cu130, CUDA build 13.0 |
| Initial claim | none; V1 offline discovery only |
| Explicitly outside this discovery | V2, serving, async, speculative, quantized, PP/DP/EP/SP, multimodal runtime |
| Auditor/date | Codex agent / 2026-08-13 UTC |

The release was verified against both the official GitHub release and PyPI.
The tag is peeled locally to the exact target commit above.

## Runtime package and registry evidence

The official wheel exposes the same DMI-used signatures as 0.25.1:

| Surface | vLLM 0.27.1 signature |
| --- | --- |
| `Worker.__init__` | `(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False)` |
| `Worker.load_model` | `(*, load_dummy_weights=False) -> None` |
| `Worker.compile_or_warm_up_model` | `() -> CompilationTimes` |
| `Worker.execute_model` | `(scheduler_output) -> ModelRunnerOutput | AsyncModelRunnerOutput | None` |
| `GPUModelRunner._prepare_inputs` | `(scheduler_output, num_scheduled_tokens)` |
| `GPUModelRunner._determine_batch_execution_and_padding` | unchanged 0.25.1 parameter and return contract |
| `GPUModelRunner.execute_model` | `(scheduler_output, intermediate_tensors=None)` |
| `ModelRegistry.register_model` | `(model_arch, model_cls) -> None` |

The versioned artifact
[`vllm-0.27.1-runtime-registry.json`](vllm-0.27.1-runtime-registry.json)
and repository verifier resolve 41/41 roadmap architectures through the exact
wheel, with zero failures. That proves only R01/R05/R07 upstream packaging and
lazy-target availability. It does not prove DMI remap validity or execution.

## Core semantic diff

AST comparison ignores line movement and formatting. The following methods are
normalized-AST identical between the exact tags:

- `Worker.load_model`, `Worker.execute_model`, and `Worker.shutdown`;
- `GPUModelRunner._prepare_inputs`;
- `GPUModelRunner._determine_batch_execution_and_padding`.

The changed methods require these semantic verdicts:

| Checklist IDs | Discovery status | Exact drift and DMI consequence |
| --- | --- | --- |
| W01 | `compatible` for the proposed V1 cell | `Worker.__init__` adds fault-tolerance sentinel and draft-buffer state. Constructor shape and V1 runner selection are unchanged. Fault tolerance and draft updates remain outside the initial cell. |
| W02 | `compatible` | `init_device` changes only an unsupported-device error message in its body. Model-runner construction and device/distributed ordering used by DMI remain unchanged. Runtime construction is still required. |
| W03 | `compatible` | `load_model` is AST-identical, including keyword forwarding and model-loader context. |
| W04 | `change-required` evidence | Warm-up now accounts memory through `total_consumed` and can persist a startup plan before returning. DMI may still clear native null mode after `super()`, but must prove startup-plan replay and graph capture do not publish warm-up rows. |
| W05 | `compatible` for non-Mamba V1 cells | Worker execution is AST-identical. Runner execution adds only `align_ctx=mamba_bufs.postprocess_align` in the Mamba branch; existing five variants cannot enter it. Hybrid/Mamba families need a separate verdict. |
| W07 | `compatible` at source level | Worker shutdown is AST-identical. Target-PyTorch native rebuild and repeated-stop tests remain required. |
| C01-C06 | `change-required` audit | Config files changed substantially, including model, parallel, compilation, cache, scheduler, speculative, and new fault-tolerance/EC-manager configuration. Every DMI-read field must be re-traced; unchanged method signatures do not approve these fields. |
| S01-S04, G01-G04, G07-G10 | `compatible` at source level for ordinary V1 | Both DMI patch points are AST-identical. Their enclosing execution path is unchanged for existing dense/MoE non-Mamba cells except unrelated branches. Request-order, actual/padded rows, early PP/SP dispatch, and exactly-once commit must be rerun. |
| G05-G06 | `blocked` outside initial cell | Async scheduling, PP, SP, fault tolerance, and newer runner branches are not covered by source identity. |
| R01, R05, R07 | `compatible` upstream evidence | Public registry shape is unchanged and all catalog targets resolve from the official wheel. |
| R02-R04, R06 | `change-required` DMI evidence | DMI P-class registration/remapping has not yet been installed in a rebuilt target runtime. Lazy import must be re-proved without parent CUDA initialization. |

The target has 367 registered architectures versus 356 in the 0.25.1 wheel.
The curated combined registry delta is recorded in the model roadmap; added or
remapped names never inherit a DMI verdict automatically.

## Existing model variants

| DMI variant | Target source drift | Discovery status | Required port action |
| --- | --- | --- | --- |
| GPT-2 | 43 changed lines | `change-required` | Replace the copied manual loader with target `AutoWeightsLoader` plus the target Conv1D transpose generator; preserve attention-mask skips and add a loader regression. |
| Qwen2 | 4 changed lines | `compatible` pending focused test | Target base now detects mixed attention through `config.layer_types`; DMI subclasses the target base at runtime, but constructor behavior and sliding-window cells must be verified. |
| Qwen3 | 4 added lines | `change-required` | Thread `per_layer_sliding_window` through the copied attention and decoder-layer constructors. Test both absent and configured values. |
| Llama | no target file diff | `compatible` at source level | Re-run constructor, loader, hook inventory, TP2, eager, graph, and storage cells. Aliases remain separate evidence rows. |
| Qwen2-MoE | 6 changed lines | `change-required` evidence | Target replaces `FusedMoE` with `FusedMoEFactory`. Revalidate `experts.router.select_experts`, routing tensor semantics, loader behavior, and TP/EP ownership before retaining the wrapper. |

`compatible` here is a pre-edit discovery status, not a final
`unchanged-verified` verdict.

## Native ABI gate

The current repository extension was built in the 0.25.1 environment against
PyTorch 2.11.0+cu130. The 0.27.1 wheel installs PyTorch 2.13.0+cu130. Reusing
the old extension is invalid:

- loading the old extension first then importing target torch produced a torch
  C-extension symbol mismatch;
- importing target torch first caused the repository extension loader to reject
  the old binary.

Therefore N10 and every runtime/import gate involving DMI remain `blocked`
until a dedicated 0.27.1 worktree rebuilds the extension with target PyTorch.
An extension import by itself will not close N01-N12.

## Implementation and validation order

1. Create root `vllm-0.27-support` and fork `dmi-v0.27.1` branches from exact
   immutable bases after the 0.25.1 GPU matrix is understood.
2. Rebuild ClickHouse and the monitoring native extension in the isolated
   0.27.1 environment; verify loaded-path and enum parity.
3. Port core V1 boundaries, fill every B/W/C/S/G/R/N/L/P checklist row, and add
   focused regressions for warm-up/startup-plan behavior and config drift.
4. Port the existing five models in the table above and resolve every declared
   hook on real models.
5. Run separate-process baseline/monitored black-box cases in eager and graph,
   then scoped storage and TP1/TP2 gates.
6. Only after phase 0 evidence is complete, add roadmap families one contract
   class at a time. Keep V2 and untested API/topology cells explicit.

No integration tag is proposed by this pre-port audit.
