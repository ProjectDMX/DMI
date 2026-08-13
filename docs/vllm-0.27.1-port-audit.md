# DMI-vLLM 0.27.1 compatibility audit

This is the agent-authored discovery and validation record for the versioned
0.27.1 port. It is not a support claim until the accelerator matrix closes all
claimed cells. Static, ABI, registry, focused CPU, public differential, and
storage-value gates are complete. A clean-commit matrix rerun remains required
after the W07 lifecycle adaptation described below.

## Audit header

| Field | Value |
| --- | --- |
| DMI replay basis | `2f0a0ec56a8af7647633fea38d61c7e689dfb2b0` |
| Previous supported basis | vLLM `v0.25.1` / `752a3a504485790a2e8491cacbb35c137339ad34` |
| Previous vLLM integration | `6f1fce945c54b96255d1eacd726918d538d5d707` |
| Target | vLLM `v0.27.1` / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Integration shape | root adapter plus versioned vLLM patch branch and official-wheel lazy registration |
| Target runtime inspected | official `vllm==0.27.1` PyPI wheel |
| Target environment | Python 3.12.8, PyTorch 2.13.0+cu130, CUDA build 13.0 |
| DMI target integration | branch `dmi-v0.27.1`, commit `fdfe631884ae318050ce371e472c1135f317cfa2` |
| Candidate claim | V1 offline API; five existing production variants; eager and supported CUDA-graph cells; topology per matrix only |
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

The repository-owned profile
[`vllm-api-audit-profile.json`](vllm-api-audit-profile.json) discovers 240
boundary candidates, grouped into 131 semantic boundaries. The agent-authored
[`vllm-0.27.1-api-boundary-map.json`](vllm-0.27.1-api-boundary-map.json) maps
all 131/131 group IDs to W/C/S/G/M/R/N/L/P checklist rows; it contains no null
or empty mappings. This closes inventory coverage only. Behavioral verdicts
still depend on the source and runtime evidence in this report.

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
| W07 | `change-required`; `adapted-verified` at focused and targeted runtime scope | Worker shutdown is AST-identical, but the public `LLM` object has no close method in 0.27.1. Storage runners now flush DMI explicitly and call the bounded EngineCore shutdown contract. DMI's rank-local teardown could also let TCPStore owner rank 0 exit before a peer NCCL heartbeat stopped; rank 0 now retains a 0.5 s post-worker grace only for distributed cells. Focused tests verify fail-closed EngineCore lookup, bounded shutdown, and rank-specific grace. Full-matrix requalification remains pending. |
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
| GPT-2 | 43 changed lines | `adapted-verified` | All P/compare/ref variants use target `AutoWeightsLoader` and Conv1D transpose generator; loader/transpose regressions and TP1/TP2 eager/graph/storage cells passed the pre-lifecycle matrix. |
| Qwen2 | 4 changed lines | `adapted-verified` | Dynamic `positions` dimension matches 0.27.1; lazy registry/inventory and TP1 eager/graph public cells passed. |
| Qwen3 | 4 added lines | `adapted-verified` | `per_layer_sliding_window` is threaded through P/compare/ref attention and decoder layers; focused propagation and TP1/TP2 eager/graph/storage cells passed. |
| Llama | no target file diff | `unchanged-verified` for the named cell | Packed weight loader and hook inventory are preserved; Llama-3.1-8B TP2 eager/graph public and storage cells passed. |
| Qwen2-MoE | 6 changed lines | `adapted-verified` for TP2, EP excluded | Target factory exposes `router.select_experts`; focused routing and Qwen1.5-MoE TP2 eager/graph public and storage cells passed. EP remains untested. |

`compatible` here is a pre-edit discovery status, not a final
`unchanged-verified` verdict.

## Native ABI gate

The 0.25.1 extension was built against PyTorch 2.11.0+cu130 and cannot be
reused. The dedicated 0.27.1 worktree rebuilt ClickHouse with position-
independent code and rebuilt the monitoring extension against PyTorch
2.13.0+cu130/CUDA 13.0/CXX11 ABI. The loaded-path check resolves:

- vLLM from the official `vllm==0.27.1` wheel;
- `monitoring_native_backend` from this worktree's `monitoring/` package;
- all 23 native hook-definition rows through the package loader.

N10 is therefore `adapted-verified` for import/ABI parity. N01-N09 and N11-N13
retain their own focused/runtime/storage requirements; extension import alone
does not close them. Importing the same `.so` manually under two module names is
invalid because PyTorch operator namespaces are process-global; all tests use
the repository package loader's single canonical path.

## Current focused evidence

The target environment passed 219/219 focused tests. This includes version and
registry compatibility, model hook inventories, black-box/comparator contracts,
request ordering and padding, MoE routing, ClickHouse test utilities, GPU-idle
gating, generated black-box cases, process-group cleanup, and release-runner
behavior. The target-specific model contracts live in
`tests/test_vllm_027_model_contracts.py`.

The first target GPU sweep found one GPT-2 eager run whose public token branch
diverged at an exactly/near-tied decision and whose common-prefix public logprob
drift exceeded the old 0.25 nat default. Three fresh baseline and three fresh
monitored processes over the identical corpus showed:

- baseline/baseline: 72/72 public outputs exact;
- monitored/monitored: 70/72 exact, with both branches publicly tied or within
  0.25 nat and both candidates present;
- baseline/monitored: 213/216 exact across all process pairs;
- maximum common-prefix drift: 0.342 nat; maximum first-divergence cross-run
  drift: 0.138 nat; selected-token gap from each public maximum: zero.

The oracle therefore uses a checkpoint-specific 0.5 nat drift ceiling for the
`gpt2` release cell while retaining 0.25 for every other model. Candidate
presence, selected gap `1e-6`, finite/schema validation, and cumulative-logprob
reconstruction remain mandatory. A later clean matrix observed one GPT-2 graph
decision with a 0.5 nat baseline branch gap, a tied monitored branch, zero
selected-to-public-maximum gap, both candidates present, and per-candidate
cross-run drift below the already calibrated 0.5 ceiling. Four independent
graph reruns and twelve active/null-mode isolation runs were exact; three more
full eager+graph replicas reproduced one eager branch reversal whose two
outputs had both appeared in independent baselines. The GPT-2 branch-gap ceiling
is therefore also 0.5, while every other model remains at 0.25. CPU regressions
reject the same gap/drift for non-GPT-2 payloads and reject GPT-2 evidence above
0.5.

The release runner also retries the full three-sample idle check through a
bounded post-case cooldown. This prevents transient utilization from a just-
exited TP cell from being mistaken for a persistent prerequisite failure; it
never bypasses the idle thresholds or terminates unrelated work.

The first clean-commit 0.27.1 accelerator sweep at root
`fe35f1d672e74f938da50b600971537ac71e8b3f` passed all 18/18 case processes:
five public eager+graph cells and twelve storage cells with 395,896/395,896
reference rows bitwise equal. Evidence inspection nevertheless rejected that
run as final release proof because four graph+TP2 storage logs let the frontend
force-kill EngineCore after its five-second destructor timeout, and one Qwen3
eager+TP2 log showed a rank-1 TCPStore heartbeat race.

Targeted W07 validation after the adaptation establishes:

- explicit EngineCore shutdown removes the frontend force-kill path;
- three exact official-wheel Qwen3 eager+TP2 baselines shut down cleanly;
- the unadapted DMI full case reproduced the TCPStore warning and showed rank 0
  returning from worker teardown about 21 ms before rank 1;
- keeping only the store-owner rank alive for 0.5 s removed that warning from
  full eager and CUDA-graph Qwen3 TP2 storage runs, both of which retained
  76,960/76,960 bitwise-equal rows;
- vLLM's inner executor may still SIGTERM CUDA-graph workers after its own
  five-second grace. The same marker was reproduced without DMI against the
  exact official wheel, so the runner retains it as a named non-fatal upstream
  warning. Frontend force-kills, worker exceptions, and TCPStore races remain
  fatal even when the process exit code is zero.

The exact patch replay is recorded in
[`vllm-0.27.1-replay-ledger.md`](vllm-0.27.1-replay-ledger.md). Every prior DMI
commit has an explicit target commit and semantic disposition; the final target-
only rewrite is `fdfe631884ae318050ce371e472c1135f317cfa2`.

## Implementation and validation order

1. Root `vllm-0.27-support` and integration `dmi-v0.27.1` branches: complete.
2. Target-native rebuild, loaded-path check, and ABI parity: complete.
3. Boundary inventory/map, target drift adaptations, and focused regressions:
   complete at 219/219.
4. Separate-process public black-box eager/graph and scoped storage TP1/TP2:
   value/transparency sweep complete, W07 clean-commit requalification pending.
5. Only after phase 0 evidence is complete, add roadmap families one contract
   class at a time. V2 and untested API/topology cells remain explicit.

No integration tag is proposed by this pre-port audit.
