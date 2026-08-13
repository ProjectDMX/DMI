# DMI-vLLM 0.25.1 compatibility audit

This report records what was actually inspected and executed for the vLLM
0.25.1 port. It is deliberately narrower than a blanket "vLLM 0.25 support"
claim. The machine-readable boundary inventory is
[`vllm-0.25.1-boundary-inventory.json`](vllm-0.25.1-boundary-inventory.json).
The public corpus and its current limits are described in
[`vllm-blackbox-test-coverage.md`](vllm-blackbox-test-coverage.md).

## Identity

| Field | Value |
| --- | --- |
| DMI implementation commit | `38ed5137aab2067d14a3f6baa3b384ffc31a3c2a` |
| Previous supported vLLM ref/commit | `v0.19.0` / `2a69949bdadf0e8942b7a1619b229cb475beef20` |
| Previous DMI vLLM integration commit | `c8e312f72f6ce67e73c85f7dd78b7f2c785ac138` |
| Target vLLM ref/commit | `v0.25.1` / `752a3a504485790a2e8491cacbb35c137339ad34` |
| Target DMI vLLM integration commit | `2228df2b07ebcdb68dcf836dc46f1587fec2cdd1` |
| DMI-only replay range | `v0.19.0..c8e312f72f6ce67e73c85f7dd78b7f2c785ac138` |
| Integration shape | root gitlink plus a versioned vLLM patch branch; model variants also register out of tree against the official wheel |
| Runtime package | official `vllm==0.25.1` wheel at `/tmp/dmi-vllm025-venv/lib/python3.12/site-packages/vllm` |
| Runtime environment | Python 3.12.8, PyTorch 2.11.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 4090 |
| Auditor/date | Codex agent / 2026-08-12 |

The adapter depends on the vLLM V1 model runner. Every validated process set
`VLLM_USE_V2_MODEL_RUNNER=0` and `VLLM_DISABLE_COMPILE_CACHE=1` before importing
vLLM. `DMXGPUWorker` rejects V2 before base worker device initialization.

## Compatibility cells

| Cell | Architecture/checkpoint | API | platform/dtype | execution | topology | storage | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0251-gpt2-eager` | `GPT2LMHeadModel` / `gpt2` | V1 offline `LLM.generate` | CUDA/BF16 | eager | TP=1 | transport, no database | experimental |
| `0251-gpt2-graph` | `GPT2LMHeadModel` / `gpt2` | V1 offline `LLM.generate` | CUDA/BF16 | default CUDA graph | TP=1 | transport, no database | experimental |
| `0251-qwen2-eager` | `Qwen2ForCausalLM` / `Qwen/Qwen2.5-0.5B-Instruct` | V1 offline `LLM.generate` | CUDA/BF16 | eager | TP=1 | transport, no database | experimental |
| `0251-qwen2-graph` | `Qwen2ForCausalLM` / `Qwen/Qwen2.5-0.5B-Instruct` | V1 offline `LLM.generate` | CUDA/BF16 | default CUDA graph | TP=1 | transport, no database | experimental |
| `0251-qwen3-eager` | `Qwen3ForCausalLM` / `Qwen/Qwen3-0.6B` | V1 offline `LLM.generate` | CUDA/BF16 | eager | TP=1 | transport, no database | experimental |
| `0251-qwen3-graph` | `Qwen3ForCausalLM` / `Qwen/Qwen3-0.6B` | V1 offline `LLM.generate` | CUDA/BF16 | default CUDA graph | TP=1 | transport, no database | experimental |
| `0251-llama-static` | `LlamaForCausalLM` aliases | import/registry only | not executed | not executed | not executed | not executed | static-only |
| `0251-qwen2moe-static` | `Qwen2MoeForCausalLM` | import/registry only | not executed | not executed | not executed | not executed | static-only |
| `0251-v2` | all architectures | V2 runner | any | any | any | any | unsupported |

`experimental` means runtime output-equivalence evidence exists, but the
storage, distributed, and negative-path release gates were not rerun for this
version. It does not mean those untested dimensions are implicitly supported.

Explicitly unsupported or untested dimensions:

- V2 runner: unsupported; its request preparation and dispatch boundaries are
  different from the V1 hooks used by DMI.
- Online serving, async engine, streaming, cancellation, speculative decoding,
  quantization, LoRA, prompt-token-ID input, logprobs, and `n > 1`: untested.
- TP>1, PP, DP, EP, and SP: untested on vLLM 0.25.1.
- ClickHouse persistence, exact-once tail flush, read reconstruction, concurrent
  capture isolation, and scoped cleanup: untested on vLLM 0.25.1.
- Llama and Qwen2-MoE: import and source-contract evidence only. Available model
  artifacts did not fit the bounded single-GPU validation cell used here.
- The Llama-compatible aliases listed in `docs/vllm.md` were not independently
  instantiated and remain untested.

## Boundary coverage

The profile-driven audit found 241 candidate occurrences grouped into 132
semantic boundaries. The agent inspected and mapped every group to the DMI-vLLM
checklist. The JSON artifact retains each occurrence, checklist mapping, and
rationale; it reports no script errors or warnings.

| Kind | candidates | groups | main semantic areas |
| --- | ---: | ---: | --- |
| vLLM imports | 68 | 36 | B02, W, C, G, M, R |
| vLLM config fields | 96 | 48 | B05, C, G, M, N, L |
| environment keys | 18 | 18 | B05, C, G, L |
| copied model implementations | 5 | 5 | B07, M |
| upstream overrides | 9 | 9 | B03, W, G, M |
| upstream inheritance | 2 | 2 | B03, W, R |
| private or patched attributes | 7 | 7 | B04, G, R |
| lazy/dynamic model targets | 6 | 6 | B06, R |
| native/generated sources | 30 | 1 | B08, N, L |
| **total** | **241** | **132** | B01-B09 and W/C/S/G/M/R/N/L/P |

Unmapped boundary groups: **0**.

## Contract audit

The table groups rows that share one verdict; every checklist ID is named. Full
boundary-group-to-ID mappings are in the JSON inventory.

| Checklist IDs | Final verdict | Contract result and evidence |
| --- | --- | --- |
| B01-B09 | unchanged/adapted-verified | Correct root, adapter, native, registry, and target vLLM trees were scanned. All 132 groups have an agent-authored semantic mapping. Five copied model families were diffed and imported. |
| W01-W02 | adapted-verified | `GPUWorker` remains the worker boundary. V2 now fails before base `init_device`; V1 initializes DMI after the worker device exists. Covered by the fail-closed focused test and runtime construction. |
| W03-W05 | adapted-verified | Official-wheel signatures resolve as `load_model(*, load_dummy_weights=False)`, `compile_or_warm_up_model() -> CompilationTimes`, and `execute_model(scheduler_output)`. DMI forwards the load keyword, accepts the new warm-up return, and preserves execute output/exception behavior. |
| W06 | N/A for claimed cells | The claimed runtime cells are TP=1 only. Distributed RPC and per-rank aggregation are not claimed. |
| W07 | unchanged-verified for claimed cells | Monitored black-box runs call public `collective_rpc("stop_monitoring")` and complete without masking generation errors. Partial-init, repeated-stop, and distributed aggregation remain outside the claim. |
| C01-C03 | adapted-verified | Official-wheel construction transports `VllmConfig`, model identity, and `additional_config` into `DMXGPUWorker`; all three runtime architectures loaded the intended P variant. |
| C04 | N/A beyond TP=1 | Rank fields are inspected but v0.25.1 distributed ownership was not executed. |
| C05 | adapted-verified | The port recognizes v0.25 graph configuration and disables the incompatible compile cache. Both eager and default CUDA-graph cells pass. |
| C06 | unchanged-verified only for tested defaults | Ordinary batched prefill/decode with the test engine defaults passes. Cache/scheduler/speculative variants are explicitly untested. |
| C07 | unchanged-verified | CPU contracts cover disabled/no-op configuration, and baseline public runs contain no DMI worker. Resource-level no-op profiling was not performed. |
| S01-S04, S07-S08 | adapted-verified for ordinary V1 generation | The V1 scheduler output, packed request order, actual token counts, padding, request-ID normalization, and empty producer paths are exercised by 117 request-order regressions plus ragged public batches. |
| S05-S06 | N/A beyond ordinary single-candidate completion | Abort/preemption/resume and multi-sequence/speculative paths are not claimed. Length/stop completion metadata is compared by the public oracle. |
| G01-G04, G07-G10 | adapted-verified for V1 | DMI still patches V1 `_prepare_inputs` and `_determine_batch_execution_and_padding`; focused tests cover ordering, padding, capacity, non-request steps, and exactly-once metadata placement. Eager and graph public runs verify output mapping. |
| G05-G06 | N/A for claimed cells | Async scheduling, PP, and SP are not part of the claimed cells. |
| M01-M04, M10-M12, M14 | adapted-verified per model status | GPT-2, Qwen2, and Qwen3 loaded real official-wheel checkpoints and passed output equivalence. All five P variants import and expose hook inventories. v0.25 constructors, `WeightsMapper`/`AutoWeightsLoader`, mixins, and auxiliary hidden-state returns were ported. Llama/MoE remain static-only. |
| M05-M09, M13 | N/A beyond claimed cells | PP, speculative, TP>1, EP/MoE runtime, quantization, and module-free distributed placement are not claimed. Qwen2-MoE source was updated from obsolete `SharedFusedMoE` assumptions to v0.25 `FusedMoE`, but has no runtime verdict. |
| R01-R07 | adapted-verified | Architecture remaps resolve to the intended lazy P classes in both the DMI source tree and official wheel. All 13 P/ref/compare modules import through the wheel; registration does not initialize CUDA. Alias runtime behavior beyond tested architectures is untested. |
| N01-N03, N05-N08, N10, N12 | unchanged-verified for single-rank transport | The current extension builds against PyTorch 2.11/CUDA 13, enum/import parity tests pass, and monitored runs transport selected hooks without storage while preserving public output. |
| N04, N09, N11 | N/A beyond claimed cells | Distributed shape/placement and injected native capacity/failure cases were not run for v0.25.1. |
| L01-L03 | unchanged-verified only for normal single-rank completion | Initialization and explicit stop execute in all monitored public runs. Failure-safe teardown and repeated stop remain untested. |
| L04-L10 | N/A for claimed cells | Claimed cells use transport with an empty database host. No v0.25.1 storage claim is made. |
| P01, P03-P05, P07-P08 | adapted-verified | Separate-process baseline and monitored runs compare prompt token IDs, generated token IDs, decoded text, finish reason, stop reason, cardinality, and ordering for eager and graph modes. |
| P02, P06, P09-P12 | N/A beyond current public cells | Text prompts and batched calls are covered; prompt-token-ID calls, logprobs, resource-level disabled mode, `DMILLM` stored results, error transparency, and serving are not claimed for v0.25.1. |
| E01-E05, E08, E10 | satisfied at the declared experimental/static scope | Audit coverage is complete; focused, static, CPU, eager, graph, and official-wheel evidence is recorded. The exact commit triplet and exclusions are explicit. |
| E06-E07, E09 | blocked for a `supported` release claim | Storage, distributed, and negative-path gates remain open, so no release tag or `supported` status is proposed. |

## Commit replay ledger

| Old DMI commit(s) | disposition | New commit(s) | Reason and checklist areas |
| --- | --- | --- | --- |
| `ca031463` | apply/rewrite | `5e5ea135`, `2228df2b` | Hooked model portion replayed; obsolete `FullHiddenStatesConnector` portion dropped because v0.25 has a different connector API. M01-M14, R |
| `a18f7ea2` | apply | `8cd08cf3` | Token-ID dtype behavior remains required. S03, N08 |
| `56dea17d` | drop/upstreamed | none | The old connector-specific slot-mapping fix targets a removed connector and cannot be carried mechanically. S/G/N |
| `02245979`..`eb93c3a5` | apply/rewrite | `e98c9255`..`a5fadb12`, `2228df2b` | Reference/compare variants, hook selections, TP-era code, Llama/MoE variants, and inventories replayed; v0.25 constructor, mixin, forward, MoE, and loader drift rewritten. M/R/N |
| `c8e312f7` | apply/rewrite | `03eb163e`, `2228df2b` | Qwen2 monitored variant replayed and rebased onto v0.25 Qwen2 contracts. M/R |
| root adapter port | rewrite | `38ed5137` | V1-only guard, official-wheel compatibility, return annotation, tests, runner environment, docs, and new gitlink. W/C/S/G/P |

## Test coverage

All runtime commands used the official `vllm==0.25.1` wheel, not an editable
installation of the fork. Baseline and monitored black-box modes ran in separate
processes against the same fixed corpus plus six seed-`20260812` generated cases.

| Test/case | Type | Main checklist IDs | Oracle | Result |
| --- | --- | --- | --- | --- |
| `test_vllm_version_compat.py`, `test_qwen2_p_inventory.py`, `test_vllm_blackbox_contract.py`, `test_vllm_comparator_contract.py`, `test_moe_v1_routing_hooks.py` | focused/CPU | W, C, M, R, P | signatures, fail-closed V2, inventory and comparator invariants | 19 passed, 8 skipped for the separate modified-Transformers prerequisite |
| `test_vllm_request_order_fix.py` | focused/CPU | S01-S08, G01-G10, N04-N07 | adversarial request order, padding, metadata, capacity, and lifecycle invariants | 117 passed |
| Qwen2.5 eager | public black box | P01, P03-P05, P07 | prompt tokens, generated tokens, text, finish/stop metadata | passed |
| Qwen2.5 CUDA graph | public black box | C05, G04, P08 | same observable fields as eager | passed |
| Qwen3 eager | public black box | P01, P03-P05, P07 | same observable fields | passed |
| Qwen3 CUDA graph | public black box | C05, G04, P08 | same observable fields as eager | passed |
| GPT-2 eager | public black box | P01, P03-P05, P07 | same observable fields | passed |
| GPT-2 CUDA graph | public black box | C05, G04, P08 | same observable fields as eager | passed |
| `compileall`, Ruff `F`, and `git diff --check` on the vLLM patch | static | E03 | import/syntax/undefined-name/patch integrity | passed |
| 13 DMI P/ref/compare module imports through the official wheel | runtime import | M/R, E08 | lazy module and class resolution | passed |

Skipped prerequisites and resulting status:

- The eight focused-test skips require DMI's separate modified Transformers
  checkout and do not represent vLLM runtime failures.
- No suitably bounded Llama or Qwen2-MoE runtime checkpoint fit the validation
  cell, so those architecture rows are `static-only`.
- ClickHouse/storage and multi-GPU resources were not included in this run;
  E06, E07, and their dependent cells remain unclaimed.

## Support decision

| Cell group | Final status | Evidence | Remaining risk |
| --- | --- | --- | --- |
| GPT-2/Qwen2/Qwen3, V1 offline, TP=1, BF16, eager, no database | experimental | official-wheel deterministic public-output equivalence | storage, distributed, failures, other public features |
| GPT-2/Qwen2/Qwen3, V1 offline, TP=1, BF16, default CUDA graph, no database | experimental | official-wheel deterministic public-output equivalence through graph dispatch | other graph modes and the same open release gates |
| Llama/Qwen2-MoE | static-only | source diff, compile, inventory, and official-wheel import | no real checkpoint execution |
| V2 runner | unsupported | fail-closed regression | adapter requires a separate V2 request/dispatch design |
| serve/async/distributed/storage/quant/speculative | untested | explicitly excluded | dedicated compatibility cells and gates required |

Proposed fork tag: **none**. A tag such as `dmi-v0.25.1-r1` should only be cut
after the desired release cells pass their storage, distributed, and failure
gates.

Exact implementation pair:

```text
DMI root:        38ed5137aab2067d14a3f6baa3b384ffc31a3c2a
vLLM integration: 2228df2b07ebcdb68dcf836dc46f1587fec2cdd1
upstream base:    752a3a504485790a2e8491cacbb35c137339ad34
```
