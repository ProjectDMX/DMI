# DMI-vLLM public black-box test coverage

This document records what the repository's generated public-API corpus can
prove. It is not a substitute for the agent-driven DMI-vLLM API checklist or
for storage/distributed evidence.

## Contract and isolation

The canonical corpus is
[`tests/blackbox/cases/transparency.json`](../tests/blackbox/cases/transparency.json).
Schema version 2 requires every case to declare:

- a stable `case_id` and relevant DMI-vLLM checklist IDs;
- only public input and `SamplingParams` values;
- covered dimensions, observable oracles, and the plausible faulty integration
  the case is intended to kill;
- a deterministic generator name/version/seed for generated cases;
- combinations omitted because they are outside the claimed cell or need a
  different prerequisite.

Baseline and monitored modes run in separate processes against the same model
artifact and corpus. Each process calls only public offline surfaces:
`LLM`, `LLM.get_tokenizer`, `SamplingParams`, `LLM.generate`,
`RequestOutput`, and `CompletionOutput`. The monitored process additionally
uses the advertised `worker_cls` construction and public collective shutdown.

Raw `cases.json`, `baseline.json`, `monitored.json`, subprocess logs, and JUnit
results are retained by the release-matrix artifact directory. A skipped test
is recorded as a blocked prerequisite, never a pass.

## Deterministic core matrix

| Case | Checklist IDs | Main partitions | Oracles | Faults rejected |
| --- | --- | --- | --- | --- |
| `core-one-token-tail` | S03, S05, P03, P05 | short prompt, exactly one generated token | differential, reverse batch | missing final completion; text-only comparison |
| `core-unicode` | P02-P04 | Unicode/emoji serialization, ragged batch | differential, reverse batch | lossy prompt serialization; text-only comparison |
| `core-shared-prefix-a/b` | S02, S04, G07, P07 | same prefix, distinct continuation, reordered execution | differential, reverse batch | scheduler-order attribution; prefix collision |
| `core-ragged-long` | S02-S03, G01-G02, P07 | one long request beside short requests | differential, reverse batch | padded rows treated as real; scheduler-order attribution |
| `core-token-ids` | P02-P03, P07 | token-ID prompt constructed by the public tokenizer | differential, reverse batch | input retokenized; token drift hidden by decoded text |
| `generated-<seed>-NNN` | P02-P05, P07 | reproducible text/Unicode/punctuation/ragged prompts | differential, reverse batch | text-only comparison; scheduler-order attribution |

The same corpus runs in eager and configured CUDA-graph modes. Graph execution
adds C05/G02/G04/G08/P08 evidence at the compatibility-cell level even though
those IDs are not duplicated onto every input row.

## Comparator and metamorphic oracle

The schema-2 comparator fails closed on missing fields, types, list cardinality,
ordering, case IDs, public request IDs, prompt token IDs, candidate count/index,
complete generated token IDs, decoded text, the public output top-logprob schema
and decision values used by the oracle, cumulative log probability when
requested, finish reason, and stop reason. It reports the exact nested field
path. It intentionally excludes timing metrics and other nondeterministic
telemetry.

Each mode executes the canonical batch and its reverse in the same engine. The
metamorphic oracle requires result order to equal input order, matches by
`case_id`, checks prompt and prompt-token attribution, requires unique public
request IDs per call, and validates candidate indices and per-case generation
bounds. Request IDs themselves are not compared across the two calls because
the public allocator advances between calls; they are compared baseline versus
monitored for the same call sequence.

Generated text and token IDs are deliberately not required to match between the
canonical and reversed calls. Changing the batch can change floating-point
reduction shape, so even greedy decoding is not a public token-invariance
guarantee. Those fields remain compared exactly between baseline and monitored
for each identical call sequence as the first oracle.

Every differential run requests the public top-20 output logprobs. If greedy
token IDs first diverge, the comparator accepts the branch only when the two
token alternatives appear in both runs' public decision maps, each actually
selected token is within `1e-6` nat of that run's public maximum, their within-run
branch gaps are at most 0.25 nat, their cross-run drifts are at most 0.25 nat, and
selected-token logprobs on the exact common prefix have no larger drift. Ordinal
rank is retained as evidence but does not decide acceptance because equal public
logprobs can receive different rank numbers; numeric gap defines the ambiguous
decision set. The trace must contain finite values for every selected token and
reconstruct `cumulative_logprob`. These thresholds are fixed before execution.
After this first-divergence proof, later generated tokens are conditioned on
different histories, so the comparator keeps schema, identity, candidate
metadata, finish/stop category, and generation bounds exact but does not require
the later paths to converge. `stability.json` records the accepted oracle and
thresholds.

When public decision logprobs are available, failing that oracle is final; extra
baseline retries cannot replace the first-divergence evidence. Only when this
public evidence is unavailable may the runner execute up to two additional
baseline processes as a bounded secondary oracle. Stable identity-matched cases
remain exact. For a case whose complete public candidate list varies across
independent baselines, the monitored list must equal one entire observed baseline
list; it cannot mix fields or introduce a new result. The runner never expands
this bound or its numeric thresholds after a failure.

## Explicit gaps

The current V1 offline corpus does not claim:

- async, serving, streaming, disconnect, cancellation, or concurrency;
- `n > 1`, beam, or speculative sequence association;
- prompt logprobs and full-vocabulary logit equivalence, which require separate
  public/focused comparators; the current output top-logprobs are decision
  evidence rather than a full-vocabulary correctness claim;
- capacity overflow, native producer/consumer failure, and ClickHouse
  exact-once/isolation, which require the transport/storage negative-path suite;
- PP/DP/EP/SP, multimodal, quantized, or LoRA cells.

These gaps remain checklist/support-matrix exclusions. Future case generation
must add the corresponding public runner and oracle before removing an entry;
adding random prompts alone is insufficient.
