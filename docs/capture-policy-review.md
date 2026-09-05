# Optimization policy capture plan: design review

Status: Review of a proposed design; no policy implementation accepted

Baseline: verified against `main` at `cb4e490`. See the addendum at the end for
what `feat/dmi-configurator` -- merged into this branch after the review was
written -- already resolves.

This document reviews a proposed design that would make optimization policy a
first-class part of the capture plan. The proposal introduces user-facing
policies (fastest / balanced / complete / budget / custom), per-selection
importance levels, a `dmi/v1 CapturePlan` YAML, a deterministic policy
resolver under `src/dmi/planning/`, native pressure semantics, and an eight-
phase delivery plan.

The review holds the proposal against the code as it exists on `main`. The
architecture is sound. Several of its premises about DMI's current runtime are
not, and the corrections change the phase ordering.

## Verdict

The core principle is right:

> The user states intent and constraints; the configurator resolves them into
> explicit capture, sampling, buffering, and persistence settings.

So is the `PolicyIntent` / `EffectiveCapturePlan` split, and the requirement
that the serialized plan — not hidden UI logic — is what the runtime consumes.

The problem is sequencing. The proposal repeatedly assumes runtime machinery
DMI does not have, and it inverts the pressure behavior DMI does have. Its
recommended first milestone — "Balanced, Fastest, and Custom using DMI's
existing schedule and buffer controls" — cannot be built as written, because
none of those three rest on controls that are wired up.

## Findings

| # | Finding | Severity |
|---|---|---|
| 1 | `CaptureSchedule` is dead config; no adapter enforces it | Blocking |
| 2 | Pressure model is inverted: DMI already blocks, cannot drop | Blocking |
| 3 | Per-layer capture selection does not exist | Blocking |
| 4 | `layers.overrides` encodes catalog-derived facts as user input | Design |
| 5 | Pressure is two domains, and a vocabulary already exists | Design |
| 6 | Binding constraint is per-step bytes, not MiB/s | Design |
| 7 | Estimator must be rank-aware; budget units are ambiguous | Design |
| 8 | `register_preset` breaks the determinism acceptance criterion | Design |
| 9 | Runtime adaptation has one lever, and it needs quiescence | Design |

### 1. `CaptureSchedule` is dead config

`src/dmi/config.py` defines `should_capture_request` and
`should_capture_step`. Across the whole repository the only callers are
`tests/test_config.py` and the documentation. `src/dmi/engine.py` assigns
`self.config = config` and never reads it again. `docs/integration-api-v1.md`
states the contract plainly:

> concrete adaptors decide whether and how to apply it; the engine does not
> enforce the schedule by itself.

No shipped adapter applies it. `docs/vllm.md`'s configuration table exposes no
stride knobs at all.

The proposal's Phase 4 ("Map effective schedules into `CaptureSchedule`") is
therefore not a mapping task; it is implementing schedule enforcement in the
adapters. Until that exists, a resolver emitting `step: {stride: 2}` produces a
YAML that does not describe what DMI executes — which violates the proposal's
own acceptance criterion.

### 2. The pressure model is inverted

`RingEnginePy::prepare_step` in `native/csrc/ring/ring_engine_py.cu` is a
pre-forward host-side capacity check. The fast path reserves and returns. When
the ring is full it synchronizes the main stream, calls
`force_flush_and_wait()`, reserves, and returns `STEP_RING_FLUSHED`. That is
lossless blocking backpressure at step granularity. `ring/ring_state.h` records
the invariant:

> Space is guaranteed by the pre-forward capacity check before kernel launch.

There is no drop path anywhere in the transport. Two consequences:

- The proposal hedges that complete capture may be "unavailable" pending
  lossless backpressure. Block is what already exists. Complete is closer to
  shippable than assumed — what it lacks is *accounting*, not mechanism.
- The proposal's Fastest preset requires "never introduce blocking behavior."
  **Fastest is the policy that is unimplementable today**, because every
  capture path blocks on a full ring. Its only honest current form is: size the
  rings so blocking is unlikely, and use null mode as the kill switch.

Phase 6 should be reordered on this basis. `drop` and `fail` are the new work;
`block` is the status quo.

### 3. Per-layer capture selection does not exist

`resolve_hook_selection` in `src/dmi/hooks/selection.py` unions comma-separated
tokens into a set of hook *types*. There is no layer dimension in selection,
in `CaptureSchedule`, or in the vLLM knobs.

So `capture.layers.default: [0, 8, 16, 24, 31]` and `capture.layers.overrides`
have no runtime counterpart, and the proposal's own decision-trace example —
"Selected five representative transformer layers" — is currently
inexpressible.

Layer sampling is also the highest-leverage byte-reduction lever every preset
depends on: Fastest's "representative layers for optional per-layer hooks" and
Budget's "layer sampling before removing high-value global signals" both reduce
to it. It is Phase 1 work, not an assumption.

### 4. `layers.overrides` encodes catalog facts as user input

`final_logits: global` and `token_ids: global` are not user choices.
`src/dmi/hooks/catalog.py` sets `per_layer=False` for both, as it does for
`embed`, `pos_embed`, `final_ln`, and `resid_final`.

Derive scope from `HOOK_DEFS`. A per-layer override on a global hook should be
a validation error, not an accepted key. The same applies to pipeline placement
(`PP_FIRST` / `PP_LAST`) and tensor-parallel sharding — all catalog-derived.

### 5. Pressure is two domains, and a vocabulary already exists

DMI has two distinct pressure stages that fail differently and on different
timescales:

| Stage | Current behavior | Telemetry |
|---|---|---|
| GPU ring transport | Block only (`prepare_step`) | None |
| Host storage spool | `BLOCK` or `DROP_NEWEST` | Loss counters and peak occupancy |

`src/dmi/storage/capture/pipeline.py` already defines `OverloadPolicy` and
`AdmissionResult{ACCEPTED, DROPPED, TIMED_OUT, TOO_LARGE, CLOSED}`.
`src/dmi/storage/capture/record_adapter.py` already tracks a `_LOSS_COUNTERS`
tuple covering dropped, timed-out, oversized, duplicate, and rejected records
plus failures. `QueueSnapshot` tracks `peak_records` and `peak_bytes` — the
proposal's `high_watermark_bytes`, implemented one layer up.

Do not introduce a third parallel `PressureStrategy` enum. Extend this
vocabulary, and make `pressure:` a per-stage mapping (`transport:`, `spool:`,
`persistence:`) instead of one global block. A single `exhausted_action` cannot
describe a system where one stage blocks and another drops.

### 6. The binding constraint is per-step bytes, not MiB/s

When `step_total_bytes > min(payload_cap, staging_cap)`, `prepare_step` returns
`STEP_OVERSIZED` and the adapter falls back to `force_eager` plus CPU-direct
dispatch. That is a large, quiet performance cliff — not data loss.

The resolver must validate peak single-step bytes against *effective* capacity
(`min(payload, pinned)`, already exposed as `RingCapacities.effective_bytes` in
`src/dmi/engine.py`). Prefill dominates that peak because `q_len` is large. The
proposal treats `phases: {prefill, decode}` as booleans, but max-prefill-step
sizing is what decides whether a plan silently degrades.

The proposal's error example is right in spirit. Make it check effective
capacity, and emit a warning with an explanation for the fallback case rather
than only a hard error — DMI degrades rather than failing.

### 7. The estimator must be rank-aware

`compute_hook_shape` divides sharded hooks by `tp_size`, and `filter_by_tp_rank`
keeps unsharded hooks only on rank 0. Aggregate bytes are therefore roughly
TP-invariant, but ring pressure is not: rank 0 carries every unsharded hook plus
its own shard, and `PP_FIRST` / `PP_LAST` skew the pipeline stages the same way.

`max_capture_mib_per_second: 800` does not say whether it is per-rank or
aggregate. Add `scope: per_rank | aggregate` to each budget constraint, and
evaluate the ring-capacity validation against the **worst rank**, not the
average.

### 8. `register_preset` breaks the determinism criterion

The acceptance criterion "the same manifest and policy always generate the same
effective plan" is violated by `register_preset` in
`src/dmi/hooks/selection.py`, which mutates a module-global dictionary at
adapter import time. The resolved plan therefore depends on which adapters were
imported.

The plan fingerprint must cover the resolved hook-type set, not the preset
string. `resolution.model_fingerprint` alone is insufficient.

### 9. Runtime adaptation has one lever, and it needs quiescence

`MonitoringEngine.set_capture_enabled` documents that callers "must ensure that
no forward pass can overlap this method." The classic producer kernels in
`native/csrc/ring/producer.cu` gate only on the device-global
`g_ring_null_mode` — all or nothing. Only the newer record producers accept a
per-launch `emit_gate` device pointer.

So the proposal's safe adaptive sequence — raise step stride for optional
hooks, raise request stride for optional hooks, disable optional hooks, and so
on — assumes per-hook schedules that exist in no form. `CaptureSchedule` is
global; hook selection is fixed at `attach_model`. Today's real lever set is one
global schedule (unenforced), one global selection (attach-time), and one kill
switch (quiescent).

The cheap unlock is extending `emit_gate` to the classic producers. That yields
a CUDA-graph-safe, per-hook, per-step, device-side gate, which is exactly what
every adaptive step needs. It should be an explicit Phase 6 deliverable; the
adaptive controller is not buildable without it.

## Smaller corrections

- **No metrics surface exists.** There is no Prometheus, Grafana, or exporter
  anywhere in the repository. The telemetry section implies a surface that is
  not there; an exporter is its own workstream, absent from the phase list.
  Name new counters off the existing `_LOSS_COUNTERS` vocabulary so the
  reference path and the transport path report in the same terms.
- **No CLI exists.** `pyproject.toml` declares no `[project.scripts]`. The
  proposed `dmi plan {validate,resolve,estimate,explain}` needs a new console
  entry point. `pyyaml` and `pydantic` are already in `requirements.txt`, so the
  schema layer adds no dependencies.
- **No model manifest exists.** `llama-3.1-8b.dmi-model.yaml` is entirely new;
  there is no YAML configuration in the repository today. `ModelShapeConfig` in
  `src/dmi/hooks/specs.py` is the natural serialization target and covers most
  of what the estimator needs.
- **`persistence.finalize_mode: wait` maps cleanly.** `flush_and_wait` in
  `src/dmi/engine.py` and the `stop_monitoring` endpoint in `docs/vllm.md` are
  real counterparts. This part of the proposal is well grounded.
- **Complete needs a fourth number.** Selection, transport, and persistence
  completeness omit *schedule* completeness. Folding stride into "eligible
  planned tensors" makes 100% completeness achievable while capturing every
  other step — not the reading a user of "Complete" expects. The proposal
  already forces stride 1 for complete mode; the report should also state the
  denominator's provenance, and both figures should be scoped per session and
  per rank.
- **Separating instantaneous capacity from sustained bandwidth is the best idea
  in the proposal.** It mirrors the real structure: `prepare_step` enforces
  instantaneous capacity, while the drain thread's `drain_flush_*` thresholds
  govern sustained drain. Keep it.

## Suggested resequencing

The proposal's phases are ordered by abstraction layer. They should be ordered
by dependency.

1. **Runtime prerequisites.** Enforce `CaptureSchedule` in the adapters; add
   per-layer selection to `src/dmi/hooks/selection.py`; extend `emit_gate` to
   the classic producers. Without these there is nothing for a resolver to
   resolve.
2. **Policy schema and static presets** (the proposal's Phase 1), with `custom`
   wrapping legacy configuration.
3. **Estimator and resolver**, made rank-aware and driven by peak-prefill-step
   bytes against effective capacity.
4. **Completeness accounting**, reusing the spool's existing counters.
5. **Drop and fail transport semantics** (block already exists), then the
   adaptive controller, then calibration.

Ship **Balanced** and **Custom** first. Hold **Fastest** until `drop` exists —
it currently promises "never blocks" over a transport that only blocks. Hold
**Complete** until accounting exists, not until backpressure exists; the
backpressure is already there.

---

## Addendum: status on this branch

The findings above were verified against `main`. `feat/dmi-configurator` has
since been merged into this branch, and it already resolves several of them.
The record above is left as written, because it reviews a proposal that
targeted `main`; this addendum states what is no longer true.

### Resolved by the configurator

| Review point | Status on this branch |
|---|---|
| "No model manifest exists" | Resolved. `ModelDescriptor` + `dmi describe-model`. |
| "No YAML configuration in the repository" | Resolved. `dmi.configuration.yaml`, golden fixtures under `tests/golden/`. |
| "No CLI exists" | Partly resolved. `dmi ui` and `dmi describe-model` ship; `dmi plan …` does not. |
| Finding 3, per-layer selection | Partly resolved. Inclusive layer *ranges* exist and are now applied by the runtime; arbitrary layer *sets* still are not. |
| MoE hooks grouped as `GROUP_OTHER` | Resolved. `catalog_adapter` layers a presentation grouping over catalog groups. |
| Availability reasons in the UI | Resolved. Derived from `select_hook_specs`, not reimplemented. |

### Resolved by this branch

**Finding 3, runtime application.** A layer range authored in the configurator
was previously never applied: `attach_model` had no way to receive it. It now
takes a keyword `layers` argument and applies `filter_by_layers` between
`apply_hook_selection` and the PP/TP filters, driven from a `DMIConfig` by
`dmi.configuration.attach_config`. This closes a silent over-promise -- the
policy control in the UI was honestly disclaimed, but layer selection, the
UI's central feature, was not.

**Findings 6 and 7, made usable.** `dmi.configuration.estimate` reports the
peak single step against `min(payload, pinned)` effective capacity and names
the worst rank, rather than reporting only a rate or a model-wide average. The
configurator shows both. It reuses `compute_hook_shape` and `plan_step`'s
alignment so the estimate cannot drift from what `prepare_step` enforces.

### Unchanged

Every blocking finding that bears on *policy* still stands:

* **Finding 1. RESOLVED on this branch.** The capture schedule is now
  enforced: `BackendAdapter.before_forward` consults
  `should_capture_step()`/`should_capture_request()` before every step, the
  transport disarms producers on refused steps (a driver-level skip alone
  left the model's HookPoints firing unreserved writes), and the estimator
  divides its volume figures by the same strides. This was the prerequisite
  for any policy that claims to change sampling; the policy module itself
  remains deferred.
* **Finding 2.** The transport still blocks losslessly and still has no drop
  path. `block` is the status quo; `drop` and `fail` are the new work.
* **Findings 4, 5, 8, 9.** Untouched -- they concern the policy module, which
  is deliberately deferred.

The resequencing above therefore holds, with its first item now half done:
layer selection is wired, and schedule enforcement landed with it (see
Finding 1 above).
