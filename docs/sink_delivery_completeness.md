# Feature request: a delivery-completeness guarantee for the callable sink

**Type:** API / guarantee gap (NOT a data-corruption bug)
**Path affected:** in-process callable Python sink (`RingEngine(cfg, <python callable>)`,
restored on the hallu / node-toggle branches). The ClickHouse `DMXHostEngine`
path is not directly exercised by this report.
**Priority:** P2 — there is a clean consumer-side workaround (route by per-slice
`req_id`, select the token by `end_token`); but the missing guarantee cost a long
debug and silently corrupts any consumer that assumes "flush ⇒ I now hold this
request's complete set".

## Summary

Captured slice **contents and metadata are CORRECT**. We verified, on Qwen3-8B
via vLLM, that the raw tensor handed to a `resid_pre` HookPoint matches HF
`output_hidden_states` at **cosine 0.9991**, and that position-aligned
ring-delivered activations are **cosine 1.0 (bit-perfect)**. The `(req_id,
start_token, end_token)` metadata on each slice is also correct.

The gap is purely **delivery timing/completeness to the sink**: after
`flush_and_wait()` + `worker.join()`, a request's single-token decode slices can be

- **delivered late** — a slice produced for request *N* arrives in the sink
  callback *after* `flush_and_wait()` returns, landing in request *N+1*'s read
  window (it still carries its correct `req_id`/position);
- **duplicated** — the same `(req_id, end_token)` slice delivered twice (possibly
  benign, e.g. a vLLM recompute/prefix-cache re-forward captured faithfully —
  **to be confirmed**);
- **dropped** — rarely (~0.3%, 1/300 final slices) never delivered.

So `flush_and_wait()` does not guarantee "all slices produced so far have reached
the sink callback" — it appears to barrier the GPU ring drain, not the full
pipeline through the Python `SubmitFn`.

### Why it matters

A naive consumer that does `generate → flush_and_wait → read what's here` and
picks "the latest / max-`end_token` slice" gets the **wrong token for a
substantial fraction of requests**, because (a) the request's own tail slice may
not have arrived yet, and (b) the *previous* request's late tail leaks into the
window. A single late tail corrupts **two** requests (the source, missing its
tail; and the next, receiving the stale one).

Concrete impact: our SEPs hallucination probe scored the wrong token and dropped
from AUROC **0.81 → 0.67**. Selecting the answer-final token by `(req_id,
end_token)` after a thorough end-of-run drain restored it to **0.811** (== the
offline HF number), confirming the data is correct and only the *delivery
timing* was the issue.

## Requested change (one of)

1. **Strengthen `flush_and_wait()`** so that when it returns, every slice
   produced so far has been delivered through the sink callback (barrier the full
   ring → drain → p2p → `SubmitFn` pipeline, not just the GPU ring). And/or add a
   separate `drain_to_sink_and_wait()` with that explicit guarantee.
2. **Per-request completion signal** — emit a marker (or invoke a callback) when
   all of a request's slices have been delivered to the sink, so consumers can
   finalize a request at exactly the right moment without guessing positions.
   This is the cleanest for streaming consumers under continuous batching.
3. **Document the current contract** of `flush_and_wait()` (drains the ring, does
   NOT guarantee sink delivery) so consumers know to route by `req_id` and select
   by `end_token`.

Separately worth confirming: whether the **duplicate** deliveries are benign
(faithful capture of a vLLM re-forward) or a real re-delivery in the drain/p2p
path; and the cause of the rare **drop**.

## Reproducer

- `hallu-monitor/tools/diag_capture.py` — `vllm` phase saves per-request
  `prompt_len`, `n_answer_tokens`, and each delivered slice's `(start, end_token)`
  ranges; shows late/dup/leaked/dropped delivery. The monkeypatch of
  `HookPoint.forward` records the RAW pre-ring tensor; `compare` shows raw==HF
  (0.999) and position-aligned ring==raw (1.0).
- `hallu-monitor/tools/eval_seps_online_v2.py` — the consumer-side workaround
  (key by `(req_id, end_token)`, drain at end) → AUROC 0.811.

## Consumer-side workaround (already in use)

Route every slice by its `req_id`; for a per-request final-token score, select the
slice whose `end_token == prompt_len + n_answer_tokens` (de-dup by keeping the
latest for a given `(req_id, end_token)`); wait for that slice to arrive (or drain
fully) before finalizing, rather than reading immediately after `flush_and_wait`.
