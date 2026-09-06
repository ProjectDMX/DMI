# ADR-0001: The configurability boundary on PR #122

**Status:** Accepted
**Date:** 2026-09-06
**Related:** docs/dmi-configurator-plan.md, docs/capture-policy-review.md,
docs/integration-api-v1.md, ProjectDMX/DMI-vLLM-Integration#20

## Context

DMI-configurator is a small local web tool that turns a model descriptor
plus a few visual selections into a validated, canonical DMI runtime YAML
configuration. It is one of three things that the user can be holding at
once:

- A **single-user desktop tool** served at `127.0.0.1`, with the URL
  bar in the same Chrome tab they are about to type a password into.
- A **config artifact** saved to disk and read by an inference host.
- A **HTTP boundary** the inference host can call from a remote machine.

These three want different things. A single-user tool wants convenience
("every field is optional, every key is a button"). A configuration
artifact wants strictness ("the round-trip is a contract, not a
suggestion"). A network boundary wants an auth gate that the desktop
tool doesn't need. The hardest part of the PR was not the UI, the
estimator, or the compile path — it was drawing the line that separates
them, and sticking to it under pressure from each.

The decisions in this record are the ones where the line moved and a
re-read six months from now would otherwise relitigate the same argument.

## Decision

Three rules, applied consistently across the PR:

### 1. The configuration artifact is strict, full stop

Every section is strict, every scalar is type-validated, unknown keys
are refused, duplicate keys are refused, non-hashable keys are refused,
and the round-trip is a contract:

    parse(serialize(config)) == config

after canonical normalization. The boundary enforcement lives in one
loader (`yaml.py`); every consumer (`yaml.load_config`,
`yaml.load_yaml_document`, the descriptor file path, the
`/api/config/parse` HTTP boundary) goes through it. This makes "merging
two configs" deterministic and "what the file says" auditable.

### 2. The runtime surface is enforced at the smallest possible layer

The capture schedule is a *runtime* concern (stride, warmup, phase
flags). It is enforced at the layer that actually makes the decision:
the adapter driver's `before_forward`, which checks the engine config's
schedule before every step. The estimator mirrors that decision in its
disclosure and in its figure reduction — the same arithmetic, the same
unit ("long-run average over many requests, NOT the cost of any one
request"). It does not try to outsmart the runtime by pre-computing
"what the runtime SHOULD do" — the disclosure is honest about the
disagreement between the figure and the production bytes.

The packed convention is a special case: the pinned vLLM integration
*executes no schedule at all*, so the figure is the unsampled cost. The
estimator does not apply the phase flags or the stride divisor on the
packed convention; it reports the authored intent as warnings so the
user knows what the integration version on the host does. The follow-up
that gates the schedule inside the vLLM integration lives in a separate
repo (DMI-vLLM-Integration#20). When it lands, the estimator disclosure
shrinks automatically — no estimator change.

### 3. The HTTP surface trusts the loopback, refuses the network

On a loopback bind, only the same user can connect — the browser tab
they are looking at, an SSH tunnel, or a developer running curl on the
host. None of those need a token. The middleware is just the
loopback-host check (TrustedHost), and a foreign-Origin refusal on
mutating requests (a CSRF defense: a web page can send `Host: 127.0.0.1`
legitimately, and `sendBeacon` can send JSON without a preflight). The
GETs stay open (the user needs to load the model description).

On a network bind, a per-launch token is generated, printed once to the
operator, and required as `X-DMI-Token` on every mutating request. The
mutating endpoints translate ConfigurationError to 400, OSError to 500,
and 401 to a persistent curl-only banner (so the browser doesn't
paint the install hint as a config error panel-by-panel). The token is
compared with `secrets.compare_digest` on bytes (header values are
latin-1 on the wire, and the comparison is constant-time). `GET /api/config`
on a network bind is also gated: the response names a server filesystem
path and the full starting config, which a LAN peer has no business
reading.

The one-owner invariant at attach is the same rule applied to the model:
a second adapter attaching over a configured model would pair one
caller's reservation with another caller's producers, and the loopback
+ CSRF defenses do not catch that. The driver's `attach_config` checks
`model._dmi_active_adapter` *before* any mutation (no schedule install,
no attach call), names the current owner, allows a same-adapter
reconfigure, and refuses a different owner loudly. The HF adapter's
`detach_model` only releases the marker if it owned it; a non-owner's
detach is a no-op.

## Alternatives considered

### Stricter defaults on the loopback

A check for "the request must come from a localhost origin" via a
reverse-proxy header, or a per-attach nonce on the URL — the kind of
defense a public tool needs. Rejected: the loopback bind is
single-user by construction, and the deployment that *does* need that
defense is the one that switches to the network bind. Two layers of
defense for the same threat class is more code to read, more code to
test, and a real chance of one shadowing the other in a future refactor.

### TLS / reverse proxy / OAuth

The local tool is not a service. Rejected: a real auth system would
add a credential, a login flow, a token store, and a refresh story,
all of which the loopback bind does not need and the network bind is
not the right place to implement. The per-launch token is the smallest
thing that addresses the actual risk (a LAN peer reaching the writes).

### Rejecting a packed configuration that names a schedule

The configurator could refuse a packed configuration whose schedule
asks for reduction, on the grounds that the vLLM integration will
ignore it. Rejected: a rejected config is a much louder failure than
a "these bytes are the unsampled cost" warning, and the user is
*editing* the config to hand it to whatever backend will run it — they
should be able to see the figure for the setting and decide. The
estimator's disclosure is the correct level of friction.

### Hooks for the schedule gate at every emit point

The capture schedule could be enforced at every producer launch, every
ring consumer, every metadata push, every output. Rejected: those
checks are how a real runtime gives the gate teeth, and they belong in
the native ring transport, not the application layer. The driver
checks the gate at the one place that exists before any emission; the
native layer is the place that has to enforce it under CUDA-graph
replay. The application-layer check is a guard, not a fence — and the
disclosure at the estimator tells the user when the guard is the
whole story.

### YAML frontmatter / schema-only header / inline JSON

Three idioms for the configuration file were considered. The YAML body
matches the schema surface and round-trips naturally; the schema is
already the contract. Rejected: any of the other two would add an
idiom to learn, plus a conversion path, for a one-file format.

## Consequences

- The configuration file is portable: a bytewise `parse(serialize(x))`
  check pins the contract, and every surface that reads it goes through
  one loader.
- The runtime surface is portable *to the extent the host's adapter
  enforces it*. The figure is an honest statement of what the
  estimator knows, not a guarantee of what the host will do.
- The host URL is portable: a config that compiles here compiles
  wherever the same DMI runs, regardless of the bind. A host that
  switched from loopback to network gets a token, not a config
  rewrite.
- The estimator disclosure is a *promise* of what the host does.
  The estimator is right when the host enforces the schedule; it is
  correct (as a disclosure) when the host doesn't. Both cases ship.
- The one-owner invariant is portable: every adapter that
  `attach_model` calls sets the marker; every adapter that
  `detach_model` checks it. Adding a third backend to the supported
  set inherits the rule for free, not by convention.
- A future change that wants to move the schedule gate to the
  producer layer shrinks the disclosure automatically: the estimator
  becomes a no-op-on-stride view, and the packed-convention disclosure
  collapses to a one-liner. No estimator change, no test change,
  no client change.
- A future change that wants to add a per-deployment knob
  (FP8 KV-cache, sliding-window attention) goes onto `Workload`,
  *not* onto the descriptor. The descriptor answers "what is this
  model?"; the workload answers "how are you running it?". The
  Knob 1 case (FP8 KV-cache) was investigated and explicitly *not*
  modeled: hooks capture forward activations in model dtype, so
  cache-store quantization changes zero hook payload bytes. The Knob
  2 case (sliding-window attention) was implemented and then
  reverted when the runtime's eager-mode behavior contradicted the
  premise (the masked attention still materializes the full
  `[heads, q_len, kv_dim]` scores matrix; reducing `kv_dim` in the
  estimate under-counts it).

## Footnotes

- The capture-schedule disclosure for compiled CUDA graphs is in
  the same file as the schedule gate, not in a separate doc: the
  compiled path bakes the gate at the first decode trace, so the
  figure that assumes "the schedule was honored step by step" is
  correct exactly until the first trace, and misleading after it.
- The one-owner invariant's marker name (`_dmi_active_adapter`) is
  deliberately not exported from the adapter namespace; it is a
  private handoff between `attach_model` and `detach_model`.
  External callers reach the model through `attach_config` and
  through the public adapter surface; the marker is the join point
  inside that surface.
- The packed-convention disclosure's mention of "Pinned (vLLM)
  until `third_party/vllm-integration` is updated" is the literal
  reason the gate is opt-out. Once the vLLM integration gains its
  own gate, that line goes.
