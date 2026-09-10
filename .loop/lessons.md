
## Run: src/dmi Python scope, 2026-09-10 (branch pr-127, baseline 480e95b)

Two macro rounds over 48 files / 16.4k lines. 73 findings reported by 8
independent lens passes; 39 verified adversarially; 18 fixed, 7 handed to the
human, 7 refuted, 11 deduped as re-discoveries, the rest optional. Tests went
1461 -> 1532 with `src/` never modified.

**The verify step paid for itself, repeatedly.** Seven findings were refuted
outright and five more had their reasoning corrected while still being
confirmed. The single most valuable catch: the `min(byte_source.numel(), ...)`
clamp in transport/ring.py's PREFIX_STRIP branch is an EQUIVALENT MUTANT --
Python slicing already clamps, so no test can ever kill it. Without that
catch a builder would have written a test asserting nothing and I would have
reported closed coverage. A close second: two consistency findings died
because the asymmetry they flagged is documented design in
docs/integration-api-v1.md. Read the published contract before filing drift.

**Mutation testing is only as good as its harness, and this repo has three
traps.** `pyproject.toml`'s `pythonpath = ["src"]` is inserted ahead of
`PYTHONPATH`; the editable install's `__editable__` finder beats both; and
`pytest-randomly` is absent so `-p no:randomly` does nothing. Four agents
reported or nearly reported false results before noticing. The rule that
fixes it: before believing any mutant survived, apply one mutant you KNOW is
covered and watch a test fail. A surviving mutant and a mutant that never
loaded look identical.

**Never let concurrent agents mutate one checkout.** I dispatched 13
verifiers against the shared tree; several mutated `src/` while others ran
the full suite, and three independently reported impossible baselines (12
failed, 8 failed, 18 collection errors) before I moved them into worktrees.
Every mutation run needs its own worktree from the first instruction, not as
a mid-flight correction. To reproduce the real baseline in a fresh worktree,
copy the prebuilt native `.so` files and `native/build` in, or native tests
skip and the number differs.

**Coverage tests need positive controls or they pass vacuously.** Several
mutants died only because of a control asserting the ACCEPTED case: the
one-open-plus-bounded payload slice (kills `> 1` -> `> 2`), the exact-bound
case (kills `>` -> `>=`), the tp_rank=0 case, and `"'hidden-states'" in
message` (because `"Available:" in message` alone survives a mutant emitting
the label with an empty list). Pin the boundary, not just the refusal.

**The queue's shape was the real finding.** Of 39 verified findings, only one
was a plain fix; everything actionable was missing coverage, and every
confirmed correctness bug was blocked on something the loop is not allowed to
decide -- GPU-only reproduction, a torch.compile trade-off, or a contract
change. When a scope's correctness bugs all land outside a behavior-
preserving remit, say so plainly instead of stretching the remit.
