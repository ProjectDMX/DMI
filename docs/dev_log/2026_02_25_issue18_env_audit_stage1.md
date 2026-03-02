# Issue #18 Stage-1: Runtime Env Audit and Dead Code Removal

Date: 2026-02-25  
Issue: https://github.com/ProjectDMX/DMI/issues/18

## Scope

Stage-1 is intentionally small:
- Audit runtime `MON_*` knobs in native-engine paths.
- Remove knobs that are read but have no behavioral effect.

No runtime behavior redesign in this PR.

## Removed Dead Knobs

The following knobs were removed because they had no effective consumers:
- `MON_NATIVE_AUTOCLEAR`
- `MON_NATIVE_STEP_STATS`
- `MON_NATIVE_PINPOOL_SLOTS_PER_BIN`
- `MON_NATIVE_PIN_THRESH_BYTES`

## Code Paths Cleaned

- Native fields removed:
  - `auto_cleanup_`
  - `stats_step_log_`
  - `pinpool_slots_per_bin_`
  - `pinpool_thresh_bytes_`
- Native env reads removed from `engine_core.cpp`.
- Test/example/benchmark/no-op env assignments removed.
- Quick-start notebook no longer sets `MON_NATIVE_AUTOCLEAR`.

## Out of Scope

Still kept (runtime-relevant today):
- `MON_NATIVE_TO_CPU`, `MON_NATIVE_PINNED`, `MON_NATIVE_PINPOOL`, `MON_NATIVE_PINPOOL_BINS_KB`
- `MON_NATIVE_HOST_COPY_THREADS`, `MON_NATIVE_HOST_COPY_QUEUE_SIZE`
- `MON_NATIVE_PINNED_INDEX`
- `MON_NATIVE_BATCH`, `MON_NATIVE_BUILDER`, `MON_NATIVE_CALLBACK`

Build-time knobs remain unchanged in `_native_engine.py`.

## Short PR Description (English)

```markdown
## Summary
Stage-1 cleanup for Issue #18: removed runtime `MON_*` knobs that were dead (read but not used).

## Removed
- MON_NATIVE_AUTOCLEAR
- MON_NATIVE_STEP_STATS
- MON_NATIVE_PINPOOL_SLOTS_PER_BIN
- MON_NATIVE_PIN_THRESH_BYTES

## Changes
- Deleted unused native fields and `getenv(...)` reads.
- Removed no-op env settings from tests/examples/benchmark scripts.
- Kept all runtime-relevant knobs unchanged.

## Validation
- Grep confirms no runtime references to the four removed knobs in code paths.
- Existing native test paths remain intact.
```
