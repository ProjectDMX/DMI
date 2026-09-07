# Findings seen this run

Keyed `file:line|summary`. Every finding ever REPORTED this run is listed,
whether it was later confirmed or refuted — dedup is against seen, not
against confirmed, so a refuted finding cannot resurface each round and keep
the loop alive forever.

## Round 1

### simplification lens (10 reported, 6 more beyond its cap)

Impact classes are the reviewer's; the orchestrator re-checked the factual
claim of every one it lists as duplicated (greps below), because a
simplification lens that invents duplication would waste a builder.

- schema.cpp:385 | inline legacy-object refusal loop unreachable after the [[noreturn]] call at 384 | **requirement** | CONFIRMED, impact DOWNGRADED to optional -> human list, not queued
- clickhouse_client.cpp:1022 | 33-line ClientOptions construction copied into both ThreadInit overloads, plus identical 9-line catch tails | optional
- pack_index.cpp:41 | local `quoted_string` is a character-for-character copy of the exported `sql_quote` already in scope | optional | FACT-CHECKED: pack_index.cpp:42 defines it, catalog_writer.h:44 declares sql_quote, pack_index.cpp:8 already includes that header
- reader.cpp:556 | row-validation/flatten block duplicated between search and get_by_ids, both messages included | optional
- catalog_writer.cpp:61 | the byte-budget chunker exists three times (two here over different element types, one open-coded in reader.cpp with the budget as a literal) | optional
- hydration.cpp:270 | estimate and hydrate each re-implement the same group/sort/coalesce planner, and a comment says they must agree | optional
- catalog_writer.cpp:218 | `quorum_write()` defined identically in three classes; `qualified()` in three | optional
- conformance_catalog.cpp:334 | three verbatim S3Config+reader blocks, two Selection decodes, two 13-line item renderers | optional
- clickhouse_client.cpp:1192 | timed-insert metrics wrapper duplicated verbatim between the two InsertBatch implementations | optional
- reader.cpp:538 | `order` is constructed always equal to `grouped`; the sort-key join is spelled four ways in one file | optional | FACT-CHECKED: reader.cpp:538-546 appends the same value to both under one condition tested on `grouped`

Reviewer also recorded as already tight, no findings: lease_coordinator.cpp,
version_allocator.cpp, indexer.cpp, common/json.cpp, ring/ headers.

### correctness lens (10 reported, 5 beyond cap)

- hydration.cpp:586 | element budget multiplies shape_product by the element count -> prod(shape)^2 | high | CONFIRMED/correctness -> QUEUED (129). Live: shape (16384,) refused at 268435456 > 64000000 where Python summarises; boundary confirmed at prod(shape)>=8001
- clickhouse_client.cpp:111 | POSTFIELDS without POSTFIELDSIZE (NUL truncates SQL); substitute escapes only 4 chars | high | CONFIRMED all 3 parts/requirement -> QUEUED (128). Live: server recorded raw 0x0D vs Python's `\r`, and recorded SQL truncated at the NUL
- pack_index.cpp:179 + indexer.cpp:36 | pack_id interpolated into SQL unescaped while neighbours are escaped | high | CONFIRMED/correctness, severity -> med on reachability -> QUEUED (128). Live: crafted pack_id returned ok:true and appended a fully attacker-chosen inventory row
- hydration.cpp:692 + :646 | abs_max_int chosen by comparing doubles, can return negative; int64 cast from widened double | high | CONFIRMED both parts/correctness -> QUEUED (129). Worse than reported: [2**63-2, 2**63-1] summarises as INT64_MIN in BOTH min and max
- spool.cpp:277 | file I/O moved outside the lock makes the pack-conflict check racy | med | **REFUTED**. The TOCTOU mechanism is structurally real, but BOTH asserted harms were disproved by execution. "One silently overwrites the other": false — `::link` cannot overwrite, it returns EEXIST and :374-389 handles the loser path; and when intents differ the ready names differ anyway, so link never collides (10/10 runs produced two distinct intact files). "The accounting double-counts": false — entries/bytes matched on-disk truth exactly in both interleavings, and the EEXIST loser's unreserve() cancels its reservation precisely (5/5 runs). Unreachable besides: the only threaded consumer is PackSink::RunStager, where every pack id is a fresh UUID v4 assigned immediately before its builder, so the race needs a UUID collision; the conformance driver is single-threaded and re-Opens per op.
  Also worth recording: the oracle IS stricter in-process (it holds its lock across the write, and refuses the second intent with PackConflictError), but threading.Lock is process-local, so for the cross-process case Python has the identical hole and parity holds. The oracle's uploader explicitly documents co-existing same-pack_id ready files as a state it must tolerate (spool.py:547-550), and its own conflict test is purely sequential — a shape native already passes.
  Residual, comment-level only: spool.h:12-13 states the same-pack_id-different-intent refusal without qualifying that it is best-effort under in-process concurrency. Explicitly NOT to be "fixed" by moving the write back under the mutex — that would serialise all stagers on disk I/O (the stated reason for the split) and still not close the cross-process hole.
- spool.cpp:190 | ".." tested as substring, not component -> legal key `a..b` rejected, and the failure LATCHES the sink | med | CONFIRMED/correctness -> QUEUED (127). Live: tenant `a..b` staged fine in Python, latched the native sink to `closed` for all later submits
- pack_builder.cpp:353 | ParseUuid result discarded -> zero-UUID pack sealed, caught only after writing | med | CONFIRMED mechanism, impact DOWNGRADED to optional, severity -> low -> human list, NOT queued. Two of the finding's three assertions were REFUTED: kBadArgument IS reachable (seven ValidateMetadata checks produce it; only the *pack_id* word in the header comment has no producer), and a zero UUID is a VALID uuid both sides accept byte-identically. The real symptom is an empty footer pack_id beside a zero header UUID. Unreachable in production: the only non-test caller feeds NewPackId(), which is always canonical, and SinkConfig has no pack_id field, so unlike Python's injectable pack_id_factory there is no injection surface at all. Python's PackReader also rejects such bytes on read.
- uploader.cpp:221 | oversized pack recorded per-pack while the comment claims the oracle's up-front batch refusal | med/requirement | CONFIRMED/requirement, severity med -> **NOT queued, escalated to the human**. Live head-to-head on an identical 3-small-plus-one-oversized batch: Python raised ValueError with 0 PUTs and all 4 packs left staged; native returned ok:true with 3 PUTs, 1 pack left staged, snapshot attempted=4/uploaded=3/failed=1. Every observable differs, and the comment at :222-223 plus uploader.h:87-88 affirmatively assert the parity they violate.
  WHY NOT QUEUED: the current native behaviour looks DELIBERATE, not accidental. Two purpose-named tests pin it green (`test_pack_over_the_byte_gate_fails_fast` and `test_mixed_batch_reports_oversized_pack_at_its_position`, tests/test_native_uploader.py:219,252, both asserting ok:true with uploaded_packs == 3), and docs/benchmarks.md:483 records a related accounting decision. The fix also needs an error channel added to UploadPending's signature. Choosing the oracle's all-or-nothing policy over the port's more forgiving one is a design decision with a paper trail behind it, so it is the human's call, not the loop's. The verifier notes native's policy is "arguably the more sensible" one; what is unambiguously broken either way is the comment asserting parity.
- indexer.cpp:105 | conflict failures emitted per input ref, not per distinct claimant | med | CONFIRMED/correctness, severity -> low -> QUEUED (128). Live: [A,A,B] gave native requested/failed 3/3 vs oracle 2/2. The verifier also found a SECOND divergence from the same loop needing NO duplicate refs: two interleaved conflicted identities produce native order [a1,b1,a2,b2] vs oracle [a1,a2,b1,b2], and ordering is load-bearing because IndexResult.merge truncates with failures[:failure_limit]. Reachability caveat: indexer.cpp compiles only into conformance_catalog and NativeIndexer::index has exactly one caller, that driver — so the blast radius is the parity harness itself, which is the port's contract.
- indexer.cpp:107 | NEW, found by the indexer verifier while verifying the above: the failure message is the bare "conflicting pack identity" while the oracle emits f"conflicting pack identity: {identity!r}" — an UNCONDITIONAL message divergence on every conflict, independent of claimant counting | requirement | queued with its neighbour (same emission site, same builder)
- reader.cpp:475 | cursor `captured_at_ns` passed through as a raw JSON token, no integer validation | med | CONFIRMED/correctness -> QUEUED (129). The worst outcome here is a SILENT WRONG ANSWER, not a refusal: with k[3] = 2**64 + 1700000000000002002, ClickHouse wraps modulo 2^64 (verified directly: `SELECT toUInt64('18446744073709551623')` -> 7), so native returned ok:true with a page starting at the wrapped position and `capture-1` silently missing, where the oracle raises InvalidCursorError "must fit UInt64".
  The verifier drove 12 crafted values and separated the two halves honestly: 10 of 12 malformed forms DO fail closed but with the wrong error class (ClickHouseError / Code: 53 instead of the ValueError the sibling test pins), and only the oversized case yields a wrong answer. Escaping was confirmed to neutralise INJECTION (`0') OR 1=1 --` came back correctly escaped) but does nothing about numeric semantics, which is the actual claim.
  Confirmed NOT already covered by this session's earlier cursor fixes: the v / field-set / member-count / fh / w checks never touch `k`, and find_uint_in works by object key so it cannot see an array element. The oversized cursor sails past all of them.
  Secondary note from the verifier, same root cause: even HONEST cursors carry captured_at_ns as a quoted string and work only via ClickHouse's implicit String->UInt64 tuple coercion — that coercion is what wraps. Fix should carry it as a typed uint64_t param so it renders bare-numeric.

### consistency lens (10 reported, 3 beyond cap)

- schema.cpp:127 | rebuild_instruction() omits legacy_objects_, so an operator who follows it is refused forever | high/requirement | CONFIRMED/requirement, severity -> med -> QUEUED (128). Live: followed the message text three times, refused every time; Python named 10 objects and succeeded on rerun
- schema.cpp:391 | the dead legacy loop again (same as simplification #1) — DUPLICATE, already verified CONFIRMED/optional
- schema.cpp:162 | require_catalog_visibility() probes only objects_, not the legacy object whose presence refuses | med/requirement | CONFIRMED/correctness, severity -> med -> QUEUED (128). Measured probe sets off system.query_log: oracle issues 10 CHECK GRANT statements including `_pack_commit_log`, native issues 9. Head-to-head on one catalog with only the legacy object's visibility restricted: native `verify_compatibility` returned {"ok":true,"state":"complete"} and `ensure_schema` returned ok, where the oracle refused with "lacks SHOW TABLES ... Grant catalog visibility". So the native path SUCCEEDS SILENTLY in exactly the "two builds share one prefix, captures are invisible" scenario the refusal exists to catch.
  NOT a duplicate of schema.cpp:127: that one is the drop-list TEXT inside a refusal (victim: an operator following a printed list); this one is the GATE, where no refusal is emitted at all. Different function, different observable, and fixing either leaves the other broken.
  Verification method worth remembering: this host's ClickHouse has NO access management (`CREATE USER` -> Code: 497), so the verifier could not build a real restricted role. It reproduced grant filtering with a local HTTP shim in front of 8123 that answered CHECK GRANT with 0 and dropped the row from system.tables — the same technique the oracle's own unit test `_OneObjectHidden` uses — and said plainly that it could not create a role. The shim log was itself decisive: 3 filtered system.tables reads, 0 CHECK GRANT hits for the legacy name.
  Refuted counter worth recording: "probing an absent object would refuse healthy catalogs" is false — `CHECK GRANT SHOW TABLES ON default.nonexistent` returns 1, and schema.cpp:160-161's own comment says the probe needs neither database nor objects to exist.
- catalog_writer.cpp:21 | escaper handles 4 of the 10 characters the driver escapes, under a byte-parity claim | med/requirement | CONFIRMED/requirement -> QUEUED (128), same fix as correctness #2
- pack_sink.cpp:64 | constructor validates only num_workers; every other bound the oracle refuses becomes a latch or a silent all-drop | med/requirement | CONFIRMED but HEAVILY SCOPE-CORRECTED, impact DOWNGRADED to optional, severity low -> human list, NOT queued. The reviewer's "every OTHER bound" is false. Of the 5 bounds PipelineConfig enforces, only 3 diverge into latch/all-drop (max_queue_records, max_queue_bytes -> all-drop; max_pack_records -> a runtime LATCH, sink dead after 1 record). Two degrade benignly (max_pack_bytes -> clean per-record `too_large`; max_linger_ns=0 -> works, one pack per record). Three adjacent bounds the claim implies were REFUTED outright: overload policy is an `enum class` with no invalid value representable; a negative admission_timeout is the DOCUMENTED encoding of Python's None (pack_sink.h:67-68), not out-of-range; and spool_max_bytes=0 is already refused at Start() with a byte-identical message to the oracle.
  Not queued because production construction is fully guarded one layer UP, deliberately: src/dmi/storage/capture/native_sink.py:42-56 refuses every one of these fields with "must be positive" before the C++ sink is built, and its docstring says so ("Field-by-field the same contract as the pipeline config the reference sink takes"), pinned by tests/test_native_rollback.py:129-139. The reachable surface is the conformance driver and the test-only pybind init.
  Useful detail for whoever picks this up: `unsigned` types buy nothing here — 0 is exactly what the oracle refuses and is storable, and the driver's static_cast<uint64_t> turns a negative into UINT64_MAX (max_queue_records=-1 gave an effectively unbounded queue). The verifier also refuted the scarier reading that kRecordLimit is latent with VALID configs: pack_sink.cpp:663 keeps `record_count < max_pack_records` at every loop top, so it fires only for the 0 config. Recommended fix is in Start(), not the constructor, since Start() already owns the error channel.
- pack_builder.cpp:353 | ParseUuid discarded, and the header documents a kBadArgument no path returns | med/requirement | duplicate of correctness #7 — and the "no path returns kBadArgument" half was REFUTED outright by the verifier
- spool.cpp:34 | IsUuid accepts uppercase hex; Python's ready-name grammar is lowercase-only, breaking the stated on-disk contract | med/requirement | CONFIRMED, impact DOWNGRADED to optional, severity low -> human list, NOT queued. The consequence is worse than reported where it is reachable: Python does not merely skip an uppercase ready file, it QUARANTINES it (`_quarantine`, spool.py:231-234), renaming and unaccounting the pack. Verified live in both directions — native staged an uppercase name and later recovered it as live, while the same input made Python rename it to `.dmi-pack.quarantined` with entries dropping 1 -> 0. Not queued because the only in-tree production caller feeds NewPackId(), which is lowercase by construction, so no real handoff can carry an uppercase name today. Notable house-pattern inconsistency: PackBuilder::ParseUuid is deliberately case-insensitive but CANONICALIZES to lowercase; IsUuid copied the leniency and dropped the canonicalization, then uses the raw string as a filename. Fix is a one-line deletion of the A-F branch; the verifier warns against lowercasing inside Stage instead, since the oracle refuses rather than repairs.
- s3_sign.cpp:75 | headers ordered by RAW map key while only the emitted text is lowercased | med/requirement | CONFIRMED, impact DOWNGRADED to optional -> human list, not queued. The reviewer's own example does NOT reproduce; verifier had to find `X-Amz-Meta-Zed`, then showed every production path lowercases before signing
- lease_coordinator.cpp:37 | quorum_write copied three times where the oracle documents one definition as a correctness requirement | med/optional | duplicate of simplification #7
- reader.cpp:396 | search() ports one CaptureQuery bound and drops the siblings | med/requirement | CONFIRMED 3 of 4 sub-claims/requirement -> QUEUED (129). Live: 1025 layer_numbers served a 4-item page where Python refuses. The reviewer's "unbounded IN list" consequence was REFUTED — the server fails closed with Code: 62

### test-coverage lens (10 reported)

- spool.cpp:190 | symlinked parent escapes the root: pack written outside, acknowledged staged, then invisible to recover — DEMONSTRATED by the reviewer running the driver | high/correctness | CONFIRMED/correctness, severity -> med on reachability (needs a pre-existing symlink; object_key is never caller-supplied in production) -> QUEUED (127). Loss is permanent, not just invisible: recursive_directory_iterator does not follow directory symlinks, so the pack is never uploaded and its bytes stay charged against the spool budget forever
- uploader.cpp:122 | the preflight's re-read-and-re-hash of a pre-existing object is untested; a regression would bless corrupt bytes and delete the only good copy | high/requirement | not yet verified
- hydration.cpp:475 | the catalog-descriptor <-> pack-footer binding, the only cross-check between row and object, is untested | high/requirement | not yet verified
- pack_index.cpp:203 | ~15 refusals over externally-supplied footer bytes, none exercised natively (Python tests the whole family) | high/requirement | not yet verified
- pack_index.cpp:62 | the dtype-width table is only ever driven with uint8; 13 of 14 admitted dtypes never reach it | high/requirement | not yet verified
- hydration.cpp:101 | bfloat16 and int64 summary decode branches unexercised, and int64 order stats provably diverge from the oracle | med/requirement | not yet verified
- hydration.cpp:193 | select()'s two refusals (exceeds one page, matches nothing) never driven | med/requirement | not yet verified
- reader.cpp:443 | two cursor refusals (filter-hash mismatch, watermark ahead of head) untested natively | med/requirement | not yet verified
- pack_sink.cpp:472 | the whole staging-failure latch path unexercised: every test opens the spool at 1<<40 bytes | med/requirement | not yet verified
- indexer.cpp:74 | max_packs never tested at either side of the limit, though its byte-budget twin is | low/optional | not yet verified

## Pre-excluded (already fixed earlier this session, told to reviewers)

These were found and fixed before this polish run, and the reviewers were
given the list so they would not re-report them. Recorded here so a later
round cannot rediscover them as "fresh":

- catalog_writer.cpp | manifest chunk read-back compared against chunk length, not distinct count
- indexer.cpp | erase from the sequence being iterated
- indexer.cpp | batch-byte guard thrown inside its own try
- indexer.cpp | publish-exhaustion guard unreachable (break skipped the increment)
- catalog_writer.cpp | quarantine never cleared after its window
- reader.cpp | cursor built from the look-ahead row
- reader.cpp | float16 subnormal decode assembled from a double's bytes
- reader.cpp | TSV escape layering (grouped columns vs tuple contents)
- reader.cpp | cursor envelope checks missing (v, unexpected/missing fields, byte limit)
- reader.cpp | empty cursor-key component accepted via raw-token fallback
- reader.cpp | published_head sent no bounded read settings
- reader.cpp | empty watermark passed the digit check vacuously
- reader.cpp | get_by_ids lacked the resolved-tuple width check
- conformance_catalog.cpp | empty JSON array became a filter on the empty string
- conformance_catalog.cpp | four duplicated op handlers (dead second copies)
- catalog_writer.cpp | parameterized statements not byte-identical to the driver's rendering
- catalog_writer.cpp | publishes not serialised per writer; no process binding
- pack_sink.cpp | std::runtime_error thrown without <stdexcept>
- reader.cpp/hydration.cpp | dead helpers (find_string_in, is_signed_dtype, unwired unescape_tsv)
- pack_index.cpp | unused bucket parameter
