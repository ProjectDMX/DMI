# Catalog descriptor key

A proposal to add pack identity to the descriptor table's sort key, and a note
on where the catalog goes after that.

## The problem

Every capture gets one row in the descriptor table saying, in effect, "capture
X lives in pack P at offset N". The table is a `ReplacingMergeTree` ordered by

```
(tenant_id, experiment_id, run_id, captured_at_ns, capture_id)
```

which is a capture's identity and nothing more. That engine collapses rows
sharing a sort key during background merges, keeping the one with the highest
`index_version` and deleting the rest.

That collapse is only safe while two rows with the same key are byte identical.
The design says they always are, because a capture belongs to exactly one pack.
Nothing enforces it.

When two rows for one capture disagree, a merge silently deletes one of them --
at an unpredictable later time, with nothing connecting the deletion to whatever
caused it. A reader whose pinned snapshot covers only the deleted row's pack
then fails with `selection no longer resolves at its catalog watermark`. The
bytes are still in object storage; the pointer to them is gone.

## When two rows disagree

Three ways, and only the third needs code that does not exist yet.

**A pack copied to a second store.** An operator copies a pack object from the
filesystem store into an S3 bucket for redundancy, then reconciles that bucket
-- a supported operation. Replay dedup is keyed on `(store_id, pack_id)`, so a
new store is not a replay: the indexer writes a fresh descriptor row for every
capture in that pack. Same captures, same sort key, different `store_id` and
`object_key`. Possible today with shipped components.

**A producer retry across a pack boundary.** `capture_id` comes from the
producer under an at-most-once contract. Within one open pack a repeat is
caught (`DuplicateCaptureError`), but once a pack is sealed and uploaded the
pipeline has no memory of it. A producer that retries after an ambiguous
failure -- a crash, a timeout, a restart -- writes the same capture into a
second pack. Possible today, and retry-after-ambiguity is the ordinary
distributed failure, not an exotic one.

**Re-packing or compaction tooling.** A maintenance job that rewrites old packs
into new ones produces exactly this shape deliberately. Does not exist yet; it
is the natural next tool for any object-store system.

## The proposal

Append pack identity to the sort key:

```
ORDER BY (tenant_id, experiment_id, run_id, captured_at_ns, capture_id,
          store_id, pack_id)
```

Rows describing one capture in different packs now have different keys, so no
merge can delete either. Rows describing the same capture in the same pack --
the ordinary replay case the engine is there for -- still share a full key and
still collapse, and those rows really are identical.

Supersession then falls out of machinery the reader already has. Keep the
`GROUP BY` on the five capture-identity columns, decoupled from the now longer
sort key, and `argMax(..., index_version)` still picks one row per capture. A
reader pinned before the new pack was committed does not see its rows at all,
because the commit-log membership clause excludes that pack; a reader on a
fresh watermark sees both and takes the newer. Re-packing goes from "would
corrupt pinned snapshots" to "supported, under the same commit-log rules as
everything else".

### What it costs

- **Query performance: nothing measurable.** The two columns append after the
  existing prefix, so tenant-range pruning and the `capture_id` bloom filter
  behave exactly as they do now.
- **Code: small.** The DDL in `ensure_schema`, decoupling `GROUP BY` from the
  sort-key constant in the two reader queries, tests, and one live test proving
  a re-packed capture resolves to the old pack under an old pin and the new pack
  under a fresh one.
- **Disk: slightly more rows** if re-packing ever happens, since superseded
  pointers are kept rather than collapsed. Negligible.
- **The public `FINAL` views** would show one row per pack for a superseded
  capture, since `FINAL` collapses on the new key. That extends the existing
  caveat that those views bypass the snapshot bound; it is not a new problem.

### Why the timing matters

ClickHouse cannot change a table's `ORDER BY` in place. Before this schema
ships anywhere, the change is one line in `ensure_schema` and every deployment
starts correct. After it ships, the same change is a create-new-table-and-copy
migration over the whole descriptor table.

That is the entire argument for doing it now rather than when the first
re-packing tool is written.

## Alternatives considered

**Enforce uniqueness at write time.** Check per capture whether it is already
described before indexing a pack. Expensive on the ingest path, racy between
concurrent indexers, and it forbids re-packing rather than supporting it.

**Leave it.** Defensible while nothing re-packs, but scenarios A and B above are
reachable today, and the cost of the fix rises sharply after the first
deployment.

## Where the catalog goes later

The deeper version of this problem is that immutable facts ("capture X is in
pack P") are stored in an engine whose job is keeping the latest version of a
mutable thing, and that a snapshot is an open predicate (`index_version <= W`)
over a table still being written. The sort-key change removes the sharp edge;
it does not change that shape. The same shape is also behind the mid-batch
membership gap and the fact that no retention or GC policy exists yet.

The clean-slate alternative is the table-format design: publish an immutable
manifest per snapshot ("snapshot 42 = 41 plus these packs"), advance a single
pointer by compare-and-swap, and demote ClickHouse to a rebuildable search
index over those manifests. Pinned snapshots become closed sets written before
they are named, so the mid-batch gap cannot occur; merges stop being
load-bearing because the index is rebuildable; and GC becomes "delete any pack
no live manifest references".

If that is ever built, evaluate Apache Iceberg and Delta Lake (`delta-rs`)
before hand-rolling manifests -- snapshot isolation, atomic multi-writer
commits, and snapshot expiry are their core competency, and they are hardened
in exactly the places our reviews keep finding holes. The deciding experiment
is whether Garage supports the conditional-put semantics a catalog-free commit
needs.

**Performance is not a reason to move.** The pack data path -- ranged reads
against immutable objects -- is identical under every option. What differs is
the metadata path, and the current design is the fastest of them: membership,
watermark and descriptors live in one engine, so a pinned search is a single
query (measured: 1.9 ms watermark pin, 120-150 ms pinned pages, point lookups
pruned to one tenant's granule range). Moving authority out of ClickHouse adds
a manifest fetch and, more importantly, makes the serving index trail the
authority. A table format adds commit round-trips and small-file compaction on
top, and still needs ClickHouse for serving latency -- so it would be adopted
for correctness generality and GC, never for speed.

Keep the catalog ClickHouse-native and harden it. Revisit when re-packing or
retention actually forces the question; the descriptor rows migrate as plain
data whichever way it is answered.
