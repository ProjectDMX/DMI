// CaptureStorageService: the production entry point for the native capture
// storage path after the spool.
//
//   NativePackSink -> spool  (the sink, in the capture process)
//   spool -> SpoolUploader -> object store -> NativeIndexer -> catalog  (here)
//
// One background thread runs a cycle: upload what is pending and index it, a
// chunk at a time, keep the publisher lease alive, and periodically reconcile
// the bucket against the catalog. The conformance drivers exercise each of
// these pieces; this is what composes them outside a test.
//
// SpoolUploader removes a pack from the spool the moment its upload is
// verified, before anything indexes it, so the spool alone cannot say what is
// still owed to the catalog. A cycle uploads a chunk of indexer.max_packs at
// a time and indexes it before it uploads the next, so at most one chunk is
// in that gap at once, and a cycle cut short leaves the rest in the spool.
// Two things cover the gap:
//   - in-process, a pack whose indexing fails stays on a retry list, and
//     flush() does not report drained until that list is empty. Nothing new
//     is uploaded while it is not, so an outage leaves new packs in the
//     durable spool, not on a list only this process remembers;
//   - across a crash, the reconciler lists the bucket, skips what the catalog
//     already committed, and indexes the rest. It runs at start(), and
//     periodically when reconcile_interval_ns is non-zero.
//
// One owner per spool directory. The service's spool is opened under the
// directory's owner lock (store/spool.h), so a second process on it is
// refused at construction, naming the holder. With adopt_sibling_spools
// the service's directory is one rank directory of the plan's section 2.3
// layout, and once started its loop adopts the siblings whose owner has
// died: a crashed process's spool is recovered by the next process on the
// node for the same catalog, whatever run it belongs to.
//
// Deployment shape: the service holds the catalog's single publisher lease, so
// run ONE service per (database, table_prefix). A second one waits up to
// start_lease_wait_ns for the lease at start(), then fails naming the holder.
// That is the in-process mode; a standalone daemon can reuse this class
// unchanged.
//
// The lease through ClickHouse errors. A catalog statement whose outcome is
// unknown (a transport error or timeout on a renewal or a publish)
// quarantines the writer: it drops its lease without a tombstone and refuses
// to publish for one TTL (catalog_writer.cpp). That is recoverable, not
// fatal: while the writer is quarantined or holds no lease the cycle skips
// the catalog phase, keeps pending_index_ and uploads nothing, so new packs
// stay in the durable spool rather than in a list only this process
// remembers. Once the window passes the service acquires a FRESH lease_id,
// as the Python oracle's writer documents
// (clickhouse_catalog.py, publish_snapshot). Only a foreign lease that stays
// live for 2 x TTL is fatal: the service stops (snapshot().failed), writes
// one line to stderr, and flush() rethrows the refusal naming the holder.
// A refusal by one of the service's own claim rows -- a claim that landed
// after its request gave up -- is no rival and restarts that clock.
//
// Bounded by the lease. Every catalog request the service makes while it
// holds the lease -- the renewal's own, and every one an index pass or a
// reconcile sends under the lease lock -- has to be answered by the lease
// deadline (lease_coordinator.h), since while that lock is held the lease
// renews only between those requests, before each one that finds it due:
// so a catalog that stops answering fails the stretch, and the lease
// quarantines, while its row still keeps rivals out. A lease whose deadline
// passes unrenewed is abandoned, never reported held (LeaseScope).
// An index pass reads its packs from the object store without the lock, so
// a stalled read cannot hold the renewal off either. Claims made with no
// lease are bounded by min(request timeout, lease_ttl / 3) per request; a
// claim that times out at start() is retried until start_lease_wait_ns ends
// (or, once, until the quarantine it left is over), and lease requests that
// keep timing out say which knobs bound them (snapshot().lease_timeout_error). The constructor refuses a clock skew
// that leaves a renewal too little time to finish.
//
// Cancelled object-store work. The uploads go through an S3 client of their
// own that shares one Cancellation with the uploader; the index reads and
// the reconcile go through the other client, which has a Cancellation of
// its own. stop() cancels both for good. A flush arms a deadline on both
// for the cycle it runs: the uploads' at the flush deadline, the reads'
// one catalog request timeout past it, since a pack cut short on upload
// stays in the durable spool but one uploaded and not yet indexed is owed,
// and only this process remembers it. A cancel starts no further request,
// aborts a transfer in flight (a multipart upload is then aborted, one
// attempt bounded by 5 s) and ends a retry backoff; a cancelled upload
// leaves its pack in the spool, a cancelled read leaves its pack owed. The
// uploads' cancel also stops a spool listing between packs, since listing
// hashes every staged pack and a backlog would hold it for as long.
// Catalog statements are never cut mid-flight: each is bounded by the
// client's request timeout, or under the lease by the lease deadline.
// An index pass ends at its first failure: a batch that threw, or a pack
// the object store did not answer for (a transport error, a timeout, a
// retryable status on every attempt), which leaves the rest of the pass
// owed without counting against the packs. Only a pack the store answered
// for and the indexer refused counts towards max_index_attempts.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "catalog/catalog_writer.h"
#include "catalog/clickhouse_client.h"
#include "catalog/indexer.h"
#include "store/cancel.h"
#include "store/s3_client.h"
#include "store/spool.h"
#include "store/uploader.h"

namespace dmi_catalog {

struct StorageServiceConfig {
  // The spool the sink stages into. The service opens its own Spool object on
  // the same root and consumes it through ListPending, which is safe while the
  // sink writes.
  std::string spool_root;
  uint64_t spool_max_bytes = 1ull << 40;
  // The spool directory's owner lock (store/spool.h). kTake owns it for the
  // service's life, and refuses a directory another process owns. A process
  // that also runs the sink on it -- the engine -- holds one SpoolOwnerLock
  // and opens both with kHeldByCaller: two takes in one process refuse each
  // other.
  dmi_store::OwnerLock spool_owner_lock = dmi_store::OwnerLock::kTake;
  bool spool_allow_shared_filesystem = false;
  // Adopt the spools of dead processes bound for this catalog. spool_root
  // must then be a rank directory of the plan's section 2.3 layout,
  //   <base>/<catalog_key>/r<producer_rank>-<incarnation>/
  // under THIS catalog's key (SpoolCatalogKey of clickhouse.host and
  // .port, writer.database and .table_prefix, s3.endpoint and .bucket, and
  // uploader.store_id), or construction throws.
  // The loop's cycles adopt, starting with the first, which start() kicks
  // off once it holds the lease and has swept its own directory; start()
  // itself uploads nothing of a dead backlog, and neither does flush(). A
  // cycle probes the owner lock of every sibling rank directory, then works
  // through those whose owner is gone: it takes one's lock and sweeps its
  // .open files, validates its .ready packs once, a pack a step -- which
  // hashes every byte of it -- and uploads and indexes them a round at a
  // time -- a chunk of the service's own upload path, at most
  // uploader.max_workers and indexer.max_packs packs, each indexed before
  // the next is uploaded -- under the cycle's upload rules, the lease and
  // nothing owed to the catalog checked before every round, until
  // adoption_slice_ns has passed, a stop is requested, or a flush() is
  // running -- a flush waits for the step the adoption is in (one round,
  // one pack's validation, or one sibling's lock and sweep), not for the
  // rest of a slice or of a listing. Its uploads and index reads go through
  // the service's own clients and Cancellations, and its listing stops
  // between packs on the uploads' cancel, so stop() cuts an adoption as it
  // cuts the service's own work: what it did not upload stays in the dead
  // spool, which is let go of and never removed while a pack of it is
  // left, for the next process on the node. The sibling's lock, its
  // listing and its remaining packs are kept between cycles, so a large
  // backlog is hashed once, not on every cycle, and neither a flush() nor
  // the service's own uploads wait behind all of it. A drained sibling is
  // removed once nothing but its lock file is left. An upload that failed
  // stays in the dead spool and is retried by
  // a later cycle, after the backoff -- unless no retry by this service can
  // ever succeed (dmi_store::UploadFailure::retryable: a pack over
  // uploader.max_in_flight_bytes, a different object at its key, bytes that
  // no longer match), or the sibling cannot be locked or opened at all.
  // Such a sibling is blocked: once the rest of its packs are up it is left
  // in place, with its lock let go, for a process that can adopt it (or a
  // person); it is reported once (last_error, snapshot blocked_siblings),
  // and in its lock file's record (dmi_store::SpoolOwner::blocked), which
  // keeps the sinks on the node from charging it against their budgets;
  // never retried or re-hashed by this service, and not owed. So is one
  // drained of packs that still holds other files. flush() covers this
  // process's records: a sibling still to adopt does not keep it from
  // reporting drained, though adopted packs uploaded and not yet indexed
  // are owed like the service's own. Live siblings -- another rank or job on this
  // node, a predecessor still closing -- are left alone, and are not owed.
  bool adopt_sibling_spools = false;
  // While a pass found a live sibling, the loop passes over the siblings
  // again this often, so one whose owner dies later -- a predecessor that
  // was still inside close() when this service started, a rank that
  // crashes while this one runs -- is adopted then, not at the next
  // restart on the node. A live sibling costs one non-blocking lock probe
  // per pass. 0 never looks again after the first pass.
  uint64_t adoption_recheck_interval_ns = 30'000'000'000ull;
  // How long one cycle may spend adopting before it lets go of the cycle --
  // to the service's own uploads, stop() -- and carries on in the next.
  // Checked between rounds and between the packs a listing validates, so a
  // cycle can outrun it by one round, or one pack's hash. A flush() does
  // not wait for it: a cycle that has made one step of adoption lets go of
  // the cycle at the next one while a flush is running.
  uint64_t adoption_slice_ns = 1'000'000'000ull;

  dmi_store::S3Config s3;
  dmi_store::UploaderConfig uploader;  // uploader.store_id names the store

  // The catalog's HTTP interface: scheme, host, port, credentials, TLS
  // trust, timeouts and retry attempts.
  ClickHouseConnection clickhouse;
  WriterConfig writer;  // database, table_prefix, lease TTLs
  IndexerConfig indexer;
  std::string holder;   // the publisher lease holder id

  // Where the reconciler lists. Object keys are
  // "v1/tenant=<t>/date=<d>/session=<s>/rank=<r>/<pack_id>.dmi-pack" and name
  // no catalog, so every pack under this prefix is indexed into THIS catalog:
  // the prefix (by default the whole bucket) must belong to one catalog.
  std::string reconcile_prefix = "v1/";
  uint64_t poll_interval_ns = 500'000'000ull;
  // While cycles keep failing (the object store or catalog is down), the
  // background loop doubles its wait up to this: every cycle re-lists the
  // spool, and listing re-hashes each pending pack. flush() does not wait.
  uint64_t max_backoff_ns = 30'000'000'000ull;
  // 0 disables the periodic pass; the start() pass is reconcile_on_start.
  uint64_t reconcile_interval_ns = 0;
  // A pack the indexer refuses on its own (not a whole-batch outage) is
  // retried this many times, then set aside: left in the object store, out
  // of the flush boundary, and reported by the next flush(). A pack too big
  // for the indexer's batch budget is set aside at once.
  int max_index_attempts = 5;
  uint64_t schema_retry_sleep_ns = 500'000'000ull;

  // How long start() waits for another holder's lease to expire before it
  // fails with the lease held. A crashed predecessor's lease stays live for
  // up to its TTL; 0 fails at once. A claim of start()'s own that timed out
  // is retried within it, and a quarantine that claim left is waited out
  // even past it, once (acquire_lease_at_start).
  uint64_t start_lease_wait_ns = 0;

  // Sweep a crashed sink's stale .open files before anything writes to the
  // spool. Recover() deletes every .open file this object does not own, so it
  // is only safe while no writer is live: start() must run before the sink
  // opens the spool. It runs after the lease is taken, so a start refused
  // the catalog never touches the spool. That keeps a second process off a
  // live spool only usually: a holder that stops renewing for a TTL
  // (quarantined, or stalled) lets its row lapse, and a second process can
  // take the lease and sweep while the first is still writing; one on
  // another (database, table_prefix) never meets the lease at all. Its
  // Recover() also lists the first's sealed packs, which the first may
  // upload too. A cycle checks the lease before each chunk it uploads, the
  // first included (after the listing, which hashes every staged pack), and
  // the uploads in flight do not stop when the lease is lost: a holder that
  // is quarantined, or refused a renewal or publish, while a chunk is in
  // flight finishes that chunk (which can outlast the TTL), and uploads
  // nothing more while it holds no lease. One whose catalog requests
  // stall gives the lease up at its deadline, before its row
  // lapses: requests made under the lease are cut off there, and a cycle's
  // check abandons a lease past it (LeaseScope), so no chunk starts after
  // that. Only a holder whose whole process stalls keeps the lease locally
  // past its row: until it resumes and next checks, or -- after a system
  // suspend, which the steady clock the deadline runs on does not count --
  // until a renewal or publish is refused. Two on different (database,
  // table_prefix) pairs each hold a lease and upload freely. The spool's
  // owner lock (spool_owner_lock) is what keeps a second process off the
  // directory itself: it is refused at construction, before any of this.
  bool sweep_spool_on_start = true;
  bool reconcile_on_start = true;
};

struct StorageServiceSnapshot {
  bool running = false;
  uint64_t cycles = 0;
  uint64_t uploaded_packs = 0;
  uint64_t uploaded_bytes = 0;
  uint64_t upload_failures = 0;
  // Uploads a flush deadline or stop() cancelled, the pack left staged.
  // Not failures: nothing was wrong with the pack or the store.
  uint64_t cancelled_uploads = 0;
  uint64_t indexed_packs = 0;
  uint64_t indexed_rows = 0;
  uint64_t index_failures = 0;
  uint64_t batch_splits = 0;
  uint64_t reconcile_passes = 0;
  uint64_t reconciled_packs = 0;  // found in the bucket, missing from the catalog
  uint64_t reconcile_skipped_objects = 0;  // not a valid DMI pack
  uint64_t reconcile_head_errors = 0;  // could not be read; retried next pass
  uint64_t lease_renewals = 0;
  uint64_t swept_on_start = 0;  // ready packs Recover() found at start
  uint64_t pending_index = 0;   // uploaded packs awaiting a retried index
  uint64_t rejected_packs = 0;  // set aside: cannot be indexed (see flush)
  // adopt_sibling_spools: dead siblings drained, the ready packs of theirs
  // that were uploaded, and whether one is still to adopt (or a look at
  // the siblings is due).
  uint64_t adopted_spools = 0;
  uint64_t adopted_packs = 0;
  bool adoption_owed = false;
  // Siblings whose owner was alive at the last adoption pass.
  uint64_t live_siblings = 0;
  // Dead siblings this service will not adopt, left in place (see
  // adopt_sibling_spools), by directory.
  std::vector<std::string> blocked_siblings;
  // A foreign lease outlived 2 x TTL: the service stopped for good.
  bool failed = false;
  // "none" before start, "held", "quarantined" (an unknown outcome set the
  // lease aside), "reacquiring" (no lease, trying for a fresh one),
  // "failed", or "released" (stop() wrote the tombstone).
  std::string lease_state = "none";
  // steady_clock ns at which the quarantine ends; 0 when not quarantined.
  uint64_t quarantined_until_ns = 0;
  uint64_t lease_reacquisitions = 0;  // fresh leases taken after a loss
  // Timeouts that cost the publisher lease -- a claim or renewal that timed
  // out (the server's own time limit included), a request made under the
  // lease that its deadline cut off, a lease abandoned at its deadline --
  // since a lease was last held for 2 x TTL. From the third on,
  // lease_timeout_error says so and names the knobs that bound them (it is
  // last_error too, when it happens); both clear once a lease has been held
  // for 2 x TTL again.
  uint64_t lease_timeouts = 0;
  std::string lease_timeout_error;
  std::string last_error;
};

class CaptureStorageService {
 public:
  explicit CaptureStorageService(StorageServiceConfig config);
  ~CaptureStorageService();
  CaptureStorageService(const CaptureStorageService&) = delete;
  CaptureStorageService& operator=(const CaptureStorageService&) = delete;

  // Ensure the catalog schema, take the publisher lease (waiting up to
  // start_lease_wait_ns for another holder's to expire, or for a claim that
  // timed out to go through -- past it, once, to wait out the quarantine
  // such a claim left), sweep the spool, reconcile once, then start the
  // background cycle -- whose first cycle, at once, starts adopting dead
  // siblings (adopt_sibling_spools). The lease renews from the moment it is
  // taken.
  // Throws if the lease is still held by another publisher when the wait
  // ends, or its claim still times out. A lease lost while the reconcile
  // runs does not fail start(): the loop takes a fresh one, as it would
  // later, and reconciles then.
  void start();

  // Run cycles until one finds the spool empty with every uploaded pack
  // indexed, or the timeout passes. Call after the sink's own flush, so
  // everything it will stage is already staged. Returns false on timeout,
  // including while a cycle already in flight outlives the deadline, and
  // once stop() has begun. The cycles it runs honour the deadline: they
  // skip the reconcile (the loop runs it), and at the deadline their
  // uploads are cancelled, each pack cut short left in the spool, and no
  // further chunk starts, so what is not uploaded by then stays in the
  // spool. The chunk a cycle has uploaded it still indexes, since until
  // then only this process remembers it -- but past the deadline no index
  // batch starts after the first: that batch's object-store reads are cut
  // one catalog request timeout past the deadline, and its catalog
  // statements are never cut mid-flight, each bounded by the client's
  // request timeout (under the lease, by the lease deadline). What it
  // leaves unindexed stays owed, for the loop or a later flush. A pass
  // ends at its first failure, so a flush overruns its deadline by about
  // one request timeout against a catalog or an object store that stopped
  // answering, and by one batch of statements against a slow catalog that
  // still answers. A multipart upload the deadline cuts adds its abort,
  // which no request timeout bounds: up to about a second for the stalled
  // transfer to see the cancel (libcurl's progress poll), then one abort
  // request of at most 5 s that nothing cuts -- past one request timeout
  // when that is under about 6 s. At zero it still runs one cycle, which
  // indexes one batch of what earlier cycles owe and uploads nothing.
  // Its cycles adopt nothing, and a dead sibling still to adopt does not
  // keep it from returning true (adopt_sibling_spools); a loop cycle that
  // is adopting when it is called lets go of the cycle after the step it
  // is in -- one round of a dead spool's uploads, one of its packs
  // validated, or one sibling's lock and sweep.
  // Throws, once, if packs were set aside since the last flush: they are in
  // the object store but can never reach the catalog.
  bool flush(double timeout_s);

  // Stop the background cycle and release the lease. Does not flush. Its
  // object-store work is cancelled first: a spool listing stops between
  // packs; an upload in flight is aborted, its pack left in the spool for
  // the next start, with every pack no upload has reached; an index read
  // in flight is cut, its pack left owed -- which stop() drops, so it waits
  // in the bucket for a start's reconcile (reconcile_on_start); that is at
  // most the one chunk a cycle has uploaded and not yet indexed. A retry
  // backoff ends, and so does the reconcile. What stop() still waits for
  // is the catalog work in flight, never cut mid-flight, the abort of a
  // multipart upload it cut (one attempt, 5 s at most), and the lease
  // release, each catalog request bounded by the client's request timeout
  // (under the lease, by the lease deadline); the lease renews until the
  // loop is done. An adoption in flight is cut the same way, its dead
  // sibling let go of with what it still holds (adopt_sibling_spools).
  void stop();

  StorageServiceSnapshot snapshot() const;

  // Rethrows a fatal failure latched by the background cycle: another
  // publisher holding the lease for longer than 2 x TTL.
  void rethrow_if_failed() const;

 private:
  struct CycleOutcome {
    bool drained = false;  // nothing pending and nothing failed
    bool failed = true;    // an upload or index failed, or the cycle threw
    // A cancel left packs in the spool: neither drained nor a failure, so
    // it moves the backoff neither way.
    bool cut_short = false;
  };

  // One chunk of the upload path, the service's own spool's or a dead
  // sibling's (upload_chunk): the batch, positional as UploadStaged returns
  // it, what went up and is to be indexed, and how many were not.
  struct ChunkOutcome {
    dmi_store::UploadBatchResult batch;
    std::vector<PackRefData> to_index;
    size_t failed = 0;     // still staged; counted in upload_failures
    size_t cancelled = 0;  // still staged: a cancel cut them short
  };

  // Holds lease_mutex_ for a stretch of catalog work, and bounds every
  // request the thread makes meanwhile by the held lease's deadline -- read
  // afresh per request, so a renewal inside the stretch extends it at once.
  // Before each request it renews the lease if that has fallen due
  // (keep_lease_in_pass). A lease whose deadline has passed is abandoned on
  // the way in and on the way out, so it is neither used nor reported held;
  // the lease state is published on the way out. Every use of writer_'s
  // lease goes through one but the schema install's at start(), before
  // anything else can use the coordinator: CatalogSchema::ensure() claims,
  // renews and releases its install lease there directly, its DDL bounded
  // by the client's timeouts rather than by that lease's deadline, and an
  // install claim that timed out is retried, not quarantined -- a row of it
  // that lands late refuses start()'s own claim until it expires, which
  // start() waits out like any holder's.
  class LeaseScope;

  // The dead sibling being adopted (storage_service.cpp).
  struct Adoption;

  void loop();
  // start()'s spool sweep and reconcile, with the lease held and the lease
  // thread renewing it. Requires cycle_mutex_.
  void sweep_and_reconcile_at_start();
  // One cycle's share of adoption (see adopt_sibling_spools): looks at the
  // siblings when that is due, then adopts a step at a time -- a sibling
  // taken and swept, one of its packs validated, a round, a finish -- until
  // the slice ends, a flush() is waiting (after one step at least), a stop
  // is requested or the uploads' Cancellation is cancelled -- which also
  // cuts a round in flight, and *cut_short then says so. Adds what its index
  // passes left owed through a cancel to *deferred (index_bounded). False
  // when an upload failed, so the cycle backs off. Requires cycle_mutex_.
  // Only a lost lease propagates.
  bool adopt_step(uint64_t deadline_ns, size_t* deferred, bool* cut_short);
  // Probes every sibling rank directory's owner lock and queues the dead
  // ones. False when the directory cannot be listed. Requires cycle_mutex_.
  bool scan_siblings();
  // Takes a queued sibling's lock, sweeps its .open files and lists its
  // ready packs into adopting_, hashing none of them (adopt_step validates
  // them, a pack a step); leaves adopting_ empty for a live or vanished
  // one, and for one it cannot lock or open, which it blocks.
  void begin_adoption(const std::string& directory);
  // Uploads and indexes one round of adopting_'s packs, one chunk of the
  // service's upload path; *cut when a cancel left some in the dead spool,
  // back at the front of adopting_. False when an upload failed.
  bool upload_adopted_round(uint64_t deadline_ns, size_t* deferred,
                            bool* cut);
  // adopting_ holds no pack any more: removes the directory, or leaves a
  // blocked one.
  void finish_adoption();
  // Leaves a dead sibling in place for good, reporting why -- and, given
  // its held lock, marking why in it (SpoolOwnerLock::MarkBlocked) before
  // letting go of it.
  void block_sibling(const std::string& directory, const std::string& reason,
                     dmi_store::SpoolOwnerLock* lock = nullptr);
  bool adoption_owed() const;  // requires cycle_mutex_
  // Lets go of the sibling being adopted, and of those queued, leaving what
  // is left of them for the next process on the node: at stop(), and once
  // the service has latched. Requires cycle_mutex_.
  void let_go_of_adoption();
  bool stop_requested();
  // Removes the staging copies (dmi_store::IsSpoolClaimStagingName) that
  // claims killed before their rename left under the catalog key.
  void clear_dead_claim_staging(
      const std::vector<std::filesystem::path>& staging);
  // Uploads one chunk -- packs a listing returned, in its order -- through
  // `uploader`, the service's own or an adoption's over a dead spool, and
  // books the outcome. The caller indexes to_index (index_or_owe) before it
  // uploads another chunk, so at most one chunk is ever out of a spool and
  // not yet in the catalog. Requires cycle_mutex_.
  ChunkOutcome upload_chunk(dmi_store::SpoolUploader* uploader,
                            std::vector<dmi_store::StagedPack> chunk);
  // Indexes refs that are gone from their spool, keeping whatever does not
  // index in pending_index_ -- the only record of it in-process. With no
  // catalog, keeps them all. Returns how many of those a cancel or the
  // deadline left owed (index_bounded). Only a lost lease propagates.
  // Requires cycle_mutex_.
  size_t index_or_owe(std::vector<PackRefData> refs, bool catalog,
                      uint64_t deadline_ns);
  // Stops the lease thread and waits for it.
  void stop_lease_thread();
  // One cycle. A non-zero deadline_ns (steady ns) cancels its uploads at
  // that moment -- flush()'s -- and allow_reconcile false skips the
  // periodic and the owed reconcile. `adopt`: the loop's cycles adopt dead
  // siblings (adopt_sibling_spools), flush()'s do not. Requires
  // cycle_mutex_.
  CycleOutcome run_cycle(uint64_t deadline_ns, bool allow_reconcile,
                         bool adopt);
  // Indexes refs in bounded batches, appending every ref that did not index
  // to *unindexed. Only a lost lease propagates; other failures are
  // recorded. A non-zero deadline_ns (steady ns) starts no batch past it
  // but the first, and none starts once read_cancel_ is cancelled. Returns
  // how many of *unindexed a cancel or the deadline left there: owed, but
  // not failed.
  size_t index_bounded(std::vector<PackRefData> refs,
                       std::vector<PackRefData>* unindexed,
                       uint64_t deadline_ns = 0);
  // False when stop() cut it short, between two of its requests.
  bool reconcile();
  void keep_lease();          // the lease thread's body
  void renew_lease_if_due();  // requires lease_mutex_
  // LeaseScope's before_request hook: renew_lease_if_due() before each
  // request a stretch under the lease lock sends.
  void keep_lease_in_pass();  // requires lease_mutex_
  // Gives up a held lease whose deadline has passed, counting it towards
  // lease_timeouts. Requires lease_mutex_.
  void abandon_lease_if_expired();
  // A lease claim or renewal failed (call from its catch block): one that
  // timed out counts towards lease_timeouts. Requires lease_mutex_.
  void note_lease_failure(const std::exception& failure);
  // Counts one timeout that cost the lease; `latest` says what it was.
  // Requires lease_mutex_.
  void count_lease_timeout(const std::string& latest);
  // Clears the timeout count once the lease now held has been held for
  // 2 x TTL (publish_lease_state runs it). Requires lease_mutex_.
  void track_stable_lease();
  // Takes the lease at start(), waiting for an expiring predecessor or
  // retrying a claim that timed out.
  void acquire_lease_at_start();  // requires lease_mutex_
  // Whether the writer holds a lease, taking a fresh one when it has none
  // and is no longer quarantined. Never throws. Requires lease_mutex_.
  bool ensure_publisher_lease();
  // Another holder refused a claim or renewal; latches once that has lasted
  // 2 x TTL. A refusal by the service's own claim rows restarts the clock
  // instead. Call from the catch block. Requires lease_mutex_.
  void lease_held_elsewhere(const CatalogError& refusal);
  void publish_lease_state();  // requires lease_mutex_
  // Sets a pack aside for good; flush() reports it. Requires cycle_mutex_.
  void reject(const PackRefData& ref, const std::string& reason);
  void record_error(const std::string& message);
  void latch_failure(std::exception_ptr failure, const std::string& message);

  const StorageServiceConfig config_;
  // Cancels the uploads: stop() for good, a flush's cycle at its deadline.
  // Before the clients that point at it.
  dmi_store::Cancellation upload_cancel_;
  // Cancels the index reads and the reconcile's requests: stop() for good,
  // a flush's cycle one catalog request timeout past its deadline.
  dmi_store::Cancellation read_cancel_;
  // Indexing and the reconcile read through s3_, which read_cancel_ cuts;
  // the uploader writes through upload_s3_, which upload_cancel_ does.
  dmi_store::S3Client s3_;
  dmi_store::S3Client upload_s3_;
  std::shared_ptr<const ClickHouseClient> clickhouse_;
  CatalogWriter writer_;
  NativeIndexer indexer_;
  dmi_store::Spool spool_;
  std::unique_ptr<dmi_store::SpoolUploader> uploader_;

  // Serialises cycles: the loop and flush() both run them, and NativeIndexer
  // is not thread-safe. Timed, so flush() can give up at its deadline while
  // a cycle is still in flight.
  std::timed_mutex cycle_mutex_;
  // Serialises every use of writer_'s lease (but the schema install's at
  // start(), see LeaseScope): the lease thread renews it while cycles
  // publish. Taken inside cycle_mutex_, never the other way, and only
  // through a LeaseScope.
  std::mutex lease_mutex_;
  // Timeouts that cost the lease since one was last held for 2 x TTL, and
  // every one ever counted (which LeaseScope compares, so that a loss is
  // counted once). Guarded by lease_mutex_.
  uint64_t lease_timeouts_ = 0;
  uint64_t lease_timeouts_counted_ = 0;
  // The lease_id held when publish_lease_state() last looked, and since
  // when (track_stable_lease). Guarded by lease_mutex_.
  std::string stable_lease_id_;
  uint64_t stable_since_ns_ = 0;
  // When a claim or renewal was first refused by another holder since the
  // lease was last held; 0 while none has been. Guarded by lease_mutex_.
  uint64_t held_elsewhere_since_ns_ = 0;
  // Earliest next claim after a refusal, so flush()'s fast cycles do not
  // hammer the lease table. Guarded by lease_mutex_.
  uint64_t next_claim_ns_ = 0;
  uint64_t last_reconcile_ns_ = 0;
  // The reconcile at start() lost the lease before it finished; the loop
  // runs one once it holds a lease again. Guarded by cycle_mutex_.
  bool reconcile_owed_ = false;
  // Adoption's state, guarded by cycle_mutex_: whether a look at the
  // siblings is due (from start() on), the dead ones the last look found,
  // the one being adopted, whether the last look found a live one and when
  // it ran (the loop looks again every adoption_recheck_interval_ns).
  bool adoption_scan_owed_ = false;
  std::deque<std::string> adoption_queue_;
  std::unique_ptr<Adoption> adopting_;
  std::set<std::string> blocked_siblings_;
  bool live_siblings_ = false;
  uint64_t last_adoption_scan_ns_ = 0;
  // flush() calls in progress. A cycle adopting lets go of the cycle at
  // its next step while one is, so a flush never waits out a slice.
  std::atomic<int> flushes_in_progress_{0};
  int failure_streak_ = 0;  // consecutive failed cycles, for the backoff
  // steady ns at which the last cycle -- the loop's or a flush's -- ended;
  // 0 before the first. The loop waits its interval from it. Guarded by
  // cycle_mutex_.
  uint64_t last_cycle_end_ns_ = 0;
  // Uploaded, so gone from the spool, but not yet in the catalog.
  std::vector<PackRefData> pending_index_;
  std::map<std::string, int> index_attempts_;  // by pack id
  std::vector<std::string> rejected_unreported_;  // for the next flush()

  std::thread thread_;
  // Renews on its own schedule, so neither the cycle backoff nor a slow
  // upload can let the lease lapse while the service still runs. It runs
  // from the moment start() takes the lease until the loop has stopped.
  std::thread lease_thread_;
  std::mutex wake_mutex_;
  std::condition_variable wake_;
  bool stop_requested_ = false;        // the loop's; guarded by wake_mutex_
  bool lease_stop_requested_ = false;  // the lease thread's; likewise
  // Set when a lease is re-acquired, so a loop in a long backoff indexes
  // what is owed now rather than after its wait. Guarded by wake_mutex_.
  bool kick_ = false;
  bool started_ = false;

  mutable std::mutex state_mutex_;
  StorageServiceSnapshot state_;
  std::exception_ptr failure_;
};

}  // namespace dmi_catalog
