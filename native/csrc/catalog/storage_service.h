// CaptureStorageService: the production entry point for the native capture
// storage path after the spool.
//
//   NativePackSink -> spool  (the sink, in the capture process)
//   spool -> SpoolUploader -> object store -> NativeIndexer -> catalog  (here)
//
// One background thread runs a cycle: upload everything pending, index what
// was uploaded, keep the publisher lease alive, and periodically reconcile the
// bucket against the catalog. The conformance drivers exercise each of these
// pieces; this is what composes them outside a test.
//
// SpoolUploader removes a pack from the spool the moment its upload is
// verified, before anything indexes it, so the spool alone cannot say what is
// still owed to the catalog. Two things cover the gap:
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
// layout, and at start() it adopts the siblings whose owner has died: a
// crashed process's spool is recovered by the next process on the node for
// the same catalog, whatever run it belongs to.
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
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "catalog/catalog_writer.h"
#include "catalog/clickhouse_client.h"
#include "catalog/indexer.h"
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
  // under THIS catalog's key (SpoolCatalogKey of writer.database,
  // writer.table_prefix and uploader.store_id), or construction throws.
  // start(), after the lease and the sweep of its own directory, tries the
  // owner lock of every sibling rank directory; each one whose owner is
  // gone has its .open files swept, its .ready packs uploaded and indexed,
  // and is removed once nothing but its lock file is left. A sibling that
  // could not be drained (no lease, an upload that failed, a pack still
  // owed) is retried by the loop's cycles, and flush() does not report
  // drained until it has been. Live siblings -- another rank or job on this
  // node -- are left alone.
  bool adopt_sibling_spools = false;

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
  // upload too. A cycle checks the lease once, before its upload batch, and
  // UploadPending does not stop when the lease is lost: a holder that is
  // quarantined, or refused a renewal or publish, while a batch is in
  // flight finishes that batch (which can outlast the TTL), and only its
  // later cycles upload nothing while it holds no lease. One whose catalog
  // requests stall gives the lease up at its deadline, before its row
  // lapses: requests made under the lease are cut off there, and a cycle's
  // check abandons a lease past it (LeaseScope), so no batch starts after
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
  // that were uploaded, and whether one is still owed a retry.
  uint64_t adopted_spools = 0;
  uint64_t adopted_packs = 0;
  bool adoption_owed = false;
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
  // such a claim left), sweep the spool, adopt dead siblings
  // (adopt_sibling_spools), reconcile once, then start the background
  // cycle. The lease renews from the moment it is taken.
  // Throws if the lease is still held by another publisher when the wait
  // ends, or its claim still times out. A lease lost while the reconcile
  // runs does not fail start(): the loop takes a fresh one, as it would
  // later, and reconciles then.
  void start();

  // Run cycles until one finds the spool empty with every uploaded pack
  // indexed, or the timeout passes. Call after the sink's own flush, so
  // everything it will stage is already staged. Returns false on timeout,
  // including while a cycle already in flight outlives the deadline; it can
  // overrun only by its own last cycle, whose requests are all bounded.
  // Throws, once, if packs were set aside since the last flush: they are in
  // the object store but can never reach the catalog.
  bool flush(double timeout_s);

  // Stop the background cycle and release the lease. Does not flush.
  void stop();

  StorageServiceSnapshot snapshot() const;

  // Rethrows a fatal failure latched by the background cycle: another
  // publisher holding the lease for longer than 2 x TTL.
  void rethrow_if_failed() const;

 private:
  struct CycleOutcome {
    bool drained = false;  // nothing pending and nothing failed
    bool failed = true;    // an upload or index failed, or the cycle threw
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

  void loop();
  // start()'s spool sweep, adoption and reconcile, with the lease held and
  // the lease thread renewing it. Requires cycle_mutex_.
  void sweep_and_reconcile_at_start();
  // One adoption pass over the sibling rank directories; sets
  // adoption_owed_ to whether one was left undrained. Requires
  // cycle_mutex_. Only a lost lease propagates.
  void adopt_siblings();
  // Adopts one sibling; false when it is owed another try.
  bool adopt_sibling(const std::string& directory);
  // Removes the staging copies (dmi_store::IsSpoolClaimStagingName) that
  // claims killed before their rename left under the catalog key.
  void clear_dead_claim_staging(
      const std::vector<std::filesystem::path>& staging);
  // Indexes refs that are gone from their spool, keeping whatever does not
  // index in pending_index_ -- the only record of it in-process. With no
  // catalog, keeps them all. Requires cycle_mutex_.
  void index_or_owe(std::vector<PackRefData> refs, bool catalog);
  // Stops the lease thread and waits for it.
  void stop_lease_thread();
  CycleOutcome run_cycle();  // requires cycle_mutex_
  // Indexes refs in bounded batches, appending every ref that did not index
  // to *unindexed. Only a lost lease propagates; other failures are recorded.
  void index_bounded(std::vector<PackRefData> refs,
                     std::vector<PackRefData>* unindexed);
  void reconcile();
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
  dmi_store::S3Client s3_;
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
  // An adoption pass left a dead sibling undrained. Guarded by cycle_mutex_.
  bool adoption_owed_ = false;
  int failure_streak_ = 0;  // consecutive failed cycles, for the backoff
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
