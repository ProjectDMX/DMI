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
// Deployment shape: the service holds the catalog's single publisher lease, so
// run ONE service per (database, table_prefix). A second one fails to acquire
// the lease at start(). That is the in-process mode; a standalone daemon can
// reuse this class unchanged.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
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

  dmi_store::S3Config s3;
  dmi_store::UploaderConfig uploader;  // uploader.store_id names the store

  std::string clickhouse_host = "127.0.0.1";
  uint16_t clickhouse_port = 8123;
  ClickHouseTimeouts clickhouse_timeouts;
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

  // Sweep a crashed sink's stale .open files before anything writes to the
  // spool. Recover() deletes every .open file this object does not own, so it
  // is only safe while no writer is live: start() must run before the sink
  // opens the spool.
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
  uint64_t lease_renewals = 0;
  uint64_t swept_on_start = 0;  // ready packs Recover() found at start
  uint64_t pending_index = 0;   // uploaded packs awaiting a retried index
  uint64_t rejected_packs = 0;  // set aside: cannot be indexed (see flush)
  std::string last_error;
};

class CaptureStorageService {
 public:
  explicit CaptureStorageService(StorageServiceConfig config);
  ~CaptureStorageService();
  CaptureStorageService(const CaptureStorageService&) = delete;
  CaptureStorageService& operator=(const CaptureStorageService&) = delete;

  // Sweep the spool, ensure the catalog schema, take the publisher lease,
  // reconcile once, then start the background cycle. Throws if the lease is
  // held by another publisher.
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

  // Rethrows a fatal failure latched by the background cycle: losing the
  // publisher lease to another holder.
  void rethrow_if_failed() const;

 private:
  struct CycleOutcome {
    bool drained = false;  // nothing pending and nothing failed
    bool failed = true;    // an upload or index failed, or the cycle threw
  };

  void loop();
  CycleOutcome run_cycle();  // requires cycle_mutex_
  // Indexes refs in bounded batches, appending every ref that did not index
  // to *unindexed. Only a lost lease propagates; other failures are recorded.
  void index_bounded(std::vector<PackRefData> refs,
                     std::vector<PackRefData>* unindexed);
  void reconcile();
  void keep_lease();          // the lease thread's body
  void renew_lease_if_due();  // requires lease_mutex_
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
  // Serialises every use of writer_'s lease: the lease thread renews it
  // while cycles publish. Taken inside cycle_mutex_, never the other way.
  std::mutex lease_mutex_;
  uint64_t last_renew_ns_ = 0;  // guarded by lease_mutex_
  uint64_t last_reconcile_ns_ = 0;
  int failure_streak_ = 0;  // consecutive failed cycles, for the backoff
  // Uploaded, so gone from the spool, but not yet in the catalog.
  std::vector<PackRefData> pending_index_;
  std::map<std::string, int> index_attempts_;  // by pack id
  std::vector<std::string> rejected_unreported_;  // for the next flush()

  std::thread thread_;
  // Renews on its own schedule, so neither the cycle backoff nor a slow
  // upload can let the lease lapse while the service still runs.
  std::thread lease_thread_;
  std::mutex wake_mutex_;
  std::condition_variable wake_;
  bool stop_requested_ = false;
  bool started_ = false;

  mutable std::mutex state_mutex_;
  StorageServiceSnapshot state_;
  std::exception_ptr failure_;
};

}  // namespace dmi_catalog
