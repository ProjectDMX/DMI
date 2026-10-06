#include "catalog/storage_service.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

#include "catalog/schema.h"
#include "pack/pack_builder.h"

namespace dmi_catalog {
namespace {

uint64_t steady_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

constexpr const char* kPackSuffix = ".dmi-pack";

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_hex64(const std::string& value) {
  if (value.size() != 64) return false;
  for (char c : value) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

// "<prefix>/<pack_id>.dmi-pack" -> pack_id, or "" if the key is not a pack.
std::string pack_id_of(const std::string& key) {
  if (!ends_with(key, kPackSuffix)) return "";
  const size_t slash = key.rfind('/');
  const size_t start = slash == std::string::npos ? 0 : slash + 1;
  return key.substr(start, key.size() - start - std::string(kPackSuffix).size());
}

// The uploader only ever writes canonical lowercase UUID pack ids, and the
// catalog's pack_id column is a UUID: anything else would fail the whole
// committed-ids query for its page, not just itself.
bool is_pack_id(const std::string& value) {
  std::array<uint8_t, 16> bytes{};
  std::string canonical;
  return dmi_pack::ParseUuid(value, &bytes, &canonical) && canonical == value;
}

// The lease thread's tick: the longest it sleeps, and the retry interval for
// a claim another holder refused or one that wrote nothing -- a sixth of the
// TTL. A renewal falls due a third of the TTL after the claim that stamped
// the row was sent; the thread wakes for it then, not on the next tick, and
// a stretch holding the lease lock renews before its next request once it
// is due. renewal_window_ns (lease_coordinator.h) allows a tick more than
// that -- a renewal is taken to start within ttl/2 of the send -- so change
// one and the other must follow.
uint64_t lease_tick_ns(uint64_t ttl_ns) {
  return std::max<uint64_t>(ttl_ns / 6, 10'000'000ull);
}

std::string seconds_text(uint64_t ns) {
  char out[32];
  std::snprintf(out, sizeof(out), "%g s", static_cast<double>(ns) / 1e9);
  return out;
}

// The service's indexer ends a read pass at a pack the object store did not
// answer for, rather than failing that pack and reading the next.
IndexerConfig service_indexer_config(IndexerConfig config) {
  config.end_read_when_store_unavailable = true;
  return config;
}

// How long past a flush's deadline its index reads may still run: the one
// catalog request timeout the flush may overrun by anyway. Capped so the
// nanoseconds cannot overflow.
uint64_t read_grace_ns(const ClickHouseConnection& connection) {
  const double seconds = std::min(connection.timeouts.request_s, 1e6);
  return seconds > 0 ? static_cast<uint64_t>(seconds * 1e9) : 0;
}

// Arms a Cancellation's deadline for one cycle, at ns (0: none), and
// disarms it at the end. Cancel(), stop()'s, outlives it.
class ArmedDeadline {
 public:
  ArmedDeadline(dmi_store::Cancellation* cancel, uint64_t ns)
      : cancel_(cancel), armed_(ns != 0) {
    if (armed_) cancel_->set_deadline(ns);
  }
  ~ArmedDeadline() {
    if (armed_) cancel_->set_deadline(0);
  }
  ArmedDeadline(const ArmedDeadline&) = delete;
  ArmedDeadline& operator=(const ArmedDeadline&) = delete;

 private:
  dmi_store::Cancellation* cancel_;
  const bool armed_;
};

}  // namespace

class CaptureStorageService::LeaseScope {
 public:
  explicit LeaseScope(CaptureStorageService* service)
      : service_(service),
        lock_(service->lease_mutex_),
        deadline_([service] { return service->writer_.lease_deadline_ns(); },
                  kLeaseDeadlineBound,
                  [service] { service->keep_lease_in_pass(); }) {
    service_->abandon_lease_if_expired();
    const PublisherLease* held = service_->writer_.held_lease();
    if (held != nullptr) held_on_entry_ = held->lease_id;
    counted_on_entry_ = service_->lease_timeouts_counted_;
  }

  ~LeaseScope() {
    try {
      service_->abandon_lease_if_expired();
      // A lease this stretch lost to a request that timed out -- one its
      // deadline cut off, or the server's own time limit -- counts towards
      // lease_timeouts, unless what lost it counted already (a renewal or
      // claim that timed out, or the abandon above).
      const PublisherLease* held = service_->writer_.held_lease();
      const bool lost = !held_on_entry_.empty() &&
                        (held == nullptr || held->lease_id != held_on_entry_);
      if (lost && service_->lease_timeouts_counted_ == counted_on_entry_ &&
          !deadline_.last_timeout().empty()) {
        service_->count_lease_timeout(deadline_.last_timeout());
      }
      service_->publish_lease_state();
    } catch (...) {
    }
  }

  LeaseScope(const LeaseScope&) = delete;
  LeaseScope& operator=(const LeaseScope&) = delete;

 private:
  CaptureStorageService* service_;
  std::unique_lock<std::mutex> lock_;
  RequestDeadline deadline_;  // after lock_: released before it
  std::string held_on_entry_;  // the lease_id held on entry, "" for none
  uint64_t counted_on_entry_ = 0;
};

// The dead sibling a cycle is adopting, kept across cycles: its owner lock,
// a Spool on it (held_by_caller, the lock being this service's), its
// recovery -- the packs listed and those validated so far, a pack a step
// (Spool::BeginRecovery) -- and, once every pack is validated, those not
// uploaded yet. So a backlog is hashed once, however many cycles its
// listing and its upload take.
struct CaptureStorageService::Adoption {
  std::string directory;
  dmi_store::SpoolOwnerLock lock;
  dmi_store::Spool spool;
  dmi_store::SpoolRecovery recovery;
  bool validated = false;  // every listed pack; `remaining` holds the valid
  std::deque<dmi_store::StagedPack> remaining;
  // The first pack this service can never upload (UploadFailure::
  // retryable false); the directory is left, with it, once the rest are up.
  std::string blocked;
};

CaptureStorageService::CaptureStorageService(StorageServiceConfig config)
    : config_(std::move(config)),
      s3_(config_.s3),
      upload_s3_(config_.s3),
      clickhouse_(std::make_shared<const ClickHouseClient>(config_.clickhouse)),
      writer_(clickhouse_, config_.writer),
      indexer_(&s3_, &writer_, service_indexer_config(config_.indexer)) {
  if (config_.spool_root.empty()) {
    throw std::invalid_argument("storage service: spool_root is required");
  }
  if (config_.holder.empty()) {
    throw std::invalid_argument("storage service: holder is required");
  }
  if (config_.uploader.store_id.empty()) {
    throw std::invalid_argument("storage service: uploader.store_id is required");
  }
  // A renewal that fails must fail while its row still keeps rivals out,
  // so it has until the lease deadline (lease_coordinator.h). A skew bound
  // that leaves it too little time is refused here rather than turned into
  // renewals that time out against a healthy server.
  const uint64_t window_ns = renewal_window_ns(config_.writer.lease_ttl_ns,
                                               config_.writer.clock_skew_ns);
  if (window_ns < kMinimumRenewalWindowNs) {
    throw std::invalid_argument(
        "storage service: clock_skew_ns leaves a lease renewal no time to "
        "finish while its row is live. A renewal starts up to lease_ttl_ns "
        "/ 2 after the claim that stamped the row was sent, and has to be "
        "answered by lease_ttl_ns less clock_skew_ns and a " +
        std::to_string(kLeaseDeadlineMarginNs / 1'000'000) +
        " ms margin after it: " + std::to_string(window_ns / 1'000'000) +
        " ms, under the " +
        std::to_string(kMinimumRenewalWindowNs / 1'000'000) +
        " ms minimum. Keep clock_skew_ns at most lease_ttl_ns / 2 - " +
        std::to_string((kMinimumRenewalWindowNs + kLeaseDeadlineMarginNs) /
                       1'000'000) +
        " ms, or raise lease_ttl_ns");
  }
  std::string error;
  dmi_store::SpoolConfig spool_config{config_.spool_root,
                                      config_.spool_max_bytes};
  spool_config.owner_lock = config_.spool_owner_lock;
  spool_config.allow_shared_filesystem = config_.spool_allow_shared_filesystem;
  if (dmi_store::Spool::Open(spool_config, &spool_, &error) !=
      dmi_store::SpoolStatus::kOk) {
    throw std::runtime_error("storage service: cannot open spool: " + error);
  }
  if (config_.adopt_sibling_spools) {
    // Siblings are adopted INTO this catalog, so this directory must sit
    // under this catalog's key: a directory under another catalog's key
    // would index that catalog's packs here.
    const std::filesystem::path own(spool_.root());
    dmi_store::SpoolDestination destination;
    destination.clickhouse_host = config_.clickhouse.host;
    destination.clickhouse_port = config_.clickhouse.port;
    destination.database = config_.writer.database;
    destination.table_prefix = config_.writer.table_prefix;
    destination.s3_endpoint = config_.s3.endpoint;
    destination.s3_bucket = config_.s3.bucket;
    destination.store_id = config_.uploader.store_id;
    const std::string key = dmi_store::SpoolCatalogKey(destination);
    uint64_t rank = 0;
    std::string incarnation;
    if (!dmi_store::ParseSpoolRankDirectoryName(own.filename().string(),
                                                &rank, &incarnation) ||
        own.parent_path().filename().string() != key) {
      throw std::invalid_argument(
          "storage service: adopt_sibling_spools needs spool_root to be a "
          "rank directory <base>/" + key + "/r<rank>-<incarnation> (this "
          "catalog's key for its ClickHouse host and port, database, "
          "table_prefix, S3 endpoint, bucket and store_id), got " +
          spool_.root());
    }
  }
  s3_.set_cancellation(&read_cancel_);
  upload_s3_.set_cancellation(&upload_cancel_);
  uploader_ = std::make_unique<dmi_store::SpoolUploader>(&spool_, &upload_s3_,
                                                          config_.uploader);
  uploader_->set_cancellation(&upload_cancel_);
}

CaptureStorageService::~CaptureStorageService() {
  try {
    stop();
  } catch (...) {
  }
}

void CaptureStorageService::start() {
  std::lock_guard<std::timed_mutex> cycle(cycle_mutex_);
  if (started_) throw std::logic_error("storage service: already started");
  // A stop() before this one cancelled both for good.
  upload_cancel_.Reset();
  read_cancel_.Reset();

  CatalogSchema(clickhouse_, config_.writer.database, config_.writer.table_prefix)
      .ensure(&writer_.leases(), config_.schema_retry_sleep_ns);
  {
    LeaseScope lease(this);
    acquire_lease_at_start();  // throws kHeld if another publisher keeps it
  }

  // The lease renews from here on, not once start() is done: the sweep and
  // the reconcile below can outlast the lease deadline (a large bucket lists
  // for longer than a TTL), and a lease nobody renewed is abandoned at the
  // next lease-locked step, when it is reported held over a dead row until
  // then.
  {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    stop_requested_ = false;
    lease_stop_requested_ = false;
    kick_ = false;
  }
  lease_thread_ = std::thread([this] { keep_lease(); });
  try {
    sweep_and_reconcile_at_start();
  } catch (...) {
    // start() must not return holding a lease stop() will never release,
    // nor leave the lease thread renewing it.
    stop_lease_thread();
    try {
      LeaseScope lease(this);
      if (writer_.held_lease() != nullptr) writer_.release_lease();
    } catch (...) {
    }
    throw;
  }
  last_reconcile_ns_ = steady_ns();

  {
    // With siblings to look at, the loop's first cycle runs at once: it,
    // not start(), adopts them.
    std::lock_guard<std::mutex> lock(wake_mutex_);
    kick_ = adoption_scan_owed_;
  }
  {
    LeaseScope lease(this);  // publishes the lease state
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.running = !state_.failed;
  }
  started_ = true;
  thread_ = std::thread([this] { loop(); });
}

void CaptureStorageService::sweep_and_reconcile_at_start() {
  // After the lease: a start refused the catalog never touches the spool.
  // The spool's owner lock (taken at construction, by this service or its
  // caller) is what keeps another process's writer out of the directory
  // this deletes .open files in.
  if (config_.sweep_spool_on_start) {
    std::vector<dmi_store::StagedPack> recovered;
    std::string error;
    if (spool_.Recover(&recovered, &error) != dmi_store::SpoolStatus::kOk) {
      throw std::runtime_error("storage service: spool recovery failed: " +
                               error);
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.swept_on_start = recovered.size();
  }

  // Dead siblings are the loop's, from its first cycle on: after the lease
  // and this directory's own sweep, as the plan orders them, but not here,
  // where a backlog the object store refuses would hold start() -- and
  // create_record_runtime -- through every pack's retry chain.
  if (config_.adopt_sibling_spools) {
    adoption_scan_owed_ = true;
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.adoption_owed = true;
  }

  // A failed pass is not fatal -- the bucket is still there next time. Nor
  // is a lease lost while it runs, to a quarantine or to another holder:
  // that is the running service's case, and the loop handles it as it does
  // there, taking a fresh lease once it can (or latching after 2 x TTL of a
  // rival). The pass it cut short is owed, and the loop runs it once it
  // holds a lease again.
  if (config_.reconcile_on_start) {
    try {
      reconcile();
    } catch (const CatalogError& exc) {
      if (is_lease_refusal(exc)) {
        reconcile_owed_ = true;
        record_error(std::string("reconcile at start lost the publisher "
                                 "lease; the loop reconciles once it holds "
                                 "one again: ") +
                     exc.what());
      } else {
        record_error(std::string("reconcile at start failed: ") + exc.what());
      }
    } catch (const std::exception& exc) {
      record_error(std::string("reconcile at start failed: ") + exc.what());
    }
  }
}

void CaptureStorageService::stop() {
  {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    stop_requested_ = true;
  }
  // Before the join: an upload or an index read the store never answers,
  // or a retry backoff, would otherwise hold it for the S3 client's
  // timeouts on every attempt, with the lease held. A cancelled upload
  // leaves its pack in the spool; a cancelled read leaves its pack owed,
  // as a read that timed out would, and so in the bucket for the next
  // start's reconcile.
  upload_cancel_.Cancel();
  read_cancel_.Cancel();
  wake_.notify_all();
  if (thread_.joinable()) thread_.join();
  // Only after the loop: its last cycle may still be indexing, and the
  // lease has to keep renewing until that is done.
  stop_lease_thread();
  std::lock_guard<std::timed_mutex> cycle(cycle_mutex_);
  let_go_of_adoption();
  if (started_) {
    started_ = false;
    // A quarantined writer holds no lease, so it writes no tombstone: the
    // outcome-unknown statement may still be running, and its row must stay
    // live until the TTL keeps a successor out of that window.
    bool released = false;
    try {
      LeaseScope lease(this);
      if (writer_.held_lease() != nullptr) {
        writer_.release_lease();
        released = true;
      }
    } catch (const std::exception& exc) {
      record_error(std::string("lease release failed: ") + exc.what());
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!state_.failed) state_.lease_state = released ? "released" : "none";
    state_.quarantined_until_ns = 0;
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.running = false;
}

void CaptureStorageService::stop_lease_thread() {
  {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    lease_stop_requested_ = true;
  }
  wake_.notify_all();
  if (lease_thread_.joinable()) lease_thread_.join();
}

bool CaptureStorageService::flush(double timeout_s) {
  // The loop's adoption gives way to this call at its next step, so the
  // cycle lock is not held for the rest of an adoption slice.
  struct InProgress {
    explicit InProgress(std::atomic<int>* count) : count_(count) {
      count_->fetch_add(1, std::memory_order_acq_rel);
    }
    ~InProgress() { count_->fetch_sub(1, std::memory_order_acq_rel); }
    std::atomic<int>* count_;
  } in_progress(&flushes_in_progress_);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::duration<double>(timeout_s));
  // The same instant for the cycles' upload cancel, on the steady clock
  // steady_ns() reads. Never 0, which would arm none.
  const uint64_t deadline_ns = std::max<uint64_t>(
      1, static_cast<uint64_t>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(
                 deadline.time_since_epoch())
                 .count()));
  while (true) {
    {
      // A cycle in flight -- the loop's, stuck on a slow catalog -- must not
      // hold this call past its deadline.
      std::unique_lock<std::timed_mutex> cycle(cycle_mutex_, std::defer_lock);
      if (!cycle.try_lock_until(deadline)) return false;
      if (!started_) throw std::logic_error("storage service: not started");
      rethrow_if_failed();
      // stop() has begun: it cancelled the uploads for good and waits for
      // this lock. Cycles now could only index, and would do so until the
      // deadline.
      if (upload_cancel_.cancelled_for_good()) return false;
      // Its own cycle, bounded by the deadline, and without the reconcile:
      // the reconcile lists the whole bucket and asks the catalog about
      // every page, which no deadline bounds. The loop runs it. Nor does it
      // adopt: a dead backlog is not this process's records.
      const bool drained = run_cycle(deadline_ns, false, false).drained;
      if (!rejected_unreported_.empty()) {
        std::string message = "storage service: " +
                              std::to_string(rejected_unreported_.size()) +
                              " pack(s) cannot be indexed and were set aside "
                              "(still in the object store): ";
        for (size_t i = 0; i < rejected_unreported_.size(); ++i) {
          if (i) message += "; ";
          message += rejected_unreported_[i];
        }
        rejected_unreported_.clear();
        throw std::runtime_error(message.substr(0, 4096));
      }
      if (drained) return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

StorageServiceSnapshot CaptureStorageService::snapshot() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_;
}

void CaptureStorageService::rethrow_if_failed() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (failure_) std::rethrow_exception(failure_);
}

void CaptureStorageService::loop() {
  uint64_t wait_ns = config_.poll_interval_ns;
  bool waited_again = false;  // the last wake gave way to a flush's cycle
  while (true) {
    bool kicked = false;
    {
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_.wait_for(lock, std::chrono::nanoseconds(wait_ns),
                     [this] { return stop_requested_ || kick_; });
      if (stop_requested_) return;
      kicked = kick_;
      kick_ = false;
    }
    bool failed = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      failed = failure_ != nullptr;
    }
    std::lock_guard<std::timed_mutex> cycle(cycle_mutex_);
    if (failed) {
      // Another publisher holds the catalog, and no cycle runs again. A
      // sibling half adopted would stay locked by this process, with
      // nobody working on it, until stop() -- the engine's close(), maybe
      // hours of capture later -- and every other process on the node
      // would read it as live meanwhile. Nor does it look again.
      adoption_scan_owed_ = false;
      let_go_of_adoption();
      return;
    }
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      if (stop_requested_) return;
    }
    // The wait runs from the end of the last cycle, anyone's. A flush that
    // held the lock while this waited for it has just run one; running
    // another straight after it would only delay a stop() that follows the
    // flush -- close()'s order -- by that cycle's catalog work, which stop()
    // cannot cut. So wait out the rest of the interval first, unless a
    // fresh lease asked for a cycle now -- once: flushes that keep coming
    // do every cycle's work but the reconcile and the adoption, which only
    // this loop runs, so the next wake runs a cycle whatever they did.
    // start() kicks the first cycle when there are siblings to look at.
    const uint64_t since_ns = steady_ns() - last_cycle_end_ns_;
    if (!kicked && !waited_again && last_cycle_end_ns_ != 0 &&
        since_ns < wait_ns) {
      wait_ns -= since_ns;
      waited_again = true;
      continue;
    }
    waited_again = false;
    run_cycle(0, true, true);
    // poll_interval * 2^streak, capped: flush() shares the streak, so an
    // outage it saw also slows the loop, and a success from either resets it.
    wait_ns = config_.poll_interval_ns;
    for (int i = 0; i < failure_streak_ && wait_ns < config_.max_backoff_ns; ++i) {
      wait_ns *= 2;
    }
    wait_ns = std::min(std::max(wait_ns, config_.poll_interval_ns),
                       std::max(config_.max_backoff_ns, config_.poll_interval_ns));
  }
}

CaptureStorageService::CycleOutcome CaptureStorageService::run_cycle(
    uint64_t deadline_ns, bool allow_reconcile, bool adopt) {
  CycleOutcome outcome;
  // stop() has begun: it cancelled the uploads and the reads for good, so a
  // cycle could only make catalog requests -- a lease claim, a replay guard
  // -- that stop() would have to wait out. Neither drained nor failed.
  if (upload_cancel_.cancelled_for_good()) {
    outcome.failed = false;
    outcome.cut_short = true;
    last_cycle_end_ns_ = steady_ns();
    return outcome;
  }
  // flush()'s deadline cancels this cycle's uploads at that moment, and its
  // index reads one catalog request timeout later: an upload cut short
  // leaves its pack in the spool, but a read cut short leaves an uploaded
  // pack owed, which only this process remembers, so what the cycle
  // uploaded gets the time a catalog statement in flight would. The loop's
  // cycles are cancelled only by stop(), which cancels both for good.
  const ArmedDeadline upload_deadline(&upload_cancel_, deadline_ns);
  const ArmedDeadline read_deadline(
      &read_cancel_,
      deadline_ns == 0 ? 0 : deadline_ns + read_grace_ns(config_.clickhouse));
  // The catalog phase needs the lease. Without one -- quarantined after an
  // unknown outcome, or refused by another holder -- the cycle uploads
  // nothing either: whatever it uploaded it could only owe, in memory.
  bool catalog = false;
  {
    LeaseScope lease(this);
    catalog = ensure_publisher_lease();
  }
  // What a cycle uploads and cannot index stays owed in pending_index_
  // (index_or_owe). Those a cancel or the deadline left owed are counted
  // here: they make the cycle cut short, not failed.
  size_t deferred = 0;
  try {
    // 1. Retry what earlier cycles uploaded but could not index. While any
    //    of it is still owed, the catalog is down or refusing: upload
    //    nothing new, so new packs stay in the durable spool rather than
    //    joining a list that only this process remembers. Past a flush's
    //    deadline no batch starts but the first (index_bounded).
    if (catalog && !pending_index_.empty()) {
      std::vector<PackRefData> owed;
      owed.swap(pending_index_);
      deferred += index_or_owe(std::move(owed), catalog, deadline_ns);
    }

    // 2. Upload what the sink has staged -- but only with the lease and
    //    nothing owed. Without the lease an uploaded pack could only be
    //    owed, and pending_index_ dies with the process: with
    //    reconcile_on_start off, a crash would leave it in the bucket and
    //    never in the catalog. Left in the spool it survives the crash.
    //    The spool is listed once, since listing hashes every staged pack.
    //    A cancel (the flush deadline, or stop()) stops the listing between
    //    packs, before the next hash, starts no upload, and cuts those in
    //    flight: those packs stay staged too. So a flush out of time before
    //    this point -- flush(0) -- hashes nothing: a listing already
    //    cancelled stops at the first staged pack, and finds an empty spool
    //    empty, which then reports drained.
    // 3. Index them -- a chunk of indexer.max_packs at a time, each chunk
    //    indexed before the next is uploaded, so at most one chunk is ever
    //    out of the spool and not yet in the catalog, remembered by this
    //    process alone (stop() drops it). A cancel starts no further chunk,
    //    so what the flush deadline or stop() finds unsent stays in the
    //    spool, which any later start uploads, not owed in memory; a chunk
    //    left owed, or a lease lost meanwhile, stops the uploads as in step
    //    1. What a chunk uploaded it indexes, a cancel of the uploads or
    //    not: past a flush's deadline, as its one batch past it
    //    (index_bounded). Its reads end at stop(), or one request timeout
    //    past a flush's deadline, leaving what they did not read owed.
    // Every staged pack listed and uploaded: drained as far as flush() is
    // concerned, since the listing came after the sink's flush.
    bool uploaded_all = false;
    bool listing_failed = false;
    bool lost_lease = false;
    size_t upload_failures = 0;
    if (catalog && pending_index_.empty()) {
      std::vector<dmi_store::StagedPack> staged;
      std::string error;
      bool cut = false;
      if (spool_.ListPending(&staged, &error, &upload_cancel_, &cut) !=
          dmi_store::SpoolStatus::kOk) {
        record_error("spool listing failed: " + error);
        listing_failed = true;
      } else if (cut) {
        outcome.cut_short = true;
      } else {
        const size_t chunk =
            static_cast<size_t>(std::max(1, config_.indexer.max_packs));
        size_t next = 0;
        while (next < staged.size()) {
          if (next != 0) {
            if (!pending_index_.empty()) break;  // the chunk before is owed
            if (upload_cancel_.cancelled()) {
              outcome.cut_short = true;
              break;
            }
            // No request: a lease quarantined or refused meanwhile has
            // already been dropped, and a fresh one is the next cycle's.
            LeaseScope lease(this);
            if (writer_.held_lease() == nullptr) {
              lost_lease = true;
              break;
            }
          }
          const size_t end = std::min(staged.size(), next + chunk);
          ChunkOutcome sent = upload_chunk(
              uploader_.get(),
              std::vector<dmi_store::StagedPack>(staged.begin() + next,
                                                 staged.begin() + end));
          next = end;
          for (size_t i = 0; i < sent.batch.failures.size(); ++i) {
            const dmi_store::UploadFailure& failure = sent.batch.failures[i];
            // A failed upload stays in the spool, so a later cycle retries
            // it; a cancelled one is still staged, and not the pack's fault.
            if (!sent.batch.refs[i].pack_id.empty() || failure.cancelled) {
              continue;
            }
            record_error("upload failed for " + failure.object_key + ": " +
                         failure.error);
          }
          if (sent.cancelled != 0) outcome.cut_short = true;
          upload_failures += sent.failed;
          deferred += index_or_owe(std::move(sent.to_index), true, deadline_ns);
        }
        uploaded_all = next == staged.size() && upload_failures == 0;
      }
    }

    // 3a. The loop's cycles adopt dead siblings, a slice at a time, under
    //     the same rule as the uploads above: only with the lease, every
    //     staged pack of this process's up, nothing owed, and no cancel.
    //     flush()'s cycles do not: a dead backlog is not this process's
    //     records. Adoption uploads and indexes through the same chunks,
    //     and the same Cancellations, as the service's own spool.
    bool adoption_failed = false;
    if (adopt && config_.adopt_sibling_spools && uploaded_all &&
        pending_index_.empty() && !upload_cancel_.cancelled()) {
      bool adoption_cut = false;
      adoption_failed = !adopt_step(deadline_ns, &deferred, &adoption_cut);
      if (adoption_cut) outcome.cut_short = true;
    }

    // 4. Reconcile on its interval, or when the pass at start() lost the
    //    lease before it finished -- in the loop's cycles only, and not
    //    once a cancel came. The lease thread keeps the lease alive.
    if (allow_reconcile && catalog && !upload_cancel_.cancelled() &&
        (reconcile_owed_ ||
         (config_.reconcile_interval_ns > 0 &&
          steady_ns() - last_reconcile_ns_ >= config_.reconcile_interval_ns))) {
      if (reconcile()) {
        reconcile_owed_ = false;
        last_reconcile_ns_ = steady_ns();
      }
    }

    // Drained: every staged pack uploaded, every uploaded pack in the
    // catalog, and nothing pending. Without the lease nothing can be
    // confirmed in the catalog, so the cycle is not drained, and it counts
    // towards the backoff. Packs a cancel or the deadline left owed make it
    // cut short, not failed. A dead sibling still to adopt is not a
    // failure, nor undrained: only an adoption upload that failed backs the
    // loop off, and adopted packs uploaded and not indexed are owed like
    // the service's own.
    if (deferred != 0) outcome.cut_short = true;
    outcome.failed = !catalog || listing_failed || lost_lease ||
                     upload_failures != 0 || adoption_failed ||
                     pending_index_.size() > deferred;
    outcome.drained = uploaded_all && !outcome.failed && !outcome.cut_short;
    std::lock_guard<std::mutex> lock(state_mutex_);
    ++state_.cycles;
  } catch (const CatalogError& exc) {
    // A lease refusal here is not fatal either: the writer has dropped or
    // been fenced out of its lease, and the next ensure_publisher_lease()
    // decides between a fresh lease and a latch.
    record_error(is_lease_refusal(exc)
                     ? std::string("publisher lease lost: ") + exc.what()
                     : std::string(exc.what()));
  } catch (const std::exception& exc) {
    record_error(exc.what());
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.pending_index = pending_index_.size();
    state_.adoption_owed = adoption_owed();
  }
  // A cycle a cancel cut short proved nothing about the store or the
  // catalog: it resets the backoff only by succeeding, and grows it only by
  // failing.
  if (outcome.failed) {
    failure_streak_ = std::min(failure_streak_ + 1, 32);
  } else if (!outcome.cut_short) {
    failure_streak_ = 0;
  }
  last_cycle_end_ns_ = steady_ns();
  return outcome;
}

size_t CaptureStorageService::index_or_owe(std::vector<PackRefData> refs,
                                           bool catalog,
                                           uint64_t deadline_ns) {
  if (!catalog) {
    pending_index_.insert(pending_index_.end(), refs.begin(), refs.end());
    return 0;
  }
  std::vector<PackRefData> unindexed;
  size_t deferred = 0;
  try {
    if (!refs.empty()) {
      deferred = index_bounded(std::move(refs), &unindexed, deadline_ns);
    }
  } catch (...) {
    pending_index_.insert(pending_index_.end(), unindexed.begin(),
                          unindexed.end());
    throw;
  }
  pending_index_.insert(pending_index_.end(), unindexed.begin(),
                        unindexed.end());
  return deferred;
}

CaptureStorageService::ChunkOutcome CaptureStorageService::upload_chunk(
    dmi_store::SpoolUploader* uploader,
    std::vector<dmi_store::StagedPack> chunk) {
  ChunkOutcome sent;
  sent.batch = uploader->UploadStaged(std::move(chunk));
  uint64_t uploaded_bytes = 0;
  for (size_t i = 0; i < sent.batch.refs.size(); ++i) {
    const dmi_store::PackRef& ref = sent.batch.refs[i];
    if (!ref.pack_id.empty()) {
      sent.to_index.push_back({ref.pack_id, ref.store_id, ref.object_key,
                               ref.object_bytes, ref.checksum,
                               ref.record_count});
      uploaded_bytes += ref.object_bytes;
    } else if (i < sent.batch.failures.size() &&
               sent.batch.failures[i].cancelled) {
      ++sent.cancelled;  // still staged; not the pack's fault
    } else {
      ++sent.failed;  // still staged too
    }
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.uploaded_packs += sent.to_index.size();
  state_.uploaded_bytes += uploaded_bytes;
  state_.upload_failures += sent.failed;
  state_.cancelled_uploads += sent.cancelled;
  return sent;
}

void CaptureStorageService::let_go_of_adoption() {
  // A sibling half adopted keeps what is left of it, for the next process
  // on the node; what was uploaded from it was indexed, or is owed. Its
  // directory is not removed: only finish_adoption() removes one, once no
  // pack of it is left.
  adopting_.reset();
  adoption_queue_.clear();
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.adoption_owed = adoption_owed();
}

bool CaptureStorageService::adoption_owed() const {
  return adoption_scan_owed_ || adopting_ != nullptr ||
         !adoption_queue_.empty();
}

bool CaptureStorageService::stop_requested() {
  std::lock_guard<std::mutex> lock(wake_mutex_);
  return stop_requested_;
}

bool CaptureStorageService::adopt_step(uint64_t deadline_ns, size_t* deferred,
                                       bool* cut_short) {
  *cut_short = false;
  const uint64_t started = steady_ns();
  if (adopting_ == nullptr && adoption_queue_.empty()) {
    // Look at the siblings when that is owed (from start() on) or, while
    // the last look found a live one, again on the recheck interval.
    const bool recheck_due =
        live_siblings_ && config_.adoption_recheck_interval_ns > 0 &&
        started - last_adoption_scan_ns_ >=
            config_.adoption_recheck_interval_ns;
    if (!adoption_scan_owed_ && !recheck_due) return true;
    if (!scan_siblings()) return false;
  }
  bool ok = true;
  // Whether this call has taken a step yet: a lock taken and a sweep, one
  // pack of a listing validated, a round, or a finish. Each call takes one
  // at least, so flushes that keep coming slow adoption down but never stop
  // it.
  bool stepped = false;
  while (!stop_requested()) {
    // The service's uploads' Cancellation is adoption's too: stop() cuts
    // it between steps -- between the packs a listing validates too -- and
    // in a round (its uploads, and, through the reads' Cancellation, its
    // index reads).
    if (upload_cancel_.cancelled()) {
      *cut_short = true;
      break;
    }
    // A flush() is waiting for the cycle: the rest of the slice is the
    // next cycle's, which the loop runs once the flush is done.
    if (stepped &&
        flushes_in_progress_.load(std::memory_order_acquire) > 0) {
      break;
    }
    stepped = true;
    if (adopting_ == nullptr) {
      if (adoption_queue_.empty()) break;
      const std::string next = adoption_queue_.front();
      adoption_queue_.pop_front();
      begin_adoption(next);
      continue;
    }
    if (!adopting_->validated) {
      // One pack of the listing: validating hashes every byte of it, so
      // over a dead backlog a whole listing takes as long as the backlog
      // is big, and neither a flush nor stop() waits for more than the
      // pack in flight. The slice bounds a listing as it does the rounds.
      Adoption& adoption = *adopting_;
      if (adoption.spool.ContinueRecovery(&adoption.recovery)) {
        // Each pack's identity and object key come from the pack and its
        // path in the dead directory, exactly as its owner would have
        // uploaded it.
        adoption.remaining.assign(
            std::make_move_iterator(adoption.recovery.valid.begin()),
            std::make_move_iterator(adoption.recovery.valid.end()));
        adoption.recovery = dmi_store::SpoolRecovery{};
        adoption.validated = true;
      }
      if (steady_ns() - started >= config_.adoption_slice_ns) break;
      continue;
    }
    if (adopting_->remaining.empty()) {
      finish_adoption();
      continue;
    }
    // The rules of the cycle's uploads, before every round: none without
    // the lease, and none while an uploaded pack is still owed to the
    // catalog -- the dead spool is where the rest are durable.
    bool catalog = false;
    {
      LeaseScope lease(this);
      catalog = writer_.held_lease() != nullptr;
    }
    if (!catalog || !pending_index_.empty()) break;
    bool cut = false;
    if (!upload_adopted_round(deadline_ns, deferred, &cut)) {
      ok = false;  // the failed packs stay in the dead spool
      break;
    }
    if (cut) {
      *cut_short = true;  // the cut packs stay in the dead spool
      break;
    }
    if (steady_ns() - started >= config_.adoption_slice_ns) break;
  }
  return ok;
}

bool CaptureStorageService::scan_siblings() {
  namespace fs = std::filesystem;
  const fs::path own(spool_.root());
  std::vector<std::string> siblings;
  std::vector<fs::path> claim_staging;
  std::error_code ec;
  for (fs::directory_iterator it(own.parent_path(), ec), end;
       !ec && it != end; it.increment(ec)) {
    uint64_t rank = 0;
    std::string incarnation;
    std::error_code type_ec;
    if (it->path() == own || it->is_symlink(type_ec) ||
        !it->is_directory(type_ec)) {
      continue;
    }
    const std::string name = it->path().filename().string();
    if (dmi_store::IsSpoolClaimStagingName(name)) {
      claim_staging.push_back(it->path());
      continue;
    }
    if (!dmi_store::ParseSpoolRankDirectoryName(name, &rank, &incarnation) ||
        blocked_siblings_.count(it->path().string()) != 0) {
      continue;
    }
    siblings.push_back(it->path().string());
  }
  if (ec) {
    record_error("adoption: cannot list " + own.parent_path().string() +
                 ": " + ec.message());
    return false;
  }
  clear_dead_claim_staging(claim_staging);
  std::sort(siblings.begin(), siblings.end());
  uint64_t live = 0;
  for (const std::string& sibling : siblings) {
    // A live owner answers a non-blocking probe at once; TryAdopt would
    // retry for a few milliseconds first, on every recheck.
    if (dmi_store::ReadSpoolOwner(sibling, nullptr)) {
      ++live;
    } else {
      adoption_queue_.push_back(sibling);
    }
  }
  adoption_scan_owed_ = false;
  live_siblings_ = live != 0;
  last_adoption_scan_ns_ = steady_ns();
  std::lock_guard<std::mutex> state(state_mutex_);
  state_.live_siblings = live;
  return true;
}

void CaptureStorageService::begin_adoption(const std::string& directory) {
  auto adoption = std::make_unique<Adoption>();
  adoption->directory = directory;
  std::string error;
  const dmi_store::SpoolStatus locked =
      dmi_store::SpoolOwnerLock::TryAdopt(directory, &adoption->lock, &error);
  if (locked == dmi_store::SpoolStatus::kOwned) {
    // Alive after all (taken since the look): its owner's, and not owed.
    live_siblings_ = true;
    std::lock_guard<std::mutex> state(state_mutex_);
    ++state_.live_siblings;
    return;
  }
  if (locked != dmi_store::SpoolStatus::kOk) {
    // Another adopter drained and removed it meanwhile: nothing is owed.
    if (!std::filesystem::exists(directory)) return;
    // Its lock cannot be taken at all -- another user's lock file, say.
    block_sibling(directory, "cannot lock it: " + error);
    return;
  }
  dmi_store::SpoolConfig config{directory, config_.spool_max_bytes};
  config.owner_lock = dmi_store::OwnerLock::kHeldByCaller;  // adoption->lock
  config.allow_shared_filesystem = config_.spool_allow_shared_filesystem;
  // Its .open files swept and its ready packs listed, none hashed yet:
  // adopt_step validates them, a pack a step.
  if (dmi_store::Spool::Open(config, &adoption->spool, &error) !=
          dmi_store::SpoolStatus::kOk ||
      adoption->spool.BeginRecovery(&adoption->recovery, &error) !=
          dmi_store::SpoolStatus::kOk) {
    block_sibling(directory, "cannot open it: " + error, &adoption->lock);
    return;
  }
  adopting_ = std::move(adoption);
}

bool CaptureStorageService::upload_adopted_round(uint64_t deadline_ns,
                                                 size_t* deferred,
                                                 bool* cut) {
  *cut = false;
  Adoption& adoption = *adopting_;
  // A round is a chunk of the service's own upload path (upload_chunk, then
  // index_or_owe): uploaded, then indexed before the next is uploaded, so
  // at most one chunk (indexer.max_packs) of it is out of the dead spool
  // and not yet in the catalog. It is at most uploader.max_workers packs as
  // well, since the adoption slice is checked between rounds.
  const size_t round = static_cast<size_t>(std::min(
      std::max(1, config_.uploader.max_workers),
      std::max(1, config_.indexer.max_packs)));
  std::vector<dmi_store::StagedPack> entries;
  while (!adoption.remaining.empty() && entries.size() < round) {
    entries.push_back(std::move(adoption.remaining.front()));
    adoption.remaining.pop_front();
  }
  // Through the service's upload client and its Cancellation, so stop()
  // cuts an adoption's transfers, retries and backoff as it does the
  // service's own.
  dmi_store::SpoolUploader uploader(&adoption.spool, &upload_s3_,
                                    config_.uploader);
  uploader.set_cancellation(&upload_cancel_);
  ChunkOutcome sent = upload_chunk(&uploader, entries);
  size_t retryable = 0;
  std::vector<dmi_store::StagedPack> cancelled;
  for (size_t i = 0; i < sent.batch.refs.size(); ++i) {
    if (!sent.batch.refs[i].pack_id.empty()) continue;
    // Still in the dead spool either way. One a cancel cut short goes
    // back to the front, as it was; one a later try could upload is
    // retried by a later cycle; one no try by this service can is not,
    // and blocks the directory once the rest are up.
    const dmi_store::UploadFailure failure =
        i < sent.batch.failures.size() ? sent.batch.failures[i]
                                       : dmi_store::UploadFailure{};
    const std::string what = failure.object_key + ": " + failure.error;
    if (failure.cancelled) {
      cancelled.push_back(entries[i]);
    } else if (failure.retryable) {
      ++retryable;
      adoption.remaining.push_back(entries[i]);
      record_error("adopting dead spool " + adoption.directory +
                   ": upload failed for " + what);
    } else if (adoption.blocked.empty()) {
      adoption.blocked = "it holds a pack this service can never upload, " +
                         what;
    }
  }
  adoption.remaining.insert(adoption.remaining.begin(), cancelled.begin(),
                            cancelled.end());
  {
    std::lock_guard<std::mutex> state(state_mutex_);
    state_.adopted_packs += sent.to_index.size();
  }
  *cut = !cancelled.empty();
  // Uploaded, so gone from the dead spool: indexed now, a cancel of the
  // uploads or not, or owed in pending_index_ like any pack of this
  // service's own. After the requeue above, since a lost lease throws out
  // of here and the packs still in the dead spool must stay in remaining.
  *deferred += index_or_owe(std::move(sent.to_index), true, deadline_ns);
  return retryable == 0;
}

void CaptureStorageService::finish_adoption() {
  std::unique_ptr<Adoption> adoption = std::move(adopting_);
  const std::string directory = adoption->directory;
  if (!adoption->blocked.empty()) {
    // The directory stays, and its lock goes.
    block_sibling(directory, adoption->blocked, &adoption->lock);
    return;
  }
  {
    std::lock_guard<std::mutex> state(state_mutex_);
    ++state_.adopted_spools;
  }
  std::string error;
  if (!adoption->lock.ReleaseAndRemoveIfEmpty(&error)) {
    // Nothing to upload is left, only files that are not its packs (a
    // quarantined one, a spool directory nested in it): the directory
    // stays for someone to look at.
    block_sibling(directory, "it was drained, but still holds files that "
                             "are not its packs (a quarantined pack, or a "
                             "spool directory nested in it)");
  }
}

void CaptureStorageService::block_sibling(const std::string& directory,
                                          const std::string& reason,
                                          dmi_store::SpoolOwnerLock* lock) {
  if (lock != nullptr) {
    // Said in its lock file, while this service still holds it, then let
    // go of: for a person, and for the sinks on the node, which charge a
    // dead directory against their budget only while an adoption can
    // drain it (SpoolConfig::charge_dead_siblings). A later take -- by a
    // process that can adopt it -- rewrites the mark.
    lock->MarkBlocked(reason);
    lock->Release();
  }
  blocked_siblings_.insert(directory);
  record_error("dead spool " + directory + " is left in place, not to be "
               "adopted by this service: " + reason);
  std::lock_guard<std::mutex> state(state_mutex_);
  state_.blocked_siblings.assign(blocked_siblings_.begin(),
                                 blocked_siblings_.end());
}

void CaptureStorageService::clear_dead_claim_staging(
    const std::vector<std::filesystem::path>& staging) {
  namespace fs = std::filesystem;
  // A claim builds its directory's staging copy and renames it into place
  // within milliseconds, so one this old whose lock nobody holds belongs
  // to a claim that died before its rename. Younger ones are left alone: a
  // claim between its mkdir and its flock holds no lock yet, and clearing
  // its copy would fail it. Nothing is owed either way.
  constexpr auto kDeadAfter = std::chrono::seconds(60);
  for (const fs::path& path : staging) {
    std::error_code ec;
    const auto written = fs::last_write_time(path, ec);
    if (ec || fs::file_time_type::clock::now() - written < kDeadAfter) {
      continue;
    }
    dmi_store::SpoolOwnerLock lock;
    std::string error;
    if (dmi_store::SpoolOwnerLock::TryAdopt(path.string(), &lock, &error) ==
        dmi_store::SpoolStatus::kOk) {
      lock.ReleaseAndRemoveIfEmpty(&error);
    }
  }
}

size_t CaptureStorageService::index_bounded(std::vector<PackRefData> refs,
                                            std::vector<PackRefData>* unindexed,
                                            uint64_t deadline_ns) {
  // Batches the indexer can take: at most max_packs, and halved again when the
  // rendered descriptors exceed max_estimated_bytes. The two bounds are
  // independent, so packs that each fit can still overflow together.
  const size_t max_packs =
      static_cast<size_t>(std::max(1, config_.indexer.max_packs));
  // A stack, the next batch at the back. The chunks go on it in reverse, so
  // the pass runs them front to back and its first batch is a full one:
  // past a flush's deadline that batch is the only one, and the remainder
  // chunk can be a single pack. A split pushes its halves the same way.
  std::vector<std::vector<PackRefData>> work;
  for (size_t chunk = (refs.size() + max_packs - 1) / max_packs; chunk-- > 0;) {
    const size_t begin = chunk * max_packs;
    work.emplace_back(refs.begin() + begin,
                      refs.begin() + std::min(refs.size(), begin + max_packs));
  }
  // What is still queued, in the order it would have run.
  const auto drain_queued = [&](uint64_t* count) {
    for (auto queued = work.rbegin(); queued != work.rend(); ++queued) {
      *count += queued->size();
      unindexed->insert(unindexed->end(), queued->begin(), queued->end());
    }
    work.clear();
  };
  // A batch that threw indexed nothing, and neither did anything still queued.
  const auto give_up = [&](std::vector<PackRefData>& failed,
                           const std::string& message) {
    uint64_t count = failed.size();
    unindexed->insert(unindexed->end(), failed.begin(), failed.end());
    drain_queued(&count);
    record_error(message);
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.index_failures += count;
  };
  // Left owed and not counted as failed: a cancel or the deadline cut the
  // pass short.
  size_t deferred = 0;
  const auto defer = [&](std::vector<PackRefData>& cut) {
    uint64_t count = cut.size();
    unindexed->insert(unindexed->end(), cut.begin(), cut.end());
    drain_queued(&count);
    deferred += static_cast<size_t>(count);
  };
  bool first = true;
  while (!work.empty()) {
    // Once the reads are cut -- stop(), or a flush one request timeout
    // past its deadline -- no batch starts, the first included: it could
    // read nothing, and would only send catalog statements (the replay
    // guard) whose answer is thrown away, and which stop() waits for.
    // Past the deadline no batch starts but the first: the pass overruns
    // it by one batch at most, never by the rest of its work, however many
    // batches that is. Catalog statements are never cut, so this is where
    // the pass can stop.
    if (read_cancel_.cancelled() ||
        (!first && deadline_ns != 0 && steady_ns() >= deadline_ns)) {
      std::vector<PackRefData> none;
      defer(none);
      break;
    }
    first = false;
    std::vector<PackRefData> batch = std::move(work.back());
    work.pop_back();
    IndexResultData result;
    try {
      // The lease lock for the catalog phases only. Between them the pass
      // reads its packs from the object store, where a stalled GET is bound
      // by the S3 client's own timeouts and read_cancel_, and must not keep
      // the lease thread from renewing meanwhile. Nothing here stamps a
      // renewal: the schedule follows the lease itself (renew_lease_if_due),
      // which only a claim that confirmed moves -- a pass that published
      // nothing, over packs already committed, renewed nothing.
      IndexPlan plan;
      {
        LeaseScope lease(this);
        plan = indexer_.plan(batch);
      }
      indexer_.read(&plan);
      LeaseScope lease(this);
      result = indexer_.commit(&plan);
    } catch (const StoreUnavailableError& exc) {
      // The object store did not answer for a pack. Every later read of
      // the pass would cost the client's timeouts and fail alike, so the
      // pass ends here, like a batch that threw: this batch and the rest
      // stay owed, and nothing counts against the packs themselves, since
      // an outage is no fault of theirs. A read a cancel cut -- stop(), or
      // a flush out of time -- is not a failure at all.
      if (exc.cancelled()) {
        defer(batch);
        return deferred;
      }
      give_up(batch, std::string("index failed: ") + exc.what());
      return deferred;
    } catch (const CatalogError& exc) {
      if (exc.kind() == CatalogError::Kind::kBatchTooLarge && batch.size() > 1) {
        const size_t middle = batch.size() / 2;
        work.emplace_back(batch.begin() + middle, batch.end());
        work.emplace_back(batch.begin(), batch.begin() + middle);
        std::lock_guard<std::mutex> lock(state_mutex_);
        ++state_.batch_splits;
        continue;
      }
      if (exc.kind() == CatalogError::Kind::kBatchTooLarge) {
        // One pack alone exceeds the budget, so no retry can ever index it.
        // Set it aside and keep draining: parking the queue behind it would
        // stop every later pack reaching the catalog.
        reject(batch.front(), exc.what());
        continue;
      }
      const bool lease_lost = is_lease_refusal(exc);
      give_up(batch, std::string("index failed: ") + exc.what());
      if (lease_lost) throw;
      return deferred;
    } catch (const std::exception& exc) {
      give_up(batch, std::string("index failed: ") + exc.what());
      return deferred;
    }
    // A failure the indexer reports for one pack is that pack's own (it was
    // read and refused), unlike a batch that threw, which an outage explains.
    // So only these count towards setting it aside.
    std::map<std::string, std::string> failed;
    for (const IndexFailureData& failure : result.failures) {
      failed[failure.pack_id] = failure.message;
      record_error("index failed for " + failure.object_key + ": " +
                   failure.message);
    }
    for (const PackRefData& ref : batch) {
      const auto it = failed.find(ref.pack_id);
      if (it == failed.end()) {
        index_attempts_.erase(ref.pack_id);
      } else if (++index_attempts_[ref.pack_id] >= config_.max_index_attempts) {
        reject(ref, "failed " + std::to_string(config_.max_index_attempts) +
                        " times: " + it->second);
      } else {
        unindexed->push_back(ref);
      }
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.indexed_packs += result.indexed_packs;
    state_.indexed_rows += result.indexed_rows;
    state_.index_failures += result.failed_packs;
  }
  return deferred;
}

void CaptureStorageService::reject(const PackRefData& ref,
                                   const std::string& reason) {
  index_attempts_.erase(ref.pack_id);
  rejected_unreported_.push_back(ref.object_key + ": " + reason.substr(0, 300));
  record_error("pack set aside, cannot be indexed: " + ref.object_key + ": " +
               reason);
  std::lock_guard<std::mutex> lock(state_mutex_);
  ++state_.rejected_packs;
  ++state_.index_failures;
}

bool CaptureStorageService::reconcile() {
  // List every pack under the prefix, ask the catalog which it already
  // committed, and HEAD only the rest -- so a steady-state pass over a large
  // bucket costs listing pages and one catalog query per page, not a HEAD per
  // object. stop() ends it between requests: what it has not reached is
  // still in the bucket, for the next pass.
  std::string token;
  uint64_t found = 0;
  uint64_t skipped = 0;
  uint64_t head_errors = 0;
  do {
    if (upload_cancel_.cancelled()) return false;
    dmi_store::ListResult page;
    std::string error;
    if (!s3_.ListObjects(config_.reconcile_prefix, "", 1000, token, &page,
                         &error)) {
      if (read_cancel_.cancelled()) return false;  // stop() cut it
      throw std::runtime_error("reconcile: listing failed: " + error);
    }
    token = page.truncated ? page.next_token : "";

    std::vector<PackIdentity> identities;
    std::vector<const dmi_store::ListedObject*> packs;
    for (const dmi_store::ListedObject& object : page.objects) {
      const std::string pack_id = pack_id_of(object.key);
      if (pack_id.empty()) continue;  // not a pack key at all
      if (!is_pack_id(pack_id)) {
        ++skipped;
        continue;
      }
      identities.emplace_back(config_.uploader.store_id, pack_id);
      packs.push_back(&object);
    }
    if (packs.empty()) continue;
    // A listing answered just as stop() came: nothing after it can be read.
    if (upload_cancel_.cancelled()) return false;
    std::set<PackIdentity> committed;
    {
      LeaseScope lease(this);
      committed = writer_.committed_pack_ids(identities);
    }

    std::vector<PackRefData> missing;
    for (size_t i = 0; i < packs.size(); ++i) {
      if (committed.count(identities[i]) != 0) continue;
      if (upload_cancel_.cancelled()) return false;
      const dmi_store::ListedObject& object = *packs[i];
      std::string head_error;
      const dmi_store::ObjectHead head = s3_.HeadObject(object.key, &head_error);
      if (!head_error.empty() && read_cancel_.cancelled()) return false;
      if (!head_error.empty()) {
        // Unread, not foreign: the object may well be a pack, so the pass
        // reports it rather than counting it as skipped. The next pass
        // retries it.
        ++head_errors;
        record_error("reconcile: HEAD failed for " + object.key + ": " +
                     head_error);
        continue;
      }
      const auto meta = [&head](const char* name) -> std::string {
        const auto it = head.metadata.find(name);
        return it == head.metadata.end() ? "" : it->second;
      };
      // The uploader's metadata, validated as the Python oracle's inspect()
      // does: a foreign object in the bucket is skipped, never indexed.
      uint64_t records = 0;
      try {
        records = std::stoull(meta("dmi-record-count"));
      } catch (...) {
        records = 0;
      }
      const std::string checksum = meta("dmi-sha256");
      if (!head.found || meta("dmi-format") != "dmi-pack-v1" ||
          meta("dmi-pack-id") != identities[i].second || !is_hex64(checksum) ||
          records < 1 || records > 1'000'000) {
        ++skipped;
        continue;
      }
      missing.push_back({identities[i].second, config_.uploader.store_id,
                         object.key, head.size, checksum, records});
    }
    found += missing.size();
    // A pack that fails here is still uncommitted in the bucket, so the next
    // pass retries it; it is not added to the flush boundary.
    std::vector<PackRefData> unindexed;
    if (!missing.empty()) index_bounded(std::move(missing), &unindexed);
  } while (!token.empty());

  std::lock_guard<std::mutex> lock(state_mutex_);
  ++state_.reconcile_passes;
  state_.reconciled_packs += found;
  state_.reconcile_skipped_objects += skipped;
  state_.reconcile_head_errors += head_errors;
  return true;
}

void CaptureStorageService::keep_lease() {
  // The lease renews only inside a publish, and the cycle loop backs off up
  // to max_backoff_ns while the object store is down -- past the lease TTL.
  // Renewing from the cycle let the lease lapse during an outage and a rival
  // take the catalog. This thread renews on its own schedule instead, and
  // takes a fresh lease once a lost one can be replaced.
  uint64_t wait_ns = lease_tick_ns(config_.writer.lease_ttl_ns);
  while (true) {
    {
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_.wait_for(lock, std::chrono::nanoseconds(wait_ns),
                     [this] { return lease_stop_requested_; });
      if (lease_stop_requested_) return;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (failure_) return;
    }
    LeaseScope lease(this);  // publishes the lease state when done
    // Wakes when the renewal falls due (a tick at most), not on a fixed
    // tick: after a pass lets go of the lease lock, the renewal it leaves
    // due has to start then, not up to a tick later.
    const auto next_wake = [this] {
      const uint64_t ttl = config_.writer.lease_ttl_ns;
      const uint64_t tick = lease_tick_ns(ttl);
      const uint64_t sent = writer_.lease_sent_ns();
      if (sent == 0) return tick;
      const uint64_t due = sent + ttl / 3;
      const uint64_t now = steady_ns();
      return std::clamp<uint64_t>(due > now ? due - now : 0, 1'000'000ull,
                                  tick);
    };
    if (writer_.held_lease() == nullptr) {
      ensure_publisher_lease();
      wait_ns = next_wake();
      continue;
    }
    wait_ns = lease_tick_ns(config_.writer.lease_ttl_ns);
    try {
      renew_lease_if_due();
      wait_ns = next_wake();
    } catch (const CatalogError& exc) {
      if (is_lease_refusal(exc)) {
        // The coordinator dropped the lease: a live foreign head refused
        // the renewal claim, or the claim was contested.
        lease_held_elsewhere(exc);
      } else {
        record_error(std::string("lease renewal failed: ") + exc.what());
      }
    } catch (const std::exception& exc) {
      // An unknown outcome: the writer quarantined itself and dropped the
      // lease. ensure_publisher_lease() replaces it after the window.
      record_error(std::string("lease renewal failed: ") + exc.what());
      note_lease_failure(exc);
    }
  }
}

void CaptureStorageService::renew_lease_if_due() {
  // Due a third of the TTL after the claim that stamped the lease row was
  // sent -- the last renewal, or the one a publish made before its fenced
  // statements -- as the writer records it, so nothing but a confirmed
  // claim moves the schedule. The lease thread wakes when it falls due, and
  // at least every sixth of the TTL, so with the lease lock free a renewal
  // starts within half the TTL of that send (in practice at a third), and
  // has until the lease deadline to be answered: the renewal window the
  // constructor checks (lease_coordinator.h). While a stretch holds the
  // lease lock, this runs before every request it sends (LeaseScope's
  // before_request hook), reads included, so the lease is never more than
  // a third of the TTL plus one request old when a renewal starts -- not
  // several requests' worth, which a slow but healthy catalog could not fit
  // a renewal after. A failed renewal costs the lease at once: a refusal
  // drops it in the coordinator, and any other error quarantines the writer
  // (renew_for_publish).
  const uint64_t ttl = config_.writer.lease_ttl_ns;
  const uint64_t sent = writer_.lease_sent_ns();
  if (ttl == 0 || sent == 0 || steady_ns() - sent < ttl / 3) return;
  writer_.renew_lease();
  held_elsewhere_since_ns_ = 0;
  std::lock_guard<std::mutex> lock(state_mutex_);
  ++state_.lease_renewals;
}

void CaptureStorageService::keep_lease_in_pass() {
  // What it throws fails the request it ran for, and so the stretch.
  try {
    renew_lease_if_due();
  } catch (const CatalogError&) {
    throw;  // a refusal: the pass fails with the lease lost
  } catch (const std::exception& exc) {
    record_error(std::string("lease renewal failed: ") + exc.what());
    note_lease_failure(exc);
    throw;
  }
}

void CaptureStorageService::abandon_lease_if_expired() {
  const uint64_t deadline = writer_.lease_deadline_ns();
  if (deadline == 0 || steady_ns() < deadline) return;
  writer_.abandon_lease();
  const std::string message =
      std::string("publisher lease abandoned: it was not renewed by ") +
      kLeaseDeadlineBound;
  record_error(message);
  count_lease_timeout(message);
}

void CaptureStorageService::note_lease_failure(const std::exception& failure) {
  const auto* error = dynamic_cast<const ClickHouseError*>(&failure);
  if (error != nullptr && error->timed_out()) count_lease_timeout(error->what());
}

void CaptureStorageService::count_lease_timeout(const std::string& latest) {
  ++lease_timeouts_;
  ++lease_timeouts_counted_;
  std::string message;
  if (lease_timeouts_ >= 3) {
    // The knobs first: last_error keeps only its first 512 bytes.
    const uint64_t claim_bound = writer_.leases().claim_bound_ns();
    message =
        std::to_string(lease_timeouts_) +
        " timeouts have cost the publisher lease since one was last held "
        "for 2 x lease_ttl_s. Requests made under the lease have until its "
        "deadline, lease_ttl_s (" +
        seconds_text(config_.writer.lease_ttl_ns) + ") less clock_skew_s (" +
        seconds_text(config_.writer.clock_skew_ns) + ") and a " +
        seconds_text(kLeaseDeadlineMarginNs) +
        " margin after the claim that stamped its row was sent; each request "
        "of a claim made without one has min(clickhouse_request_timeout_s, "
        "lease_ttl_s / 3) = " + seconds_text(claim_bound) +
        ". A catalog this slow needs a longer lease_ttl_s. The latest: " +
        latest;
    record_error(message);
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.lease_timeouts = lease_timeouts_;
  state_.lease_timeout_error = message.substr(0, 1024);
}

void CaptureStorageService::track_stable_lease() {
  // Not on every claim or renewal that succeeds: against a catalog too slow
  // to keep a lease, each claim goes through and the lease is then lost to
  // a timeout, and a count cleared by the claim never got past one. A lease
  // held for 2 x TTL -- renewed through several deadlines -- is one the
  // catalog can keep. A lease_id names one holding: every claim after a
  // loss mints a fresh one, and renewals keep it.
  const PublisherLease* held = writer_.held_lease();
  if (held == nullptr) {
    stable_lease_id_.clear();
    return;
  }
  const uint64_t now = steady_ns();
  if (held->lease_id != stable_lease_id_) {
    stable_lease_id_ = held->lease_id;
    stable_since_ns_ = now;
    return;
  }
  if (lease_timeouts_ == 0 ||
      now - stable_since_ns_ < 2 * config_.writer.lease_ttl_ns) {
    return;
  }
  lease_timeouts_ = 0;
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.lease_timeouts = 0;
  state_.lease_timeout_error.clear();
}

void CaptureStorageService::acquire_lease_at_start() {
  // A crashed predecessor's lease stays live for up to its TTL. Waiting it
  // out here turns a restart inside that window into a short delay instead
  // of a failed start; a live publisher keeps renewing, so the wait ends in
  // the same refusal as before, naming the holder.
  uint64_t deadline = steady_ns() + config_.start_lease_wait_ns;
  const uint64_t poll = std::clamp<uint64_t>(
      config_.writer.lease_ttl_ns / 10, 50'000'000ull, 500'000'000ull);
  // Whether the wait has already been stretched for a quarantine this
  // start()'s own claim left (below): once only.
  bool stretched = false;
  while (true) {
    try {
      writer_.acquire_lease(config_.holder);
      return;
    } catch (const CatalogError& exc) {
      if (!is_lease_refusal(exc) || config_.start_lease_wait_ns == 0) throw;
      const uint64_t now = steady_ns();
      if (now >= deadline) {
        throw CatalogError(
            exc.kind(),
            "storage service: waited " +
                std::to_string(config_.start_lease_wait_ns / 1'000'000) +
                " ms for the publisher lease and it is still held: " +
                exc.what());
      }
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(std::min(poll, deadline - now)));
    } catch (const ClickHouseError& exc) {
      // A claim that timed out -- a cold catalog's first reads can outlast
      // the claim bound, and so can a slow INSERT -- is retried within the
      // same wait. One that wrote nothing (its head read timed out) goes
      // again at once. One whose INSERT may have landed quarantined the
      // writer for a TTL: its row, if it landed, is live that long, as a
      // crashed predecessor's is. The wait is sized for one of those, and a
      // claim that used its bound has spent enough of it that the
      // quarantine always ended past it at the defaults; so the quarantine
      // is waited out even then, once, with time for a claim after it (one
      // the late row refuses until it expires, if it landed). Any other
      // error fails start() as before.
      record_error(std::string("publisher lease claim at start failed: ") +
                   exc.what());
      note_lease_failure(exc);
      if (!exc.timed_out() || config_.start_lease_wait_ns == 0) throw;
      uint64_t resume = steady_ns();
      uint64_t until = 0;
      if (writer_.quarantined(&until)) {
        resume = std::max(resume, until);
        // A claim's three requests, each within the claim bound, and a poll
        // for the late row to expire.
        const uint64_t claim_ns =
            3 * writer_.leases().claim_bound_ns() + poll;
        if (!stretched && until + claim_ns > deadline) {
          stretched = true;
          deadline = until + claim_ns;
        }
      }
      if (resume >= deadline) {
        throw ClickHouseError(
            "storage service: the publisher lease claim timed out, and "
            "start_lease_wait_ns (" +
                std::to_string(config_.start_lease_wait_ns / 1'000'000) +
                " ms) leaves no time to try again: " + exc.what(),
            true, exc.sent());
      }
      const uint64_t now = steady_ns();
      if (resume > now) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(resume - now));
      }
    }
  }
}

bool CaptureStorageService::ensure_publisher_lease() {
  if (writer_.held_lease() != nullptr) return true;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (failure_) return false;
  }
  if (writer_.quarantined()) {
    // The unknown-outcome statement may still be running under the lease
    // it dropped; no claim, not even a fresh lease_id, until the window
    // (one TTL) has passed and that lease's row has expired with it.
    publish_lease_state();
    return false;
  }
  const uint64_t now = steady_ns();
  if (now < next_claim_ns_) return false;
  try {
    // A fresh lease_id: the coordinator mints one because the quarantine
    // or refusal already dropped the old lease.
    writer_.acquire_lease(config_.holder);
  } catch (const CatalogError& exc) {
    if (is_lease_refusal(exc)) {
      lease_held_elsewhere(exc);
    } else {
      record_error(std::string("publisher lease acquisition failed: ") +
                   exc.what());
    }
    publish_lease_state();
    return false;
  } catch (const std::exception& exc) {
    // An unknown outcome if the claim INSERT may have reached the server:
    // the writer quarantined itself for a TTL. A claim that wrote nothing
    // (its head read timed out or could not connect) is not quarantined,
    // and goes again a tick from now, so that flush()'s fast cycles do not
    // hammer a catalog that cannot answer.
    record_error(std::string("publisher lease acquisition failed: ") +
                 exc.what());
    note_lease_failure(exc);
    if (writer_.quarantined()) {
      // A claim sends its INSERT only once its head read found no live
      // holder, so whoever refused this service before had left by then:
      // the run of refusals ends here, as it does at a refusal by our own
      // late row (lease_held_elsewhere). Otherwise a rival that left, this
      // claim timing out and a second rival refusing the claims after the
      // quarantine add up to 2 x TTL, and latch the service over a catalog
      // nobody held in between.
      held_elsewhere_since_ns_ = 0;
    } else {
      next_claim_ns_ = steady_ns() + lease_tick_ns(config_.writer.lease_ttl_ns);
    }
    publish_lease_state();
    return false;
  }
  held_elsewhere_since_ns_ = 0;
  next_claim_ns_ = 0;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    ++state_.lease_reacquisitions;
  }
  publish_lease_state();
  {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    kick_ = true;
  }
  wake_.notify_all();
  return true;
}

void CaptureStorageService::lease_held_elsewhere(const CatalogError& refusal) {
  const uint64_t now = steady_ns();
  const uint64_t ttl = config_.writer.lease_ttl_ns;
  next_claim_ns_ = now + lease_tick_ns(ttl);
  if (writer_.refused_by_own_claims()) {
    // Refused by this process's own claim row: one whose request gave up
    // but which the server still ran -- late, past the cap on the lease
    // INSERT (lease_coordinator.cpp says when that can happen). It expires
    // one TTL after it landed and is no rival, so it restarts the refusal
    // clock rather than counting towards the latch. Sending that claim at
    // all meant the head read found no live rival, so a rival's earlier
    // refusals ended there too. Without this, refusals by a rival that has
    // since left, then by our own late row, could add up to 2 x TTL and
    // latch the service against itself.
    held_elsewhere_since_ns_ = 0;
    record_error(std::string("publisher lease refused by this service's "
                             "own earlier claim, which landed after its "
                             "request gave up; retrying: ") +
                 refusal.what());
    return;
  }
  if (held_elsewhere_since_ns_ == 0) held_elsewhere_since_ns_ = now;
  // Our own dropped row is dead within one TTL of the loss, and a handover
  // (a rival that stops, releasing with a tombstone) ends sooner still. A
  // refusal that has lasted 2 x TTL is a publisher that means to stay.
  if (now - held_elsewhere_since_ns_ >= 2 * ttl) {
    latch_failure(std::make_exception_ptr(refusal),
                  std::string("publisher lease held by another publisher "
                              "for over 2 x TTL: ") + refusal.what());
  } else {
    record_error(std::string("publisher lease held elsewhere: ") +
                 refusal.what());
  }
}

void CaptureStorageService::publish_lease_state() {
  track_stable_lease();
  uint64_t until = 0;
  const char* lease_state = "reacquiring";
  if (writer_.held_lease() != nullptr) {
    lease_state = "held";
  } else if (writer_.quarantined(&until)) {
    lease_state = "quarantined";
  } else {
    until = 0;
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (state_.failed) return;
  state_.lease_state = lease_state;
  state_.quarantined_until_ns = until;
}

void CaptureStorageService::record_error(const std::string& message) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.last_error = message.substr(0, 512);
}

void CaptureStorageService::latch_failure(std::exception_ptr failure,
                                          const std::string& message) {
  std::string line = message.substr(0, 512);
  std::replace(line.begin(), line.end(), '\n', ' ');
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (failure_) return;
    failure_ = std::move(failure);
    state_.last_error = line;
    state_.failed = true;
    state_.running = false;
    state_.lease_state = "failed";
    state_.quarantined_until_ns = 0;
  }
  // Once, and on one line: the loop and the lease thread both stop here,
  // and nothing else will say why indexing did.
  std::fprintf(stderr, "dmi capture storage: indexing stopped: %s\n",
               line.c_str());
  std::fflush(stderr);
  // Woken now, not after its wait (up to max_backoff_ns in an outage), so
  // the loop lets go of a sibling it was adopting at once.
  {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    kick_ = true;
  }
  wake_.notify_all();
}

}  // namespace dmi_catalog
