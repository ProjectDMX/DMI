#include "catalog/storage_service.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
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

CaptureStorageService::CaptureStorageService(StorageServiceConfig config)
    : config_(std::move(config)),
      s3_(config_.s3),
      clickhouse_(std::make_shared<const ClickHouseClient>(config_.clickhouse)),
      writer_(clickhouse_, config_.writer),
      indexer_(&s3_, &writer_, config_.indexer) {
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
  if (dmi_store::Spool::Open({config_.spool_root, config_.spool_max_bytes},
                             &spool_, &error) != dmi_store::SpoolStatus::kOk) {
    throw std::runtime_error("storage service: cannot open spool: " + error);
  }
  uploader_ = std::make_unique<dmi_store::SpoolUploader>(&spool_, &s3_,
                                                          config_.uploader);
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
    std::lock_guard<std::mutex> lock(wake_mutex_);
    kick_ = false;
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
  // After the lease, never before, so a second process pointed at this spool
  // usually learns that the catalog is held before it can delete a live
  // sink's .open files. Only usually: a holder that is quarantined has let
  // its row lapse, and a second process can take the lease in that gap. The
  // spool itself is not locked; one process per spool is the caller's job
  // until the spool gets an owner lock.
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
  wake_.notify_all();
  if (thread_.joinable()) thread_.join();
  // Only after the loop: its last cycle may still be indexing, and the
  // lease has to keep renewing until that is done.
  stop_lease_thread();
  std::lock_guard<std::timed_mutex> cycle(cycle_mutex_);
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
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::duration<double>(timeout_s));
  while (true) {
    {
      // A cycle in flight -- the loop's, stuck on a slow catalog -- must not
      // hold this call past its deadline.
      std::unique_lock<std::timed_mutex> cycle(cycle_mutex_, std::defer_lock);
      if (!cycle.try_lock_until(deadline)) return false;
      if (!started_) throw std::logic_error("storage service: not started");
      rethrow_if_failed();
      const bool drained = run_cycle().drained;
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
  while (true) {
    {
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_.wait_for(lock, std::chrono::nanoseconds(wait_ns),
                     [this] { return stop_requested_ || kick_; });
      if (stop_requested_) return;
      kick_ = false;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (failure_) return;  // another publisher holds the catalog
    }
    std::lock_guard<std::timed_mutex> cycle(cycle_mutex_);
    run_cycle();
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

CaptureStorageService::CycleOutcome CaptureStorageService::run_cycle() {
  CycleOutcome outcome;
  // The catalog phase needs the lease. Without one -- quarantined after an
  // unknown outcome, or refused by another holder -- the cycle uploads
  // nothing either: whatever it uploaded it could only owe, in memory.
  bool catalog = false;
  {
    LeaseScope lease(this);
    catalog = ensure_publisher_lease();
  }
  // Indexes refs, keeping whatever does not index owed: it is already gone
  // from the spool, so pending_index_ is the only record of it in-process.
  const auto index_or_owe = [this, catalog](std::vector<PackRefData> refs) {
    if (!catalog) {
      pending_index_.insert(pending_index_.end(), refs.begin(), refs.end());
      return;
    }
    std::vector<PackRefData> unindexed;
    try {
      if (!refs.empty()) index_bounded(std::move(refs), &unindexed);
    } catch (...) {
      pending_index_.insert(pending_index_.end(), unindexed.begin(),
                            unindexed.end());
      throw;
    }
    pending_index_.insert(pending_index_.end(), unindexed.begin(),
                          unindexed.end());
  };
  try {
    // 1. Retry what earlier cycles uploaded but could not index. While any
    //    of it is still owed, the catalog is down or refusing: upload
    //    nothing new, so new packs stay in the durable spool rather than
    //    joining a list that only this process remembers.
    if (catalog && !pending_index_.empty()) {
      std::vector<PackRefData> owed;
      owed.swap(pending_index_);
      index_or_owe(std::move(owed));
    }

    // 2. Upload everything the sink has staged -- but only with the lease
    //    and nothing owed. Without the lease an uploaded pack could only be
    //    owed, and pending_index_ dies with the process: with
    //    reconcile_on_start off, a crash would leave it in the bucket and
    //    never in the catalog. Left in the spool it survives the crash.
    dmi_store::UploadBatchResult batch;
    if (catalog && pending_index_.empty()) {
      batch = uploader_->UploadPending(-1);
    }
    std::vector<PackRefData> to_index;
    uint64_t uploaded_packs = 0;
    uint64_t uploaded_bytes = 0;
    size_t upload_failures = 0;
    for (size_t i = 0; i < batch.refs.size(); ++i) {
      const dmi_store::PackRef& ref = batch.refs[i];
      if (!ref.pack_id.empty()) {
        to_index.push_back({ref.pack_id, ref.store_id, ref.object_key,
                            ref.object_bytes, ref.checksum, ref.record_count});
        ++uploaded_packs;
        uploaded_bytes += ref.object_bytes;
      } else {
        // A failed upload stays in the spool, so the next cycle retries it.
        ++upload_failures;
        if (i < batch.failures.size()) {
          record_error("upload failed for " + batch.failures[i].object_key +
                       ": " + batch.failures[i].error);
        }
      }
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state_.uploaded_packs += uploaded_packs;
      state_.uploaded_bytes += uploaded_bytes;
      state_.upload_failures += upload_failures;
    }

    // 3. Index them.
    index_or_owe(std::move(to_index));

    // 4. Reconcile on its interval, or when the pass at start() lost the
    //    lease before it finished. The lease thread keeps the lease alive.
    if (catalog &&
        (reconcile_owed_ ||
         (config_.reconcile_interval_ns > 0 &&
          steady_ns() - last_reconcile_ns_ >= config_.reconcile_interval_ns))) {
      reconcile();
      reconcile_owed_ = false;
      last_reconcile_ns_ = steady_ns();
    }

    // Drained: nothing failed to upload, every uploaded pack is in the
    // catalog, and nothing is pending. A batch that uploaded everything it
    // listed is drained as far as flush() is concerned -- its listing came
    // after the sink's flush -- so the spool is re-listed only when the batch
    // was empty, which UploadPending also returns when its listing FAILED.
    // A cycle that already failed is not drained whatever the spool holds,
    // so it skips the listing: an empty spool lists for free, but a backlog
    // would be re-hashed on every cycle of an outage.
    // Without the lease nothing can be confirmed in the catalog, so the
    // cycle is not drained, and it counts towards the backoff.
    outcome.failed =
        !catalog || upload_failures != 0 || !pending_index_.empty();
    bool nothing_pending = !batch.refs.empty();
    if (batch.refs.empty() && !outcome.failed) {
      std::vector<dmi_store::StagedPack> pending;
      std::string error;
      const bool listed =
          spool_.ListPending(&pending, &error) == dmi_store::SpoolStatus::kOk;
      if (!listed) {
        record_error("spool listing failed: " + error);
        outcome.failed = true;
      }
      nothing_pending = listed && pending.empty();
    }
    outcome.drained = nothing_pending && !outcome.failed;
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
  }
  failure_streak_ = outcome.failed ? std::min(failure_streak_ + 1, 32) : 0;
  return outcome;
}

void CaptureStorageService::index_bounded(std::vector<PackRefData> refs,
                                          std::vector<PackRefData>* unindexed) {
  // Batches the indexer can take: at most max_packs, and halved again when the
  // rendered descriptors exceed max_estimated_bytes. The two bounds are
  // independent, so packs that each fit can still overflow together.
  const size_t max_packs =
      static_cast<size_t>(std::max(1, config_.indexer.max_packs));
  std::vector<std::vector<PackRefData>> work;
  for (size_t i = 0; i < refs.size(); i += max_packs) {
    work.emplace_back(refs.begin() + i,
                      refs.begin() + std::min(refs.size(), i + max_packs));
  }
  // A batch that threw indexed nothing, and neither did anything still queued.
  const auto give_up = [&](std::vector<PackRefData>& failed,
                           const std::string& message) {
    uint64_t count = failed.size();
    unindexed->insert(unindexed->end(), failed.begin(), failed.end());
    for (std::vector<PackRefData>& queued : work) {
      count += queued.size();
      unindexed->insert(unindexed->end(), queued.begin(), queued.end());
    }
    work.clear();
    record_error(message);
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.index_failures += count;
  };
  while (!work.empty()) {
    std::vector<PackRefData> batch = std::move(work.back());
    work.pop_back();
    IndexResultData result;
    try {
      // The lease lock for the catalog phases only. Between them the pass
      // reads its packs from the object store, where a stalled GET is bound
      // only by the S3 client's own timeouts, and must not keep the lease
      // thread from renewing meanwhile. Nothing here stamps a renewal: the
      // schedule follows the lease itself (renew_lease_if_due), which only
      // a claim that confirmed moves -- a pass that published nothing, over
      // packs already committed, renewed nothing.
      IndexPlan plan;
      {
        LeaseScope lease(this);
        plan = indexer_.plan(batch);
      }
      indexer_.read(&plan);
      LeaseScope lease(this);
      result = indexer_.commit(&plan);
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
      return;
    } catch (const std::exception& exc) {
      give_up(batch, std::string("index failed: ") + exc.what());
      return;
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

void CaptureStorageService::reconcile() {
  // List every pack under the prefix, ask the catalog which it already
  // committed, and HEAD only the rest -- so a steady-state pass over a large
  // bucket costs listing pages and one catalog query per page, not a HEAD per
  // object.
  std::string token;
  uint64_t found = 0;
  uint64_t skipped = 0;
  uint64_t head_errors = 0;
  do {
    dmi_store::ListResult page;
    std::string error;
    if (!s3_.ListObjects(config_.reconcile_prefix, "", 1000, token, &page,
                         &error)) {
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
    std::set<PackIdentity> committed;
    {
      LeaseScope lease(this);
      committed = writer_.committed_pack_ids(identities);
    }

    std::vector<PackRefData> missing;
    for (size_t i = 0; i < packs.size(); ++i) {
      if (committed.count(identities[i]) != 0) continue;
      const dmi_store::ListedObject& object = *packs[i];
      std::string head_error;
      const dmi_store::ObjectHead head = s3_.HeadObject(object.key, &head_error);
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
    if (!writer_.quarantined()) {
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
}

}  // namespace dmi_catalog
