#include "pack_sink.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
// std::runtime_error is thrown below. g++ 11 on 20.04 happens to reach it
// through another header, so the omission is invisible on a dev box; the
// newer libstdc++ on the CI image does not, and this file failed to
// compile there the moment a job actually asked for the sink driver.
#include <stdexcept>

#include "object_key.h"

namespace dmi_sink {

namespace {

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double NowS() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string NewPackId() {
  // UUID v4, canonical lowercase form (matches PackBuilder::ParseUuid).
  std::random_device rd;
  uint64_t hi = (static_cast<uint64_t>(rd()) << 32) | rd();
  uint64_t lo = (static_cast<uint64_t>(rd()) << 32) | rd();
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
  char buf[37];
  std::snprintf(buf, sizeof(buf),
                "%08x-%04x-%04x-%04x-%012llx",
                static_cast<unsigned>(hi >> 32),
                static_cast<unsigned>((hi >> 16) & 0xFFFF),
                static_cast<unsigned>(hi & 0xFFFF),
                static_cast<unsigned>(lo >> 48),
                static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return buf;
}

void CountFlush(SinkSnapshot* counters, FlushReason reason) {
  switch (reason) {
    case FlushReason::kSize: ++counters->flush_size; break;
    case FlushReason::kRecords: ++counters->flush_records; break;
    case FlushReason::kLinger: ++counters->flush_linger; break;
    case FlushReason::kSession: ++counters->flush_session; break;
    case FlushReason::kManual: ++counters->flush_manual; break;
    case FlushReason::kShutdown: ++counters->flush_shutdown; break;
  }
}

}  // namespace

PackSink::PackSink(SinkConfig config) : config_(std::move(config)) {
  if (config_.num_workers <= 0) config_.num_workers = 1;
}

PackSink::~PackSink() {
  if (thread_started_) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
    for (auto& cv : worker_cv_) cv.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    // Stage queues close only after all workers finished forwarding —
    // otherwise a stager could exit with a pack still inbound.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::fill(stage_closed_.begin(), stage_closed_.end(), true);
    }
    cv_.notify_all();
    for (auto& cv : stage_cv_) cv.notify_all();
    for (auto& stager : stagers_) {
      if (stager.joinable()) stager.join();
    }
  }
}

std::string PackSink::Start(std::string* spool_error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (thread_started_) return "sink has already been started";
  dmi_store::SpoolConfig spool_config;
  spool_config.root = config_.spool_root;
  spool_config.max_bytes = config_.spool_max_bytes;
  std::string error;
  const dmi_store::SpoolStatus st =
      dmi_store::Spool::Open(spool_config, &spool_, &error);
  if (st != dmi_store::SpoolStatus::kOk) {
    if (spool_error) *spool_error = error;
    return "cannot open spool: " + error;
  }
  thread_started_ = true;
  const size_t n = static_cast<size_t>(config_.num_workers);
  queues_.resize(n);
  assemblers_.resize(n);
  stage_queues_.resize(n);
  stage_closed_.assign(n, false);
  for (size_t i = 0; i < n; ++i) {
    worker_cv_.emplace_back();
    stage_cv_.emplace_back();
  }
  for (size_t i = 0; i < n; ++i) {
    workers_.emplace_back(&PackSink::Run, this, i);
    stagers_.emplace_back(&PackSink::RunStager, this, i);
  }
  return "";
}

bool PackSink::IsRunning() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return thread_started_ && !closed_ && latched_error_.empty();
}

size_t PackSink::RouteWorker(const std::string& tenant,
                             const std::string& session,
                             uint64_t rank) const {
  // FNV-1a over tenant\0session\0rank, then a splitmix64-style finalizer.
  // The finalizer is load-bearing, not decoration: FNV's low bits are weak
  // for similar inputs (sessions "0".."7" all landed on worker 0 at N=2,4 —
  // measured, and the flat "scaling" curve below it), and `% workers` reads
  // exactly those bits. fmix64 avalanches them.
  uint64_t hash = 14695981039346656037ULL;
  auto mix = [&](const std::string& s) {
    for (unsigned char c : s) {
      hash ^= c;
      hash *= 1099511628211ULL;
    }
    // Domain separator between the two strings (NUL cannot appear in them:
    // the reference validates text fields, and NUL would fail UTF-8 text
    // rules downstream — but the separator costs nothing regardless).
    hash ^= 0xFF;
    hash *= 1099511628211ULL;
  };
  mix(tenant);
  mix(session);
  for (int i = 0; i < 8; ++i) {
    hash ^= static_cast<unsigned char>((rank >> (8 * i)) & 0xFF);
    hash *= 1099511628211ULL;
  }
  uint64_t z = hash;
  z ^= z >> 33;
  z *= 0xff51afd7ed558ccdULL;
  z ^= z >> 33;
  z *= 0xc4ceb9fe1a85ec53ULL;
  z ^= z >> 33;
  return static_cast<size_t>(z % static_cast<uint64_t>(queues_.size()));
}

Admission PackSink::Submit(dmi_pack::RecordMetadata metadata,
                           const uint8_t* payload, size_t n) {
  size_t worker = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_started_) return Admission::kClosed;
    ++counters_.submitted_records;
    if (!latched_error_.empty()) return Admission::kClosed;
    worker = RouteWorker(metadata.tenant_id, metadata.session_id,
                         metadata.producer_rank);
  }
  if (n > config_.max_pack_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++counters_.oversized_records;
    return Admission::kTooLarge;
  }
  const bool block = config_.overload == Overload::kBlock;
  const double deadline =
      config_.admission_timeout_s < 0
          ? -1.0
          : NowS() + config_.admission_timeout_s;
  SinkRecord record;
  record.metadata = std::move(metadata);
  record.payload.assign(payload, payload + n);
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    if (closed_) {
      ++counters_.rejected_closed_records;
      return Admission::kClosed;
    }
    if (queue_records_ < config_.max_queue_records &&
        queue_bytes_ + n <= config_.max_queue_bytes) {
      queues_[worker].emplace_back(std::move(record));
      ++queue_records_;
      queue_bytes_ += n;
      ++counters_.admitted_records;
      counters_.admitted_bytes += n;
      queue_peak_records_ = std::max(queue_peak_records_, queue_records_);
      queue_peak_bytes_ = std::max(queue_peak_bytes_, queue_bytes_);
      lock.unlock();
      // Signal ONLY the routed worker: every other worker would wake to
      // find another queue's record. (Broadcast here was the measured
      // scaling inhibitor.)
      worker_cv_[worker].notify_one();
      return Admission::kAccepted;
    }
    if (!block) {
      ++counters_.dropped_records;
      return Admission::kDropped;
    }
    if (deadline >= 0 && NowS() >= deadline) {
      ++counters_.timed_out_records;
      return Admission::kTimedOut;
    }
    if (deadline < 0) {
      cv_.wait(lock);
    } else {
      cv_.wait_until(
          lock, std::chrono::steady_clock::now() +
                    std::chrono::duration<double>(deadline - NowS()));
    }
  }
}

bool PackSink::Flush(double timeout_s, std::string* error) {
  const double deadline = timeout_s < 0 ? -1.0 : NowS() + timeout_s;
  uint64_t requested = 0;
  size_t workers = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_started_) {
      if (error) *error = "sink is not started";
      return false;
    }
    requested = counters_.admitted_records;
    workers = queues_.size();
    if (!latched_error_.empty()) {
      if (error) *error = latched_error_;
      return false;
    }
  }
  std::lock_guard<std::mutex> flush_lock(flush_mutex_);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!latched_error_.empty()) {
      if (error) *error = latched_error_;
      return false;
    }
    if (pending_) {
      // Reuse the in-flight barrier (mirrors the Python flush coalescing:
      // repeated timeouts cannot grow an unbounded control queue).
      auto barrier = pending_;
      lock.unlock();
      bool done;
      if (deadline < 0) {
        std::unique_lock<std::mutex> wait_lock(mutex_);
        cv_.wait(wait_lock, [&] { return barrier->completed; });
        done = true;
      } else {
        std::unique_lock<std::mutex> wait_lock(mutex_);
        done = cv_.wait_until(
            wait_lock,
            std::chrono::steady_clock::now() +
                std::chrono::duration<double>(
                    std::max(0.0, deadline - NowS())),
            [&] { return barrier->completed; });
      }
      if (!done) return false;
      if (!barrier->error.empty()) {
        if (error) *error = barrier->error;
        return false;
      }
      std::lock_guard<std::mutex> clear_lock(mutex_);
      if (pending_ == barrier) pending_.reset();
      if (barrier->target_admitted >= requested) return true;
    }
  }
  auto barrier = std::make_shared<FlushBarrier>();
  barrier->target_admitted = requested;
  barrier->remaining = static_cast<int>(workers);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ = barrier;
    if (closed_) {
      if (!latched_error_.empty()) {
        if (error) *error = latched_error_;
      }
      if (pending_ == barrier) pending_.reset();
      return false;
    }
    for (auto& queue : queues_) {
      queue.emplace_back(barrier);
    }
  }
  for (auto& cv : worker_cv_) cv.notify_one();
  std::unique_lock<std::mutex> lock(mutex_);
  bool done;
  if (deadline < 0) {
    cv_.wait(lock, [&] { return barrier->completed; });
    done = true;
  } else {
    done = cv_.wait_until(
        lock,
        std::chrono::steady_clock::now() +
            std::chrono::duration<double>(std::max(0.0, deadline - NowS())),
        [&] { return barrier->completed; });
  }
  if (!done) return false;  // keep the barrier for the next call to reuse
  if (!barrier->error.empty()) {
    if (error) *error = barrier->error;
    return false;
  }
  if (pending_ == barrier) pending_.reset();
  return true;
}

SinkSnapshot PackSink::Close(double timeout_s, std::string* error) {
  (void)timeout_s;  // workers always terminate on close: each drains its
                    // queue, forwards its packs and barriers, and returns;
                    // stagers then drain and return. A stuck spool
                    // filesystem surfaces as a latched error, not a hang,
                    // so no timed join is needed.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_started_) {
      if (error) *error = "sink is not started";
      return SnapshotLocked();
    }
    closed_ = true;
  }
  cv_.notify_all();
  for (auto& cv : worker_cv_) cv.notify_all();
  for (auto& cv : stage_cv_) cv.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
  // All packs and barrier copies are forwarded; no worker will push again.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(stage_closed_.begin(), stage_closed_.end(), true);
  }
  cv_.notify_all();
  for (auto& cv : stage_cv_) cv.notify_all();
  for (auto& stager : stagers_) {
    if (stager.joinable()) stager.join();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!latched_error_.empty() && error) *error = latched_error_;
  return SnapshotLocked();
}

SinkSnapshot PackSink::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return SnapshotLocked();
}

std::string PackSink::LastError() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return latched_error_;
}

SinkSnapshot PackSink::SnapshotLocked() const {
  SinkSnapshot snapshot = counters_;
  snapshot.queue_records = queue_records_;
  snapshot.queue_bytes = queue_bytes_;
  snapshot.queue_peak_records = queue_peak_records_;
  snapshot.queue_peak_bytes = queue_peak_bytes_;
  snapshot.stage_packs = stage_packs_;
  snapshot.stage_bytes = stage_bytes_;
  snapshot.stage_peak_packs = stage_peak_packs_;
  snapshot.stage_peak_bytes = stage_peak_bytes_;
  return snapshot;
}

void PackSink::SealForStager(size_t w, FlushReason reason) {
  // Owning worker's thread only; throws std::runtime_error on seal failure
  // (latched by Run). Staging happens on the stager thread, overlapping
  // this worker's next pack. Blocks when this pair's stage queue is full
  // (backpressure, still overlapped with other pairs).
  Assembler& assembler = assemblers_[w];
  if (!assembler.builder) return;
  dmi_pack::SealedPack sealed;
  const dmi_pack::Status st = assembler.builder->Seal(&sealed);
  const uint64_t records = assembler.builder->record_count();
  const std::string pack_id = assembler.open_pack_id;
  dmi_pack::RecordMetadata first = assembler.first_metadata;
  assembler.builder.reset();
  assembler.has_first = false;
  assembler.opened_ns = -1;
  if (st != dmi_pack::Status::kOk) {
    throw std::runtime_error(std::string("seal failed: ") +
                             dmi_pack::StatusName(st));
  }
  StageItem item;
  item.is_barrier = false;
  item.pack = std::move(sealed);
  item.pack_id = pack_id;
  item.first_metadata = std::move(first);
  item.record_count = records;
  item.reason = reason;
  const uint64_t bytes = item.pack.data.size();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // closed_ in the predicate: on failure/close the stagers may be gone,
    // and this wait must not block shutdown. Pushed items are still
    // processed — stagers are joined only after all workers finish.
    stage_cv_[w].wait(lock, [&] {
      return closed_ || stage_queues_[w].size() <
                            static_cast<size_t>(config_.stage_queue_packs);
    });
    stage_queues_[w].push_back(std::move(item));
    stage_packs_ += 1;
    stage_bytes_ += bytes;
    stage_peak_packs_ = std::max(stage_peak_packs_, stage_packs_);
    stage_peak_bytes_ = std::max(stage_peak_bytes_, stage_bytes_);
  }
  stage_cv_[w].notify_one();
}

void PackSink::RunStager(size_t w) {
  try {
    for (;;) {
      StageItem item;
      bool have_item = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        stage_cv_[w].wait(lock, [&] {
          return stage_closed_[w] || !stage_queues_[w].empty();
        });
        if (!stage_queues_[w].empty()) {
          item = std::move(stage_queues_[w].front());
          stage_queues_[w].pop_front();
          if (!item.is_barrier) {
            stage_packs_ -= 1;
            stage_bytes_ -= item.pack.data.size();
          }
          have_item = true;
          lock.unlock();
          // Space freed in this pair's stage queue: wake its packer (which
          // may wait for room) without disturbing other pairs.
          stage_cv_[w].notify_one();
        } else if (stage_closed_[w]) {
          return;
        } else {
          continue;
        }
      }
      if (!have_item) continue;
      if (item.is_barrier) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (--item.barrier->remaining <= 0) {
            item.barrier->completed = true;
          }
        }
        cv_.notify_all();
        continue;
      }
      const std::string key = ObjectKeyFor(
          item.first_metadata.tenant_id, item.first_metadata.session_id,
          item.first_metadata.producer_rank,
          item.first_metadata.captured_at_ns, item.pack_id);
      dmi_store::StagedPack staged;
      std::string error;
      const dmi_store::SpoolStatus sst = spool_.Stage(
          item.pack_id, item.first_metadata.captured_at_ns,
          item.record_count, item.pack.checksum, key, item.pack.data.data(),
          item.pack.data.size(), &staged, &error);
      if (sst != dmi_store::SpoolStatus::kOk) {
        throw std::runtime_error("spool stage failed: " + error);
      }
      std::lock_guard<std::mutex> lock(mutex_);
      counters_.persisted_records += item.record_count;
      ++counters_.packs_persisted;
      counters_.packed_bytes += item.pack.data.size();
      CountFlush(&counters_, item.reason);
    }
  } catch (const std::exception& exc) {
    FailWorker(exc.what());
  } catch (...) {
    FailWorker("unknown pipeline failure");
  }
}

void PackSink::Run(size_t w) {
  try {
    Assembler& assembler = assemblers_[w];
    for (;;) {
      // Linger deadline for this worker's open pack, if any.
      double wait_s = -1.0;
      if (assembler.opened_ns >= 0) {
        const int64_t remaining = static_cast<int64_t>(
            config_.max_linger_ns - (NowNs() - assembler.opened_ns));
        wait_s = remaining <= 0 ? 0.0 : remaining / 1e9;
      }
      Item item;
      bool have_item = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        // This worker's own cv: no other worker can consume its queue, so
        // waking them all on every push is pure overhead (measured).
        if (wait_s < 0) {
          worker_cv_[w].wait(lock,
                             [&] { return closed_ || !queues_[w].empty(); });
        } else if (wait_s > 0) {
          worker_cv_[w].wait_for(lock, std::chrono::duration<double>(wait_s),
                                 [&] {
                                   return closed_ || !queues_[w].empty();
                                 });
        }
        if (!queues_[w].empty()) {
          item = std::move(queues_[w].front());
          queues_[w].pop_front();
          if (std::holds_alternative<SinkRecord>(item)) {
            const auto& record = std::get<SinkRecord>(item);
            queue_records_ -= 1;
            queue_bytes_ -= record.payload.size();
          }
          have_item = true;
          lock.unlock();
          // Space freed / progress made: the BLOCK producer and the flush
          // waiter listen on the shared cv (at most a couple of waiters).
          cv_.notify_all();
        } else if (closed_) {
          // Records are counted per pop and barriers never touch the
          // counters, so an empty queue with no records outstanding means
          // this worker is drained. (A barrier copy can still sit in
          // another worker's queue; each worker completes its own copy.)
          lock.unlock();
          SealForStager(w, FlushReason::kShutdown);
          return;
        } else {
          lock.unlock();
          // Idle wakeup: seal an expired open pack.
          if (assembler.opened_ns >= 0 &&
              NowNs() - assembler.opened_ns >=
                  static_cast<int64_t>(config_.max_linger_ns)) {
            SealForStager(w, FlushReason::kLinger);
          }
          continue;
        }
      }
      if (!have_item) continue;
      if (std::holds_alternative<std::shared_ptr<FlushBarrier>>(item)) {
        auto barrier = std::get<std::shared_ptr<FlushBarrier>>(item);
        // Seal first: everything this worker admitted before the flush must
        // be durable before the barrier completes. The sealed pack AND the
        // barrier copy both travel through this pair's stage queue in order,
        // so the stager stages the pack before counting the copy down.
        // (The stager is alive until Close joins it, so forwarding during
        // close is safe; the copy completes there.)
        try {
          SealForStager(w, FlushReason::kManual);
        } catch (...) {
          std::string barrier_error;
          try {
            throw;
          } catch (const std::exception& exc) {
            barrier_error = exc.what();
          } catch (...) {
            barrier_error = "unknown pipeline failure";
          }
          {
            std::lock_guard<std::mutex> lock(mutex_);
            if (barrier->error.empty()) barrier->error = barrier_error;
            barrier->completed = true;
          }
          cv_.notify_all();
          throw std::runtime_error(barrier_error);
        }
        StageItem forward;
        forward.is_barrier = true;
        forward.barrier = barrier;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          // Pair cv, not shared: the stager signals stage_cv_ when it frees
          // room, and nothing else notifies shared cv_ on a pop. Waiting on
          // shared here missed the wakeup and hung flush under load.
          stage_cv_[w].wait(lock, [&] {
            return closed_ || stage_queues_[w].size() <
                                  static_cast<size_t>(
                                      config_.stage_queue_packs);
          });
          stage_queues_[w].push_back(std::move(forward));
        }
        stage_cv_[w].notify_one();
        continue;
      }
      auto record = std::move(std::get<SinkRecord>(item));
      // Enforce linger on the append path too: under continuous traffic the
      // idle wakeup never fires and an open pack would linger unbounded.
      if (assembler.opened_ns >= 0 &&
          NowNs() - assembler.opened_ns >=
              static_cast<int64_t>(config_.max_linger_ns)) {
        SealForStager(w, FlushReason::kLinger);
      }
      const bool scope_changed =
          assembler.has_first &&
          (record.metadata.tenant_id != assembler.scope_tenant ||
           record.metadata.session_id != assembler.scope_session ||
           record.metadata.producer_rank != assembler.scope_rank);
      if (scope_changed) {
        SealForStager(w, FlushReason::kSession);
      }
      if (!assembler.builder) {
        assembler.open_pack_id = NewPackId();
        assembler.builder = std::make_unique<dmi_pack::PackBuilder>(
            assembler.open_pack_id, record.metadata.captured_at_ns,
            config_.max_pack_bytes, config_.max_pack_records);
        assembler.first_metadata = record.metadata;
        assembler.has_first = true;
        assembler.scope_tenant = record.metadata.tenant_id;
        assembler.scope_session = record.metadata.session_id;
        assembler.scope_rank = record.metadata.producer_rank;
        assembler.opened_ns = NowNs();
      }
      dmi_pack::PackRecord pack_record;
      pack_record.metadata = &record.metadata;
      pack_record.payload = record.payload.data();
      pack_record.payload_bytes = record.payload.size();
      dmi_pack::Status st = assembler.builder->Append(pack_record);
      if (st == dmi_pack::Status::kDuplicateId) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.duplicate_records;
        continue;
      }
      if (st == dmi_pack::Status::kCapacity) {
        // Full pack: seal with the reason the reference uses (records bound
        // hit → RECORDS, else SIZE), then retry into a fresh builder. A
        // record that fits no empty pack is oversized framing, not a
        // pipeline failure.
        const FlushReason reason =
            assembler.builder->record_count() >= config_.max_pack_records
                ? FlushReason::kRecords
                : FlushReason::kSize;
        SealForStager(w, reason);
        assembler.open_pack_id = NewPackId();
        assembler.builder = std::make_unique<dmi_pack::PackBuilder>(
            assembler.open_pack_id, record.metadata.captured_at_ns,
            config_.max_pack_bytes, config_.max_pack_records);
        assembler.first_metadata = record.metadata;
        assembler.has_first = true;
        assembler.scope_tenant = record.metadata.tenant_id;
        assembler.scope_session = record.metadata.session_id;
        assembler.scope_rank = record.metadata.producer_rank;
        assembler.opened_ns = NowNs();
        st = assembler.builder->Append(pack_record);
        if (st == dmi_pack::Status::kCapacity) {
          assembler.builder.reset();
          assembler.has_first = false;
          assembler.opened_ns = -1;
          std::lock_guard<std::mutex> lock(mutex_);
          ++counters_.oversized_records;
          continue;
        }
      }
      if (st != dmi_pack::Status::kOk) {
        throw std::runtime_error(std::string("append failed: ") +
                                 dmi_pack::StatusName(st));
      }
      if (assembler.builder->record_count() >= config_.max_pack_records) {
        SealForStager(w, FlushReason::kRecords);
      }
    }
  } catch (const std::exception& exc) {
    FailWorker(exc.what());
  } catch (...) {
    FailWorker("unknown pipeline failure");
  }
}

void PackSink::FailWorker(const std::string& what) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // First failure wins (mirrors the latched _error).
    if (latched_error_.empty()) {
      latched_error_ = what;
      ++counters_.failures;
    }
    closed_ = true;
    if (pending_ && !pending_->completed) {
      pending_->error = latched_error_;
      pending_->completed = true;
    }
  }
  cv_.notify_all();
  for (auto& cv : worker_cv_) cv.notify_all();
  for (auto& cv : stage_cv_) cv.notify_all();
}

}  // namespace dmi_sink
