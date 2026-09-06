#include "pack_sink.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>

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

PackSink::PackSink(SinkConfig config) : config_(std::move(config)) {}

PackSink::~PackSink() {
  if (thread_started_ && worker_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
    worker_.join();
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
  worker_ = std::thread(&PackSink::Run, this);
  return "";
}

bool PackSink::IsRunning() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return thread_started_ && !closed_ && latched_error_.empty();
}

Admission PackSink::Submit(dmi_pack::RecordMetadata metadata,
                           const uint8_t* payload, size_t n) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_started_) return Admission::kClosed;
    ++counters_.submitted_records;
    if (!latched_error_.empty()) return Admission::kClosed;
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
      queue_.emplace_back(std::move(record));
      ++queue_records_;
      queue_bytes_ += n;
      ++counters_.admitted_records;
      counters_.admitted_bytes += n;
      queue_peak_records_ = std::max(queue_peak_records_, queue_records_);
      queue_peak_bytes_ = std::max(queue_peak_bytes_, queue_bytes_);
      lock.unlock();
      cv_.notify_one();
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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_started_) {
      if (error) *error = "sink is not started";
      return false;
    }
    requested = counters_.admitted_records;
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
    queue_.emplace_back(barrier);
  }
  cv_.notify_one();
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
  (void)timeout_s;  // the worker always terminates on close: it drains the
                    // queue, persists the open pack, and returns. A stuck
                    // spool filesystem surfaces as a latched error, not a
                    // hang, so no timed join is needed.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!thread_started_) {
      if (error) *error = "sink is not started";
      return SnapshotLocked();
    }
    closed_ = true;
  }
  cv_.notify_all();
  worker_.join();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!latched_error_.empty() && error) *error = latched_error_;
  return SnapshotLocked();
}

SinkSnapshot PackSink::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return SnapshotLocked();
}

SinkSnapshot PackSink::SnapshotLocked() const {
  SinkSnapshot snapshot = counters_;
  snapshot.queue_records = queue_records_;
  snapshot.queue_bytes = queue_bytes_;
  snapshot.queue_peak_records = queue_peak_records_;
  snapshot.queue_peak_bytes = queue_peak_bytes_;
  return snapshot;
}

void PackSink::PersistOpenPack(FlushReason reason) {
  // Worker thread only; throws std::runtime_error on failure (latched).
  if (!builder_) return;
  dmi_pack::SealedPack sealed;
  const dmi_pack::Status st = builder_->Seal(&sealed);
  const uint64_t records = builder_->record_count();
  const std::string pack_id = open_pack_id_;
  builder_.reset();
  has_first_ = false;
  opened_ns_ = -1;
  if (st != dmi_pack::Status::kOk) {
    throw std::runtime_error(std::string("seal failed: ") +
                             dmi_pack::StatusName(st));
  }
  const std::string key = ObjectKeyFor(first_metadata_.tenant_id,
                                       first_metadata_.session_id,
                                       first_metadata_.producer_rank,
                                       first_metadata_.captured_at_ns,
                                       pack_id);
  dmi_store::StagedPack staged;
  std::string error;
  const dmi_store::SpoolStatus sst =
      spool_.Stage(pack_id, first_metadata_.captured_at_ns, records,
                   sealed.checksum, key, sealed.data.data(),
                   sealed.data.size(), &staged, &error);
  if (sst != dmi_store::SpoolStatus::kOk) {
    throw std::runtime_error("spool stage failed: " + error);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  counters_.persisted_records += records;
  ++counters_.packs_persisted;
  counters_.packed_bytes += sealed.data.size();
  CountFlush(&counters_, reason);
}

void PackSink::Run() {
  try {
    for (;;) {
      // Linger deadline for the open pack, if any.
      double wait_s = -1.0;
      if (opened_ns_ >= 0) {
        const int64_t remaining = static_cast<int64_t>(
            config_.max_linger_ns - (NowNs() - opened_ns_));
        wait_s = remaining <= 0 ? 0.0 : remaining / 1e9;
      }
      Item item;
      bool have_item = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (wait_s < 0) {
          cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        } else if (wait_s > 0) {
          cv_.wait_for(lock, std::chrono::duration<double>(wait_s),
                       [&] { return closed_ || !queue_.empty(); });
        }
        if (!queue_.empty()) {
          item = std::move(queue_.front());
          queue_.pop_front();
          if (std::holds_alternative<SinkRecord>(item)) {
            const auto& record =
                std::get<SinkRecord>(item);
            queue_records_ -= 1;
            queue_bytes_ -= record.payload.size();
          }
          have_item = true;
          lock.unlock();
          cv_.notify_all();
        } else if (closed_ && queue_records_ == 0) {
          lock.unlock();
          PersistOpenPack(FlushReason::kShutdown);
          return;
        } else {
          lock.unlock();
          // Idle wakeup: seal an expired open pack.
          if (opened_ns_ >= 0 &&
              NowNs() - opened_ns_ >=
                  static_cast<int64_t>(config_.max_linger_ns)) {
            PersistOpenPack(FlushReason::kLinger);
          }
          continue;
        }
      }
      if (!have_item) continue;
      if (std::holds_alternative<std::shared_ptr<FlushBarrier>>(item)) {
        auto barrier = std::get<std::shared_ptr<FlushBarrier>>(item);
        std::string barrier_error;
        try {
          PersistOpenPack(FlushReason::kManual);
        } catch (const std::exception& exc) {
          barrier_error = exc.what();
        } catch (...) {
          barrier_error = "unknown pipeline failure";
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          barrier->error = barrier_error;
          barrier->completed = true;
        }
        cv_.notify_all();
        if (!barrier_error.empty()) {
          throw std::runtime_error(barrier_error);
        }
        continue;
      }
      auto record = std::move(std::get<SinkRecord>(item));
      // Enforce linger on the append path too: under continuous traffic the
      // idle wakeup never fires and an open pack would linger unbounded.
      if (opened_ns_ >= 0 &&
          NowNs() - opened_ns_ >=
              static_cast<int64_t>(config_.max_linger_ns)) {
        PersistOpenPack(FlushReason::kLinger);
      }
      const bool scope_changed =
          has_first_ && (record.metadata.tenant_id != scope_tenant_ ||
                         record.metadata.session_id != scope_session_ ||
                         record.metadata.producer_rank != scope_rank_);
      if (scope_changed) {
        PersistOpenPack(FlushReason::kSession);
      }
      if (!builder_) {
        open_pack_id_ = NewPackId();
        builder_ = std::make_unique<dmi_pack::PackBuilder>(
            open_pack_id_, record.metadata.captured_at_ns,
            config_.max_pack_bytes, config_.max_pack_records);
        first_metadata_ = record.metadata;
        has_first_ = true;
        scope_tenant_ = record.metadata.tenant_id;
        scope_session_ = record.metadata.session_id;
        scope_rank_ = record.metadata.producer_rank;
        opened_ns_ = NowNs();
      }
      dmi_pack::PackRecord pack_record;
      pack_record.metadata = record.metadata;
      pack_record.payload = record.payload.data();
      pack_record.payload_bytes = record.payload.size();
      dmi_pack::Status st = builder_->Append(pack_record);
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
            builder_->record_count() >= config_.max_pack_records
                ? FlushReason::kRecords
                : FlushReason::kSize;
        PersistOpenPack(reason);
        open_pack_id_ = NewPackId();
        builder_ = std::make_unique<dmi_pack::PackBuilder>(
            open_pack_id_, record.metadata.captured_at_ns,
            config_.max_pack_bytes, config_.max_pack_records);
        first_metadata_ = record.metadata;
        has_first_ = true;
        scope_tenant_ = record.metadata.tenant_id;
        scope_session_ = record.metadata.session_id;
        scope_rank_ = record.metadata.producer_rank;
        opened_ns_ = NowNs();
        st = builder_->Append(pack_record);
        if (st == dmi_pack::Status::kCapacity) {
          builder_.reset();
          has_first_ = false;
          opened_ns_ = -1;
          std::lock_guard<std::mutex> lock(mutex_);
          ++counters_.oversized_records;
          continue;
        }
      }
      if (st != dmi_pack::Status::kOk) {
        throw std::runtime_error(std::string("append failed: ") +
                                 dmi_pack::StatusName(st));
      }
      if (builder_->record_count() >= config_.max_pack_records) {
        PersistOpenPack(FlushReason::kRecords);
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
}

}  // namespace dmi_sink
