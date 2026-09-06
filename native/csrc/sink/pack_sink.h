// Native capture sink: bounded queue → one pack assembler → durable spool.
//
// Mirrors HostCapturePipeline (pipeline.py) single-worker semantics:
// admission policy (block / drop-newest), linger + size + record + session
// sealing, non-closing flush barriers, terminal close, latched failures,
// and the same counter snapshot. The multi-worker pool (scope-hash
// partition) is A5a; the ring::RecordSink adapter is A5b.

#ifndef DMI_SINK_PACK_SINK_H_
#define DMI_SINK_PACK_SINK_H_

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "../pack/pack_builder.h"
#include "../store/spool.h"

namespace dmi_sink {

enum class Admission {
  kAccepted,
  kDropped,
  kTimedOut,
  kTooLarge,
  kClosed,
};

enum class Overload {
  kBlock,
  kDropNewest,
};

inline const char* AdmissionName(Admission a) {
  switch (a) {
    case Admission::kAccepted: return "accepted";
    case Admission::kDropped: return "dropped";
    case Admission::kTimedOut: return "timed_out";
    case Admission::kTooLarge: return "too_large";
    case Admission::kClosed: return "closed";
  }
  return "unknown";
}

enum class FlushReason {
  kSize,
  kRecords,
  kLinger,
  kSession,
  kManual,
  kShutdown,
};

struct SinkConfig {
  uint64_t max_queue_records = 256;
  uint64_t max_queue_bytes = 16 * 1024 * 1024;
  uint64_t max_pack_bytes = 128ull * 1024 * 1024;
  uint64_t max_pack_records = 10'000;
  uint64_t max_linger_ns = 1'000'000'000;
  Overload overload = Overload::kDropNewest;
  // < 0 waits forever under kBlock; matches admission_timeout=None.
  double admission_timeout_s = -1.0;
  std::string spool_root;
  uint64_t spool_max_bytes = 1ull << 40;
};

struct SinkSnapshot {
  uint64_t submitted_records = 0;
  uint64_t admitted_records = 0;
  uint64_t admitted_bytes = 0;
  uint64_t dropped_records = 0;
  uint64_t timed_out_records = 0;
  uint64_t oversized_records = 0;
  uint64_t duplicate_records = 0;
  uint64_t rejected_closed_records = 0;
  uint64_t persisted_records = 0;
  uint64_t packs_persisted = 0;
  uint64_t packed_bytes = 0;
  uint64_t flush_size = 0;
  uint64_t flush_records = 0;
  uint64_t flush_linger = 0;
  uint64_t flush_session = 0;
  uint64_t flush_manual = 0;
  uint64_t flush_shutdown = 0;
  uint64_t failures = 0;
  uint64_t queue_records = 0;
  uint64_t queue_bytes = 0;
  uint64_t queue_peak_records = 0;
  uint64_t queue_peak_bytes = 0;
};

// One admitted record: metadata by value (26 strings — moved, not copied,
// through the queue) plus an owned payload.
struct SinkRecord {
  dmi_pack::RecordMetadata metadata;
  std::vector<uint8_t> payload;
};

// Non-closing durability barrier: completes once every record admitted
// before it has been persisted.
struct FlushBarrier {
  uint64_t target_admitted = 0;
  bool completed = false;
  std::string error;
};

class PackSink {
 public:
  explicit PackSink(SinkConfig config);
  ~PackSink();

  PackSink(const PackSink&) = delete;
  PackSink& operator=(const PackSink&) = delete;

  // Starts the worker. Returns an error string, empty on success.
  std::string Start(std::string* spool_error = nullptr);
  bool IsRunning() const;

  Admission Submit(dmi_pack::RecordMetadata metadata,
                   const uint8_t* payload, size_t n);

  // Persist everything admitted before this call. False on timeout; the
  // in-flight barrier is kept for the next call to reuse. timeout_s < 0
  // waits forever. Error string set on pipeline failure.
  bool Flush(double timeout_s, std::string* error);
  // Terminal close: drains, joins, returns the final snapshot.
  SinkSnapshot Close(double timeout_s, std::string* error);
  SinkSnapshot Snapshot() const;

 private:
  using Item = std::variant<SinkRecord, std::shared_ptr<FlushBarrier>>;

  void Run();
  void FailWorker(const std::string& what);
  // Seal the open pack for `reason`, stage it, count it. Throws on failure
  // (latched by Run).
  void PersistOpenPack(FlushReason reason);
  // Snapshot body with mutex_ already held.
  SinkSnapshot SnapshotLocked() const;

  SinkConfig config_;
  dmi_store::Spool spool_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
  uint64_t queue_records_ = 0;
  uint64_t queue_bytes_ = 0;
  bool closed_ = false;
  bool thread_started_ = false;
  std::thread worker_;

  std::mutex flush_mutex_;
  std::shared_ptr<FlushBarrier> pending_;

  std::string latched_error_;
  SinkSnapshot counters_;
  uint64_t queue_peak_records_ = 0;
  uint64_t queue_peak_bytes_ = 0;

  // Assembler state (worker thread only).
  std::unique_ptr<dmi_pack::PackBuilder> builder_;
  dmi_pack::RecordMetadata first_metadata_;
  bool has_first_ = false;
  std::string scope_tenant_;
  std::string scope_session_;
  uint64_t scope_rank_ = 0;
  int64_t opened_ns_ = -1;
  std::string open_pack_id_;
};

}  // namespace dmi_sink

#endif  // DMI_SINK_PACK_SINK_H_
