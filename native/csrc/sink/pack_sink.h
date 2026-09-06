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
  // Pack assembler workers. Records route by scope hash
  // (tenant, session, producer_rank), so one scope always lands on one
  // worker: per-scope ordering and single-scope packs are preserved at any
  // N, and N=1 is exactly the A3b behavior. 0 means 1.
  int num_workers = 1;
  // Sealed packs awaiting spool staging, per packer/stager pair. The stager
  // overlaps disk I/O with the next pack's assembly; the bound caps
  // in-memory packs (each up to max_pack_bytes).
  int stage_queue_packs = 2;
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
  // Sealed packs handed to stagers but not yet durable. Not part of the
  // Python snapshot (no stage queue exists there); native-only visibility
  // into the second pipeline stage.
  uint64_t stage_packs = 0;
  uint64_t stage_bytes = 0;
  uint64_t stage_peak_packs = 0;
  uint64_t stage_peak_bytes = 0;
};

// One admitted record: metadata by value (26 strings — moved, not copied,
// through the queue) plus an owned payload.
struct SinkRecord {
  dmi_pack::RecordMetadata metadata;
  std::vector<uint8_t> payload;
};

// Non-closing durability barrier: completes once every record admitted
// before it has been persisted. With N workers each queue carries one copy;
// `remaining` counts the copies still outstanding.
struct FlushBarrier {
  uint64_t target_admitted = 0;
  int remaining = 1;
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

  void Run(size_t worker);
  void FailWorker(const std::string& what);
  // Seal worker `w`'s open pack for `reason` and hand it to its stager.
  // Throws on seal failure (latched by Run); staging errors surface at the
  // stager and complete the in-flight barrier with the error.
  void SealForStager(size_t w, FlushReason reason);
  // Stager thread for pair `w`: stages sealed packs, counts down barriers.
  void RunStager(size_t w);
  // Snapshot body with mutex_ already held.
  SinkSnapshot SnapshotLocked() const;
  // Worker index for a scope. FNV-1a over tenant\0session\0rank — stable
  // across calls so one scope always lands on one worker.
  size_t RouteWorker(const std::string& tenant, const std::string& session,
                     uint64_t rank) const;

  SinkConfig config_;
  dmi_store::Spool spool_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  // Per-worker wakeup channels. A broadcast notify_all on the shared cv
  // wakes every worker on every submit/pop (~180k thundering-herd wakeups
  // per 10k-record trial at N=8 — measured as the scaling inhibitor), so
  // the producer signals ONLY the routed worker's cv. The shared cv stays
  // for the few waiters that need broadcast: the BLOCK producer (space
  // freed by any pop), the flush waiter (barrier completion), and close.
  // deque: condition_variable is neither movable nor copyable, and deque
  // emplace_back never moves existing elements.
  std::deque<std::condition_variable> worker_cv_;
  std::deque<std::condition_variable> stage_cv_;
  // One FIFO per worker; admission bounds (queue_records_/bytes_) are global
  // across them so single-scope throughput never depends on N.
  std::vector<std::deque<Item>> queues_;
  uint64_t queue_records_ = 0;
  uint64_t queue_bytes_ = 0;
  bool closed_ = false;
  bool thread_started_ = false;
  std::vector<std::thread> workers_;
  std::vector<std::thread> stagers_;

  std::mutex flush_mutex_;
  std::shared_ptr<FlushBarrier> pending_;

  std::string latched_error_;
  SinkSnapshot counters_;
  uint64_t queue_peak_records_ = 0;
  uint64_t queue_peak_bytes_ = 0;

  // Stage queues: sealed packs (and forwarded barriers) awaiting the spool.
  // One per packer/stager pair, so each pair is an independent ordered
  // pipeline and barrier countdown needs no cross-ordering logic. Bounded
  // by stage_queue_packs; the packer blocks when full (backpressure, still
  // overlapped with staging).
  struct StageItem {
    // Exactly one of pack / barrier is set.
    bool is_barrier = false;
    dmi_pack::SealedPack pack;
    std::string pack_id;
    dmi_pack::RecordMetadata first_metadata;
    uint64_t record_count = 0;
    FlushReason reason = FlushReason::kManual;
    std::shared_ptr<FlushBarrier> barrier;
  };
  std::vector<std::deque<StageItem>> stage_queues_;
  std::vector<bool> stage_closed_;
  // Packs currently held in stage queues (bytes), for the snapshot.
  uint64_t stage_packs_ = 0;
  uint64_t stage_bytes_ = 0;
  uint64_t stage_peak_packs_ = 0;
  uint64_t stage_peak_bytes_ = 0;

  // Per-worker assembler state (touched by its worker thread only, except
  // through PersistOpenPack which runs on the owning worker too).
  struct Assembler {
    std::unique_ptr<dmi_pack::PackBuilder> builder;
    dmi_pack::RecordMetadata first_metadata;
    bool has_first = false;
    std::string scope_tenant;
    std::string scope_session;
    uint64_t scope_rank = 0;
    int64_t opened_ns = -1;
    std::string open_pack_id;
  };
  std::vector<Assembler> assemblers_;
};

}  // namespace dmi_sink

#endif  // DMI_SINK_PACK_SINK_H_
