#ifndef DMX_HOST_CLICKHOUSE_CLIENT_H_
#define DMX_HOST_CLICKHOUSE_CLIENT_H_

#include <any>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <ATen/ATen.h>
#include <clickhouse/client.h>

#include "dmx_host_utils.h"
#include "record_schema.h"

namespace dmx_host {

// Settings value types for session SET ... (bound from Python too).
using ClickHouseSettingValue = std::variant<std::string, std::int64_t, bool>;

struct ClickHouseWorkerMetrics {
  int worker_index = 0;
  std::uint64_t batches = 0;
  std::uint64_t rows = 0;
  std::uint64_t logical_bytes = 0;
  double insert_seconds = 0.0;
};

struct ClickHouseMetricsSnapshot {
  int expected_workers = 0;
  int ready_workers = 0;
  std::uint64_t active_inserts = 0;
  std::uint64_t peak_active_inserts = 0;
  std::uint64_t batches = 0;
  std::uint64_t rows = 0;
  std::uint64_t logical_bytes = 0;
  double insert_seconds = 0.0;
  std::vector<ClickHouseWorkerMetrics> workers;
};

class ClickHouseRuntimeMetrics {
 public:
  explicit ClickHouseRuntimeMetrics(int expected_workers = 0);

  void WorkerReady(int worker_index);
  bool WaitUntilReady(std::chrono::milliseconds timeout);
  void BeginInsert();
  void EndInsert(int worker_index, std::uint64_t rows,
                 std::uint64_t logical_bytes, double seconds);
  ClickHouseMetricsSnapshot Snapshot() const;

 private:
  mutable std::mutex mu_;
  std::condition_variable ready_cv_;
  int expected_workers_ = 0;
  int ready_workers_ = 0;
  std::uint64_t active_inserts_ = 0;
  std::uint64_t peak_active_inserts_ = 0;
  std::uint64_t batches_ = 0;
  std::uint64_t rows_ = 0;
  std::uint64_t logical_bytes_ = 0;
  double insert_seconds_ = 0.0;
  std::vector<bool> ready_;
  std::vector<ClickHouseWorkerMetrics> workers_;
};

/**
 * ClickHouse connection + schema init configuration.
 */
struct ClickHouseClientConfig {
  std::string host = "localhost";
  int port = 9000;
  std::string username = "default";
  std::string password = "";

  std::string database = "default";
  std::string table = "offload";

  bool secure = false;

  // Optional per-session SET key=value applied after USE db.
  std::unordered_map<std::string, ClickHouseSettingValue> client_settings;

  bool create_database_if_missing = true;
  bool drop_existing_database = false;  // for testing

  // "none" | "lz4" | "zstd" | "true" | "false"
  std::string client_side_compress = "none";

  int index_granularity = 8192;
  int connect_timeout_ms = 5000;
  int receive_timeout_ms = 0;
  int send_timeout_ms = 0;

  std::shared_ptr<ClickHouseRuntimeMetrics> runtime_metrics =
      std::make_shared<ClickHouseRuntimeMetrics>();
};

// Internal, immutable worker initialization value for the additive generic
// record stage. ClickHouseClientConfig itself remains the released connection
// configuration and is intentionally not extended with schema state.
struct ClickHouseRecordStageConfig {
  ClickHouseClientConfig client;
  RecordSchema schema;
};

/**
 * ClickHouse sink stage for PipelinedEngine.
 *
 * Thread model:
 * - clickhouse-cpp Client is NOT thread-safe -> we keep a thread_local Client.
 * - call ThreadInit once per worker thread, and ThreadCleanup at thread end.
 *
 * Row format (exactly 8 cells):
 *   0 model_id (string)
 *   1 request_id (string)
 *   2 act_name (string)
 *   3 layer_no (int32)
 *   4 shard_rank (int32)
 *   5 start_token_idx (int32)
 *   6 end_token_idx (int32)
 *   7 tensor (at::Tensor)
 *
 * The stage derives CH columns:
 *   dtype: String
 *   shape: Array(Int64)
 *   bytes: String (binary-safe)
 */
class ClickHouseInsertStage final {
 public:
  static void ThreadInit(int thread_idx, const ClickHouseClientConfig& cfg);
  static void ThreadCleanup() noexcept;

  // Insert a batch; does not forward.
  static void InsertBatch(std::vector<dmx_host_queue_item>&& batch);

  // Engine-compatible wrappers:
  static inline void ThreadInitAny(int thread_idx, const std::any& cfg_any) {
    ThreadInit(thread_idx, std::any_cast<const ClickHouseClientConfig&>(cfg_any));
  }

  static inline void ThreadCleanupAny() noexcept { ThreadCleanup(); }

  template <typename QueueT>
  static inline std::optional<std::vector<dmx_host_queue_item>> ProcessFn(std::vector<dmx_host_queue_item> batch,
                                                                    QueueT* /*next_q*/) {
    InsertBatch(std::move(batch));
    return std::vector<dmx_host_queue_item>{};  // sink: no outputs
  }
};

/** Schema-driven typed-row sink. This is a separate opt-in stage; it does not
 * route through or reinterpret ClickHouseInsertStage's fixed inference row.
 */
class ClickHouseRecordInsertStage final {
 public:
  static void ThreadInit(int thread_idx,
                         const ClickHouseRecordStageConfig& cfg);
  static void ThreadCleanup() noexcept;
  static void InsertBatch(std::vector<dmx_host_queue_item>&& batch);

  static inline void ThreadInitAny(int thread_idx, const std::any& cfg_any) {
    ThreadInit(
        thread_idx,
        std::any_cast<const ClickHouseRecordStageConfig&>(cfg_any));
  }

  static inline void ThreadCleanupAny() noexcept { ThreadCleanup(); }

  template <typename QueueT>
  static inline std::optional<std::vector<dmx_host_queue_item>> ProcessFn(
      std::vector<dmx_host_queue_item> batch, QueueT* /*next_q*/) {
    InsertBatch(std::move(batch));
    return std::vector<dmx_host_queue_item>{};
  }
};

}  // namespace dmx_host

#endif  // DMX_HOST_CLICKHOUSE_CLIENT_H_
