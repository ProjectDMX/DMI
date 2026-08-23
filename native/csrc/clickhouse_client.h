#ifndef DMX_HOST_CLICKHOUSE_CLIENT_H_
#define DMX_HOST_CLICKHOUSE_CLIENT_H_

#include <any>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <ATen/ATen.h>
#include <clickhouse/client.h>

#include "dmx_host_utils.h"

namespace dmx_host {

// Settings value types for session SET ... (bound from Python too).
using ClickHouseSettingValue = std::variant<std::string, std::int64_t, bool>;

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

}  // namespace dmx_host

#endif  // DMX_HOST_CLICKHOUSE_CLIENT_H_
