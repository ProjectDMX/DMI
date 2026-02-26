#ifndef DMX_FUTURE_PROCESS_H__
#define DMX_FUTURE_PROCESS_H__

#include "dmx_host_utils.h"
#include <any>

namespace dmx_host{

class ProcessFutureStage final {
 public:
  static void ThreadInit(int thread_idx, bool debug_log);
  static void ThreadCleanup() noexcept;

  template <typename QueueT>
  static std::optional<std::vector<dmx_host_queue_item>> ProcessFuture(std::vector<dmx_host_queue_item>&& batch, QueueT* next_q);

  // Engine-compatible wrappers:
  static inline void ThreadInitAny(int thread_idx, const std::any& cfg_any) {
    bool debug_log = false;
    if (const bool* debug_ptr = std::any_cast<bool>(&cfg_any)) {
      debug_log = *debug_ptr;
    }
    ThreadInit(thread_idx, debug_log);
  }

  static inline void ThreadCleanupAny() noexcept { ThreadCleanup(); }

  template <typename QueueT>
  static inline std::optional<std::vector<dmx_host_queue_item>> ProcessFn(std::vector<dmx_host_queue_item> batch,
                                                                    QueueT* next_q) {
    return ProcessFuture<QueueT> (std::move(batch), next_q);
  }
};
}

#endif
