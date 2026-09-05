#pragma once

#include "d2h_window_pattern.h"
#include "d2h_window_progress_layout.h"

#include <cstdint>

namespace ring {

enum class D2HWindowCompletionResult : uint8_t {
    SUCCESS = 0,
    OVERRUN = 1,
    VERSION_CHANGE_DISCARD = 2,
};

class D2HWindowDebugLogger {
  public:
    void log_issue(D2HWindowPackedProgressLayout::Version version,
                   const D2HWindowOccurrence& window,
                   D2HWindowPackedProgressLayout::Counter counter, uint64_t bytes,
                   bool minimum_record_probe) const noexcept;
    void log_completion(D2HWindowPackedProgressLayout::Version version,
                        D2HWindowPackedProgressLayout::Counter counter,
                        D2HWindowCompletionResult result) const noexcept;
};

}  // namespace ring
