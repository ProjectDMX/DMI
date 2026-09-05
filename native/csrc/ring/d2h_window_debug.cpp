#include "d2h_window_debug.h"

#include <cstdio>

namespace ring {

void D2HWindowDebugLogger::log_issue(D2HWindowPackedProgressLayout::Version version,
                                     const D2HWindowOccurrence& window,
                                     D2HWindowPackedProgressLayout::Counter counter,
                                     uint64_t bytes,
                                     bool minimum_record_probe) const noexcept {
    std::fprintf(stderr,
                 "[d2h_window] issue version=%u window=%lu occurrence=%lu "
                 "counter=%lu bytes=%lu minimum_record_probe=%d\n",
                 static_cast<unsigned>(version),
                 static_cast<unsigned long>(window.window_index),
                 static_cast<unsigned long>(window.occurrence),
                 static_cast<unsigned long>(counter), static_cast<unsigned long>(bytes),
                 minimum_record_probe ? 1 : 0);
    std::fflush(stderr);
}

void D2HWindowDebugLogger::log_completion(
    D2HWindowPackedProgressLayout::Version version,
    D2HWindowPackedProgressLayout::Counter counter,
    D2HWindowCompletionResult result) const noexcept {
    const char* text = "success";
    if (result == D2HWindowCompletionResult::OVERRUN)
        text = "overrun";
    if (result == D2HWindowCompletionResult::VERSION_CHANGE_DISCARD) {
        text = "version-change-discard";
    }
    std::fprintf(stderr, "[d2h_window] complete version=%u counter=%lu result=%s\n",
                 static_cast<unsigned>(version), static_cast<unsigned long>(counter),
                 text);
    std::fflush(stderr);
}

}  // namespace ring
