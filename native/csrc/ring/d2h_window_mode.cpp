#include "d2h_window_mode.h"

#include <stdexcept>

namespace ring {

D2HWindowModeController::D2HWindowModeController(uint64_t fallback_threshold)
    : capacity_flush_fallback_threshold_(fallback_threshold) {
    if (fallback_threshold == 0) {
        throw std::invalid_argument(
            "D2H capacity flush fallback threshold must be > 0");
    }
}

D2HWindowMode D2HWindowModeController::mode() const noexcept {
    return mode_.load(std::memory_order_acquire);
}

bool D2HWindowModeController::window_scheduling_in_effect() const noexcept {
    return mode() == D2HWindowMode::ENABLED_ACTIVE;
}

void D2HWindowModeController::record_pattern_version_activation() noexcept {
    if (mode() == D2HWindowMode::ENABLED_FALLBACK)
        return;
    capacity_forced_flush_count_.store(0, std::memory_order_relaxed);
    mode_.store(D2HWindowMode::ENABLED_ACTIVE, std::memory_order_release);
}

bool D2HWindowModeController::record_capacity_forced_flush() noexcept {
    if (mode() != D2HWindowMode::ENABLED_ACTIVE)
        return false;
    const uint64_t count =
        capacity_forced_flush_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count < capacity_flush_fallback_threshold_)
        return false;
    D2HWindowMode expected = D2HWindowMode::ENABLED_ACTIVE;
    return mode_.compare_exchange_strong(expected, D2HWindowMode::ENABLED_FALLBACK,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
}

void D2HWindowModeController::reset_for_version_reuse() noexcept {
    capacity_forced_flush_count_.store(0, std::memory_order_relaxed);
    mode_.store(D2HWindowMode::ENABLED_NO_PATTERN, std::memory_order_release);
}

D2HWindowRuntimeSnapshot D2HWindowModeController::snapshot() const noexcept {
    return D2HWindowRuntimeSnapshot{
        mode(),
        capacity_forced_flush_count_.load(std::memory_order_relaxed),
        capacity_flush_fallback_threshold_,
    };
}

}  // namespace ring
