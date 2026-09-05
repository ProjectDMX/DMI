#pragma once

#include <atomic>
#include <cstdint>

namespace ring {

enum class D2HWindowMode : uint8_t {
    DISABLED = 0,
    ENABLED_NO_PATTERN = 1,
    ENABLED_ACTIVE = 2,
    ENABLED_FALLBACK = 3,
};

struct D2HWindowRuntimeSnapshot {
    D2HWindowMode mode{D2HWindowMode::DISABLED};
    uint64_t capacity_forced_flush_count{0};
    uint64_t capacity_flush_fallback_threshold{0};
};

class D2HWindowModeController {
  public:
    explicit D2HWindowModeController(uint64_t fallback_threshold);

    D2HWindowMode mode() const noexcept;
    bool window_scheduling_in_effect() const noexcept;
    void record_pattern_version_activation() noexcept;
    bool record_capacity_forced_flush() noexcept;
    void reset_for_version_reuse() noexcept;
    D2HWindowRuntimeSnapshot snapshot() const noexcept;

  private:
    std::atomic<D2HWindowMode> mode_{D2HWindowMode::ENABLED_NO_PATTERN};
    std::atomic<uint64_t> capacity_forced_flush_count_{0};
    const uint64_t capacity_flush_fallback_threshold_;
};

}  // namespace ring
