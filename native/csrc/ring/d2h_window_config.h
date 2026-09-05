#pragma once

#include <cstdint>

namespace ring {

enum class D2HWindowProgressKind : uint8_t {
    PACKED_VERSION_COUNTER = 0,
};

enum class D2HWindowGrantPolicyKind : uint8_t {
    LAST_K_ADAPTIVE = 0,
};

struct D2HWindowOffset {
    uint64_t begin{0};
    uint64_t end{0};
};

struct RecurringD2HWindowConfig {
    bool enabled{false};
    D2HWindowProgressKind progress{D2HWindowProgressKind::PACKED_VERSION_COUNTER};
    D2HWindowGrantPolicyKind grant_policy{D2HWindowGrantPolicyKind::LAST_K_ADAPTIVE};
    uint64_t history_size{0};
    uint64_t minimum_record_probe_retry_interval_occurrences{0};
    uint64_t capacity_flush_fallback_threshold{0};
    bool debug_enabled{false};
};

}  // namespace ring
