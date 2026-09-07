#pragma once

#include "d2h_window_config.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace ring {

struct D2HWindowAvailability {
    uint64_t full_grant_bytes{0};
    std::optional<uint64_t> first_record_bytes;
};

enum class D2HWindowGrantKind : uint8_t {
    FULL_AVAILABLE = 0,
    HALF_UNSAFE = 1,
    MAX_SAFE = 2,
    MIDPOINT = 3,
    TIMING_ESTIMATE = 4,
    MINIMUM_RECORD_PROBE = 5,
};

struct D2HWindowGrantDecision {
    uint64_t byte_limit{0};
    D2HWindowGrantKind kind{D2HWindowGrantKind::FULL_AVAILABLE};
    bool timing_revalidation{false};
    uint64_t full_grant_bytes{0};
    std::optional<uint64_t> prior_max_safe;
    std::optional<uint64_t> prior_min_unsafe;

    bool minimum_record_probe() const noexcept {
        return kind == D2HWindowGrantKind::MINIMUM_RECORD_PROBE;
    }
};

struct D2HWindowAttempt {
    uint64_t occurrence{0};
    uint64_t bytes{0};
    bool overran{false};
    D2HWindowGrantDecision decision;
};

struct D2HWindowPolicyObservation {
    bool entered_record_granularity_stall{false};
};

class D2HWindowGrantPolicy {
  public:
    virtual ~D2HWindowGrantPolicy() = default;
    virtual std::optional<D2HWindowGrantDecision>
    choose(uint64_t occurrence, D2HWindowAvailability availability) const = 0;
    virtual D2HWindowPolicyObservation observe(D2HWindowAttempt attempt) = 0;
    virtual void observe_timing_estimate(uint64_t measured_bytes,
                                         uint64_t estimate_bytes) = 0;
};

class BinaryAdaptiveGrantPolicy final : public D2HWindowGrantPolicy {
  public:
    BinaryAdaptiveGrantPolicy(
        uint64_t minimum_record_probe_retry_interval_occurrences,
        uint64_t timing_revalidation_retry_interval_occurrences);

    std::optional<D2HWindowGrantDecision>
    choose(uint64_t occurrence, D2HWindowAvailability availability) const override;
    D2HWindowPolicyObservation observe(D2HWindowAttempt attempt) override;
    void observe_timing_estimate(uint64_t measured_bytes,
                                 uint64_t estimate_bytes) override;

  private:
    D2HWindowGrantDecision
    choose_base(uint64_t occurrence, D2HWindowAvailability availability) const;
    bool minimum_record_probe_eligible(uint64_t occurrence) const noexcept;
    bool timing_revalidation_eligible(uint64_t occurrence) const noexcept;
    void reset_timing_revalidation() noexcept;
    void clear_stall() noexcept;

    uint64_t minimum_record_retry_interval_{0};
    uint64_t timing_revalidation_retry_interval_{0};

    std::optional<uint64_t> max_safe_;
    std::optional<uint64_t> min_unsafe_;
    std::optional<uint64_t> timing_estimate_;

    uint64_t minimum_record_failure_count_{0};
    std::optional<uint64_t> next_minimum_record_probe_occurrence_;

    std::optional<uint64_t> failed_revalidation_min_unsafe_;
    uint64_t revalidation_failure_count_{0};
    std::optional<uint64_t> next_revalidation_occurrence_;

    bool binary_search_stalled_{false};
    std::optional<uint64_t> stalled_timing_estimate_;
};

std::unique_ptr<D2HWindowGrantPolicy>
make_d2h_window_grant_policy(D2HWindowGrantPolicyKind kind,
                             const RecurringD2HWindowConfig& config);

}  // namespace ring
