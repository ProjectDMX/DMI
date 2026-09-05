#pragma once

#include "d2h_window_config.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>

namespace ring {

struct D2HWindowAttempt {
    uint64_t occurrence{0};
    uint64_t bytes{0};
    bool overran{false};
    bool minimum_record_probe{false};
};

struct D2HWindowAvailability {
    uint64_t full_grant_bytes{0};
    std::optional<uint64_t> first_record_bytes;
};

struct D2HWindowGrantDecision {
    uint64_t byte_limit{0};
    bool minimum_record_probe{false};
};

class D2HWindowGrantPolicy {
  public:
    virtual ~D2HWindowGrantPolicy() = default;
    virtual std::optional<D2HWindowGrantDecision>
    choose(uint64_t occurrence, D2HWindowAvailability availability) const = 0;
    virtual void observe(D2HWindowAttempt attempt) = 0;
};

class LastKAdaptiveGrantPolicy final : public D2HWindowGrantPolicy {
  public:
    LastKAdaptiveGrantPolicy(uint64_t history_size, uint64_t probe_retry_interval);

    std::optional<D2HWindowGrantDecision>
    choose(uint64_t occurrence, D2HWindowAvailability availability) const override;
    void observe(D2HWindowAttempt attempt) override;

    size_t history_size() const noexcept { return history_.size(); }
    std::optional<uint64_t> next_probe_occurrence() const noexcept {
        return next_probe_occurrence_;
    }

  private:
    uint64_t target(uint64_t full_grant_bytes) const noexcept;
    uint64_t next_retry(uint64_t occurrence) const noexcept;

    uint64_t history_limit_{0};
    uint64_t probe_retry_interval_{0};
    std::deque<D2HWindowAttempt> history_;
    std::optional<uint64_t> next_probe_occurrence_;
};

std::unique_ptr<D2HWindowGrantPolicy>
make_d2h_window_grant_policy(D2HWindowGrantPolicyKind kind,
                             const RecurringD2HWindowConfig& config);

}  // namespace ring
