#include "d2h_window_grant_policy.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ring {

LastKAdaptiveGrantPolicy::LastKAdaptiveGrantPolicy(uint64_t history_size,
                                                   uint64_t probe_retry_interval)
    : history_limit_(history_size), probe_retry_interval_(probe_retry_interval) {
    if (history_limit_ < 2) {
        throw std::invalid_argument("D2H window history_size must be >= 2");
    }
    if (probe_retry_interval_ == 0) {
        throw std::invalid_argument(
            "D2H minimum-record probe retry interval must be > 0");
    }
}

uint64_t LastKAdaptiveGrantPolicy::target(uint64_t full_grant_bytes) const noexcept {
    if (history_.empty())
        return full_grant_bytes;

    bool any_success = false;
    bool any_failure = false;
    uint64_t maximum_success = 0;
    uint64_t minimum_failure = std::numeric_limits<uint64_t>::max();
    for (const auto& attempt : history_) {
        if (attempt.overran) {
            any_failure = true;
            minimum_failure = std::min(minimum_failure, attempt.bytes);
        } else {
            any_success = true;
            maximum_success = std::max(maximum_success, attempt.bytes);
        }
    }

    if (!any_failure)
        return full_grant_bytes;
    if (!any_success)
        return history_.back().bytes / 2;
    if (maximum_success < minimum_failure) {
        return maximum_success + (minimum_failure - maximum_success) / 2;
    }
    return minimum_failure / 2;
}

std::optional<D2HWindowGrantDecision>
LastKAdaptiveGrantPolicy::choose(uint64_t occurrence,
                                 D2HWindowAvailability availability) const {
    if (!availability.first_record_bytes.has_value())
        return std::nullopt;

    const uint64_t limit =
        std::min(target(availability.full_grant_bytes), availability.full_grant_bytes);
    if (*availability.first_record_bytes <= limit) {
        return D2HWindowGrantDecision{limit, false};
    }

    if (next_probe_occurrence_.has_value() && occurrence < *next_probe_occurrence_) {
        return std::nullopt;
    }
    return D2HWindowGrantDecision{*availability.first_record_bytes, true};
}

uint64_t LastKAdaptiveGrantPolicy::next_retry(uint64_t occurrence) const noexcept {
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (occurrence > maximum - probe_retry_interval_)
        return maximum;
    return occurrence + probe_retry_interval_;
}

void LastKAdaptiveGrantPolicy::observe(D2HWindowAttempt attempt) {
    if (attempt.minimum_record_probe && !attempt.overran) {
        history_.clear();
        history_.push_back(attempt);
        next_probe_occurrence_.reset();
        return;
    }

    if (!attempt.overran)
        next_probe_occurrence_.reset();
    history_.push_back(attempt);
    while (history_.size() > history_limit_)
        history_.pop_front();

    if (attempt.minimum_record_probe && attempt.overran) {
        bool all_failed_probes = history_.size() == history_limit_;
        for (const auto& retained : history_) {
            all_failed_probes =
                all_failed_probes && retained.overran && retained.minimum_record_probe;
        }
        if (next_probe_occurrence_.has_value() || all_failed_probes) {
            next_probe_occurrence_ = next_retry(attempt.occurrence);
        }
    }
}

std::unique_ptr<D2HWindowGrantPolicy>
make_d2h_window_grant_policy(D2HWindowGrantPolicyKind kind,
                             const RecurringD2HWindowConfig& config) {
    switch (kind) {
    case D2HWindowGrantPolicyKind::LAST_K_ADAPTIVE:
        return std::make_unique<LastKAdaptiveGrantPolicy>(
            config.history_size,
            config.minimum_record_probe_retry_interval_occurrences);
    }
    throw std::invalid_argument("unknown D2H window grant policy kind");
}

}  // namespace ring
