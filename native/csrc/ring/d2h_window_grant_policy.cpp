#include "d2h_window_grant_policy.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ring {
namespace {

uint64_t saturating_increment(uint64_t value) noexcept {
    if (value == std::numeric_limits<uint64_t>::max())
        return value;
    return value + 1;
}

uint64_t saturating_multiply(uint64_t lhs, uint64_t rhs) noexcept {
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (lhs != 0 && rhs > maximum / lhs)
        return maximum;
    return lhs * rhs;
}

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) noexcept {
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (rhs > maximum - lhs)
        return maximum;
    return lhs + rhs;
}

uint64_t next_retry(uint64_t occurrence, uint64_t failures,
                    uint64_t interval) noexcept {
    return saturating_add(occurrence, saturating_multiply(failures, interval));
}

}  // namespace

BinaryAdaptiveGrantPolicy::BinaryAdaptiveGrantPolicy(
    uint64_t minimum_record_probe_retry_interval_occurrences,
    uint64_t timing_revalidation_retry_interval_occurrences)
    : minimum_record_retry_interval_(
          minimum_record_probe_retry_interval_occurrences),
      timing_revalidation_retry_interval_(
          timing_revalidation_retry_interval_occurrences) {
    if (minimum_record_retry_interval_ == 0) {
        throw std::invalid_argument(
            "D2H minimum-record probe retry interval must be > 0");
    }
    if (timing_revalidation_retry_interval_ == 0) {
        throw std::invalid_argument(
            "D2H timing revalidation retry interval must be > 0");
    }
}

bool BinaryAdaptiveGrantPolicy::minimum_record_probe_eligible(
    uint64_t occurrence) const noexcept {
    return !next_minimum_record_probe_occurrence_.has_value() ||
        occurrence >= *next_minimum_record_probe_occurrence_;
}

bool BinaryAdaptiveGrantPolicy::timing_revalidation_eligible(
    uint64_t occurrence) const noexcept {
    return !next_revalidation_occurrence_.has_value() ||
        occurrence >= *next_revalidation_occurrence_;
}

D2HWindowGrantDecision BinaryAdaptiveGrantPolicy::choose_base(
    uint64_t occurrence, D2HWindowAvailability availability) const {
    D2HWindowGrantDecision decision;
    decision.full_grant_bytes = availability.full_grant_bytes;
    decision.prior_max_safe = max_safe_;
    decision.prior_min_unsafe = min_unsafe_;

    if (!max_safe_.has_value() && !min_unsafe_.has_value()) {
        decision.byte_limit = std::numeric_limits<uint64_t>::max();
        decision.kind = D2HWindowGrantKind::FULL_AVAILABLE;
        return decision;
    }

    if (!max_safe_.has_value()) {
        decision.byte_limit = *min_unsafe_ / 2;
        decision.kind = D2HWindowGrantKind::HALF_UNSAFE;
        return decision;
    }

    if (!min_unsafe_.has_value()) {
        if (timing_estimate_.has_value() && *timing_estimate_ > *max_safe_) {
            decision.byte_limit = *timing_estimate_;
            decision.kind = D2HWindowGrantKind::TIMING_ESTIMATE;
        } else {
            decision.byte_limit = *max_safe_;
            decision.kind = D2HWindowGrantKind::MAX_SAFE;
        }
        return decision;
    }

    const uint64_t safe = *max_safe_;
    const uint64_t unsafe = *min_unsafe_;
    const bool has_new_timing_estimate =
        timing_estimate_.has_value() &&
        (!stalled_timing_estimate_.has_value() ||
         *timing_estimate_ != *stalled_timing_estimate_);

    if (binary_search_stalled_) {
        if (has_new_timing_estimate && *timing_estimate_ > safe &&
            *timing_estimate_ < unsafe) {
            decision.byte_limit = *timing_estimate_;
            decision.kind = D2HWindowGrantKind::TIMING_ESTIMATE;
            return decision;
        }
        if (has_new_timing_estimate && *timing_estimate_ > unsafe &&
            timing_revalidation_eligible(occurrence)) {
            decision.byte_limit = *timing_estimate_;
            decision.kind = D2HWindowGrantKind::TIMING_ESTIMATE;
            decision.timing_revalidation = true;
            return decision;
        }
        decision.byte_limit = safe;
        decision.kind = D2HWindowGrantKind::MAX_SAFE;
        return decision;
    }

    if (timing_estimate_.has_value() && *timing_estimate_ > unsafe &&
        timing_revalidation_eligible(occurrence)) {
        decision.byte_limit = *timing_estimate_;
        decision.kind = D2HWindowGrantKind::TIMING_ESTIMATE;
        decision.timing_revalidation = true;
        return decision;
    }

    decision.byte_limit = safe + (unsafe - safe) / 2;
    decision.kind = D2HWindowGrantKind::MIDPOINT;
    return decision;
}

std::optional<D2HWindowGrantDecision> BinaryAdaptiveGrantPolicy::choose(
    uint64_t occurrence, D2HWindowAvailability availability) const {
    if (!availability.first_record_bytes.has_value())
        return std::nullopt;

    auto decision = choose_base(occurrence, availability);
    if (*availability.first_record_bytes <= decision.byte_limit)
        return decision;

    if (!minimum_record_probe_eligible(occurrence))
        return std::nullopt;

    decision.byte_limit = *availability.first_record_bytes;
    decision.kind = D2HWindowGrantKind::MINIMUM_RECORD_PROBE;
    decision.timing_revalidation = false;
    return decision;
}

void BinaryAdaptiveGrantPolicy::reset_timing_revalidation() noexcept {
    failed_revalidation_min_unsafe_.reset();
    revalidation_failure_count_ = 0;
    next_revalidation_occurrence_.reset();
}

void BinaryAdaptiveGrantPolicy::clear_stall() noexcept {
    binary_search_stalled_ = false;
    stalled_timing_estimate_.reset();
}

D2HWindowPolicyObservation BinaryAdaptiveGrantPolicy::observe(
    D2HWindowAttempt attempt) {
    D2HWindowPolicyObservation observation;
    if (attempt.bytes == 0)
        return observation;

    const auto prior_safe = attempt.decision.prior_max_safe;
    const auto prior_unsafe = attempt.decision.prior_min_unsafe;

    if (attempt.overran) {
        clear_stall();
        if (prior_safe.has_value() && attempt.bytes <= *prior_safe) {
            max_safe_.reset();
            min_unsafe_ = attempt.bytes;
            timing_estimate_.reset();
            reset_timing_revalidation();
        } else if (!min_unsafe_.has_value() || attempt.bytes < *min_unsafe_) {
            min_unsafe_ = attempt.bytes;
            reset_timing_revalidation();
        }
    } else if (prior_unsafe.has_value() && attempt.bytes >= *prior_unsafe) {
        max_safe_ = attempt.bytes;
        min_unsafe_.reset();
        reset_timing_revalidation();
        clear_stall();
    } else {
        if (!max_safe_.has_value() || attempt.bytes > *max_safe_)
            max_safe_ = attempt.bytes;

        const bool safe_bound_advanced =
            prior_safe.has_value() && attempt.bytes > *prior_safe;
        if (safe_bound_advanced)
            clear_stall();

        const bool search_attempt =
            attempt.decision.kind == D2HWindowGrantKind::MIDPOINT ||
            attempt.decision.kind == D2HWindowGrantKind::TIMING_ESTIMATE;
        if (!safe_bound_advanced && search_attempt && prior_safe.has_value() &&
            attempt.bytes <= *prior_safe &&
            attempt.decision.full_grant_bytes > attempt.bytes) {
            observation.entered_record_granularity_stall =
                !binary_search_stalled_;
            binary_search_stalled_ = true;
            if (attempt.decision.kind == D2HWindowGrantKind::TIMING_ESTIMATE)
                stalled_timing_estimate_ = attempt.decision.byte_limit;
        }
    }

    if (attempt.decision.minimum_record_probe()) {
        if (attempt.overran) {
            minimum_record_failure_count_ =
                saturating_increment(minimum_record_failure_count_);
            next_minimum_record_probe_occurrence_ = next_retry(
                attempt.occurrence, minimum_record_failure_count_,
                minimum_record_retry_interval_);
        } else {
            minimum_record_failure_count_ = 0;
            next_minimum_record_probe_occurrence_.reset();
        }
    }

    if (attempt.decision.timing_revalidation && attempt.overran &&
        prior_unsafe.has_value() && attempt.bytes >= *prior_unsafe &&
        min_unsafe_.has_value() && *min_unsafe_ == *prior_unsafe) {
        if (!failed_revalidation_min_unsafe_.has_value() ||
            *failed_revalidation_min_unsafe_ != *prior_unsafe) {
            failed_revalidation_min_unsafe_ = *prior_unsafe;
            revalidation_failure_count_ = 1;
        } else {
            revalidation_failure_count_ =
                saturating_increment(revalidation_failure_count_);
        }
        next_revalidation_occurrence_ = next_retry(
            attempt.occurrence, revalidation_failure_count_,
            timing_revalidation_retry_interval_);
    }

    return observation;
}

void BinaryAdaptiveGrantPolicy::observe_timing_estimate(
    uint64_t measured_bytes, uint64_t estimate_bytes) {
    if (max_safe_.has_value() && measured_bytes == *max_safe_ &&
        estimate_bytes <= *max_safe_) {
        return;
    }
    timing_estimate_ = estimate_bytes;
}

std::unique_ptr<D2HWindowGrantPolicy> make_d2h_window_grant_policy(
    D2HWindowGrantPolicyKind kind, const RecurringD2HWindowConfig& config) {
    switch (kind) {
    case D2HWindowGrantPolicyKind::BINARY_ADAPTIVE:
        return std::make_unique<BinaryAdaptiveGrantPolicy>(
            config.minimum_record_probe_retry_interval_occurrences,
            config.timing_revalidation_retry_interval_occurrences);
    }
    throw std::invalid_argument("unknown D2H window grant policy kind");
}

}  // namespace ring
