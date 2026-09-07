#include "recurring_d2h_grant_controller.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ring {

RecurringD2HGrantController::RecurringD2HGrantController(
    D2HWindowProgressSource& progress, D2HWindowModeController& mode,
    D2HWindowGrantPolicyFactory policy_factory, D2HWindowDebugLogger* debug_logger,
    D2HWindowNowFunction now)
    : progress_(progress), mode_(mode), policy_factory_(std::move(policy_factory)),
      debug_logger_(debug_logger), now_(std::move(now)) {
    if (!policy_factory_) {
        throw std::invalid_argument("D2H window policy factory is required");
    }
    if (!now_)
        now_ = [] { return D2HWindowClock::now(); };
}

std::unique_ptr<RecurringD2HGrantController::VersionBundle>
RecurringD2HGrantController::make_bundle(
    D2HWindowPackedProgressLayout::Version version, uint64_t period,
    const std::vector<D2HWindowOffset>& windows) const {
    auto bundle = std::make_unique<VersionBundle>(VersionBundle{
        version,
        D2HWindowPatternMatcher(period, windows),
        {},
        std::nullopt,
    });
    bundle->windows.reserve(windows.size());
    for (size_t index = 0; index < windows.size(); ++index) {
        bundle->windows.push_back(WindowState{
            policy_factory_(),
            std::nullopt,
            std::nullopt,
            std::nullopt,
        });
        if (!bundle->windows.back().policy) {
            throw std::logic_error("D2H window policy factory returned null");
        }
    }
    return bundle;
}

void RecurringD2HGrantController::install_pending(
    D2HWindowPackedProgressLayout::Version version, uint64_t period,
    const std::vector<D2HWindowOffset>& windows) {
    auto bundle = make_bundle(version, period, windows);
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    if (!pending_bundles_.empty() &&
        version <= pending_bundles_.back()->version) {
        throw std::logic_error(
            "pending D2H window pattern versions must increase monotonically");
    }
    if (pending_bundles_.empty() && current_bundle_ &&
        version <= current_bundle_->version) {
        throw std::logic_error(
            "pending D2H window pattern version must exceed the current version");
    }
    pending_bundles_.push_back(std::move(bundle));
}

void RecurringD2HGrantController::cancel_pending(
    D2HWindowPackedProgressLayout::Version version) noexcept {
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    for (auto it = pending_bundles_.begin(); it != pending_bundles_.end(); ++it) {
        if ((*it)->version == version) {
            pending_bundles_.erase(it);
            return;
        }
    }
}

void RecurringD2HGrantController::reset_for_version_reuse() {
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    current_bundle_.reset();
    pending_bundles_.clear();
    cached_progress_.reset();
}

void RecurringD2HGrantController::cancel_pending_for_fallback() noexcept {
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    pending_bundles_.clear();
}

bool RecurringD2HGrantController::record_capacity_forced_flush(
    uint64_t count_reset_interval_periods) {
    const auto observed = progress_.load();
    bool reset_accumulated_count = false;
    if (current_bundle_ && observed.version == current_bundle_->version) {
        const auto previous = current_bundle_->last_capacity_forced_flush_counter;
        if (previous.has_value() && observed.counter >= *previous) {
            const uint64_t elapsed_periods =
                (observed.counter - *previous) / current_bundle_->matcher.period();
            reset_accumulated_count = elapsed_periods >= count_reset_interval_periods;
        }
        current_bundle_->last_capacity_forced_flush_counter = observed.counter;
    }

    return mode_.record_capacity_forced_flush(reset_accumulated_count);
}

void RecurringD2HGrantController::reconcile_progress() {
    auto observed = progress_.load();
    if (current_bundle_ && observed.version == current_bundle_->version) {
        cached_progress_ = observed;
        return;
    }

    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    observed = progress_.load();
    if (current_bundle_ && observed.version == current_bundle_->version) {
        cached_progress_ = observed;
        return;
    }
    size_t matching_index = 0;
    while (matching_index < pending_bundles_.size() &&
           pending_bundles_[matching_index]->version != observed.version) {
        ++matching_index;
    }
    if (matching_index < pending_bundles_.size()) {
        for (size_t index = 0; index < matching_index; ++index)
            pending_bundles_.pop_front();
        current_bundle_ = std::move(pending_bundles_.front());
        pending_bundles_.pop_front();
        cached_progress_ = observed;
        mode_.record_pattern_version_activation();
        std::fprintf(stderr, "[d2h_window] active version=%u\n",
                     static_cast<unsigned>(current_bundle_->version));
        std::fflush(stderr);
        return;
    }
    cached_progress_ = observed;
}

std::optional<uint64_t> RecurringD2HGrantController::estimate_window_bytes(
    const TimingObservation& timing, D2HWindowClock::time_point close) {
    if (!timing.clean_open || !timing.issue.has_value() ||
        !timing.completion.has_value()) {
        return std::nullopt;
    }
    if (!(timing.open <= *timing.issue &&
          *timing.issue < *timing.completion && *timing.completion < close)) {
        return std::nullopt;
    }

    const auto transfer_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 *timing.completion - *timing.issue)
                                 .count();
    const auto window_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               close - timing.open)
                               .count();
    if (transfer_ns <= 0 || window_ns <= 0)
        return std::nullopt;

    const unsigned __int128 numerator =
        static_cast<unsigned __int128>(timing.bytes) *
        static_cast<uint64_t>(window_ns);
    const unsigned __int128 estimate =
        numerator / static_cast<uint64_t>(transfer_ns);
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (estimate > maximum)
        return maximum;
    return static_cast<uint64_t>(estimate);
}

void RecurringD2HGrantController::observe_progress(
    const D2HWindowProgressSnapshot& progress, D2HWindowClock::time_point now) {
    if (!current_bundle_ || progress.version != current_bundle_->version)
        return;

    for (auto& state : current_bundle_->windows) {
        if (!state.timing.has_value() ||
            progress.counter < state.timing->absolute_close) {
            continue;
        }
        const auto estimate = estimate_window_bytes(*state.timing, now);
        if (estimate.has_value()) {
            state.policy->observe_timing_estimate(state.timing->bytes, *estimate);
        }
        state.timing.reset();
    }

    const auto matched = current_bundle_->matcher.match(progress.counter);
    if (!matched.has_value())
        return;
    auto& state = current_bundle_->windows.at(matched->window_index);
    if (state.timing.has_value() &&
        state.timing->occurrence == matched->occurrence) {
        return;
    }

    bool clean_open = true;
    if (state.missed_open_occurrence.has_value()) {
        if (*state.missed_open_occurrence == matched->occurrence)
            clean_open = false;
        if (*state.missed_open_occurrence <= matched->occurrence)
            state.missed_open_occurrence.reset();
    }
    state.timing = TimingObservation{
        matched->occurrence,
        matched->absolute_end,
        now,
        clean_open,
        std::nullopt,
        std::nullopt,
        0,
    };
}

std::optional<D2HWindowAdmission> RecurringD2HGrantController::poll(
    D2HWindowAvailability availability) {
    reconcile_progress();
    const auto now = now_();
    if (!current_bundle_ || !cached_progress_.has_value() ||
        cached_progress_->version != current_bundle_->version) {
        return std::nullopt;
    }

    observe_progress(*cached_progress_, now);
    const auto occurrence =
        current_bundle_->matcher.match(cached_progress_->counter);
    if (!occurrence.has_value())
        return std::nullopt;
    auto& state = current_bundle_->windows.at(occurrence->window_index);
    if (state.spent_occurrence == occurrence->occurrence)
        return std::nullopt;
    auto decision = state.policy->choose(occurrence->occurrence, availability);
    if (!decision.has_value())
        return std::nullopt;
    return D2HWindowAdmission{
        current_bundle_->version,
        *occurrence,
        *decision,
    };
}

bool RecurringD2HGrantController::commit(
    const D2HWindowAdmission& admission, uint64_t actual_bytes) {
    if (!current_bundle_ || admission.version != current_bundle_->version ||
        actual_bytes > admission.decision.byte_limit) {
        return false;
    }
    const auto observed = progress_.load();
    if (observed.version != admission.version)
        return false;
    const auto matched = current_bundle_->matcher.match(observed.counter);
    if (!matched.has_value() ||
        matched->window_index != admission.window.window_index ||
        matched->occurrence != admission.window.occurrence) {
        return false;
    }
    auto& state = current_bundle_->windows.at(admission.window.window_index);
    if (state.spent_occurrence == admission.window.occurrence)
        return false;

    state.spent_occurrence = admission.window.occurrence;
    if (actual_bytes != 0 && state.timing.has_value() &&
        state.timing->occurrence == admission.window.occurrence) {
        state.timing->issue = now_();
        state.timing->bytes = actual_bytes;
    }
    if (debug_logger_ && actual_bytes != 0) {
        debug_logger_->log_issue(
            admission.version, admission.window, observed.counter, actual_bytes,
            admission.decision.minimum_record_probe());
    }
    return true;
}

void RecurringD2HGrantController::warn_record_granularity_stall(
    const D2HWindowAdmission& admission, uint64_t actual_bytes) noexcept {
    if (warned_record_granularity_stall_)
        return;
    warned_record_granularity_stall_ = true;
    std::fprintf(
        stderr,
        "[d2h_window] warning=record-granularity-stall version=%u window=%lu "
        "max_safe=%lu min_unsafe=%lu grant=%lu bytes=%lu available=%lu\n",
        static_cast<unsigned>(admission.version),
        static_cast<unsigned long>(admission.window.window_index),
        static_cast<unsigned long>(
            admission.decision.prior_max_safe.value_or(0)),
        static_cast<unsigned long>(
            admission.decision.prior_min_unsafe.value_or(0)),
        static_cast<unsigned long>(admission.decision.byte_limit),
        static_cast<unsigned long>(actual_bytes),
        static_cast<unsigned long>(admission.decision.full_grant_bytes));
    std::fflush(stderr);
}

void RecurringD2HGrantController::warn_cross_window_overrun(
    const D2HWindowAdmission& admission, uint64_t completion_counter) noexcept {
    if (warned_cross_window_overrun_)
        return;
    warned_cross_window_overrun_ = true;
    std::fprintf(
        stderr,
        "[d2h_window] warning=cross-window-overrun version=%u window=%lu "
        "occurrence=%lu close=%lu completion_counter=%lu\n",
        static_cast<unsigned>(admission.version),
        static_cast<unsigned long>(admission.window.window_index),
        static_cast<unsigned long>(admission.window.occurrence),
        static_cast<unsigned long>(admission.window.absolute_end),
        static_cast<unsigned long>(completion_counter));
    std::fflush(stderr);
}

void RecurringD2HGrantController::complete(
    const D2HWindowAdmission& admission, uint64_t actual_bytes) {
    const auto observed = progress_.load();
    const auto now = now_();
    if (observed.version != admission.version) {
        if (debug_logger_ && actual_bytes != 0) {
            debug_logger_->log_completion(
                admission.version, observed.counter,
                D2HWindowCompletionResult::VERSION_CHANGE_DISCARD);
        }
        if (current_bundle_ && current_bundle_->version == admission.version) {
            current_bundle_->windows.at(admission.window.window_index)
                .timing.reset();
        }
        return;
    }

    const bool overran = observed.counter >= admission.window.absolute_end;
    if (debug_logger_ && actual_bytes != 0) {
        debug_logger_->log_completion(
            admission.version, observed.counter,
            overran ? D2HWindowCompletionResult::OVERRUN
                    : D2HWindowCompletionResult::SUCCESS);
    }
    if (actual_bytes == 0 || !current_bundle_ ||
        current_bundle_->version != admission.version) {
        return;
    }

    auto& state = current_bundle_->windows.at(admission.window.window_index);
    const auto policy_observation = state.policy->observe(D2HWindowAttempt{
        admission.window.occurrence,
        actual_bytes,
        overran,
        admission.decision,
    });
    if (policy_observation.entered_record_granularity_stall)
        warn_record_granularity_stall(admission, actual_bytes);

    if (!overran) {
        if (state.timing.has_value() &&
            state.timing->occurrence == admission.window.occurrence &&
            state.timing->issue.has_value()) {
            state.timing->completion = now;
        }
        return;
    }

    state.timing.reset();
    const uint64_t next_open =
        current_bundle_->matcher.next_window_begin(admission.window);
    if (observed.counter < next_open)
        return;

    warn_cross_window_overrun(admission, observed.counter);
    const auto crossed = current_bundle_->matcher.match(observed.counter);
    if (crossed.has_value() &&
        (crossed->window_index != admission.window.window_index ||
         crossed->occurrence != admission.window.occurrence)) {
        current_bundle_->windows.at(crossed->window_index)
            .missed_open_occurrence = crossed->occurrence;
    }
}

}  // namespace ring
