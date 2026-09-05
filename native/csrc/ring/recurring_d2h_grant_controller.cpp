#include "recurring_d2h_grant_controller.h"

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ring {

RecurringD2HGrantController::RecurringD2HGrantController(
    D2HWindowProgressSource& progress, D2HWindowModeController& mode,
    D2HWindowGrantPolicyFactory policy_factory, D2HWindowDebugLogger* debug_logger)
    : progress_(progress), mode_(mode), policy_factory_(std::move(policy_factory)),
      debug_logger_(debug_logger) {
    if (!policy_factory_) {
        throw std::invalid_argument("D2H window policy factory is required");
    }
}

std::unique_ptr<RecurringD2HGrantController::VersionBundle>
RecurringD2HGrantController::make_bundle(
    D2HWindowPackedProgressLayout::Version version, uint64_t period,
    const std::vector<D2HWindowOffset>& windows) const {
    auto bundle = std::make_unique<VersionBundle>(VersionBundle{
        version,
        D2HWindowPatternMatcher(period, windows),
        {},
    });
    bundle->windows.reserve(windows.size());
    for (size_t index = 0; index < windows.size(); ++index) {
        bundle->windows.push_back(WindowState{
            policy_factory_(),
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
    if (pending_bundle_) {
        throw std::logic_error("a D2H window pattern version is already pending");
    }
    pending_bundle_ = std::move(bundle);
}

bool RecurringD2HGrantController::has_pending() const {
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    return static_cast<bool>(pending_bundle_);
}

void RecurringD2HGrantController::cancel_pending(
    D2HWindowPackedProgressLayout::Version version) noexcept {
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    if (pending_bundle_ && pending_bundle_->version == version) {
        pending_bundle_.reset();
    }
}

void RecurringD2HGrantController::reset_for_version_reuse() {
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    current_bundle_.reset();
    pending_bundle_.reset();
    cached_progress_.reset();
}

void RecurringD2HGrantController::cancel_pending_for_fallback() noexcept {
    std::lock_guard<std::mutex> lock(bundle_control_mu_);
    pending_bundle_.reset();
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
    if (pending_bundle_ && observed.version == pending_bundle_->version) {
        current_bundle_ = std::move(pending_bundle_);
        cached_progress_ = observed;
        mode_.record_pattern_version_activation();
        std::fprintf(stderr, "[d2h_window] active version=%u\n",
                     static_cast<unsigned>(current_bundle_->version));
        std::fflush(stderr);
        return;
    }
    cached_progress_ = observed;
}

std::optional<D2HWindowAdmission>
RecurringD2HGrantController::consider(D2HWindowAvailability availability) {
    if (!current_bundle_ || !cached_progress_.has_value() ||
        cached_progress_->version != current_bundle_->version) {
        return std::nullopt;
    }
    auto occurrence = current_bundle_->matcher.match(cached_progress_->counter);
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
        decision->byte_limit,
        decision->minimum_record_probe,
    };
}

bool RecurringD2HGrantController::commit(const D2HWindowAdmission& admission,
                                         uint64_t actual_bytes) {
    if (!current_bundle_ || admission.version != current_bundle_->version ||
        actual_bytes > admission.byte_limit) {
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
    if (debug_logger_ && actual_bytes != 0) {
        debug_logger_->log_issue(admission.version, admission.window, observed.counter,
                                 actual_bytes, admission.minimum_record_probe);
    }
    return true;
}

void RecurringD2HGrantController::complete(const D2HWindowAdmission& admission,
                                           uint64_t actual_bytes) {
    const auto observed = progress_.load();
    if (observed.version != admission.version) {
        if (debug_logger_ && actual_bytes != 0) {
            debug_logger_->log_completion(
                admission.version, observed.counter,
                D2HWindowCompletionResult::VERSION_CHANGE_DISCARD);
        }
        return;
    }
    const bool overran = observed.counter >= admission.window.absolute_end;
    if (debug_logger_ && actual_bytes != 0) {
        debug_logger_->log_completion(admission.version, observed.counter,
                                      overran ? D2HWindowCompletionResult::OVERRUN
                                              : D2HWindowCompletionResult::SUCCESS);
    }
    if (actual_bytes == 0 || !current_bundle_ ||
        current_bundle_->version != admission.version) {
        return;
    }
    current_bundle_->windows.at(admission.window.window_index)
        .policy->observe(D2HWindowAttempt{
            admission.window.occurrence,
            actual_bytes,
            overran,
            admission.minimum_record_probe,
        });
}

}  // namespace ring
