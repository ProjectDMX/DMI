#include "d2h_window_subsystem.h"

#include "d2h_window_pattern.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace ring {
namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return;
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(error));
}

void validate_configuration(const RecurringD2HWindowConfig& config) {
    if (config.history_size < 2) {
        throw std::invalid_argument("recurring D2H window history_size must be >= 2");
    }
    if (config.minimum_record_probe_retry_interval_occurrences == 0) {
        throw std::invalid_argument(
            "recurring D2H window probe retry interval must be > 0");
    }
    if (config.capacity_flush_fallback_threshold == 0) {
        throw std::invalid_argument(
            "recurring D2H window capacity flush fallback threshold must be > 0");
    }
    switch (config.progress) {
    case D2HWindowProgressKind::PACKED_VERSION_COUNTER:
        break;
    default:
        throw std::invalid_argument("unknown D2H window progress kind");
    }
    switch (config.grant_policy) {
    case D2HWindowGrantPolicyKind::LAST_K_ADAPTIVE:
        break;
    default:
        throw std::invalid_argument("unknown D2H window grant policy kind");
    }
}

}  // namespace

RecurringD2HWindowSubsystem::RecurringD2HWindowSubsystem(
    RecurringD2HWindowConfig config, int owner_device)
    : config_(config) {
    validate_configuration(config_);
    switch (config_.progress) {
    case D2HWindowProgressKind::PACKED_VERSION_COUNTER:
        progress_ = std::make_unique<PackedVersionCounterProgressSource>(owner_device);
        break;
    }
    mode_controller_ = std::make_unique<D2HWindowModeController>(
        config_.capacity_flush_fallback_threshold);
    if (config_.debug_enabled) {
        debug_logger_ = std::make_unique<D2HWindowDebugLogger>();
    }
    D2HWindowGrantPolicyFactory policy_factory = [config = config_] {
        return make_d2h_window_grant_policy(config.grant_policy, config);
    };
    grant_controller_ = std::make_unique<RecurringD2HGrantController>(
        *progress_, *mode_controller_, std::move(policy_factory), debug_logger_.get());
}

void RecurringD2HWindowSubsystem::define_with_new_version_locked(
    D2HWindowPackedProgressLayout::Version version, uint64_t period,
    const std::vector<D2HWindowOffset>& windows, uint64_t initial_counter,
    cudaStream_t framework_stream) {
    grant_controller_->install_pending(version, period, windows);
    try {
        progress_->enqueue_reset(version, initial_counter, framework_stream);
    } catch (...) {
        grant_controller_->cancel_pending(version);
        throw;
    }
    last_allocated_version_ = version;
}

void RecurringD2HWindowSubsystem::define_pattern(
    uint64_t period, std::vector<D2HWindowOffset> windows,
    std::optional<uint64_t> initial_counter, cudaStream_t framework_stream,
    DrainPauseControl& drain_pause) {
    const uint64_t counter = initial_counter.value_or(0);
    if (counter >= D2HWindowPackedProgressLayout::kCounterLimit) {
        throw std::invalid_argument(
            "D2H window initial counter is outside the packed counter domain");
    }
    D2HWindowPatternMatcher validation(period, windows);

    std::unique_lock<std::mutex> lock(control_mu_);
    if (version_reuse_in_progress_) {
        throw std::logic_error("D2H window version reuse is already in progress");
    }
    if (mode_controller_->mode() == D2HWindowMode::ENABLED_FALLBACK) {
        throw std::logic_error(
            "D2H window pattern cannot be defined after terminal fallback");
    }
    if (grant_controller_->has_pending()) {
        throw std::logic_error("a D2H window pattern version is already pending");
    }
    if (last_allocated_version_ < D2HWindowPackedProgressLayout::kMaxPatternVersion) {
        const auto version = static_cast<D2HWindowPackedProgressLayout::Version>(
            last_allocated_version_ + 1);
        define_with_new_version_locked(version, period, windows, counter,
                                       framework_stream);
        return;
    }

    version_reuse_in_progress_ = true;
    lock.unlock();
    define_after_version_exhaustion(period, windows, counter, framework_stream,
                                    drain_pause);
}

void RecurringD2HWindowSubsystem::define_after_version_exhaustion(
    uint64_t period, const std::vector<D2HWindowOffset>& windows,
    uint64_t initial_counter, cudaStream_t framework_stream,
    DrainPauseControl& drain_pause) {
    std::optional<DrainPauseToken> pause_token;
    try {
        check_cuda(cudaStreamSynchronize(framework_stream),
                   "synchronize framework stream for D2H window version reuse");
        pause_token = drain_pause.pause_after_flush_and_wait();

        std::unique_lock<std::mutex> lock(control_mu_);
        if (mode_controller_->mode() == D2HWindowMode::ENABLED_FALLBACK) {
            version_reuse_in_progress_ = false;
            lock.unlock();
            drain_pause.resume(*pause_token);
            pause_token.reset();
            throw std::logic_error(
                "D2H window version reuse lost to terminal fallback");
        }

        grant_controller_->reset_for_version_reuse();
        mode_controller_->reset_for_version_reuse();
        progress_->enqueue_reset(D2HWindowPackedProgressLayout::kNoPatternVersion, 0,
                                 framework_stream);
        last_allocated_version_ = D2HWindowPackedProgressLayout::kNoPatternVersion;
        define_with_new_version_locked(
            D2HWindowPackedProgressLayout::kFirstPatternVersion, period, windows,
            initial_counter, framework_stream);
        version_reuse_in_progress_ = false;
        lock.unlock();
        drain_pause.resume(*pause_token);
        pause_token.reset();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(control_mu_);
            version_reuse_in_progress_ = false;
        }
        if (pause_token.has_value())
            drain_pause.resume(*pause_token);
        throw;
    }
}

void RecurringD2HWindowSubsystem::record_capacity_forced_flush() {
    std::lock_guard<std::mutex> lock(control_mu_);
    if (!mode_controller_->record_capacity_forced_flush())
        return;
    grant_controller_->cancel_pending_for_fallback();
    const auto state = mode_controller_->snapshot();
    std::fprintf(stderr, "[d2h_window] terminal fallback count=%lu threshold=%lu\n",
                 static_cast<unsigned long>(state.capacity_forced_flush_count),
                 static_cast<unsigned long>(state.capacity_flush_fallback_threshold));
    std::fflush(stderr);
}

std::unique_ptr<RecurringD2HWindowSubsystem>
make_recurring_d2h_subsystem(const RecurringD2HWindowConfig& config, int owner_device) {
    if (!config.enabled)
        return nullptr;
    return std::make_unique<RecurringD2HWindowSubsystem>(config, owner_device);
}

}  // namespace ring
