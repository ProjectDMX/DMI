#pragma once

#include "d2h_window_debug.h"
#include "d2h_window_grant_policy.h"
#include "d2h_window_mode.h"
#include "d2h_window_pattern.h"
#include "d2h_window_progress.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace ring {

using D2HWindowGrantPolicyFactory =
    std::function<std::unique_ptr<D2HWindowGrantPolicy>()>;
using D2HWindowClock = std::chrono::steady_clock;
using D2HWindowNowFunction = std::function<D2HWindowClock::time_point()>;

struct D2HWindowAdmission {
    D2HWindowPackedProgressLayout::Version version{
        D2HWindowPackedProgressLayout::kNoPatternVersion};
    D2HWindowOccurrence window;
    D2HWindowGrantDecision decision;
};

class D2HGrantController {
  public:
    virtual ~D2HGrantController() = default;
    virtual std::optional<D2HWindowAdmission>
    poll(D2HWindowAvailability availability) = 0;
    virtual bool commit(const D2HWindowAdmission& admission,
                        uint64_t actual_bytes) = 0;
    virtual void complete(const D2HWindowAdmission& admission,
                          uint64_t actual_bytes) = 0;
};

class RecurringD2HGrantController final : public D2HGrantController {
  public:
    RecurringD2HGrantController(D2HWindowProgressSource& progress,
                                D2HWindowModeController& mode,
                                D2HWindowGrantPolicyFactory policy_factory,
                                D2HWindowDebugLogger* debug_logger,
                                D2HWindowNowFunction now = {});

    void install_pending(D2HWindowPackedProgressLayout::Version version,
                         uint64_t period,
                         const std::vector<D2HWindowOffset>& windows);
    void cancel_pending(D2HWindowPackedProgressLayout::Version version) noexcept;
    void reset_for_version_reuse();
    void cancel_pending_for_fallback() noexcept;
    bool record_capacity_forced_flush(uint64_t count_reset_interval_periods);

    std::optional<D2HWindowAdmission>
    poll(D2HWindowAvailability availability) override;
    bool commit(const D2HWindowAdmission& admission,
                uint64_t actual_bytes) override;
    void complete(const D2HWindowAdmission& admission,
                  uint64_t actual_bytes) override;

  private:
    struct TimingObservation {
        uint64_t occurrence{0};
        uint64_t absolute_close{0};
        D2HWindowClock::time_point open;
        bool clean_open{true};
        std::optional<D2HWindowClock::time_point> issue;
        std::optional<D2HWindowClock::time_point> completion;
        uint64_t bytes{0};
    };

    struct WindowState {
        std::unique_ptr<D2HWindowGrantPolicy> policy;
        std::optional<uint64_t> spent_occurrence;
        std::optional<TimingObservation> timing;
        std::optional<uint64_t> missed_open_occurrence;
    };

    struct VersionBundle {
        D2HWindowPackedProgressLayout::Version version;
        D2HWindowPatternMatcher matcher;
        std::vector<WindowState> windows;
        std::optional<uint64_t> last_capacity_forced_flush_counter;
    };

    std::unique_ptr<VersionBundle>
    make_bundle(D2HWindowPackedProgressLayout::Version version, uint64_t period,
                const std::vector<D2HWindowOffset>& windows) const;
    void reconcile_progress();
    void observe_progress(const D2HWindowProgressSnapshot& progress,
                          D2HWindowClock::time_point now);
    static std::optional<uint64_t>
    estimate_window_bytes(const TimingObservation& timing,
                          D2HWindowClock::time_point close);
    void warn_record_granularity_stall(const D2HWindowAdmission& admission,
                                       uint64_t actual_bytes) noexcept;
    void warn_cross_window_overrun(const D2HWindowAdmission& admission,
                                   uint64_t completion_counter) noexcept;

    D2HWindowProgressSource& progress_;
    D2HWindowModeController& mode_;
    D2HWindowGrantPolicyFactory policy_factory_;
    D2HWindowDebugLogger* debug_logger_{nullptr};
    D2HWindowNowFunction now_;

    mutable std::mutex bundle_control_mu_;
    std::unique_ptr<VersionBundle> current_bundle_;
    std::deque<std::unique_ptr<VersionBundle>> pending_bundles_;
    std::optional<D2HWindowProgressSnapshot> cached_progress_;
    bool warned_record_granularity_stall_{false};
    bool warned_cross_window_overrun_{false};
};

}  // namespace ring
