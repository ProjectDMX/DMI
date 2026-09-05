#pragma once

#include "d2h_window_debug.h"
#include "d2h_window_grant_policy.h"
#include "d2h_window_mode.h"
#include "d2h_window_pattern.h"
#include "d2h_window_progress.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace ring {

using D2HWindowGrantPolicyFactory =
    std::function<std::unique_ptr<D2HWindowGrantPolicy>()>;

struct D2HWindowAdmission {
    D2HWindowPackedProgressLayout::Version version{
        D2HWindowPackedProgressLayout::kNoPatternVersion};
    D2HWindowOccurrence window;
    uint64_t byte_limit{0};
    bool minimum_record_probe{false};
};

class D2HGrantController {
  public:
    virtual ~D2HGrantController() = default;
    virtual void reconcile_progress() = 0;
    virtual std::optional<D2HWindowAdmission>
    consider(D2HWindowAvailability availability) = 0;
    virtual bool commit(const D2HWindowAdmission& admission, uint64_t actual_bytes) = 0;
    virtual void complete(const D2HWindowAdmission& admission,
                          uint64_t actual_bytes) = 0;
};

class RecurringD2HGrantController final : public D2HGrantController {
  public:
    RecurringD2HGrantController(D2HWindowProgressSource& progress,
                                D2HWindowModeController& mode,
                                D2HWindowGrantPolicyFactory policy_factory,
                                D2HWindowDebugLogger* debug_logger);

    void install_pending(D2HWindowPackedProgressLayout::Version version,
                         uint64_t period, const std::vector<D2HWindowOffset>& windows);
    bool has_pending() const;
    void cancel_pending(D2HWindowPackedProgressLayout::Version version) noexcept;
    void reset_for_version_reuse();
    void cancel_pending_for_fallback() noexcept;

    void reconcile_progress() override;
    std::optional<D2HWindowAdmission>
    consider(D2HWindowAvailability availability) override;
    bool commit(const D2HWindowAdmission& admission, uint64_t actual_bytes) override;
    void complete(const D2HWindowAdmission& admission, uint64_t actual_bytes) override;

  private:
    struct WindowState {
        std::unique_ptr<D2HWindowGrantPolicy> policy;
        std::optional<uint64_t> spent_occurrence;
    };

    struct VersionBundle {
        D2HWindowPackedProgressLayout::Version version;
        D2HWindowPatternMatcher matcher;
        std::vector<WindowState> windows;
    };

    std::unique_ptr<VersionBundle>
    make_bundle(D2HWindowPackedProgressLayout::Version version, uint64_t period,
                const std::vector<D2HWindowOffset>& windows) const;

    D2HWindowProgressSource& progress_;
    D2HWindowModeController& mode_;
    D2HWindowGrantPolicyFactory policy_factory_;
    D2HWindowDebugLogger* debug_logger_{nullptr};

    mutable std::mutex bundle_control_mu_;
    std::unique_ptr<VersionBundle> current_bundle_;
    std::unique_ptr<VersionBundle> pending_bundle_;
    std::optional<D2HWindowProgressSnapshot> cached_progress_;
};

}  // namespace ring
