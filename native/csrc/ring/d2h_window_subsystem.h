#pragma once

#include "d2h_window_config.h"
#include "d2h_window_debug.h"
#include "d2h_window_mode.h"
#include "d2h_window_pause.h"
#include "d2h_window_progress.h"
#include "recurring_d2h_grant_controller.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace ring {

class RecurringD2HWindowSubsystem {
  public:
    RecurringD2HWindowSubsystem(RecurringD2HWindowConfig config, int owner_device);

    void define_pattern(uint64_t period, std::vector<D2HWindowOffset> windows,
                        std::optional<uint64_t> initial_counter,
                        cudaStream_t framework_stream, DrainPauseControl& drain_pause);

    void record_capacity_forced_flush();

    D2HGrantController& grant_controller() noexcept { return *grant_controller_; }
    D2HWindowModeController& mode_controller() noexcept { return *mode_controller_; }
    D2HWindowProgressState progress_state() const { return progress_->state(); }
    D2HWindowRuntimeSnapshot snapshot() const noexcept {
        return mode_controller_->snapshot();
    }

  private:
    void define_with_new_version_locked(D2HWindowPackedProgressLayout::Version version,
                                        uint64_t period,
                                        const std::vector<D2HWindowOffset>& windows,
                                        uint64_t initial_counter,
                                        cudaStream_t framework_stream);
    void define_after_version_exhaustion(uint64_t period,
                                         const std::vector<D2HWindowOffset>& windows,
                                         uint64_t initial_counter,
                                         cudaStream_t framework_stream,
                                         DrainPauseControl& drain_pause);

    RecurringD2HWindowConfig config_;
    std::unique_ptr<D2HWindowProgressSource> progress_;
    std::unique_ptr<D2HWindowModeController> mode_controller_;
    std::unique_ptr<D2HWindowDebugLogger> debug_logger_;
    std::unique_ptr<RecurringD2HGrantController> grant_controller_;

    std::mutex control_mu_;
    bool version_reuse_in_progress_{false};
    D2HWindowPackedProgressLayout::Version last_allocated_version_{
        D2HWindowPackedProgressLayout::kNoPatternVersion};
};

std::unique_ptr<RecurringD2HWindowSubsystem>
make_recurring_d2h_subsystem(const RecurringD2HWindowConfig& config, int owner_device);

}  // namespace ring
