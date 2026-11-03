// ring/pipeline_telemetry.h -- host-only accounting of the drain and P2P
// stages.  No device reads and no added synchronization.
#pragma once
#include "host_wait_counter.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace ring {

class DrainTelemetry {
public:
    // Wait for staging space under the caller's lock.  An uncontended check
    // reads no clock.
    template <class Predicate>
    void wait_for_space(std::condition_variable& cv,
                        std::unique_lock<std::mutex>& lock, Predicate ready) {
        if (!ready()) staging_wait.measure([&] { cv.wait(lock, ready); });
    }
    // One task's latency from first seen published to committed to the P2P
    // queue (after its D2H copy completed).
    void record_commit(int64_t ready_us, int64_t committed_us) {
        if (ready_us <= 0 || committed_us < ready_us) {
            ready_commit_invalid.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        ready_commit_us.fetch_add(static_cast<uint64_t>(committed_us - ready_us),
                                  std::memory_order_relaxed);
        ready_commit_tasks.fetch_add(1, std::memory_order_relaxed);
    }
    ring_py::HostWaitCounter staging_wait;
    std::atomic<uint64_t> ready_commit_tasks{0}, ready_commit_us{0},
                          ready_commit_invalid{0};
    std::atomic<uint64_t> d2h_copy_errors{0}, d2h_sync_errors{0};
};

class PageableCopyTelemetry {
public:
    // Staging -> pageable copy of one task (one or two spans when the staging
    // ring wraps; counted as one operation).  Destination allocation and the
    // staging release are outside the measurement.
    void copy(uint8_t* dst, const uint8_t* a, uint64_t n,
              const uint8_t* b, uint64_t m) {
        if (n == 0 && m == 0) return;
        elapsed.measure([&] {
            if (n) std::memcpy(dst, a, n);
            if (m) std::memcpy(dst + n, b, m);
        });
        bytes.fetch_add(n + m, std::memory_order_relaxed);
    }
    ring_py::HostWaitCounter elapsed;
    std::atomic<uint64_t> bytes{0};
};

}  // namespace ring
