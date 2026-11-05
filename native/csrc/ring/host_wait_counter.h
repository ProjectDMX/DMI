// ring/host_wait_counter.h -- host elapsed-time counters for waits the ring
// already performs.  No CUDA work, no device memory, no allocation, no
// polling: a counter only times a call that would run anyway.  Snapshot
// fields are individually monotonic, not one atomic multi-field read while a
// measured call is completing.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <type_traits>

namespace ring_py {

struct HostWaitSnapshot {
    uint64_t calls{0};
    uint64_t ns{0};
    uint64_t errors{0};
};

class HostWaitCounter {
public:
    // Time f().  A non-void result is a CUDA status: nonzero counts as an
    // error.  An exception counts as an error and propagates.
    template <class F> auto measure(F&& f) -> decltype(f()) {
        const auto start = std::chrono::steady_clock::now();
        try {
            if constexpr (std::is_void_v<decltype(f())>) {
                f();
                record(start, false);
            } else {
                auto result = f();
                record(start, result != 0);
                return result;
            }
        } catch (...) {
            record(start, true);
            throw;
        }
    }
    HostWaitSnapshot snapshot() const {
        return {calls_.load(std::memory_order_relaxed),
                ns_.load(std::memory_order_relaxed),
                errors_.load(std::memory_order_relaxed)};
    }

private:
    void record(std::chrono::steady_clock::time_point start, bool error) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        ns_.fetch_add(static_cast<uint64_t>(elapsed), std::memory_order_relaxed);
        if (error) errors_.fetch_add(1, std::memory_order_relaxed);
        calls_.fetch_add(1, std::memory_order_relaxed);
    }
    std::atomic<uint64_t> calls_{0}, ns_{0}, errors_{0};
};

}  // namespace ring_py
