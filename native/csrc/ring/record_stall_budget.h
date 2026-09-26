// How long a record reservation past a spent step stall budget may wait for
// the drain before the ring is failed.
//
// Plain C++ (no ATen/CUDA) so the CPU tests can check it.

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>

namespace ring {

// Once a step's stall budget is spent the sink is out of the path, but the
// drain still has to move what the ring holds -- device ring to pinned
// staging, then the worker takes each payload off the staging and discards
// it -- before the reservation fits.  The reservation waits for that at most
// the sink's admission bound plus this grace: a fixed 2 s, plus the bytes the
// payload ring and its pinned staging can hold, at a conservative 1 GB/s,
// which is one byte per nanosecond.  With the engine's default 4 GiB ring
// and 4 GiB staging that is about 10.6 s.
inline constexpr std::chrono::milliseconds kRecordDrainGraceBase{2000};
inline constexpr uint64_t kRecordDrainGraceBytesPerSecond = 1'000'000'000;
static_assert(kRecordDrainGraceBytesPerSecond == 1'000'000'000,
              "record_drain_grace converts one byte to one nanosecond");

inline std::chrono::nanoseconds record_drain_grace(
    uint64_t payload_ring_bytes, uint64_t staging_bytes) {
    // Saturate far below the clock's range, so that a deadline adding this
    // grace to steady_clock::now() cannot overflow.
    constexpr uint64_t kCap =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / 4);
    const uint64_t base = static_cast<uint64_t>(
        std::chrono::nanoseconds(kRecordDrainGraceBase).count());
    uint64_t total = base;
    for (const uint64_t bytes : {payload_ring_bytes, staging_bytes}) {
        total = bytes > kCap - total ? kCap : total + bytes;
    }
    return std::chrono::nanoseconds(static_cast<int64_t>(total));
}

}  // namespace ring
