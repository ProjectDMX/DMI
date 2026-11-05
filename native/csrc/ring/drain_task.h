// ring/drain_task.h -- Task descriptor passed from drain thread to p2p thread.

#pragma once
#include <ATen/ATen.h>
#include <cstdint>

namespace ring {

struct DrainTask {
    // Pinned staging data (1-2 segments for staging ring wrap)
    uint8_t* data_ptr1 = nullptr;
    uint64_t data_len1 = 0;
    uint8_t* data_ptr2 = nullptr;   // non-null only if staging ring wraps
    uint64_t data_len2 = 0;
    uint64_t alloc_bytes = 0;       // staging bytes to release (0 for cpu-direct)

    // Tensor size
    uint64_t tensor_total_bytes = 0;

    // CPU-direct path: tensor already in pageable CPU memory.
    // When defined(), skip staging copy in p2p thread.
    at::Tensor cpu_paged_tensor;

    // steady_clock microseconds when the drain thread first saw this task's
    // publication on the host: the capture exists and is host-visible.
    // Stamped before the flush wait and the D2H copy, so a consumer's
    // (now - capture_ts_us) includes the drain-flush latency the drain_flush
    // settings control.  0 = unstamped.
    int64_t capture_ts_us = 0;
};

}  // namespace ring
