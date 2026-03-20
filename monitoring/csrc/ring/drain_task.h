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
};

}  // namespace ring
