// ring/drain_task.h — Task descriptor passed from drain thread to p2p thread.

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
    uint64_t alloc_bytes = 0;       // staging bytes to release (0 for large/drop)

    // Metadata from TaskEntry
    uint64_t logical_task_id = 0;
    uint64_t chunk_offset_bytes = 0;
    uint64_t tensor_total_bytes = 0;
    uint32_t hook_type = 0;
    uint32_t hook_id   = 0;
    uint32_t flags     = 0;         // IS_FIRST, IS_LAST, IS_DROP, LARGE_TENSOR

    // Large tensor bypass: tensor already in pageable memory.
    // Only set when TASK_FLAG_LARGE_TENSOR is set in flags.
    at::Tensor large_tensor;
};

}  // namespace ring
