#pragma once

#include "d2h_window_progress_layout.h"

#include <cuda_runtime.h>

namespace ring {

void launch_d2h_window_boundary(
    D2HWindowPackedProgressLayout::Word* device_packed_progress,
    D2HWindowPackedProgressLayout::Word* cpu_visible_packed_progress,
    cudaStream_t stream);

void launch_d2h_window_progress_reset(
    D2HWindowPackedProgressLayout::Word* device_packed_progress,
    D2HWindowPackedProgressLayout::Word* cpu_visible_packed_progress,
    D2HWindowPackedProgressLayout::Version version,
    D2HWindowPackedProgressLayout::Counter counter, cudaStream_t stream);

}  // namespace ring
