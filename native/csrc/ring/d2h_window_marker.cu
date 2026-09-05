#include "d2h_window_marker.h"

#include <cuda/atomic>
#include <stdexcept>
#include <string>

namespace ring {
namespace {

__global__ void advance_d2h_window_boundary_kernel(
    D2HWindowPackedProgressLayout::Word* device_packed_progress,
    D2HWindowPackedProgressLayout::Word* cpu_visible_packed_progress) {
    const auto next = *device_packed_progress + 1;
    *device_packed_progress = next;
    cuda::atomic_ref<D2HWindowPackedProgressLayout::Word, cuda::thread_scope_system>
        published(*cpu_visible_packed_progress);
    published.store(next, cuda::memory_order_relaxed);
}

__global__ void reset_d2h_window_progress_kernel(
    D2HWindowPackedProgressLayout::Word* device_packed_progress,
    D2HWindowPackedProgressLayout::Word* cpu_visible_packed_progress,
    D2HWindowPackedProgressLayout::Version version,
    D2HWindowPackedProgressLayout::Counter counter) {
    const auto packed = D2HWindowPackedProgressLayout::pack(version, counter);
    *device_packed_progress = packed;
    cuda::atomic_ref<D2HWindowPackedProgressLayout::Word, cuda::thread_scope_system>
        published(*cpu_visible_packed_progress);
    published.store(packed, cuda::memory_order_relaxed);
}

void check_launch(const char* operation) {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess)
        return;
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(error));
}

}  // namespace

void launch_d2h_window_boundary(
    D2HWindowPackedProgressLayout::Word* device_packed_progress,
    D2HWindowPackedProgressLayout::Word* cpu_visible_packed_progress,
    cudaStream_t stream) {
    advance_d2h_window_boundary_kernel<<<1, 1, 0, stream>>>(
        device_packed_progress, cpu_visible_packed_progress);
    check_launch("advance D2H window boundary kernel launch");
}

void launch_d2h_window_progress_reset(
    D2HWindowPackedProgressLayout::Word* device_packed_progress,
    D2HWindowPackedProgressLayout::Word* cpu_visible_packed_progress,
    D2HWindowPackedProgressLayout::Version version,
    D2HWindowPackedProgressLayout::Counter counter, cudaStream_t stream) {
    reset_d2h_window_progress_kernel<<<1, 1, 0, stream>>>(
        device_packed_progress, cpu_visible_packed_progress, version, counter);
    check_launch("reset D2H window progress kernel launch");
}

}  // namespace ring
