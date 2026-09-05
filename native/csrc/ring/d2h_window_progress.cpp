#include "d2h_window_progress.h"
#include "d2h_window_marker.h"

#include <stdexcept>
#include <string>

namespace ring {
namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return;
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(error));
}

}  // namespace

PackedVersionCounterProgressSource::PackedVersionCounterProgressSource(int owner_device)
    : owner_device_(owner_device) {
    check_cuda(cudaSetDevice(owner_device_), "cudaSetDevice");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_packed_progress_),
                              sizeof(*device_packed_progress_)),
                   "cudaMalloc device D2H window progress");
        check_cuda(
            cudaMallocManaged(reinterpret_cast<void**>(&cpu_visible_packed_progress_),
                              sizeof(*cpu_visible_packed_progress_)),
            "cudaMallocManaged CPU-visible D2H window progress");
        *cpu_visible_packed_progress_ = 0;
        check_cuda(
            cudaMemset(device_packed_progress_, 0, sizeof(*device_packed_progress_)),
            "initialize device D2H window progress");
        check_cuda(cudaMemAdvise(cpu_visible_packed_progress_,
                                 sizeof(*cpu_visible_packed_progress_),
                                 cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId),
                   "set CPU-visible D2H window progress preferred location");
        check_cuda(cudaMemAdvise(cpu_visible_packed_progress_,
                                 sizeof(*cpu_visible_packed_progress_),
                                 cudaMemAdviseSetAccessedBy, owner_device_),
                   "make CPU-visible D2H window progress GPU-accessible");
        check_cuda(cudaMemPrefetchAsync(cpu_visible_packed_progress_,
                                        sizeof(*cpu_visible_packed_progress_),
                                        cudaCpuDeviceId, nullptr),
                   "prefetch CPU-visible D2H window progress");
        check_cuda(cudaDeviceSynchronize(),
                   "initialize D2H window progress allocations");

        auto options =
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, owner_device_);
        device_view_ = at::from_blob(device_packed_progress_, {1}, options);
        cpu_visible_view_ = at::from_blob(cpu_visible_packed_progress_, {1}, options);
    } catch (...) {
        if (cpu_visible_packed_progress_) {
            cudaFree(cpu_visible_packed_progress_);
            cpu_visible_packed_progress_ = nullptr;
        }
        if (device_packed_progress_) {
            cudaFree(device_packed_progress_);
            device_packed_progress_ = nullptr;
        }
        throw;
    }
}

PackedVersionCounterProgressSource::~PackedVersionCounterProgressSource() {
    device_view_ = at::Tensor();
    cpu_visible_view_ = at::Tensor();
    if (cpu_visible_packed_progress_)
        cudaFree(cpu_visible_packed_progress_);
    if (device_packed_progress_)
        cudaFree(device_packed_progress_);
}

D2HWindowProgressSnapshot PackedVersionCounterProgressSource::load() const noexcept {
    const auto packed = __atomic_load_n(cpu_visible_packed_progress_, __ATOMIC_RELAXED);
    return D2HWindowProgressSnapshot{
        D2HWindowPackedProgressLayout::version(packed),
        D2HWindowPackedProgressLayout::counter(packed),
    };
}

D2HWindowProgressState PackedVersionCounterProgressSource::state() const {
    return D2HWindowProgressState{device_view_, cpu_visible_view_};
}

void PackedVersionCounterProgressSource::enqueue_reset(
    D2HWindowPackedProgressLayout::Version version,
    D2HWindowPackedProgressLayout::Counter counter, cudaStream_t stream) {
    launch_d2h_window_progress_reset(device_packed_progress_,
                                     cpu_visible_packed_progress_, version, counter,
                                     stream);
}

}  // namespace ring
