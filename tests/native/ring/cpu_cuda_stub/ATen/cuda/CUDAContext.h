// CPU-only stand-in for <ATen/cuda/CUDAContext.h>, enough for
// ring/ring_torch_op.cpp: the current-stream query its producer ops make.
// Placed ahead of the Torch include directories, it shadows the real header,
// so the ops compile against CPU-only Torch without the CUDA toolkit.  No
// CUDA library is linked or initialized.
#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>

namespace at::cuda {

// Device index the last getCurrentCUDAStream() asked for.
inline c10::DeviceIndex fake_last_stream_device = -1;

struct FakeCUDAStream {
    cudaStream_t stream() const { return reinterpret_cast<cudaStream_t>(0x5); }
};

inline FakeCUDAStream getCurrentCUDAStream(c10::DeviceIndex device = -1) {
    fake_last_stream_device = device;
    return {};
}

}  // namespace at::cuda
