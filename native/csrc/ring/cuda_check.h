#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ring {

// Shared by the D2H window translation units so a CUDA failure reports the
// same "<operation> failed: <reason>" shape wherever it is raised.
inline void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return;
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(error));
}

// Kernel launches report asynchronously, so their errors surface through
// cudaGetLastError() rather than a return value.
inline void check_cuda_launch(const char* operation) {
    check_cuda(cudaGetLastError(), operation);
}

}  // namespace ring
