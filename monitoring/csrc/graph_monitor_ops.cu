#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace vllm {

struct __align__(128) TensorMetadata {
  uint64_t data_ptr;
  int64_t shape[4];
  int64_t stride[4];
  int32_t ndim;
  int32_t dtype_id;
  int32_t device_idx;
  char padding[44];
};

static_assert(sizeof(TensorMetadata) == 128,
              "TensorMetadata must be 128 bytes to match Python parser.");

__global__ void record_metadata_kernel(
    TensorMetadata* buffer, int slot_id, uint64_t data_ptr,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int64_t st0, int64_t st1, int64_t st2, int64_t st3,
    int ndim, int dtype_id, int device_idx) {
  TensorMetadata& slot = buffer[slot_id];
  slot.data_ptr = data_ptr;
  slot.shape[0] = s0;
  slot.shape[1] = s1;
  slot.shape[2] = s2;
  slot.shape[3] = s3;
  slot.stride[0] = st0;
  slot.stride[1] = st1;
  slot.stride[2] = st2;
  slot.stride[3] = st3;
  slot.ndim = ndim;
  slot.dtype_id = dtype_id;
  slot.device_idx = device_idx;
}

__global__ void sink_kernel(uint64_t ptr_value) {
  if (threadIdx.x == 0) {
    asm volatile("" ::"l"(ptr_value));
  }
}

void record_op(const at::Tensor& tensor, const at::Tensor& buffer,
               int64_t slot_id) {
  TORCH_CHECK(tensor.is_cuda(), "record() expects CUDA tensor input.");
  TORCH_CHECK(buffer.is_cuda(), "Metadata buffer must live on CUDA.");
  TORCH_CHECK(buffer.dtype() == at::kByte,
              "Metadata buffer must be uint8 tensor.");
  TORCH_CHECK(buffer.is_contiguous(),
              "Metadata buffer must be contiguous.");
  TORCH_CHECK(slot_id >= 0, "slot_id must be non-negative.");

  const int64_t slot_idx = slot_id;
  auto sizes = tensor.sizes();
  auto strides = tensor.strides();
  const int ndim = static_cast<int>(tensor.dim());
  int64_t shape_vals[4] = {0, 0, 0, 0};
  int64_t stride_vals[4] = {0, 0, 0, 0};
  for (int i = 0; i < ndim && i < 4; ++i) {
    shape_vals[i] = sizes[i];
    stride_vals[i] = strides[i];
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  auto* metadata_buffer =
      reinterpret_cast<TensorMetadata*>(buffer.data_ptr<uint8_t>());

  record_metadata_kernel<<<1, 1, 0, stream>>>(
      metadata_buffer, static_cast<int>(slot_idx),
      reinterpret_cast<uint64_t>(tensor.data_ptr()),
      shape_vals[0], shape_vals[1], shape_vals[2], shape_vals[3],
      stride_vals[0], stride_vals[1], stride_vals[2], stride_vals[3],
      ndim, static_cast<int>(tensor.scalar_type()), tensor.get_device());
  AT_CUDA_CHECK(cudaGetLastError());
}

void sink_op(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  for (const auto& tensor : tensors) {
    if (!tensor.defined()) {
      continue;
    }
    TORCH_CHECK(tensor.is_cuda(), "sink() expects CUDA tensors.");
    sink_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint64_t>(tensor.data_ptr()));
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

at::Tensor alias_tensor(
    int64_t data_ptr,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int64_t dtype_id,
    int64_t device_index) {
  auto scalar_type = static_cast<at::ScalarType>(dtype_id);
  auto options = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA, device_index);
  auto tensor = at::from_blob(
      reinterpret_cast<void*>(data_ptr),
      c10::IntArrayRef(sizes),
      c10::IntArrayRef(strides),
      [](void*) {},
      options);
  return tensor;
}

}  // namespace vllm

TORCH_LIBRARY(graphmonitor_ops, m) {
  m.def("record(Tensor tensor, Tensor buffer, int slot_id) -> ()");
  m.def("sink(Tensor[] tensors) -> ()");
}

TORCH_LIBRARY_IMPL(graphmonitor_ops, CUDA, m) {
  m.impl("record", &vllm::record_op);
  m.impl("sink", &vllm::sink_op);
  m.impl("alias_tensor", &vllm::alias_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
