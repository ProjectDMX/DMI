#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

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

// --- Per-slot D2H event barrier for inline wait inside CUDA Graph ---
// wait_d2h() uses cudaEventWaitExternal so the wait is captured as an
// event-wait node in the graph.  On every replay the node checks the
// *current* state of the external event (recorded on copy_stream).
#ifndef cudaEventWaitExternal
#define cudaEventWaitExternal 0x2
#endif

static constexpr int kMaxD2HEvents = 8192;  // 2 frames × 4096 max_slots
static cudaEvent_t g_d2h_events[kMaxD2HEvents];
static int g_d2h_event_count = 0;

void init_d2h_events_op(int64_t num_events) {
  TORCH_CHECK(num_events > 0 && num_events <= kMaxD2HEvents,
              "num_events must be in [1, ", kMaxD2HEvents, "], got ", num_events);
  // Destroy any existing events first
  for (int i = 0; i < g_d2h_event_count; i++) {
    cudaEventDestroy(g_d2h_events[i]);
  }
  g_d2h_event_count = 0;
  for (int64_t i = 0; i < num_events; i++) {
    AT_CUDA_CHECK(cudaEventCreateWithFlags(
        &g_d2h_events[i], cudaEventDisableTiming));
  }
  g_d2h_event_count = static_cast<int>(num_events);
}

void wait_d2h_op(const at::Tensor& buf, int64_t slot_id) {
  TORCH_CHECK(slot_id >= 0 && slot_id < g_d2h_event_count,
              "wait_d2h: slot_id ", slot_id,
              " out of range [0, ", g_d2h_event_count, ")");
  auto stream = at::cuda::getCurrentCUDAStream();
  // Detect stream capture mode: external events require special flag
  cudaStreamCaptureStatus capture_status;
  AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  unsigned int flags = (capture_status != cudaStreamCaptureStatusNone)
      ? cudaEventWaitExternal : 0;
  AT_CUDA_CHECK(cudaStreamWaitEvent(stream, g_d2h_events[slot_id], flags));
}

void record_d2h_event_op(int64_t slot_id) {
  TORCH_CHECK(slot_id >= 0 && slot_id < g_d2h_event_count,
              "record_d2h_event: slot_id ", slot_id,
              " out of range [0, ", g_d2h_event_count, ")");
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaEventRecord(g_d2h_events[slot_id], stream));
}

void destroy_d2h_events_op() {
  for (int i = 0; i < g_d2h_event_count; i++) {
    cudaEventDestroy(g_d2h_events[i]);
  }
  g_d2h_event_count = 0;
}

// --- Design C: held_tensors for cross-graph address isolation ---
// During recording, sink_hold() pushes real at::Tensor refs here, keeping
// allocator refcount > 0.  After both graphs are recorded,
// clear_held_tensors() releases everything (addresses are baked into graphs).
static std::vector<at::Tensor> g_held_tensors[2];
static std::mutex g_held_mutex;

void sink_hold_op(const std::vector<at::Tensor>& tensors, int64_t frame) {
  if (tensors.empty()) return;
  TORCH_CHECK(frame == 0 || frame == 1,
              "sink_hold frame must be 0 or 1, got ", frame);
  auto stream = at::cuda::getCurrentCUDAStream();
  for (const auto& tensor : tensors) {
    if (!tensor.defined()) continue;
    TORCH_CHECK(tensor.is_cuda(), "sink_hold() expects CUDA tensors.");
    // GPU side: dummy kernel (same as sink) — creates data dependency in graph
    sink_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<uint64_t>(tensor.data_ptr()));
    // C++ side: hold real at::Tensor ref to keep allocator refcount > 0.
    // This host code only runs during recording/eager, NOT during graph replay.
    std::lock_guard<std::mutex> lock(g_held_mutex);
    g_held_tensors[frame].push_back(tensor);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void clear_held_tensors_op() {
  std::lock_guard<std::mutex> lock(g_held_mutex);
  g_held_tensors[0].clear();
  g_held_tensors[0].shrink_to_fit();
  g_held_tensors[1].clear();
  g_held_tensors[1].shrink_to_fit();
}

int64_t held_tensors_count_op(int64_t frame) {
  TORCH_CHECK(frame == 0 || frame == 1,
              "held_tensors_count frame must be 0 or 1, got ", frame);
  std::lock_guard<std::mutex> lock(g_held_mutex);
  return static_cast<int64_t>(g_held_tensors[frame].size());
}

}  // namespace vllm

TORCH_LIBRARY(graphmonitor_ops, m) {
  m.def("record(Tensor tensor, Tensor(a!) buffer, int slot_id) -> ()");
  m.def("sink(Tensor[] tensors) -> ()");
  m.def("alias_tensor(int data_ptr, int[] sizes, int[] strides, int dtype_id, int device_index) -> Tensor");
  m.def("sink_hold(Tensor[] tensors, int frame) -> ()");
  m.def("clear_held_tensors() -> ()");
  m.def("held_tensors_count(int frame) -> int");
  // Per-slot D2H event barrier ops
  m.def("init_d2h_events(int num_events) -> ()");
  m.def("wait_d2h(Tensor(a!) buffer, int slot_id) -> ()");
  m.def("record_d2h_event(int slot_id) -> ()");
  m.def("destroy_d2h_events() -> ()");
}

TORCH_LIBRARY_IMPL(graphmonitor_ops, CUDA, m) {
  m.impl("record", &vllm::record_op);
  m.impl("sink", &vllm::sink_op);
  m.impl("sink_hold", &vllm::sink_hold_op);
  m.impl("wait_d2h", &vllm::wait_d2h_op);
}

TORCH_LIBRARY_IMPL(graphmonitor_ops, Meta, m) {
  m.impl("record", [](const at::Tensor&, const at::Tensor&, int64_t) {});
  m.impl("sink", [](const std::vector<at::Tensor>&) {});
  m.impl("sink_hold", [](const std::vector<at::Tensor>&, int64_t) {});
  m.impl("wait_d2h", [](const at::Tensor&, int64_t) {});
}

// Ops with no tensor args cannot be dispatched by device.
// Register as CompositeImplicitAutograd (fallback for all backends).
TORCH_LIBRARY_IMPL(graphmonitor_ops, CompositeImplicitAutograd, m) {
  m.impl("alias_tensor", &vllm::alias_tensor);
  m.impl("clear_held_tensors", &vllm::clear_held_tensors_op);
  m.impl("held_tensors_count", &vllm::held_tensors_count_op);
  m.impl("init_d2h_events", &vllm::init_d2h_events_op);
  m.impl("record_d2h_event", &vllm::record_d2h_event_op);
  m.impl("destroy_d2h_events", &vllm::destroy_d2h_events_op);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
