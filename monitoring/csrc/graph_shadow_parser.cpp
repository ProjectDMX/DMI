#include "graph_shadow_parser.h"

#include <ATen/ops/from_blob.h>
#include <pybind11/stl.h>

namespace monitoring {

namespace py = pybind11;

namespace {

struct alignas(128) ShadowSlotRow {
  uint64_t data_ptr;
  int64_t shape[4];
  int64_t stride[4];
  int32_t ndim;
  int32_t dtype_id;
  int32_t device_idx;
  char padding[44];
};

static_assert(sizeof(ShadowSlotRow) == 128,
              "ShadowSlotRow must match metadata layout.");

bool ends_with(const std::string& value, const char* suffix) {
  size_t suffix_len = std::char_traits<char>::length(suffix);
  if (value.size() < suffix_len) {
    return false;
  }
  return value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

int64_t infer_pos_dim(const std::string& name) {
  if (ends_with(name, "hook_q") ||
      ends_with(name, "hook_k") ||
      ends_with(name, "hook_v") ||
      ends_with(name, "hook_z") ||
      ends_with(name, "hook_result")) {
    return -3;
  }
  return -2;
}

bool is_cuda_pointer(const ShadowSlotRow& row) {
  // record_op() already enforces tensor.is_cuda() and writes device_idx
  // via tensor.get_device() inside a GPU kernel.  A non-negative device_idx
  // therefore guarantees a valid CUDA (device or managed) pointer — no need
  // to round-trip through the CUDA driver with cudaPointerGetAttributes().
  return row.device_idx >= 0;
}

at::Tensor alias_from_row(const ShadowSlotRow& row, int64_t ndim) {
  TORCH_CHECK(ndim >= 0 && ndim <= 4,
              "Graph monitor metadata supports up to 4 dims, got ",
              ndim);
  std::vector<int64_t> sizes(ndim);
  std::vector<int64_t> strides(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    sizes[i] = row.shape[i];
    strides[i] = row.stride[i];
  }
  auto dtype = static_cast<at::ScalarType>(row.dtype_id);
  TORCH_CHECK(dtype >= at::ScalarType::Byte &&
                  dtype < at::ScalarType::NumOptions,
              "Unsupported dtype id in metadata: ", row.dtype_id);
  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .device(torch::kCUDA, row.device_idx);
  return at::from_blob(reinterpret_cast<void*>(row.data_ptr),
                       c10::IntArrayRef(sizes),
                       c10::IntArrayRef(strides),
                       [](void*) {},
                       options);
}

}  // namespace

py::dict parse_shadow_block(const at::Tensor& metadata,
                            const std::vector<int64_t>& slot_ids,
                            const std::vector<std::string>& hook_names) {
  TORCH_CHECK(metadata.device().is_cpu(),
              "Shadow block metadata must live on CPU/pinned memory.");
  TORCH_CHECK(metadata.dtype() == at::kByte,
              "Metadata tensor must be uint8.");
  TORCH_CHECK(metadata.is_contiguous(),
              "Metadata tensor must be contiguous.");
  TORCH_CHECK(metadata.dim() == 2 && metadata.size(1) == sizeof(ShadowSlotRow),
              "Metadata tensor must have shape [N, 128].");
  TORCH_CHECK(slot_ids.size() == hook_names.size(),
              "slot_ids and hook_names must have the same length.");

  const auto* rows =
      reinterpret_cast<const ShadowSlotRow*>(metadata.data_ptr<uint8_t>());
  const auto num_rows = static_cast<int64_t>(metadata.size(0));

  py::list tensors;
  py::list slice_dims;
  py::list remove_batch;
  py::list can_slice;
  py::list slice_modes;
  py::list target_devices;

  for (size_t i = 0; i < slot_ids.size(); ++i) {
    int64_t slot_idx = slot_ids[i];
    if (slot_idx < 0 || slot_idx >= num_rows) {
      continue;
    }
    const ShadowSlotRow& row = rows[slot_idx];
    if (row.data_ptr == 0 || row.ndim <= 0) {
      continue;
    }
    if (!is_cuda_pointer(row)) {
      continue;
    }
    int64_t ndim = static_cast<int64_t>(row.ndim);
    if (ndim > 4) {
      TORCH_CHECK(false,
                  "Metadata row for slot %lld exceeds 4 dims (ndim=%lld)",
                  static_cast<long long>(slot_idx),
                  static_cast<long long>(ndim));
    }
    at::Tensor tensor;
    try {
      tensor = alias_from_row(row, ndim);
    } catch (const c10::Error&) {
      // Stale metadata row — pointer was valid in a previous step but the
      // underlying GPU memory has since been freed or remapped.  Skip it
      // rather than querying the driver with cudaPointerGetAttributes.
      continue;
    }
    int64_t pos_dim = infer_pos_dim(hook_names[i]);
    bool can_slice_flag =
        (pos_dim < 0) ? (ndim >= -pos_dim) : (ndim > pos_dim);

    tensors.append(tensor);
    slice_dims.append(py::int_(pos_dim));
    remove_batch.append(py::bool_(false));
    can_slice.append(py::bool_(can_slice_flag));
    slice_modes.append(py::int_(0));  // identity
    target_devices.append(py::none());
  }

  py::dict spec;
  spec["tensors"] = tensors;
  spec["slice_dims"] = slice_dims;
  spec["remove_batch"] = remove_batch;
  spec["can_slice"] = can_slice;
  spec["slice_modes"] = slice_modes;
  spec["target_devices"] = target_devices;
  return spec;
}

}  // namespace monitoring
