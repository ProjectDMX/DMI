#include "graph_native_delegate.h"
#include "graph_shadow_parser.h"

#include <pybind11/stl.h>

#include <algorithm>

namespace monitoring {

namespace py = pybind11;

GraphNativeDelegate::GraphNativeDelegate(
    std::shared_ptr<NativeMonitoringEngine> backend)
    : backend_(std::move(backend)) {
  TORCH_CHECK(backend_ != nullptr, "GraphNativeDelegate requires a backend");
}

void GraphNativeDelegate::submit_and_resolve(
    int64_t step_id,
    const at::Tensor& metadata,
    const std::vector<int64_t>& slot_ids,
    const std::vector<std::string>& hook_names,
    std::optional<uint64_t> stream_handle) {
  if (!backend_ || slot_ids.empty()) {
    return;
  }
  py::dict spec = parse_shadow_block(metadata, slot_ids, hook_names);
  auto tensors = spec["tensors"].cast<py::list>();
  if (tensors.empty()) {
    return;
  }
  backend_->submit_step_soa(step_id, spec, stream_handle);
  backend_->seal_step(step_id, stream_handle);
  backend_->resolve_all();
}

std::shared_ptr<GraphNativeDelegate> create_graph_delegate(
    const std::shared_ptr<NativeMonitoringEngine>& backend) {
  return std::make_shared<GraphNativeDelegate>(backend);
}

}  // namespace monitoring
