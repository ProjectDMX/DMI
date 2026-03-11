#ifndef MONITORING_GRAPH_NATIVE_DELEGATE_H_
#define MONITORING_GRAPH_NATIVE_DELEGATE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "native_engine.h"

namespace monitoring {

class GraphNativeDelegate {
 public:
  explicit GraphNativeDelegate(std::shared_ptr<NativeMonitoringEngine> backend);

  void submit_and_resolve(int64_t step_id,
                          const at::Tensor& metadata,
                          const std::vector<int64_t>& slot_ids,
                          const std::vector<std::string>& hook_names,
                          std::optional<uint64_t> stream_handle);

 private:
  std::shared_ptr<NativeMonitoringEngine> backend_;
};

std::shared_ptr<GraphNativeDelegate> create_graph_delegate(
    const std::shared_ptr<NativeMonitoringEngine>& backend);

}  // namespace monitoring

#endif  // MONITORING_GRAPH_NATIVE_DELEGATE_H_
