#ifndef MONITORING_GRAPH_SHADOW_PARSER_H_
#define MONITORING_GRAPH_SHADOW_PARSER_H_

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace monitoring {

namespace py = pybind11;

py::dict parse_shadow_block(const at::Tensor& metadata,
                            const std::vector<int64_t>& slot_ids,
                            const std::vector<std::string>& hook_names);

}  // namespace monitoring

#endif  // MONITORING_GRAPH_SHADOW_PARSER_H_
