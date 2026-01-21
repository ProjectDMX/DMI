// Pybind11 module + factory

#include "native_engine_internal.h"

namespace monitoring {

std::shared_ptr<NativeMonitoringEngine> create_engine(int64_t queue_size,
                                                      py::object cache_dtype,
                                                      int64_t delay_steps) {
  std::optional<at::ScalarType> dtype;
  if (!cache_dtype.is_none()) {
    dtype = cache_dtype.cast<at::ScalarType>();
  }
  return std::make_shared<NativeMonitoringEngine>(queue_size, dtype, delay_steps);
}

}  // namespace monitoring

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<monitoring::NativeMonitoringEngine, std::shared_ptr<monitoring::NativeMonitoringEngine>>(m,
                                                                                                     "NativeMonitoringEngine")
      .def("submit_step", &monitoring::NativeMonitoringEngine::submit_step,
           py::arg("step_id"), py::arg("tasks"), py::arg("stream_handle") = std::optional<uint64_t>())
      .def("set_capture_schedule", &monitoring::NativeMonitoringEngine::set_capture_schedule,
           py::arg("step_stride"), py::arg("step_offset"), py::arg("warmup_steps"),
           py::arg("capture_prefill"), py::arg("capture_decode"),
           py::arg("request_stride"), py::arg("request_offset"), py::arg("warmup_requests"))
      .def("begin_request", &monitoring::NativeMonitoringEngine::begin_request,
           py::arg("request_id"))
      .def("begin_step", &monitoring::NativeMonitoringEngine::begin_step,
           py::arg("step_id"), py::arg("phase") = static_cast<int64_t>(monitoring::StepPhase::kUnknown))
      .def("create_hook_callback", &monitoring::NativeMonitoringEngine::create_hook_callback,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("pos_slice"),
           py::arg("target_device") = py::none())
      .def("create_hook_callback_with_cache", &monitoring::NativeMonitoringEngine::create_hook_callback_with_cache,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("pos_slice"),
           py::arg("target_device") = py::none(), py::arg("cache"))
      .def("create_hook_callback_with_cache_sig", &monitoring::NativeMonitoringEngine::create_hook_callback_with_cache_sig,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("slice_tuple"),
           py::arg("target_device") = py::none(), py::arg("cache"))
      .def("create_global_hook_callback_sig", &monitoring::NativeMonitoringEngine::create_global_hook_callback_sig,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("slice_tuple"),
           py::arg("target_device") = py::none())
      .def("register_hook_callback", &monitoring::NativeMonitoringEngine::register_hook_callback,
           py::arg("hook_point"), py::arg("hook_name"), py::arg("cache_name"), py::arg("is_backward"),
           py::arg("remove_batch_dim"), py::arg("slice_tuple"), py::arg("target_device"),
           py::arg("prepend") = false)
      .def("create_inline_hook_ticket", &monitoring::NativeMonitoringEngine::create_inline_hook_ticket,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("slice_tuple"),
           py::arg("target_device") = py::none())
      .def("monitor_inline", &monitoring::NativeMonitoringEngine::monitor_inline,
           py::arg("ticket"), py::arg("hook_name"), py::arg("cache_name"), py::arg("tensor"))
      .def("set_enabled_hooks", &monitoring::NativeMonitoringEngine::set_enabled_hooks,
           py::arg("enabled_names"))
      .def("collect_step_futures_into", &monitoring::NativeMonitoringEngine::collect_step_futures_into,
           py::arg("step_id"), py::arg("cache"))
      .def("submit_step_soa", &monitoring::NativeMonitoringEngine::submit_step_soa,
           py::arg("step_id"), py::arg("spec"), py::arg("stream_handle") = std::optional<uint64_t>())
      .def("add_task", &monitoring::NativeMonitoringEngine::add_task,
           py::arg("step_id"), py::arg("task"))
      .def("seal_step", &monitoring::NativeMonitoringEngine::seal_step,
           py::arg("step_id"), py::arg("stream_handle") = std::optional<uint64_t>())
      .def("append_hook", &monitoring::NativeMonitoringEngine::append_hook,
           py::arg("step_id"), py::arg("hook_name"), py::arg("tensor"),
           py::arg("remove_batch_dim"), py::arg("pos_slice"), py::arg("target_device") = py::none())
      .def("resolve_all", &monitoring::NativeMonitoringEngine::resolve_all)
      .def("future_ready", &monitoring::NativeMonitoringEngine::future_ready)
      .def("future_wait", &monitoring::NativeMonitoringEngine::future_wait,
           py::arg("token"), py::arg("timeout") = std::optional<double>())
      .def("future_result", &monitoring::NativeMonitoringEngine::future_result,
           py::arg("token"), py::arg("timeout") = std::optional<double>())
      .def("close", &monitoring::NativeMonitoringEngine::close)
      .def("clear_completed_results", &monitoring::NativeMonitoringEngine::clear_completed_results)
      .def("get_stats", &monitoring::NativeMonitoringEngine::get_stats);

  m.def("create_engine", &monitoring::create_engine,
        py::arg("queue_size"), py::arg("cache_dtype"), py::arg("delay_steps"));
}
