// Top-level class thin wrappers and hook callback creation.

#include "native_engine_internal.h"

namespace monitoring {

namespace py = pybind11;

NativeMonitoringEngine::NativeMonitoringEngine(int64_t queue_size,
                                               std::optional<at::ScalarType> cache_dtype,
                                               int64_t delay_steps)
    : impl_(std::make_unique<Impl>(queue_size, cache_dtype, delay_steps)) {}

NativeMonitoringEngine::~NativeMonitoringEngine() {
  close();
}

py::dict NativeMonitoringEngine::get_stats() { return impl_->get_stats(); }

std::vector<int64_t> NativeMonitoringEngine::submit_step(int64_t step_id,
                                                         const py::list& tasks,
                                                         std::optional<uint64_t> stream_handle) {
  return impl_->submit_step(step_id, tasks, stream_handle);
}

void NativeMonitoringEngine::begin_step(int64_t step_id) { impl_->begin_step(step_id); }
void NativeMonitoringEngine::record_callback_duration(int64_t us) { impl_->record_callback_duration(us); }

std::vector<int64_t> NativeMonitoringEngine::submit_step_soa(int64_t step_id,
                                                             const py::dict& spec,
                                                             std::optional<uint64_t> stream_handle) {
  return impl_->submit_step_soa(step_id, spec, stream_handle);
}

int64_t NativeMonitoringEngine::add_task(int64_t step_id, const py::tuple& task_tuple) {
  return impl_->add_task(step_id, task_tuple);
}

void NativeMonitoringEngine::seal_step(int64_t step_id, std::optional<uint64_t> stream_handle) {
  impl_->seal_step(step_id, stream_handle);
}

py::object NativeMonitoringEngine::create_hook_callback(const std::string& hook_name,
                                                        bool remove_batch_dim,
                                                        py::object pos_slice,
                                                        py::object target_device) {
  HookConfig* cfg_ptr = impl_->upsert_hook_config(hook_name, remove_batch_dim,
                                                  std::move(pos_slice), std::move(target_device));
  auto engine = shared_from_this();
  return py::cpp_function(
      [engine, cfg_ptr](py::args args, py::kwargs /*kwargs*/) -> py::object {
        if (args.size() == 0) {
          throw std::runtime_error("Native callback expected tensor argument");
        }
        at::Tensor tensor = args[0].cast<at::Tensor>();
        auto t0 = std::chrono::steady_clock::now();
        {
          py::gil_scoped_release release;
          if (tensor.requires_grad()) {
            tensor = tensor.detach();
          }
          engine->impl_->append_hook_current_step(*cfg_ptr, std::move(tensor));
        }
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        engine->record_callback_duration(static_cast<int64_t>(us));
        return py::none();
      });
}

py::object NativeMonitoringEngine::create_hook_callback_with_cache(const std::string& hook_name,
                                                                   bool remove_batch_dim,
                                                                   py::object pos_slice,
                                                                   py::object target_device,
                                                                   py::dict cache) {
  HookConfig* cfg_ptr = impl_->upsert_hook_config(hook_name, remove_batch_dim,
                                                  std::move(pos_slice), std::move(target_device));
  auto engine = shared_from_this();
  std::string hook_name_copy = hook_name;  // ensure lifetime inside lambda
  return py::cpp_function(
      [engine, cfg_ptr, cache, hook_name_copy](py::args args, py::kwargs /*kwargs*/) -> py::object {
        if (args.size() == 0) {
          throw std::runtime_error("Native callback expected tensor argument");
        }
        at::Tensor tensor = args[0].cast<at::Tensor>();
        int64_t token = 0;
        auto t0 = std::chrono::steady_clock::now();
        {
          py::gil_scoped_release release;
          if (tensor.requires_grad()) {
            tensor = tensor.detach();
          }
          token = engine->impl_->add_task_from_config(*cfg_ptr, std::move(tensor));
        }
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        engine->record_callback_duration(static_cast<int64_t>(us));

        // Write BackendFuture(native_backend, token) into the provided cache dict
        try {
          py::object task_mod = py::module::import("monitoring.task");
          py::object backend_future_cls = task_mod.attr("BackendFuture");
          py::object py_future = backend_future_cls(engine, py::int_(token));
          cache[py::str(hook_name_copy.c_str())] = py_future;
        } catch (const std::exception& e) {
          // Fallback: leave cache entry as None on error
          cache[py::str(hook_name_copy.c_str())] = py::none();
        }
        return py::none();
      });
}

void NativeMonitoringEngine::append_hook(int64_t step_id,
                                         const std::string& hook_name,
                                         at::Tensor tensor,
                                         bool remove_batch_dim,
                                         py::object pos_slice,
                                         py::object target_device) {
  impl_->append_hook(step_id, hook_name, std::move(tensor), remove_batch_dim,
                     std::move(pos_slice), std::move(target_device));
}

void NativeMonitoringEngine::resolve_all() { impl_->resolve_all(); }
bool NativeMonitoringEngine::future_ready(int64_t token) { return impl_->future_ready(token); }
bool NativeMonitoringEngine::future_wait(int64_t token, std::optional<double> timeout) {
  return impl_->future_wait(token, timeout);
}
at::Tensor NativeMonitoringEngine::future_result(int64_t token, std::optional<double> timeout) {
  return impl_->future_result(token, timeout);
}
void NativeMonitoringEngine::clear_completed_results() { impl_->clear_completed_results(); }
void NativeMonitoringEngine::close() {
  if (impl_) impl_->close();
}

}  // namespace monitoring
