// Top-level class thin wrappers and hook callback creation.

#include "native_engine_internal.h"
#include "nvtx_shim.h"

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

void NativeMonitoringEngine::set_capture_schedule(int64_t step_stride,
                                                  int64_t step_offset,
                                                  int64_t warmup_steps,
                                                  bool capture_prefill,
                                                  bool capture_decode,
                                                  int64_t request_stride,
                                                  int64_t request_offset,
                                                  int64_t warmup_requests) {
  impl_->set_capture_schedule(step_stride, step_offset, warmup_steps, capture_prefill,
                              capture_decode, request_stride, request_offset, warmup_requests);
}

void NativeMonitoringEngine::begin_request(int64_t request_id) { impl_->begin_request(request_id); }

void NativeMonitoringEngine::begin_step(int64_t step_id, int64_t phase) {
  impl_->begin_step(step_id, phase);
}
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
        if (!engine->impl_->is_capture_enabled()) {
          return py::none();
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
        if (!engine->impl_->is_capture_enabled()) {
          cache[py::str(hook_name_copy.c_str())] = py::none();
          return py::none();
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

py::object NativeMonitoringEngine::create_hook_callback_with_cache_sig(const std::string& hook_name,
                                                                       bool remove_batch_dim,
                                                                       py::tuple slice_tuple,
                                                                       py::object target_device,
                                                                       py::dict cache) {
  HookConfig* cfg_ptr = impl_->upsert_hook_config_tuple(hook_name, remove_batch_dim,
                                                        std::move(slice_tuple), std::move(target_device));
  auto engine = shared_from_this();
  std::string hook_name_copy = hook_name;
  return py::cpp_function(
      [engine, cfg_ptr, cache, hook_name_copy](py::args args, py::kwargs /*kwargs*/) -> py::object {
        if (args.size() == 0) {
          throw std::runtime_error("Native callback expected tensor argument");
        }
        if (!engine->impl_->is_capture_enabled()) {
          cache[py::str(hook_name_copy.c_str())] = py::none();
          return py::none();
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

        try {
          py::object task_mod = py::module::import("monitoring.task");
          py::object backend_future_cls = task_mod.attr("BackendFuture");
          py::object py_future = backend_future_cls(engine, py::int_(token));
          cache[py::str(hook_name_copy.c_str())] = py_future;
        } catch (const std::exception& e) {
          cache[py::str(hook_name_copy.c_str())] = py::none();
        }
        return py::none();
      });
}

py::object NativeMonitoringEngine::create_global_hook_callback_sig(const std::string& hook_name,
                                                                   bool remove_batch_dim,
                                                                   py::tuple slice_tuple,
                                                                   py::object target_device) {
  // Pre-create and store HookConfig; callback will only append tasks (no Python cache writes)
  HookConfig* cfg_ptr = impl_->upsert_hook_config_tuple(hook_name, remove_batch_dim,
                                                        std::move(slice_tuple), std::move(target_device));
  auto engine = shared_from_this();
  std::string hook_name_copy = hook_name;
  return py::cpp_function(
      [engine, cfg_ptr, hook_name_copy](py::args args, py::kwargs /*kwargs*/) -> py::object {
        if (args.size() == 0) {
          throw std::runtime_error("Native callback expected tensor argument");
        }
        if (!engine->impl_->is_capture_enabled()) {
          return py::none();
        }
        at::Tensor tensor = args[0].cast<at::Tensor>();
        auto t0 = std::chrono::steady_clock::now();
        {
          py::gil_scoped_release release;
          if (tensor.requires_grad()) {
            tensor = tensor.detach();
          }
          bool enabled = false;
          {
            std::lock_guard<std::mutex> lk(engine->impl_->enabled_mutex_);
            enabled = engine->impl_->is_hook_enabled_unlocked(hook_name_copy);
          }
          if (enabled) {
            int64_t token = engine->impl_->add_task_from_config(*cfg_ptr, std::move(tensor));
            int64_t step_id = engine->impl_->current_step_id_.load(std::memory_order_acquire);
            engine->impl_->record_step_name_token(step_id, hook_name_copy, token);
          }
        }
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        engine->record_callback_duration(static_cast<int64_t>(us));
        return py::none();
      });
}

py::object NativeMonitoringEngine::register_hook_callback(py::object hook_point,
                                                          const std::string& hook_name,
                                                          const std::string& cache_name,
                                                          bool is_backward,
                                                          bool remove_batch_dim,
                                                          py::tuple slice_tuple,
                                                          py::object target_device,
                                                          bool prepend) {
  if (hook_point.is_none()) {
    throw std::runtime_error("register_hook_callback requires a HookPoint module");
  }

  auto py_module = hook_point.cast<py::object>();
  auto handle_attr_name = is_backward ? "register_full_backward_hook" : "register_forward_hook";
  py::object register_fn = py_module.attr(handle_attr_name);

  HookConfig* cfg_ptr = impl_->upsert_hook_config_tuple(hook_name, remove_batch_dim,
                                                        std::move(slice_tuple), std::move(target_device));
  auto engine = shared_from_this();
  std::string gate_name = hook_name;
  std::string cache_name_copy = cache_name;

  std::string nvtx_label = std::string("MonEng::Hook[") + gate_name + (is_backward ? ":bwd]" : ":fwd]");

  py::object full_hook = py::cpp_function(
      [engine, cfg_ptr, gate_name, cache_name_copy, is_backward, nvtx_label](py::object /*module*/,
                                                                            py::object /*module_input*/,
                                                                            py::object module_output) -> py::object {
        py::object tensor_obj;
        if (is_backward) {
          if (py::isinstance<py::tuple>(module_output)) {
            py::tuple tup = module_output.cast<py::tuple>();
            if (tup.size() == 0) {
              return py::none();
            }
            tensor_obj = tup[0];
          } else {
            tensor_obj = module_output;
          }
        } else {
          tensor_obj = module_output;
        }

        at::Tensor tensor = tensor_obj.cast<at::Tensor>();
        auto t0 = std::chrono::steady_clock::now();
        mon_nvtx_push(nvtx_label.c_str());
        {
          py::gil_scoped_release release;
          engine->impl_->process_native_hook(*cfg_ptr, std::move(tensor), gate_name, cache_name_copy);
        }
        mon_nvtx_pop();
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        engine->record_callback_duration(static_cast<int64_t>(us));
        return py::none();
      });

  py::function reg_fn = register_fn.cast<py::function>();
  if (prepend) {
    return reg_fn(full_hook, py::arg("prepend") = true);
  }
  return reg_fn(full_hook);
}

void NativeMonitoringEngine::set_enabled_hooks(py::object names_iterable) {
  impl_->set_enabled_hooks(std::move(names_iterable));
}

void NativeMonitoringEngine::collect_step_futures_into(int64_t step_id, py::dict cache) {
  // Move out name->token pairs for this step
  std::vector<std::pair<std::string, int64_t>> items;
  {
    std::lock_guard<std::mutex> lock(impl_->staging_mutex_);
    auto it = impl_->step_name_tokens_.find(step_id);
    if (it != impl_->step_name_tokens_.end()) {
      items = std::move(it->second);
      impl_->step_name_tokens_.erase(it);
    }
  }
  if (items.empty()) return;
  // Create BackendFuture(native_backend, token) and fill into provided cache dict
  py::object task_mod = py::module::import("monitoring.task");
  py::object backend_future_cls = task_mod.attr("BackendFuture");
  for (auto& kv : items) {
    const std::string& name = kv.first;
    int64_t token = kv.second;
    py::object py_future = backend_future_cls(shared_from_this(), py::int_(token));
    cache[py::str(name.c_str())] = py_future;
  }
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
