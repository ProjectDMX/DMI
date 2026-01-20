// Copyright (internal) – native monitoring engine (public API)
#ifndef MONITORING_NATIVE_ENGINE_H_
#define MONITORING_NATIVE_ENGINE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ATen/Tensor.h>
#include <pybind11/pytypes.h>

namespace monitoring {

namespace py = pybind11;

class NativeMonitoringEngine : public std::enable_shared_from_this<NativeMonitoringEngine> {
 public:
  NativeMonitoringEngine(int64_t queue_size,
                         std::optional<at::ScalarType> cache_dtype,
                         int64_t delay_steps);
  ~NativeMonitoringEngine();

  py::dict get_stats();

  std::vector<int64_t> submit_step(int64_t step_id,
                                   const py::list& tasks,
                                   std::optional<uint64_t> stream_handle);

  void set_capture_schedule(int64_t step_stride,
                            int64_t step_offset,
                            int64_t warmup_steps,
                            bool capture_prefill,
                            bool capture_decode,
                            int64_t request_stride,
                            int64_t request_offset,
                            int64_t warmup_requests);
  void begin_request(int64_t request_id);
  void begin_step(int64_t step_id, int64_t phase);
  void record_callback_duration(int64_t us);

  // Struct-of-arrays submit to minimize Python overhead
  std::vector<int64_t> submit_step_soa(int64_t step_id,
                                       const py::dict& spec,
                                       std::optional<uint64_t> stream_handle);

  // Low-overhead single task append
  int64_t add_task(int64_t step_id, const py::tuple& task_tuple);

  // Seal an open step and dispatch according to delay policy
  void seal_step(int64_t step_id, std::optional<uint64_t> stream_handle);

  // Create a Python hook callback (captures engine lifetime)
  py::object create_hook_callback(const std::string& hook_name,
                                  bool remove_batch_dim,
                                  py::object pos_slice,
                                  py::object target_device);
  // Create a Python hook callback that also writes BackendFuture into cache dict
  py::object create_hook_callback_with_cache(const std::string& hook_name,
                                             bool remove_batch_dim,
                                             py::object pos_slice,
                                             py::object target_device,
                                             py::dict cache);
  py::object create_hook_callback_with_cache_sig(const std::string& hook_name,
                                                 bool remove_batch_dim,
                                                 py::tuple slice_tuple,
                                                 py::object target_device,
                                                 py::dict cache);
  py::object create_global_hook_callback_sig(const std::string& hook_name,
                                             bool remove_batch_dim,
                                             py::tuple slice_tuple,
                                             py::object target_device);
  py::object register_hook_callback(py::object hook_point,
                                    const std::string& hook_name,
                                    const std::string& cache_name,
                                    bool is_backward,
                                    bool remove_batch_dim,
                                    py::tuple slice_tuple,
                                    py::object target_device,
                                    bool prepend);
  void set_enabled_hooks(py::object names_iterable);
  void collect_step_futures_into(int64_t step_id, py::dict cache);

  // Append a hook record directly (tokens assigned at seal time)
  void append_hook(int64_t step_id,
                   const std::string& hook_name,
                   at::Tensor tensor,
                   bool remove_batch_dim,
                   py::object pos_slice,
                   py::object target_device);

  // Synchronization / futures
  void resolve_all();
  bool future_ready(int64_t token);
  bool future_wait(int64_t token, std::optional<double> timeout);
  at::Tensor future_result(int64_t token, std::optional<double> timeout);
  void clear_completed_results();
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::shared_ptr<NativeMonitoringEngine> create_engine(int64_t queue_size,
                                                      py::object cache_dtype,
                                                      int64_t delay_steps);

}  // namespace monitoring

#endif  // MONITORING_NATIVE_ENGINE_H_
