// Pybind11 module + factory

#include "native_engine_internal.h"
#include "clickhouse_client.h"
// dmx_host pipeline
#include "dmx_host_engine.h"
// ring offload engine
#include "ring/ring_engine_py.h"
#include <ATen/cuda/CUDAContext.h>

namespace monitoring {

std::shared_ptr<NativeMonitoringEngine> create_engine(int64_t queue_size,
                                                      py::object cache_dtype,
                                                      int64_t delay_steps,
                                                      const std::vector<int64_t>& pinpool_bins_kb,
                                                      int64_t pinpool_max_mb,
                                                      int64_t host_copy_threads,
                                                      int64_t host_copy_queue_size) {
  std::optional<at::ScalarType> dtype;
  if (!cache_dtype.is_none()) {
    dtype = cache_dtype.cast<at::ScalarType>();
  }
  return std::make_shared<NativeMonitoringEngine>(
      queue_size,
      dtype,
      delay_steps,
      pinpool_bins_kb,
      pinpool_max_mb,
      host_copy_threads,
      host_copy_queue_size);
}

}  // namespace monitoring

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<monitoring::NativeMonitoringEngine,
             std::shared_ptr<monitoring::NativeMonitoringEngine>>(m, "NativeMonitoringEngine")
      .def("submit_step", &monitoring::NativeMonitoringEngine::submit_step,
           py::arg("step_id"), py::arg("tasks"),
           py::arg("stream_handle") = std::optional<uint64_t>())
      .def("set_capture_schedule", &monitoring::NativeMonitoringEngine::set_capture_schedule,
           py::arg("step_stride"), py::arg("step_offset"), py::arg("warmup_steps"),
           py::arg("capture_prefill"), py::arg("capture_decode"),
           py::arg("request_stride"), py::arg("request_offset"), py::arg("warmup_requests"))
      .def("begin_request", &monitoring::NativeMonitoringEngine::begin_request,
           py::arg("request_id"))
      .def("begin_step", &monitoring::NativeMonitoringEngine::begin_step,
           py::arg("step_id"),
           py::arg("phase") = static_cast<int64_t>(monitoring::StepPhase::kUnknown))
      .def("set_partial_seal_config",
           &monitoring::NativeMonitoringEngine::set_partial_seal_config,
           py::arg("enabled"),
           py::arg("chunk_bytes"),
           py::arg("cap_enabled"),
           py::arg("cap_ratio"),
           py::arg("driver_guard_mb"))
      .def("create_hook_callback", &monitoring::NativeMonitoringEngine::create_hook_callback,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("pos_slice"),
           py::arg("target_device") = py::none())
      .def("create_hook_callback_with_cache",
           &monitoring::NativeMonitoringEngine::create_hook_callback_with_cache,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("pos_slice"),
           py::arg("target_device") = py::none(), py::arg("cache"))
      .def("create_hook_callback_with_cache_sig",
           &monitoring::NativeMonitoringEngine::create_hook_callback_with_cache_sig,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("slice_tuple"),
           py::arg("target_device") = py::none(), py::arg("cache"))
      .def("create_global_hook_callback_sig",
           &monitoring::NativeMonitoringEngine::create_global_hook_callback_sig,
           py::arg("hook_name"), py::arg("remove_batch_dim"), py::arg("slice_tuple"),
           py::arg("target_device") = py::none())
      .def("set_enabled_hooks", &monitoring::NativeMonitoringEngine::set_enabled_hooks,
           py::arg("enabled_names"))
      .def("collect_step_futures_into",
           &monitoring::NativeMonitoringEngine::collect_step_futures_into,
           py::arg("step_id"), py::arg("cache"))
      .def("add_task", &monitoring::NativeMonitoringEngine::add_task,
           py::arg("step_id"), py::arg("task"))
      .def("seal_step", &monitoring::NativeMonitoringEngine::seal_step,
           py::arg("step_id"),
           py::arg("stream_handle") = std::optional<uint64_t>())
      .def("append_hook", &monitoring::NativeMonitoringEngine::append_hook,
           py::arg("step_id"), py::arg("hook_name"), py::arg("tensor"),
           py::arg("remove_batch_dim"), py::arg("pos_slice"),
           py::arg("target_device") = py::none())
      .def("resolve_all", &monitoring::NativeMonitoringEngine::resolve_all)
      .def("future_ready", &monitoring::NativeMonitoringEngine::future_ready,
           py::arg("token"))
      .def("future_wait", &monitoring::NativeMonitoringEngine::future_wait,
           py::arg("token"), py::arg("timeout") = std::optional<double>(),
           py::arg("called_from_cpp") = false)
      .def("future_result", &monitoring::NativeMonitoringEngine::future_result,
           py::arg("token"), py::arg("timeout") = std::optional<double>(),
           py::arg("called_from_cpp") = false)
      .def("close", &monitoring::NativeMonitoringEngine::close)
      .def("clear_completed_results", &monitoring::NativeMonitoringEngine::clear_completed_results)
      .def("get_stats", &monitoring::NativeMonitoringEngine::get_stats);

  // C++ BackendFuture class exposed to Python with the same interface as before.
  py::class_<monitoring::BackendFuture>(m, "BackendFuture")
      .def(py::init<std::shared_ptr<monitoring::NativeMonitoringEngine>, int64_t>(),
           py::arg("backend"), py::arg("token"))
      .def("size", &monitoring::BackendFuture::size)
      .def("ready", &monitoring::BackendFuture::ready)
      .def("wait", &monitoring::BackendFuture::wait,
           py::arg("timeout") = std::optional<double>(),
           py::arg("called_from_cpp") = false)
      .def("result", &monitoring::BackendFuture::result,
           py::arg("timeout") = std::optional<double>(),
           py::arg("called_from_cpp") = false);

  m.def("create_engine", &monitoring::create_engine,
        py::arg("queue_size"),
        py::arg("cache_dtype"),
        py::arg("delay_steps"),
        py::arg("pinpool_bins_kb") = std::vector<int64_t>{256, 512, 1024, 2048, 4096, 8192},
        py::arg("pinpool_max_mb") = 512,
        py::arg("host_copy_threads") = 0,
        py::arg("host_copy_queue_size") = 512);


  // ---- ClickHouseClientConfig (config only; stage is C++-only) ----
  py::class_<dmx_host::ClickHouseClientConfig>(m, "ClickHouseClientConfig")
      .def(py::init<>())

      .def_readwrite("host", &dmx_host::ClickHouseClientConfig::host)
      .def_readwrite("port", &dmx_host::ClickHouseClientConfig::port)
      .def_readwrite("username", &dmx_host::ClickHouseClientConfig::username)
      .def_readwrite("password", &dmx_host::ClickHouseClientConfig::password)
      .def_readwrite("database", &dmx_host::ClickHouseClientConfig::database)
      .def_readwrite("table", &dmx_host::ClickHouseClientConfig::table)
      .def_readwrite("secure", &dmx_host::ClickHouseClientConfig::secure)

      .def_readwrite("create_database_if_missing",
                     &dmx_host::ClickHouseClientConfig::create_database_if_missing)
      .def_readwrite("drop_existing_database",
                     &dmx_host::ClickHouseClientConfig::drop_existing_database)
      .def_readwrite("client_side_compress",
                     &dmx_host::ClickHouseClientConfig::client_side_compress)
      .def_readwrite("index_granularity",
                     &dmx_host::ClickHouseClientConfig::index_granularity)

      // Expose client_settings as a dict, store internally as unordered_map<string, variant<...>>.
      // This avoids requiring <pybind11/stl_variant.h>.
      .def_property(
          "client_settings",
          [](const dmx_host::ClickHouseClientConfig& self) {
            py::dict d;
            for (const auto& kv : self.client_settings) {
              const auto& key = kv.first;
              const auto& val = kv.second;
              if (std::holds_alternative<bool>(val)) {
                d[py::str(key)] = py::bool_(std::get<bool>(val));
              } else if (std::holds_alternative<std::int64_t>(val)) {
                d[py::str(key)] = py::int_(std::get<std::int64_t>(val));
              } else {
                d[py::str(key)] = py::str(std::get<std::string>(val));
              }
            }
            return d;
          },
          [](dmx_host::ClickHouseClientConfig& self, py::object obj) {
            self.client_settings.clear();
            if (obj.is_none()) return;

            py::dict d = obj.cast<py::dict>();
            for (auto item : d) {
              std::string key = py::cast<std::string>(item.first);
              py::handle v = item.second;

              // bool must be checked before int (Python bool is an int subclass)
              if (py::isinstance<py::bool_>(v)) {
                self.client_settings.emplace(std::move(key), py::cast<bool>(v));
              } else if (py::isinstance<py::int_>(v)) {
                self.client_settings.emplace(
                    std::move(key),
                    static_cast<std::int64_t>(py::cast<long long>(v)));
              } else if (py::isinstance<py::str>(v)) {
                self.client_settings.emplace(std::move(key), py::cast<std::string>(v));
              } else {
                throw py::type_error("client_settings values must be bool/int/str (or None)");
              }
            }
          });

  // ---- dmx_host StageConfig + DMXHostEngine ----
  using DMXHostEngine = dmx_host::DMXHostEngine;
  using StageConfig = DMXHostEngine::StageConfig;
  using ThreadFailure = DMXHostEngine::ThreadFailure;
  using QueueT = DMXHostEngine::QueueT;
  using QueueConfig = DMXHostEngine::QueueConfig;
  using EnqueuePolicy = DMXHostEngine::EnqueuePolicy;
  using Duration = DMXHostEngine::Duration;

  py::enum_<dmx_host::OnFullPolicy>(m, "OnFullPolicy")
      .value("RAISE", dmx_host::OnFullPolicy::RAISE)
      .value("DROP", dmx_host::OnFullPolicy::DROP)
      .value("RETRY", dmx_host::OnFullPolicy::RETRY)
      .value("ABORT", dmx_host::OnFullPolicy::ABORT)
      .export_values();

  py::enum_<dmx_host::OnClosedPolicy>(m, "OnClosedPolicy")
      .value("RAISE", dmx_host::OnClosedPolicy::RAISE)
      .value("DROP", dmx_host::OnClosedPolicy::DROP)
      .export_values();

  py::class_<QueueConfig>(m, "QueueConfig")
      .def(py::init<>())
      .def_readwrite("min_batch_items", &QueueConfig::min_batch_items)
      .def_readwrite("min_batch_size", &QueueConfig::min_batch_size)
      .def_property(
          "max_linger_s",
          [](const QueueConfig& q) -> std::optional<double> {
            if (!q.max_linger) return std::nullopt;
            return q.max_linger->count();
          },
          [](QueueConfig& q, std::optional<double> v) {
            if (v) q.max_linger = Duration(*v);
            else q.max_linger.reset();
          })
      .def_readwrite("max_batch_items", &QueueConfig::max_batch_items)
      .def_readwrite("max_batch_size", &QueueConfig::max_batch_size)
      .def_readwrite("high_watermark_items", &QueueConfig::high_watermark_items)
      .def_readwrite("high_watermark_size", &QueueConfig::high_watermark_size);

  py::class_<EnqueuePolicy>(m, "EnqueuePolicy")
      .def(py::init<>())
      .def_readwrite("block", &EnqueuePolicy::block)
      .def_property(
          "timeout_s",
          [](const EnqueuePolicy& p) -> std::optional<double> {
            if (!p.timeout) return std::nullopt;
            return p.timeout->count();
          },
          [](EnqueuePolicy& p, std::optional<double> v) {
            if (v) p.timeout = Duration(*v);
            else p.timeout.reset();
          })
      .def_readwrite("on_full", &EnqueuePolicy::on_full)
      .def_readwrite("max_retries", &EnqueuePolicy::max_retries)
      .def_property(
          "retry_backoff_s",
          [](const EnqueuePolicy& p) { return p.retry_backoff.count(); },
          [](EnqueuePolicy& p, double v) { p.retry_backoff = Duration(v); })
      .def_readwrite("on_closed", &EnqueuePolicy::on_closed)
      .def_readwrite("drop_if_stopping", &EnqueuePolicy::drop_if_stopping);

  py::class_<ThreadFailure>(m, "ThreadFailure")
      .def_readonly("stage", &ThreadFailure::stage)
      .def_readonly("thread_name", &ThreadFailure::thread_name)
      .def_readonly("where", &ThreadFailure::where)
      .def_readonly("exc_type", &ThreadFailure::exc_type)
      .def_readonly("exc_what", &ThreadFailure::exc_what);

  py::class_<StageConfig>(m, "StageConfig")
      .def(py::init<>())
      .def_readwrite("name", &StageConfig::name)
      .def_readwrite("parallelism", &StageConfig::parallelism)
      .def_readwrite("input_queue", &StageConfig::input_queue)
      .def_readwrite("ingress_policy", &StageConfig::ingress_policy)
      .def_property(
          "thread_name_prefix",
          [](const StageConfig& s) { return s.thread_name_prefix; },
          [](StageConfig& s, std::optional<std::string> v) { s.thread_name_prefix = std::move(v); })
      // ClickHouse insert stage (the only stage in DMXHostEngine)
      .def_static(
          "clickhouse_insert",
          [](const dmx_host::ClickHouseClientConfig& ch_cfg, int parallelism, std::string name) {
            StageConfig cfg;
            cfg.name = std::move(name);
            cfg.parallelism = parallelism;
            cfg.process_fn = [](std::vector<dmx_host::dmx_host_queue_item> batch, QueueT* next_q) {
              return dmx_host::ClickHouseInsertStage::ProcessFn<QueueT>(std::move(batch), next_q);
            };
            // Stored by value in std::any; ClickHouseInsertStage::ThreadInitAny will any_cast it.
            cfg.thread_init_config = ch_cfg;
            cfg.thread_init = &dmx_host::ClickHouseInsertStage::ThreadInitAny;
            cfg.thread_cleanup = &dmx_host::ClickHouseInsertStage::ThreadCleanupAny;
            return cfg;
          },
          py::arg("clickhouse_config"),
          py::arg("parallelism") = 1,
          py::arg("name") = "clickhouse_insert");

  py::class_<DMXHostEngine, std::shared_ptr<DMXHostEngine>>(m, "DMXHostEngine")
      .def(py::init<StageConfig>(), py::arg("insert_stage"))
      .def("start", &DMXHostEngine::start)
      .def("stop",
           [](DMXHostEngine& self, bool graceful, std::optional<double> timeout_s) {
             if (timeout_s) {
               return self.stop(graceful, DMXHostEngine::Duration(*timeout_s));
             }
             return self.stop(graceful, std::nullopt);
           },
           py::arg("graceful") = true,
           py::arg("timeout_s") = std::optional<double>(),
           py::call_guard<py::gil_scoped_release>())
      .def("close_input", &DMXHostEngine::close_input)
      .def("request_abort", &DMXHostEngine::request_abort)
      .def("join",
           [](DMXHostEngine& self, std::optional<double> timeout_s) {
             if (timeout_s) return self.join(DMXHostEngine::Duration(*timeout_s));
             return self.join(std::nullopt);
           },
           py::arg("timeout_s") = std::optional<double>(),
           py::call_guard<py::gil_scoped_release>())
      .def("failures", &DMXHostEngine::failures)
      .def("raise_if_failed", &DMXHostEngine::raise_if_failed)
      // Submit a pre-formatted ClickHouseRow directly to the insert stage.
      // Called from the ring transport drain callback after format processing.
      .def("submit_direct",
           [](DMXHostEngine& self,
              const std::string& model_id, int32_t shard_rank,
              const std::string& req_id, const std::string& act_name,
              int32_t layer_no, int32_t start_token, int32_t end_token,
              at::Tensor tensor) {
             at::Tensor t = tensor.is_contiguous() ? tensor : tensor.contiguous();
             uint64_t nbytes = static_cast<uint64_t>(t.nbytes());
             dmx_host::ClickHouseRow row;
             row.push_back(model_id);
             row.push_back(req_id);
             row.push_back(act_name);
             row.push_back(layer_no);
             row.push_back(shard_rank);
             row.push_back(start_token);
             row.push_back(end_token);
             row.push_back(std::move(t));
             self.submit_direct(std::move(row), nbytes);
           },
           py::arg("model_id"), py::arg("shard_rank"),
           py::arg("req_id"), py::arg("act_name"),
           py::arg("layer_no"), py::arg("start_token"), py::arg("end_token"),
           py::arg("tensor"),
           py::call_guard<py::gil_scoped_release>());

  // -------------------------------------------------------------------------
  // Ring offload engine
  // -------------------------------------------------------------------------
  py::class_<ring_py::RingConfig>(m, "RingConfig")
      .def(py::init<>())
      .def_readwrite("task_ring_entries",         &ring_py::RingConfig::task_ring_entries)
      .def_readwrite("payload_ring_bytes",        &ring_py::RingConfig::payload_ring_bytes)
      .def_readwrite("chunk_bytes",               &ring_py::RingConfig::chunk_bytes)
      .def_readwrite("pinned_pool_bytes",         &ring_py::RingConfig::pinned_pool_bytes)
      .def_readwrite("wait_policy",               &ring_py::RingConfig::wait_policy)
      .def_readwrite("no_progress_timeout_cycles",&ring_py::RingConfig::no_progress_timeout_cycles)
      .def_readwrite("drop_reporting",            &ring_py::RingConfig::drop_reporting);

  py::class_<ring_py::RingEnginePy, std::shared_ptr<ring_py::RingEnginePy>>(m, "RingEngine")
      .def(py::init([](ring_py::RingConfig cfg, py::object host_engine_obj) {
             // Build a C++ SubmitFn from the DMXHostEngine shared_ptr.
             // The callback runs from the C++ callback thread without the GIL.
             ring_py::SubmitFn submit_fn;
             if (!host_engine_obj.is_none()) {
                 auto host = host_engine_obj.cast<std::shared_ptr<dmx_host::DMXHostEngine>>();
                 submit_fn = [host](const std::string& model_id, int32_t shard_rank,
                                    const std::string& req_id, const std::string& act_name,
                                    int32_t layer_no, int32_t start_token, int32_t end_token,
                                    at::Tensor slice) {
                     dmx_host::ClickHouseRow row;
                     row.emplace_back(model_id);
                     row.emplace_back(req_id);
                     row.emplace_back(act_name);
                     row.emplace_back(layer_no);
                     row.emplace_back(shard_rank);
                     row.emplace_back(start_token);
                     row.emplace_back(end_token);
                     uint64_t nbytes = static_cast<uint64_t>(slice.nbytes());
                     row.emplace_back(std::move(slice));
                     host->submit_direct(std::move(row), nbytes);
                 };
             }
             return std::make_shared<ring_py::RingEnginePy>(
                 std::move(cfg), std::move(submit_fn));
           }),
           py::arg("config"), py::arg("host_engine") = py::none())
      .def("init",  &ring_py::RingEnginePy::init,
           py::arg("stream_handle") = uint64_t{0})
      .def("start", &ring_py::RingEnginePy::start)
      .def("stop",  &ring_py::RingEnginePy::stop,
           py::call_guard<py::gil_scoped_release>())
      // Low-level hook: d_ptr and stream_handle are raw CUDA pointers as ints.
      .def("hook",
           &ring_py::RingEnginePy::hook,
           py::arg("d_ptr"), py::arg("nbytes"),
           py::arg("logical_task_id"),
           py::arg("hook_type") = uint32_t{0},
           py::arg("hook_id")   = uint32_t{0},
           py::arg("stream_handle") = uint64_t{0},
           py::call_guard<py::gil_scoped_release>())
      // Push metadata for the next tensor before calling hook().
      .def("push_meta",
           [](ring_py::RingEnginePy& self,
              const std::string& hook_name,
              const std::string& model_id,
              int32_t shard_rank,
              py::list req_ids_py,
              py::list token_ranges_py,
              py::list shape_py,
              py::object dtype_obj) {
               ring_py::TensorMeta meta;
               meta.hook_name  = hook_name;
               meta.model_id   = model_id;
               meta.shard_rank = shard_rank;
               meta.dtype      = static_cast<int>(dtype_obj.cast<at::ScalarType>());
               for (auto h : shape_py)
                   meta.shape.push_back(py::cast<int64_t>(h));
               for (size_t i = 0; i < static_cast<size_t>(py::len(req_ids_py)); ++i) {
                   ring_py::RequestMeta rm;
                   rm.req_id      = py::cast<std::string>(req_ids_py[i]);
                   py::tuple tr   = token_ranges_py[i].cast<py::tuple>();
                   rm.start_token = py::cast<int32_t>(tr[0]);
                   rm.end_token   = py::cast<int32_t>(tr[1]);
                   meta.requests.push_back(std::move(rm));
               }
               self.push_meta(std::move(meta));
           },
           py::arg("hook_name"), py::arg("model_id"), py::arg("shard_rank"),
           py::arg("req_ids"), py::arg("token_ranges"),
           py::arg("shape"), py::arg("dtype"))
      .def("pop_last_meta", &ring_py::RingEnginePy::pop_last_meta,
           py::call_guard<py::gil_scoped_release>());
}
