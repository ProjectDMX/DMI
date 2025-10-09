#include <torch/extension.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

namespace monitoring {

namespace py = pybind11;

enum class SliceMode { Identity, Int, Range, Array };

struct SliceSpec {
  SliceMode mode{SliceMode::Identity};
  int64_t int_value{0};
  std::optional<int64_t> start;
  std::optional<int64_t> stop;
  std::optional<int64_t> step;
  std::vector<int64_t> indices;
};

struct TaskSpec {
  at::Tensor tensor;
  int64_t slice_dim{-2};
  bool remove_batch_dim{false};
  bool can_slice{true};
  SliceSpec slice;
  std::optional<c10::Device> target_device;
};

struct TaskEntry {
  TaskSpec spec;
  int64_t token{0};
};

struct StepWork {
  int64_t step_id{0};
  std::vector<TaskEntry> tasks;
  cudaEvent_t event{nullptr};
};

struct ResultSlot {
  std::mutex mutex;
  std::condition_variable cv;
  bool ready{false};
  bool has_error{false};
  bool consumed{false};
  at::Tensor tensor;
  std::string error;
};

class NativeMonitoringEngine : public std::enable_shared_from_this<NativeMonitoringEngine> {
 public:
  NativeMonitoringEngine(int64_t queue_size,
                         std::optional<at::ScalarType> cache_dtype,
                         int64_t delay_steps)
      : max_queue_size_(queue_size > 0 ? static_cast<size_t>(queue_size) : 0),
        cache_dtype_(cache_dtype),
        delay_steps_(delay_steps > 0 ? delay_steps : 0),
        cache_stream_(at::cuda::getStreamFromPool(/*isHighPriority=*/false)) {
    worker_ = std::thread(&NativeMonitoringEngine::worker_loop, this);
  }

  ~NativeMonitoringEngine() {
    close();
  }

  py::dict get_stats() {
    py::dict d;
    d["total_steps"] = stats_total_steps_.load(std::memory_order_relaxed);
    d["total_tasks"] = stats_total_tasks_.load(std::memory_order_relaxed);
    d["submit_us"] = stats_submit_us_.load(std::memory_order_relaxed);
    d["process_us"] = stats_process_us_.load(std::memory_order_relaxed);
    return d;
  }

  std::vector<int64_t> submit_step(int64_t step_id,
                                   const py::list& tasks,
                                   std::optional<uint64_t> stream_handle) {
    auto t0 = std::chrono::steady_clock::now();
    StepWork work;
    work.step_id = step_id;

    std::vector<int64_t> tokens;
    tokens.reserve(tasks.size());

    for (auto task_obj : tasks) {
      TaskSpec spec;
      if (PyTuple_Check(task_obj.ptr())) {
        spec = parse_task_tuple(py::reinterpret_borrow<py::tuple>(task_obj));
      } else {
        throw std::runtime_error("Native backend expects a tuple task payload");
      }

      int64_t token = next_token_++;
      tokens.push_back(token);

      auto slot = std::make_shared<ResultSlot>();
      {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        slots_.emplace(token, std::move(slot));
      }

      TaskEntry entry;
      entry.spec = std::move(spec);
      entry.token = token;
      work.tasks.emplace_back(std::move(entry));
    }

    if (work.tasks.empty()) {
      if (stream_handle.has_value()) {
        cudaEvent_t event;
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_handle);
        C10_CUDA_CHECK(cudaEventRecord(event, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(event));
        C10_CUDA_CHECK(cudaEventDestroy(event));
      }
      return tokens;
    }

    if (stream_handle.has_value()) {
      cudaEvent_t event;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_handle);
      C10_CUDA_CHECK(cudaEventRecord(event, stream));
      work.event = event;
    }

    py::gil_scoped_release release;

    pending_tasks_.fetch_add(static_cast<int64_t>(work.tasks.size()), std::memory_order_relaxed);

    std::vector<StepWork> ready;
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      sealed_steps_.emplace_back(std::move(work));
      while (static_cast<int64_t>(sealed_steps_.size()) > delay_steps_) {
        ready.emplace_back(std::move(sealed_steps_.front()));
        sealed_steps_.pop_front();
      }
    }

    for (auto& item : ready) {
      dispatch_step(std::move(item));
    }

    // Stats: count steps/tasks and submit CPU time
    stats_total_steps_.fetch_add(1, std::memory_order_relaxed);
    stats_total_tasks_.fetch_add(static_cast<int64_t>(tokens.size()), std::memory_order_relaxed);
    auto t1 = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    stats_submit_us_.fetch_add(static_cast<int64_t>(us), std::memory_order_relaxed);
    return tokens;
  }

  // Add a single task to an open step (low-overhead path).
  int64_t add_task(int64_t step_id, const py::tuple& task_tuple) {
    TaskSpec spec = parse_task_tuple(task_tuple);

    // Allocate token + result slot
    int64_t token = next_token_++;
    {
      auto slot = std::make_shared<ResultSlot>();
      std::lock_guard<std::mutex> lock(slots_mutex_);
      slots_.emplace(token, std::move(slot));
    }

    // Append into open step
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      StepWork& work = open_steps_[step_id];
      work.step_id = step_id;

      TaskEntry entry;
      entry.spec = std::move(spec);
      entry.token = token;
      work.tasks.emplace_back(std::move(entry));
    }

    pending_tasks_.fetch_add(1, std::memory_order_relaxed);
    return token;
  }

  // Seal an open step and dispatch according to delay policy.
  void seal_step(int64_t step_id, std::optional<uint64_t> stream_handle) {
    StepWork work;
    bool has_work = false;
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      auto it = open_steps_.find(step_id);
      if (it != open_steps_.end()) {
        work = std::move(it->second);
        open_steps_.erase(it);
        has_work = true;
      }
    }

    if (!has_work) {
      // No tasks were added for this step. If a stream is provided, just sync an event to
      // maintain ordering guarantees, then return.
      if (stream_handle.has_value()) {
        cudaEvent_t event;
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_handle);
        C10_CUDA_CHECK(cudaEventRecord(event, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(event));
        C10_CUDA_CHECK(cudaEventDestroy(event));
      }
      return;
    }

    if (stream_handle.has_value()) {
      cudaEvent_t event;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_handle);
      C10_CUDA_CHECK(cudaEventRecord(event, stream));
      work.event = event;
    }

    std::vector<StepWork> ready;
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      sealed_steps_.emplace_back(std::move(work));
      while (static_cast<int64_t>(sealed_steps_.size()) > delay_steps_) {
        ready.emplace_back(std::move(sealed_steps_.front()));
        sealed_steps_.pop_front();
      }
    }

    for (auto& item : ready) {
      dispatch_step(std::move(item));
    }
  }

  void resolve_all() {
    py::gil_scoped_release release;

    std::vector<StepWork> ready;
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      while (!sealed_steps_.empty()) {
        ready.emplace_back(std::move(sealed_steps_.front()));
        sealed_steps_.pop_front();
      }
    }

    for (auto& item : ready) {
      dispatch_step(std::move(item));
    }

    std::unique_lock<std::mutex> lock(pending_mutex_);
    pending_cv_.wait(lock, [&] { return pending_tasks_.load(std::memory_order_acquire) == 0; });
  }

  bool future_ready(int64_t token) {
    auto slot = get_slot(token);
    std::lock_guard<std::mutex> lock(slot->mutex);
    return slot->ready;
  }

  bool future_wait(int64_t token, std::optional<double> timeout) {
    auto slot = get_slot(token);
    py::gil_scoped_release release;
    std::unique_lock<std::mutex> lock(slot->mutex);
    if (timeout.has_value()) {
      return slot->cv.wait_for(lock,
                               std::chrono::duration<double>(*timeout),
                               [&] { return slot->ready; });
    }
    slot->cv.wait(lock, [&] { return slot->ready; });
    return true;
  }

  at::Tensor future_result(int64_t token, std::optional<double> timeout) {
    auto slot = get_slot(token);
    {
      py::gil_scoped_release release;
      std::unique_lock<std::mutex> lock(slot->mutex);
      if (timeout.has_value()) {
        if (!slot->cv.wait_for(lock,
                               std::chrono::duration<double>(*timeout),
                               [&] { return slot->ready; })) {
          PyErr_SetString(PyExc_TimeoutError, "Future timed out before data was ready");
          throw py::error_already_set();
        }
      } else {
        slot->cv.wait(lock, [&] { return slot->ready; });
      }

      if (slot->has_error) {
        std::string message = slot->error;
        lock.unlock();
        remove_slot(token);
        throw std::runtime_error(message);
      }

      if (slot->consumed) {
        at::Tensor tensor = slot->tensor;
        return tensor;
      }

      slot->consumed = true;
    }

    at::Tensor tensor = slot->tensor;
    remove_slot(token);
    return tensor;
  }

  void clear_completed_results() {
    std::vector<int64_t> tokens;
    std::vector<std::shared_ptr<ResultSlot>> slot_refs;
    {
      std::lock_guard<std::mutex> lock(slots_mutex_);
      tokens.reserve(slots_.size());
      slot_refs.reserve(slots_.size());
      for (const auto& kv : slots_) {
        tokens.push_back(kv.first);
        slot_refs.push_back(kv.second);
      }
    }

    std::vector<int64_t> ready_tokens;
    ready_tokens.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
      auto& slot = slot_refs[i];
      std::lock_guard<std::mutex> slot_lock(slot->mutex);
      if (slot->ready) {
        ready_tokens.push_back(tokens[i]);
      }
    }

    if (!ready_tokens.empty()) {
      std::lock_guard<std::mutex> lock(slots_mutex_);
      for (int64_t token : ready_tokens) {
        slots_.erase(token);
      }
    }
  }

  void close() {
    bool expected = false;
    if (!closed_.compare_exchange_strong(expected, true)) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      while (!sealed_steps_.empty()) {
        dispatch_step(std::move(sealed_steps_.front()));
        sealed_steps_.pop_front();
      }
    }

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    queue_cv_.notify_all();

    if (worker_.joinable()) {
      worker_.join();
    }

    // Destroy any remaining result slots to avoid memory leaks.
    {
      std::lock_guard<std::mutex> lock(slots_mutex_);
      slots_.clear();
    }
  }

 private:
  TaskSpec parse_task_tuple(const py::tuple& task_tuple) {
    TORCH_CHECK(task_tuple.size() == 6,
                "Native backend expects task tuple of length 6, got ",
                task_tuple.size());

    TaskSpec spec;
    spec.tensor = task_tuple[0].cast<at::Tensor>();
    spec.slice_dim = task_tuple[1].cast<int64_t>();
    spec.remove_batch_dim = task_tuple[2].cast<bool>();
    spec.can_slice = task_tuple[3].cast<bool>();

    spec.slice = parse_slice_tuple(task_tuple[4].cast<py::tuple>());

    py::object device_obj = task_tuple[5];
    if (!device_obj.is_none()) {
      spec.target_device = device_obj.cast<c10::Device>();
    }
    return spec;
  }

  SliceSpec parse_slice_tuple(const py::tuple& slice_tuple) {
    TORCH_CHECK(slice_tuple.size() > 0, "Slice tuple cannot be empty");
    SliceSpec spec;
    std::string mode = slice_tuple[0].cast<std::string>();
    if (mode == "identity") {
      spec.mode = SliceMode::Identity;
      return spec;
    }

    if (mode == "int") {
      TORCH_CHECK(slice_tuple.size() >= 2,
                  "Slice tuple with mode=int must include value");
      spec.mode = SliceMode::Int;
      spec.int_value = slice_tuple[1].cast<int64_t>();
      return spec;
    }

    if (mode == "slice") {
      spec.mode = SliceMode::Range;
      py::object start = slice_tuple.size() > 1 ? slice_tuple[1] : py::none();
      py::object stop = slice_tuple.size() > 2 ? slice_tuple[2] : py::none();
      py::object step = slice_tuple.size() > 3 ? slice_tuple[3] : py::none();
      if (!start.is_none()) {
        spec.start = start.cast<int64_t>();
      }
      if (!stop.is_none()) {
        spec.stop = stop.cast<int64_t>();
      }
      if (!step.is_none()) {
        spec.step = step.cast<int64_t>();
      }
      return spec;
    }

    if (mode == "array") {
      spec.mode = SliceMode::Array;
      if (slice_tuple.size() > 1) {
        py::object values_obj = slice_tuple[1];
        if (py::isinstance<py::list>(values_obj)) {
          spec.indices = values_obj.cast<std::vector<int64_t>>();
        } else if (py::isinstance<py::tuple>(values_obj)) {
          auto values_tuple = values_obj.cast<py::tuple>();
          spec.indices.reserve(values_tuple.size());
          for (auto item : values_tuple) {
            spec.indices.push_back(item.cast<int64_t>());
          }
        } else if (!values_obj.is_none()) {
          spec.indices.push_back(values_obj.cast<int64_t>());
        }
      }
      return spec;
    }

    spec.mode = SliceMode::Identity;
    return spec;
  }

  std::shared_ptr<ResultSlot> get_slot(int64_t token) {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    auto it = slots_.find(token);
    if (it == slots_.end()) {
      throw std::runtime_error("Unknown future token");
    }
    return it->second;
  }

  void remove_slot(int64_t token) {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slots_.erase(token);
  }

  void dispatch_step(StepWork&& work) {
    if (work.tasks.empty()) {
      if (work.event != nullptr) {
        C10_CUDA_CHECK(cudaEventDestroy(work.event));
      }
      return;
    }

    if (max_queue_size_ > 0) {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (!stop_ && queue_.size() >= max_queue_size_) {
        lock.unlock();
        process_step(std::move(work));
        return;
      }
      queue_.push_back(std::move(work));
      lock.unlock();
      queue_cv_.notify_one();
      return;
    }

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      queue_.push_back(std::move(work));
    }
    queue_cv_.notify_one();
  }

  void process_step(StepWork&& work) {
    auto t0 = std::chrono::steady_clock::now();
    if (work.event != nullptr) {
      C10_CUDA_CHECK(cudaStreamWaitEvent(cache_stream_.stream(), work.event, 0));
      C10_CUDA_CHECK(cudaEventDestroy(work.event));
      work.event = nullptr;
    }

    // Set cache stream as current
    auto prev_stream = at::cuda::getCurrentCUDAStream(cache_stream_.device_index());
    at::cuda::setCurrentCUDAStream(cache_stream_);

    for (auto& entry : work.tasks) {
      try {
        at::Tensor result = run_task(entry.spec);
        store_result(entry.token, std::move(result));
      } catch (const c10::Error& err) {
        store_exception(entry.token, err.what());
      } catch (const std::exception& err) {
        store_exception(entry.token, err.what());
      }
    }

    // Restore previous stream
    at::cuda::setCurrentCUDAStream(prev_stream);

    auto t1 = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    stats_process_us_.fetch_add(static_cast<int64_t>(us), std::memory_order_relaxed);
  }

  at::Tensor run_task(const TaskSpec& spec) {
    at::Tensor tensor = spec.tensor;

    if (spec.target_device.has_value() && tensor.device() != *spec.target_device) {
      tensor = tensor.to(*spec.target_device, /*non_blocking=*/true, /*copy=*/false);
    }

    if (spec.remove_batch_dim) {
      TORCH_CHECK(tensor.dim() > 0, "Cannot remove batch dimension from scalar tensor");
      tensor = tensor.index({0});
    }

    if (spec.can_slice && spec.slice.mode != SliceMode::Identity) {
      tensor = apply_slice(tensor, spec.slice, spec.slice_dim);
    }

    if (cache_dtype_.has_value() && tensor.scalar_type() != *cache_dtype_) {
      tensor = tensor.to(*cache_dtype_, /*non_blocking=*/true, /*copy=*/false);
    }

    return tensor;
  }

  at::Tensor apply_slice(at::Tensor tensor, const SliceSpec& spec, int64_t slice_dim) {
    int64_t dim = slice_dim;
    if (dim < 0) {
      dim += tensor.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < tensor.dim(), "slice_dim out of range");

    switch (spec.mode) {
      case SliceMode::Identity:
        return tensor;
      case SliceMode::Int: {
        return tensor.select(dim, spec.int_value);
      }
      case SliceMode::Range: {
        int64_t start = spec.start.value_or(0);
        int64_t stop = spec.stop.value_or(tensor.size(dim));
        int64_t step = spec.step.value_or(1);

        if (step == 1) {
          return tensor.narrow(dim, start, stop - start);
        } else {
          return tensor.slice(dim, start, stop, step);
        }
      }
      case SliceMode::Array: {
        auto options = torch::TensorOptions().dtype(torch::kLong).device(tensor.device());
        at::Tensor idx = torch::tensor(spec.indices, options);
        return tensor.index_select(dim, idx);
      }
    }
    return tensor;
  }

  void store_result(int64_t token, at::Tensor&& tensor) {
    auto slot = get_slot(token);
    {
      std::lock_guard<std::mutex> lock(slot->mutex);
      slot->tensor = std::move(tensor);
      slot->ready = true;
    }
    slot->cv.notify_all();
    pending_tasks_.fetch_sub(1, std::memory_order_acq_rel);
    pending_cv_.notify_all();
  }

  void store_exception(int64_t token, const std::string& error) {
    auto slot = get_slot(token);
    {
      std::lock_guard<std::mutex> lock(slot->mutex);
      slot->has_error = true;
      slot->error = error;
      slot->ready = true;
    }
    slot->cv.notify_all();
    pending_tasks_.fetch_sub(1, std::memory_order_acq_rel);
    pending_cv_.notify_all();
  }

  void worker_loop() {
    while (true) {
      StepWork work;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) {
          break;
        }
        work = std::move(queue_.front());
        queue_.pop_front();
      }
      process_step(std::move(work));
    }
  }

  std::mutex staging_mutex_;
  // Open steps being built via add_task before they are sealed.
  std::unordered_map<int64_t, StepWork> open_steps_;
  std::deque<StepWork> sealed_steps_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<StepWork> queue_;
  size_t max_queue_size_{0};
  bool stop_{false};

  std::mutex slots_mutex_;
  std::unordered_map<int64_t, std::shared_ptr<ResultSlot>> slots_;

  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;
  std::atomic<int64_t> pending_tasks_{0};

  std::atomic<int64_t> next_token_{1};

  at::cuda::CUDAStream cache_stream_;
  std::optional<at::ScalarType> cache_dtype_;
  int64_t delay_steps_{0};

  std::thread worker_;
  std::atomic<bool> closed_{false};

  // Stats ---------------------------------------------------------------
  std::atomic<int64_t> stats_total_steps_{0};
  std::atomic<int64_t> stats_total_tasks_{0};
  std::atomic<int64_t> stats_submit_us_{0};
  std::atomic<int64_t> stats_process_us_{0};
};

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
      .def("add_task", &monitoring::NativeMonitoringEngine::add_task,
           py::arg("step_id"), py::arg("task"))
      .def("seal_step", &monitoring::NativeMonitoringEngine::seal_step,
           py::arg("step_id"), py::arg("stream_handle") = std::optional<uint64_t>())
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
