// Public API implementations: submit/add/seal/resolve/future/stats/close

#include "native_engine_internal.h"
#include "engine_utils.h"
#include "nvtx_shim.h"

namespace monitoring {

namespace py = pybind11;

py::dict NativeMonitoringEngine::Impl::get_stats() {
  py::dict d;
  d["total_steps"] = stats_total_steps_.load(std::memory_order_relaxed);
  d["total_tasks"] = stats_total_tasks_.load(std::memory_order_relaxed);
  d["submit_us"] = stats_submit_us_.load(std::memory_order_relaxed);
  d["process_us"] = stats_process_us_.load(std::memory_order_relaxed);
  d["callback_us"] = stats_callback_us_.load(std::memory_order_relaxed);
  d["hook_calls"] = stats_hook_calls_.load(std::memory_order_relaxed);
  d["hook_enqueued"] = stats_hook_enqueued_.load(std::memory_order_relaxed);
  d["pool_hits"] = stats_pool_hits_.load(std::memory_order_relaxed);
  d["pool_misses"] = stats_pool_misses_.load(std::memory_order_relaxed);
  d["pool_high_watermark_bytes"] = stats_pool_high_watermark_bytes_.load(std::memory_order_relaxed);
  d["pool_fallbacks"] = stats_pool_fallbacks_.load(std::memory_order_relaxed);
  d["host_memcpy_mb"] = static_cast<double>(stats_memcpy_bytes_.load(std::memory_order_relaxed)) / (1024.0 * 1024.0);
  d["pending_notifies"] = stats_pending_notifies_.load(std::memory_order_relaxed);
  d["inflight_bytes"] = inflight_bytes_.load(std::memory_order_relaxed);
  d["cap_enabled"] = congestion_cap_enabled_;
  d["cap_ratio"] = congestion_cap_ratio_;
  d["driver_guard_bytes"] = driver_guard_bytes_;
  d["driver_free_bytes"] = last_driver_free_bytes_.load(std::memory_order_relaxed);
  d["allocator_allocated_bytes"] = last_allocated_bytes_.load(std::memory_order_relaxed);
  d["allocator_reserved_bytes"] = last_reserved_bytes_.load(std::memory_order_relaxed);
  d["clear_calls"] = stats_clear_calls_.load(std::memory_order_relaxed);
  d["clear_ms_total"] = static_cast<double>(stats_clear_us_.load(std::memory_order_relaxed)) / 1000.0;
  d["clear_scanned_total"] = stats_clear_scanned_.load(std::memory_order_relaxed);
  d["clear_ready_total"] = stats_clear_ready_.load(std::memory_order_relaxed);
  if (enable_host_copy_pool_ && host_copy_pool_) {
    d["host_copy_queue_depth"] = host_copy_pool_->queue_depth_.load(std::memory_order_relaxed);
    d["host_copy_total_tasks"] = host_copy_pool_->total_tasks_.load(std::memory_order_relaxed);
    d["host_copy_total_mb"] = static_cast<double>(host_copy_pool_->total_bytes_.load(std::memory_order_relaxed)) / (1024.0 * 1024.0);
  }
  if (enable_gather_h2h_) {
    d["gather_h2h_mb"] = static_cast<double>(stats_gather_h2h_bytes_.load(std::memory_order_relaxed)) / (1024.0 * 1024.0);
    d["gather_h2h_calls"] = stats_gather_h2h_calls_.load(std::memory_order_relaxed);
  }
  return d;
}

void NativeMonitoringEngine::Impl::set_capture_schedule(int64_t step_stride,
                                                        int64_t step_offset,
                                                        int64_t warmup_steps,
                                                        bool capture_prefill,
                                                        bool capture_decode,
                                                        int64_t request_stride,
                                                        int64_t request_offset,
                                                        int64_t warmup_requests) {
  schedule_.step_stride = std::max<int64_t>(1, step_stride);
  schedule_.step_offset = std::max<int64_t>(0, step_offset);
  schedule_.warmup_steps = std::max<int64_t>(0, warmup_steps);
  schedule_.capture_prefill = capture_prefill;
  schedule_.capture_decode = capture_decode;
  schedule_.request_stride = std::max<int64_t>(1, request_stride);
  schedule_.request_offset = std::max<int64_t>(0, request_offset);
  schedule_.warmup_requests = std::max<int64_t>(0, warmup_requests);
}

bool NativeMonitoringEngine::Impl::should_capture_request(int64_t request_id) const {
  if (request_id < schedule_.warmup_requests) {
    return false;
  }
  int64_t effective = request_id - schedule_.warmup_requests;
  if (effective < schedule_.request_offset) {
    return false;
  }
  return ((effective - schedule_.request_offset) % schedule_.request_stride) == 0;
}

bool NativeMonitoringEngine::Impl::should_capture_step(int64_t step_id, int64_t phase) const {
  StepPhase step_phase = static_cast<StepPhase>(phase);
  if (step_phase == StepPhase::kPrefill && !schedule_.capture_prefill) {
    return false;
  }
  if (step_phase == StepPhase::kDecode && !schedule_.capture_decode) {
    return false;
  }
  if (step_id < schedule_.warmup_steps) {
    return false;
  }
  int64_t effective = step_id - schedule_.warmup_steps;
  if (effective < schedule_.step_offset) {
    return false;
  }
  return ((effective - schedule_.step_offset) % schedule_.step_stride) == 0;
}

void NativeMonitoringEngine::Impl::update_capture_enabled(int64_t step_id, int64_t phase) {
  bool req_enabled = request_capture_enabled_.load(std::memory_order_acquire);
  bool step_enabled = should_capture_step(step_id, phase);
  capture_enabled_.store(req_enabled && step_enabled, std::memory_order_release);
}

void NativeMonitoringEngine::Impl::begin_request(int64_t request_id) {
  current_request_id_.store(request_id, std::memory_order_release);
  bool enabled = should_capture_request(request_id);
  request_capture_enabled_.store(enabled, std::memory_order_release);
}

void NativeMonitoringEngine::Impl::begin_step(int64_t step_id, int64_t phase) {
  current_step_id_.store(step_id, std::memory_order_release);
  current_phase_.store(phase, std::memory_order_release);
  update_capture_enabled(step_id, phase);
}

void NativeMonitoringEngine::Impl::record_callback_duration(int64_t us) {
  stats_callback_us_.fetch_add(us, std::memory_order_relaxed);
}

void NativeMonitoringEngine::Impl::set_partial_seal_config(bool enabled,
                                                           int64_t chunk_bytes,
                                                           bool cap_enabled,
                                                           double cap_ratio,
                                                           int64_t driver_guard_mb) {
  partial_seal_enabled_ = enabled;
  partial_seal_chunk_bytes_ = std::max<int64_t>(0, chunk_bytes);
  congestion_cap_enabled_ = cap_enabled;
  congestion_cap_ratio_ = clamp_ratio(cap_ratio);
  driver_guard_bytes_ = mb_to_bytes(driver_guard_mb);
  memory_stats_refresh_tick_.store(0, std::memory_order_release);
}

std::vector<int64_t> NativeMonitoringEngine::Impl::submit_step(int64_t step_id,
                                                               const py::list& tasks,
                                                               std::optional<uint64_t> stream_handle) {
  mon_nvtx_push("MonEng::submit_step");
  auto t0 = std::chrono::steady_clock::now();
  StepWork work;
  work.step_id = step_id;
  work.final_chunk = true;

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
    work.bytes += estimate_task_bytes(entry.spec);
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
    mon_nvtx_pop();
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
  mon_nvtx_pop();
  return tokens;
}

int64_t NativeMonitoringEngine::Impl::add_task(int64_t step_id, const py::tuple& task_tuple) {
  mon_nvtx_push("MonEng::add_task");
  auto t_add0 = std::chrono::steady_clock::now();
  TaskSpec spec = parse_task_tuple(task_tuple);

  // Allocate token + result slot
  int64_t token = next_token_++;
  {
    auto slot = std::make_shared<ResultSlot>();
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slots_.emplace(token, std::move(slot));
  }

  TaskEntry entry;
  entry.spec = std::move(spec);
  entry.token = token;
  pending_tasks_.fetch_add(1, std::memory_order_relaxed);
  stats_total_tasks_.fetch_add(1, std::memory_order_relaxed);
  append_task_entry_and_maybe_seal(step_id, std::move(entry));
  auto t_add1 = std::chrono::steady_clock::now();
  auto us_add = std::chrono::duration_cast<std::chrono::microseconds>(t_add1 - t_add0).count();
  stats_submit_us_.fetch_add(static_cast<int64_t>(us_add), std::memory_order_relaxed);
  mon_nvtx_pop();
  return token;
}

void NativeMonitoringEngine::Impl::seal_step(int64_t step_id, std::optional<uint64_t> stream_handle) {
  mon_nvtx_push("MonEng::seal_step");
  auto t0 = std::chrono::steady_clock::now();
  std::optional<StepWork> maybe_work;
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    maybe_work = maybe_cut_open_step_chunk_locked(step_id, /*force_tail=*/true);
  }

  if (!maybe_work.has_value()) {
    if (stream_handle.has_value()) {
      cudaEvent_t event;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(*stream_handle);
      C10_CUDA_CHECK(cudaEventRecord(event, stream));
      C10_CUDA_CHECK(cudaEventSynchronize(event));
      C10_CUDA_CHECK(cudaEventDestroy(event));
    }
    mon_nvtx_pop();
    return;
  }
  StepWork work = std::move(*maybe_work);

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
  stats_total_steps_.fetch_add(1, std::memory_order_relaxed);
  auto t1 = std::chrono::steady_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  stats_submit_us_.fetch_add(static_cast<int64_t>(us), std::memory_order_relaxed);
  mon_nvtx_pop();
}

void NativeMonitoringEngine::Impl::resolve_all() {
  mon_nvtx_push("MonEng::resolve_all");
  py::gil_scoped_release release;

  std::vector<StepWork> ready;
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    for (auto& kv : open_steps_) {
      if (!kv.second.tasks.empty()) {
        kv.second.final_chunk = true;
        stats_total_steps_.fetch_add(1, std::memory_order_relaxed);
        ready.emplace_back(std::move(kv.second));
      }
    }
    open_steps_.clear();
  }

  for (auto& item : ready) {
    dispatch_step(std::move(item));
  }

  ready.clear();
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
  mon_nvtx_push("MonEng::resolve_wait");
  pending_cv_.wait(lock, [&] { return pending_tasks_.load(std::memory_order_acquire) == 0; });
  mon_nvtx_pop();
  mon_nvtx_pop();
}

bool NativeMonitoringEngine::Impl::future_ready(int64_t token) {
  auto slot = get_slot(token);
  std::lock_guard<std::mutex> lock(slot->mutex);
  return slot->ready;
}

bool NativeMonitoringEngine::Impl::future_wait(int64_t token,
                                               std::optional<double> timeout,
                                               bool called_from_cpp /* = false */ ) {
  auto slot = get_slot(token);
  std::unique_ptr<py::gil_scoped_release> gil_release;
  if (!called_from_cpp) {
    gil_release = std::make_unique<py::gil_scoped_release>();
  }
  std::unique_lock<std::mutex> lock(slot->mutex);
  if (timeout.has_value()) {
    return slot->cv.wait_for(lock,
                             std::chrono::duration<double>(*timeout),
                             [&] { return slot->ready; });
  }
  slot->cv.wait(lock, [&] { return slot->ready; });
  return true;
}

at::Tensor NativeMonitoringEngine::Impl::future_result(int64_t token,
                                                       std::optional<double> timeout,
                                                       bool called_from_cpp /* = false */ ) {
  mon_nvtx_push("MonEng::future_result");
  auto slot = get_slot(token);
  {
    // Python-facing calls: release GIL while waiting.
    // C++ calls: do NOT touch GIL or Python exception APIs.
    std::unique_ptr<py::gil_scoped_release> gil_release;
    if (!called_from_cpp) {
      gil_release = std::make_unique<py::gil_scoped_release>();
    }

    std::unique_lock<std::mutex> lock(slot->mutex);
    if (timeout.has_value()) {
      if (!slot->cv.wait_for(lock,
                             std::chrono::duration<double>(*timeout),
                             [&] { return slot->ready; })) {
        if (!called_from_cpp) {
          // Re-acquire the GIL before touching Python exception state.
          gil_release.reset();
          PyErr_SetString(PyExc_TimeoutError,
                          "Future timed out before data was ready");
          throw py::error_already_set();
        }
        throw std::runtime_error("Future timed out before data was ready");
      }
    } else {
      slot->cv.wait(lock, [&] { return slot->ready; });
    }

    if (slot->has_error) {
      std::string message = slot->error;
      lock.unlock();
      remove_slot(token);
      mon_nvtx_pop();
      throw std::runtime_error(message);
    }

    if (slot->consumed) {
      at::Tensor tensor = slot->tensor;
      mon_nvtx_pop();
      return tensor;
    }

    slot->consumed = true;
  }

  at::Tensor tensor = slot->tensor;
  remove_slot(token);
  mon_nvtx_pop();
  return tensor;
}

void NativeMonitoringEngine::Impl::clear_completed_results() {
  clear_completed_results_internal();
}

void NativeMonitoringEngine::Impl::close() {
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
    py::gil_scoped_release release;
    worker_.join();
  }

  // Stop host-copy pool after draining queued jobs
  if (host_copy_pool_) {
    {
      std::lock_guard<std::mutex> lock(host_copy_pool_->queue_mutex_);
      host_copy_pool_->stop_.store(true, std::memory_order_relaxed);
    }
    host_copy_pool_->queue_cv_.notify_all();
    {
      py::gil_scoped_release release;
      for (auto& t : host_copy_pool_->workers_) {
        if (t.joinable()) t.join();
      }
    }
    host_copy_pool_.reset();
  }

  // Destroy any remaining result slots to avoid memory leaks.
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slots_.clear();
  }
}

}  // namespace monitoring
