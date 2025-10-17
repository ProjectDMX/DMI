// Core execution: dispatch/worker, stream/event sync, result storage.

#include "native_engine_internal.h"
#include <cstdlib>

namespace monitoring {

NativeMonitoringEngine::Impl::Impl(int64_t queue_size,
                                   std::optional<at::ScalarType> cache_dtype,
                                   int64_t delay_steps)
    : max_queue_size_(queue_size > 0 ? static_cast<size_t>(queue_size) : 0),
      cache_stream_(at::cuda::getStreamFromPool(/*isHighPriority=*/false)),
      cache_dtype_(cache_dtype),
      delay_steps_(delay_steps > 0 ? delay_steps : 0) {
  // Runtime controls for D2H offload (single-stream version)
  if (const char* v = std::getenv("MON_NATIVE_TO_CPU")) {
    move_to_cpu_ = (*v == '1');
  }
  if (const char* v = std::getenv("MON_NATIVE_PINNED")) {
    use_pinned_ = (*v != '0');
  }
  worker_ = std::thread(&NativeMonitoringEngine::Impl::worker_loop, this);
}

NativeMonitoringEngine::Impl::~Impl() = default;

std::shared_ptr<ResultSlot> NativeMonitoringEngine::Impl::get_slot(int64_t token) {
  std::lock_guard<std::mutex> lock(slots_mutex_);
  auto it = slots_.find(token);
  if (it == slots_.end()) {
    throw std::runtime_error("Unknown future token");
  }
  return it->second;
}

void NativeMonitoringEngine::Impl::remove_slot(int64_t token) {
  std::lock_guard<std::mutex> lock(slots_mutex_);
  slots_.erase(token);
}

void NativeMonitoringEngine::Impl::dispatch_step(StepWork&& work) {
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

void NativeMonitoringEngine::Impl::process_step(StepWork&& work) {
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
      if (entry.token == 0) {
        // Assign token lazily for builder/append_hook path
        entry.token = next_token_++;
        auto slot = std::make_shared<ResultSlot>();
        {
          std::lock_guard<std::mutex> lock(slots_mutex_);
          slots_.emplace(entry.token, std::move(slot));
        }
      }
      at::Tensor result = run_task(entry.spec);
      // Ensure async D2H completes before exposing CPU tensor to consumers
      if (result.device().is_cpu() && entry.spec.tensor.defined() && entry.spec.tensor.is_cuda()) {
        C10_CUDA_CHECK(cudaStreamSynchronize(cache_stream_.stream()));
      }
      store_result(entry.token, std::move(result));

      // Immediately release the input tensor to free GPU memory
      entry.spec.tensor = at::Tensor();
    } catch (const c10::Error& err) {
      store_exception(entry.token, err.what());
      entry.spec.tensor = at::Tensor();
    } catch (const std::exception& err) {
      store_exception(entry.token, err.what());
      entry.spec.tensor = at::Tensor();
    }
  }

  // Restore previous stream
  at::cuda::setCurrentCUDAStream(prev_stream);

  auto t1 = std::chrono::steady_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  stats_process_us_.fetch_add(static_cast<int64_t>(us), std::memory_order_relaxed);
}

at::Tensor NativeMonitoringEngine::Impl::run_task(const TaskSpec& spec) {
  at::Tensor tensor = spec.tensor;

  if (spec.target_device.has_value() && tensor.device() != *spec.target_device) {
    if (spec.target_device->is_cpu()) {
      // Defer GPU->CPU copy to the pinned-memory path below
    } else {
      tensor = tensor.to(*spec.target_device, /*non_blocking=*/true, /*copy=*/false);
    }
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

  // Optional D2H offload on current cache stream
  bool want_cpu = move_to_cpu_;
  if (spec.target_device.has_value() && spec.target_device->is_cpu()) {
    want_cpu = true;
  }
  if (want_cpu && tensor.is_cuda()) {
    if (use_pinned_) {
      auto opts = tensor.options().device(torch::kCPU).pinned_memory(true);
      at::Tensor dst = at::empty_like(tensor, opts);
      dst.copy_(tensor, /*non_blocking=*/true);
      tensor = dst;
    } else {
      tensor = tensor.to(torch::kCPU, /*non_blocking=*/true, /*copy=*/true);
    }
  }

  return tensor;
}

at::Tensor NativeMonitoringEngine::Impl::apply_slice(at::Tensor tensor, const SliceSpec& spec, int64_t slice_dim) {
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

void NativeMonitoringEngine::Impl::store_result(int64_t token, at::Tensor&& tensor) {
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

void NativeMonitoringEngine::Impl::store_exception(int64_t token, const std::string& error) {
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

void NativeMonitoringEngine::Impl::clear_completed_results_internal() {
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

void NativeMonitoringEngine::Impl::worker_loop() {
  constexpr int64_t CLEANUP_INTERVAL = 8;  // Clean completed results every 8 steps
  int64_t steps_since_cleanup = 0;

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

    // Periodic cleanup to prevent ResultSlot accumulation
    ++steps_since_cleanup;
    if (steps_since_cleanup >= CLEANUP_INTERVAL) {
      clear_completed_results_internal();
      steps_since_cleanup = 0;
    }
  }
}

}  // namespace monitoring
