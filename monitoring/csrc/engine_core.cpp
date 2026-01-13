// Core execution: dispatch/worker, stream/event sync, result storage.

#include "native_engine_internal.h"
#include "nvtx_shim.h"
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
  if (const char* v = std::getenv("MON_NATIVE_AUTOCLEAR")) {
    auto_cleanup_ = (*v != '0');
  }

  // Configure pinned pool (enabled only when pinned offload is in use)
  enable_pinpool_ = false;
  if (use_pinned_ && move_to_cpu_) {
    // Default bins (bytes)
    pinpool_bins_bytes_ = {256ull * 1024ull, 512ull * 1024ull, 1024ull * 1024ull,
                           2ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull,
                           8ull * 1024ull * 1024ull};
    // Parse env vars
    if (const char* v = std::getenv("MON_NATIVE_PINPOOL")) {
      enable_pinpool_ = (*v != '0');
    } else {
      enable_pinpool_ = true; // default on when pinned offload is used
    }
    if (const char* v = std::getenv("MON_NATIVE_PINPOOL_BINS_KB")) {
      // format: 256,512,1024,2048
      pinpool_bins_bytes_.clear();
      std::string s(v);
      size_t start = 0;
      while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
        if (!tok.empty()) {
          size_t kb = static_cast<size_t>(std::strtoull(tok.c_str(), nullptr, 10));
          if (kb > 0) pinpool_bins_bytes_.push_back(kb * 1024ull);
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
      if (pinpool_bins_bytes_.empty()) {
        pinpool_bins_bytes_ = {256ull * 1024ull, 512ull * 1024ull, 1024ull * 1024ull,
                               2ull * 1024ull * 1024ull, 4ull * 1024ull * 1024ull,
                               8ull * 1024ull * 1024ull};
      }
    }
    if (const char* v = std::getenv("MON_NATIVE_PINPOOL_SLOTS_PER_BIN")) {
      int slots = std::atoi(v);
      if (slots > 0) pinpool_slots_per_bin_ = slots;
    }
    if (const char* v = std::getenv("MON_NATIVE_PINPOOL_MAX_MB")) {
      size_t mb = static_cast<size_t>(std::strtoull(v, nullptr, 10));
      if (mb > 0) pinpool_max_bytes_ = mb * 1024ull * 1024ull;
    }
    if (const char* v = std::getenv("MON_NATIVE_PIN_THRESH_BYTES")) {
      size_t th = static_cast<size_t>(std::strtoull(v, nullptr, 10));
      if (th > 0) pinpool_thresh_bytes_ = th;
    }
  }

  // Host-copy pool configuration (optional parallel memcpy)
  if (const char* v = std::getenv("MON_NATIVE_HOST_COPY_THREADS")) {
    host_copy_threads_ = std::atoi(v);
    if (host_copy_threads_ > 0) {
      enable_host_copy_pool_ = true;
      host_copy_pool_ = std::make_unique<HostCopyThreadPool>();
      if (const char* qv = std::getenv("MON_NATIVE_HOST_COPY_QUEUE_SIZE")) {
        size_t q = static_cast<size_t>(std::strtoull(qv, nullptr, 10));
        if (q > 0) host_copy_pool_->max_queue_size_ = q;
      }
      // Spawn workers
      for (int i = 0; i < host_copy_threads_; ++i) {
        host_copy_pool_->workers_.emplace_back(&NativeMonitoringEngine::Impl::host_copy_worker, this);
      }
    }
  }
  worker_ = std::thread(&NativeMonitoringEngine::Impl::worker_loop, this);

  // Set thread name for profiling tools
#ifdef __linux__
  pthread_setname_np(worker_.native_handle(), "MonEngWorker");
#endif
}

NativeMonitoringEngine::Impl::~Impl() {
  // Ensure threads are stopped even if close() wasn't called explicitly
  try {
    // Stop host-copy workers
    if (host_copy_pool_) {
      {
        std::lock_guard<std::mutex> lock(host_copy_pool_->queue_mutex_);
        host_copy_pool_->stop_.store(true, std::memory_order_release);
      }
      host_copy_pool_->queue_cv_.notify_all();
      for (auto& w : host_copy_pool_->workers_) {
        if (w.joinable()) w.join();
      }
      host_copy_pool_.reset();
    }

    // Stop main worker
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  } catch (...) {
    // best-effort cleanup
  }
}

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
  mon_nvtx_push("MonEng::dispatch_step");
  if (work.tasks.empty()) {
    if (work.event != nullptr) {
      C10_CUDA_CHECK(cudaEventDestroy(work.event));
    }
    mon_nvtx_pop();
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
    mon_nvtx_pop();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push_back(std::move(work));
  }
  queue_cv_.notify_one();
  mon_nvtx_pop();
}

void NativeMonitoringEngine::Impl::process_step(StepWork&& work) {
  mon_nvtx_push("MonEng::process_step");
  auto t0 = std::chrono::steady_clock::now();
  if (work.event != nullptr) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(cache_stream_.stream(), work.event, 0));
    C10_CUDA_CHECK(cudaEventDestroy(work.event));
    work.event = nullptr;
  }

  // Set cache stream as current
  auto prev_stream = at::cuda::getCurrentCUDAStream(cache_stream_.device_index());
  at::cuda::setCurrentCUDAStream(cache_stream_);
  mon_nvtx_push("MonEng::process_tasks");
  struct PendingResult { at::Tensor tensor; int64_t token; int64_t block_id; bool needs_sync; };
  std::vector<PendingResult> results;
  results.reserve(work.tasks.size());
  bool any_sync = false;
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
      mon_nvtx_push("MonEng::run_task");
      at::Tensor result = run_task(entry.spec);
      mon_nvtx_pop();
      bool needs_sync = result.device().is_cpu() && entry.spec.tensor.defined() && entry.spec.tensor.is_cuda();
      any_sync = any_sync || needs_sync;
      int64_t blk_id = -1;
      if (enable_pinpool_ && result.defined() && result.device().is_cpu()) {
        blk_id = find_pool_block_id(result.data_ptr());
      }
      results.push_back(PendingResult{std::move(result), entry.token, blk_id, needs_sync});
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
  mon_nvtx_pop();

  // Ensure async D2H completes before exposing CPU tensors: single sync per step
  if (any_sync) {
    mon_nvtx_push("MonEng::sync_d2h");
    C10_CUDA_CHECK(cudaStreamSynchronize(cache_stream_.stream()));
    mon_nvtx_pop();
  }

  // Restore previous stream
  at::cuda::setCurrentCUDAStream(prev_stream);

  // Host memcpy (pinned -> pageable) and store results
  mon_nvtx_push("MonEng::finalize_results");
  // Strategy: Always convert pinned CPU results to pageable to avoid exhausting pinned memory.
  // Three scenarios for CPU results:
  // 1. Pool-backed pinned (block_id >= 0): copy to pageable + release pool block
  // 2. Non-pool pinned (block_id < 0): copy to pageable + auto-release via refcount
  // 3. Already pageable or GPU: store directly without host copy
  if (enable_host_copy_pool_) {
    for (auto& pr : results) {
      const bool need_repage = pr.tensor.defined() && pr.tensor.device().is_cpu() && pr.tensor.is_pinned();
      if (need_repage) {
        // Enqueue copy job or process inline if queue is full
        CopyJob job{pr.tensor, pr.block_id, pr.token};
        bool enqueued = false;
        {
          std::unique_lock<std::mutex> lock(host_copy_pool_->queue_mutex_);
          if (host_copy_pool_->queue_.size() < host_copy_pool_->max_queue_size_) {
            host_copy_pool_->queue_.push_back(std::move(job));
            host_copy_pool_->queue_depth_.fetch_add(1, std::memory_order_relaxed);
            enqueued = true;
          }
        }
        if (enqueued) {
          host_copy_pool_->queue_cv_.notify_one();
        } else {
          // Fallback: process in current thread to apply backpressure
          process_copy_job(job);
        }
      } else {
        store_result(pr.token, std::move(pr.tensor));
      }
    }
  } else {
    for (auto& pr : results) {
      const bool need_repage = pr.tensor.defined() && pr.tensor.device().is_cpu() && pr.tensor.is_pinned();
      if (need_repage) {
        CopyJob job{pr.tensor, pr.block_id, pr.token};
        process_copy_job(job);
      } else {
        store_result(pr.token, std::move(pr.tensor));
      }
    }
  }
  mon_nvtx_pop();

  auto t1 = std::chrono::steady_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  stats_process_us_.fetch_add(static_cast<int64_t>(us), std::memory_order_relaxed);
  mon_nvtx_pop();
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
      // Always prefer pinned destinations for D2H, regardless of size.
      size_t nbytes = static_cast<size_t>(tensor.nbytes());
      at::ScalarType dt = tensor.scalar_type();
      if (enable_pinpool_) {
        // Try the pinned pool first (no size threshold)
        auto got = acquire_pinned_block(nbytes, dt);
        if (got.first.defined()) {
          at::Tensor dst = got.first.view(tensor.sizes());
          dst.copy_(tensor, /*non_blocking=*/true);
          return dst;
        }
        // Pool could not provide a block (capacity/pressure) — fall back to direct pinned alloc
        auto opts = tensor.options().device(torch::kCPU).pinned_memory(true);
        at::Tensor dst = at::empty_like(tensor, opts);
        dst.copy_(tensor, /*non_blocking=*/true);
        tensor = dst;
      } else {
        // Pool disabled — allocate pinned destination directly
        auto opts = tensor.options().device(torch::kCPU).pinned_memory(true);
        at::Tensor dst = at::empty_like(tensor, opts);
        dst.copy_(tensor, /*non_blocking=*/true);
        tensor = dst;
      }
    } else {
      // Pinned off not requested — keep legacy pageable transfer
      tensor = tensor.to(torch::kCPU, /*non_blocking=*/true, /*copy=*/true);
    }
  }

  return tensor;
}

// --------------------- Pinned pool helpers ---------------------

size_t NativeMonitoringEngine::Impl::pick_bin_bytes(size_t nbytes) {
  for (size_t b : pinpool_bins_bytes_) {
    if (b >= nbytes) return b;
  }
  // If larger than largest bin, round up to next multiple of largest bin
  if (!pinpool_bins_bytes_.empty()) {
    size_t last = pinpool_bins_bytes_.back();
    size_t mult = (nbytes + last - 1) / last;
    return mult * last;
  }
  return nbytes;
}

std::pair<at::Tensor, int64_t> NativeMonitoringEngine::Impl::acquire_pinned_block(size_t nbytes, at::ScalarType dtype) {
  if (!enable_pinpool_) return {at::Tensor(), -1};

  size_t cap = pick_bin_bytes(nbytes);
  at::Tensor view;
  int64_t blk_id = -1;
  void* base_ptr = nullptr;

  {
    // Minimize time under pool_mutex_; do not nest ptr_mutex_ while holding this lock.
    std::lock_guard<std::mutex> lock(pool_mutex_);
    // Try to find a free block with same dtype and sufficient capacity
    for (auto& blk : pinpool_blocks_) {
      if (!blk.in_use && blk.dtype == dtype && blk.capacity_bytes >= cap) {
        blk.in_use = true;
        stats_pool_hits_.fetch_add(1, std::memory_order_relaxed);
        size_t elem_size = c10::elementSize(dtype);
        int64_t numel = static_cast<int64_t>(nbytes / elem_size);
        view = blk.buf.narrow(0, 0, numel);
        blk_id = blk.id;
        base_ptr = view.data_ptr();
        break;
      }
    }

    if (!view.defined()) {
      // Can we allocate a new block?
      if (pinpool_total_bytes_ + cap <= pinpool_max_bytes_) {
        at::TensorOptions opts = torch::TensorOptions().device(torch::kCPU).pinned_memory(true).dtype(dtype);
        size_t elem_size = c10::elementSize(dtype);
        int64_t capacity_elems = static_cast<int64_t>(cap / elem_size);
        at::Tensor buf = at::empty({capacity_elems}, opts);
        PinnedBlock blk;
        blk.buf = buf;
        blk.capacity_bytes = cap;
        blk.dtype = dtype;
        blk.in_use = true;
        blk.id = next_block_id_++;
        pinpool_blocks_.push_back(blk);
        pinpool_total_bytes_ += cap;
        // Update high watermark
        int64_t hw = static_cast<int64_t>(pinpool_total_bytes_);
        int64_t prev = stats_pool_high_watermark_bytes_.load(std::memory_order_relaxed);
        while (hw > prev && !stats_pool_high_watermark_bytes_.compare_exchange_weak(prev, hw)) {}
        stats_pool_misses_.fetch_add(1, std::memory_order_relaxed);
        view = buf.narrow(0, 0, static_cast<int64_t>(nbytes / elem_size));
        blk_id = blk.id;
        base_ptr = view.data_ptr();
      } else {
        // Pool limit reached: fail
        stats_pool_fallbacks_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  if (!view.defined()) {
    return {at::Tensor(), -1};
  }

  // Record pointer mapping outside pool_mutex_ to reduce nested lock contention
  {
    std::lock_guard<std::mutex> lk(ptr_mutex_);
    ptr_to_block_id_[base_ptr] = blk_id;
  }

  return {view, blk_id};
}

void NativeMonitoringEngine::Impl::release_pool_block(int64_t block_id) {
  if (block_id < 0) return;
  void* ptr_to_clear = nullptr;
  // Find block and mark free with minimal critical section.
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto& blk : pinpool_blocks_) {
      if (blk.id == block_id) {
        blk.in_use = false;
        if (blk.buf.defined()) ptr_to_clear = blk.buf.data_ptr();
        break;
      }
    }
  }
  // Erase pointer mapping directly by key (O(log N)) outside pool_mutex_
  if (ptr_to_clear != nullptr) {
    std::lock_guard<std::mutex> lk(ptr_mutex_);
    ptr_to_block_id_.erase(ptr_to_clear);
  }
}

int64_t NativeMonitoringEngine::Impl::find_pool_block_id(void* ptr) {
  std::lock_guard<std::mutex> lk(ptr_mutex_);
  auto it = ptr_to_block_id_.find(ptr);
  if (it == ptr_to_block_id_.end()) return -1;
  return it->second; // do not erase here; erased on release
}

void NativeMonitoringEngine::Impl::process_copy_job(const CopyJob& job) {
  // host-to-host memcpy from pinned to pageable, then release block and store result
  auto pageable_opts = job.pinned_tensor.options().device(torch::kCPU).pinned_memory(false);
  at::Tensor dst = at::empty_like(job.pinned_tensor, pageable_opts);
  dst.copy_(job.pinned_tensor);
  if (job.block_id >= 0) {
    release_pool_block(job.block_id);
  }
  {
    // stats
    stats_memcpy_bytes_.fetch_add(dst.nbytes(), std::memory_order_relaxed);
    if (host_copy_pool_) host_copy_pool_->total_bytes_.fetch_add(dst.nbytes(), std::memory_order_relaxed);
  }
  store_result(job.token, std::move(dst));
}

void NativeMonitoringEngine::Impl::host_copy_worker() {
  // Worker loop for host-side memcpy and pool return
  while (true) {
    CopyJob job;
    bool got = false;
    {
      std::unique_lock<std::mutex> lock(host_copy_pool_->queue_mutex_);
      host_copy_pool_->queue_cv_.wait(lock, [&]{ return host_copy_pool_->stop_.load(std::memory_order_relaxed) || !host_copy_pool_->queue_.empty(); });
      if (!host_copy_pool_->queue_.empty()) {
        job = std::move(host_copy_pool_->queue_.front());
        host_copy_pool_->queue_.pop_front();
        host_copy_pool_->queue_depth_.fetch_sub(1, std::memory_order_relaxed);
        got = true;
      } else if (host_copy_pool_->stop_.load(std::memory_order_relaxed)) {
        break; // stop requested and queue drained
      }
    }
    if (got) {
      mon_nvtx_push("HostCopy::memcpy");
      process_copy_job(job);
      mon_nvtx_pop();
      host_copy_pool_->total_tasks_.fetch_add(1, std::memory_order_relaxed);
    }
  }
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
      // Optional: build index path using pinned host staging to produce
      // "Memcpy HtoD (Pinned)" instead of pageable in profilers.
      bool pin_index = false;
      if (const char* v = std::getenv("MON_NATIVE_PINNED_INDEX")) {
        pin_index = (*v != '0');
      }
      if (pin_index) {
        // Create a temporary CPU Long tensor from indices, then copy into
        // a pinned host buffer and transfer to device non-blocking.
        at::Tensor tmp = torch::tensor(spec.indices, torch::TensorOptions().dtype(torch::kLong));
        auto host_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU).pinned_memory(true);
        at::Tensor idx_host = at::empty_like(tmp, host_opts);
        idx_host.copy_(tmp, /*non_blocking=*/false);
        at::Tensor idx = idx_host.to(tensor.device(), /*non_blocking=*/true, /*copy=*/true);
        return tensor.index_select(dim, idx);
      } else {
        auto options = torch::TensorOptions().dtype(torch::kLong).device(tensor.device());
        at::Tensor idx = torch::tensor(spec.indices, options);
        return tensor.index_select(dim, idx);
      }
    }
  }
  return tensor;
}

void NativeMonitoringEngine::Impl::store_result(int64_t token, at::Tensor&& tensor) {
  mon_nvtx_push("MonEng::store_result");
  auto slot = get_slot(token);
  {
    std::lock_guard<std::mutex> lock(slot->mutex);
    slot->tensor = std::move(tensor);
    slot->ready = true;
  }
  slot->cv.notify_all();
  pending_tasks_.fetch_sub(1, std::memory_order_acq_rel);
  mon_nvtx_push("MonEng::pending_notify");
  pending_cv_.notify_all();
  stats_pending_notifies_.fetch_add(1, std::memory_order_relaxed);
  mon_nvtx_pop();
  mon_nvtx_pop();
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
  mon_nvtx_push("MonEng::pending_notify");
  pending_cv_.notify_all();
  stats_pending_notifies_.fetch_add(1, std::memory_order_relaxed);
  mon_nvtx_pop();
}

void NativeMonitoringEngine::Impl::clear_completed_results_internal() {
  mon_nvtx_push("MonEng::clear_results");
  auto t0 = std::chrono::steady_clock::now();
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
  auto t1 = std::chrono::steady_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  stats_clear_calls_.fetch_add(1, std::memory_order_relaxed);
  stats_clear_us_.fetch_add(static_cast<int64_t>(us), std::memory_order_relaxed);
  stats_clear_scanned_.fetch_add(static_cast<int64_t>(tokens.size()), std::memory_order_relaxed);
  stats_clear_ready_.fetch_add(static_cast<int64_t>(ready_tokens.size()), std::memory_order_relaxed);
  mon_nvtx_pop();
}

void NativeMonitoringEngine::Impl::worker_loop() {
  constexpr int64_t CLEANUP_INTERVAL = 8;  // Clean completed results every 8 steps
  int64_t steps_since_cleanup = 0;

  while (true) {
    StepWork work;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      mon_nvtx_push("MonEng::worker_wait");
      queue_cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      mon_nvtx_pop();
      if (stop_ && queue_.empty()) {
        break;
      }
      work = std::move(queue_.front());
      queue_.pop_front();
    }
    process_step(std::move(work));

    // Periodic cleanup to prevent ResultSlot accumulation
    ++steps_since_cleanup;
    if (auto_cleanup_ && steps_since_cleanup >= CLEANUP_INTERVAL) {
      mon_nvtx_push("MonEng::cleanup");
      clear_completed_results_internal();
      steps_since_cleanup = 0;
      mon_nvtx_pop();
    }
  }
}

}  // namespace monitoring
