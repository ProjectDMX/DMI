// Internal-only header for NativeMonitoringEngine implementation details.
// This file is not part of the public API surface and should only be
// included by .cpp files within this directory.

#ifndef MONITORING_NATIVE_ENGINE_INTERNAL_H_
#define MONITORING_NATIVE_ENGINE_INTERNAL_H_

#include "native_engine.h"

#include <torch/extension.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
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

struct HookConfig {
  std::string name;
  int64_t pos_dim{-2};
  bool remove_batch_dim{false};
  SliceSpec slice;
  std::optional<c10::Device> target_device;
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

struct NativeMonitoringEngine::Impl {
  // Lifecycle
  Impl(int64_t queue_size,
       std::optional<at::ScalarType> cache_dtype,
       int64_t delay_steps);
  ~Impl();

  // Public API (mirrors NativeMonitoringEngine)
  py::dict get_stats();

  std::vector<int64_t> submit_step(int64_t step_id,
                                   const py::list& tasks,
                                   std::optional<uint64_t> stream_handle);

  void begin_step(int64_t step_id);
  void record_callback_duration(int64_t us);

  std::vector<int64_t> submit_step_soa(int64_t step_id,
                                       const py::dict& spec,
                                       std::optional<uint64_t> stream_handle);

  int64_t add_task(int64_t step_id, const py::tuple& task_tuple);
  void seal_step(int64_t step_id, std::optional<uint64_t> stream_handle);

  void append_hook(int64_t step_id,
                   const std::string& hook_name,
                   at::Tensor tensor,
                   bool remove_batch_dim,
                   py::object pos_slice,
                   py::object target_device);

  void resolve_all();
  bool future_ready(int64_t token);
  bool future_wait(int64_t token, std::optional<double> timeout);
  at::Tensor future_result(int64_t token, std::optional<double> timeout);
  void clear_completed_results();
  void close();

  // Hook helpers exposed for top-level callback creation
  HookConfig* upsert_hook_config(const std::string& hook_name,
                                 bool remove_batch_dim,
                                 py::object pos_slice,
                                 py::object target_device);
  HookConfig* upsert_hook_config_tuple(const std::string& hook_name,
                                       bool remove_batch_dim,
                                       py::tuple slice_tuple,
                                       py::object target_device);
  void append_hook_current_step(const HookConfig& cfg, at::Tensor tensor);
  int64_t add_task_from_config(const HookConfig& cfg, at::Tensor tensor);

  // Internal helpers ----------------------------------------------------
  int64_t deduce_pos_dim(const std::string& name);
  SliceSpec parse_slice_py(py::object obj);
  TaskSpec parse_task_tuple(const py::tuple& task_tuple);
  SliceSpec parse_slice_tuple(const py::tuple& slice_tuple);

  std::shared_ptr<ResultSlot> get_slot(int64_t token);
  void remove_slot(int64_t token);
  void clear_completed_results_internal();

  void dispatch_step(StepWork&& work);
  void process_step(StepWork&& work);
  at::Tensor run_task(const TaskSpec& spec);
  at::Tensor apply_slice(at::Tensor tensor, const SliceSpec& spec, int64_t slice_dim);
  void store_result(int64_t token, at::Tensor&& tensor);
  void store_exception(int64_t token, const std::string& error);
  void worker_loop();

  // Data ----------------------------------------------------------------
  // D2H offload controls
  bool move_to_cpu_{false};
  bool use_pinned_{true};

  // Pinned memory pool (for stable GPU->CPU offload throughput)
  struct PinnedBlock {
    at::Tensor buf;              // 1D pinned tensor with element dtype
    size_t capacity_bytes{0};    // total bytes the block can hold
    at::ScalarType dtype{at::kByte};
    bool in_use{false};
    int64_t id{-1};
  };

  // Pool controls and state
  bool enable_pinpool_{false};
  size_t pinpool_thresh_bytes_{64 * 1024}; // small tensors fallback to pageable
  size_t pinpool_max_bytes_{512ull * 1024ull * 1024ull}; // total pool cap
  std::vector<size_t> pinpool_bins_bytes_; // capacity bins in bytes (ascending)
  int pinpool_slots_per_bin_{8};

  // Storage
  std::mutex pool_mutex_;
  std::vector<PinnedBlock> pinpool_blocks_; // all blocks
  std::unordered_map<void*, int64_t> ptr_to_block_id_; // view.data_ptr() -> block id
  std::mutex ptr_mutex_;
  int64_t next_block_id_{1};
  size_t pinpool_total_bytes_{0};

  // Pool helpers
  size_t pick_bin_bytes(size_t nbytes);
  std::pair<at::Tensor, int64_t> acquire_pinned_block(size_t nbytes, at::ScalarType dtype);
  void release_pool_block(int64_t block_id);
  int64_t find_pool_block_id(void* ptr);

  // Pool stats
  std::atomic<int64_t> stats_pool_hits_{0};
  std::atomic<int64_t> stats_pool_misses_{0};
  std::atomic<int64_t> stats_pool_high_watermark_bytes_{0};
  std::atomic<int64_t> stats_pool_fallbacks_{0};
  std::atomic<int64_t> stats_memcpy_bytes_{0};

  // Diagnostics: pending notify/wake/clear timings
  std::atomic<int64_t> stats_pending_notifies_{0};
  std::atomic<int64_t> stats_clear_calls_{0};
  std::atomic<int64_t> stats_clear_us_{0};
  std::atomic<int64_t> stats_clear_scanned_{0};
  std::atomic<int64_t> stats_clear_ready_{0};

  // Host-copy thread pool (optional)
  struct CopyJob {
    at::Tensor pinned_tensor; // pinned CPU tensor (from pool) to be copied to pageable
    int64_t block_id{-1};     // pool block id
    int64_t token{0};         // result token
  };

  struct HostCopyThreadPool {
    std::vector<std::thread> workers_;
    std::deque<CopyJob> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_{false};
    size_t max_queue_size_{512};
    // Stats
    std::atomic<int64_t> queue_depth_{0};
    std::atomic<int64_t> total_tasks_{0};
    std::atomic<int64_t> total_bytes_{0};
  };

  std::unique_ptr<HostCopyThreadPool> host_copy_pool_;
  bool enable_host_copy_pool_{false};
  int host_copy_threads_{0};

  void host_copy_worker();
  void process_copy_job(const CopyJob& job);

  std::mutex staging_mutex_;
  std::unordered_map<int64_t, StepWork> open_steps_;
  std::deque<StepWork> sealed_steps_;

  std::mutex hook_config_mutex_;
  std::unordered_map<std::string, std::unique_ptr<HookConfig>> hook_configs_;

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
  std::atomic<int64_t> current_step_id_{0};

  std::thread worker_;
  std::atomic<bool> closed_{false};

  // Stats ---------------------------------------------------------------
  std::atomic<int64_t> stats_total_steps_{0};
  std::atomic<int64_t> stats_total_tasks_{0};
  std::atomic<int64_t> stats_submit_us_{0};
  std::atomic<int64_t> stats_process_us_{0};
  std::atomic<int64_t> stats_callback_us_{0};
};

}  // namespace monitoring

#endif  // MONITORING_NATIVE_ENGINE_INTERNAL_H_
