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
#include <unordered_set>
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
  int64_t bytes{0};
  bool counted_inflight{false};
  bool final_chunk{false};
};

struct HookConfig {
  std::string name;
  int64_t pos_dim{-2};
  bool remove_batch_dim{false};
  SliceSpec slice;
  std::optional<c10::Device> target_device;
};

enum class StepPhase : int64_t {
  kUnknown = 0,
  kPrefill = 1,
  kDecode = 2,
};

struct CaptureSchedule {
  int64_t step_stride{1};
  int64_t step_offset{0};
  int64_t warmup_steps{0};
  bool capture_prefill{true};
  bool capture_decode{true};

  int64_t request_stride{1};
  int64_t request_offset{0};
  int64_t warmup_requests{0};
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
  void set_partial_seal_config(bool enabled,
                               int64_t chunk_bytes,
                               bool cap_enabled,
                               double cap_ratio,
                               int64_t driver_guard_mb);

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
  bool future_wait(int64_t token, std::optional<double> timeout, bool called_from_cpp = false);
  at::Tensor future_result(int64_t token, std::optional<double> timeout, bool called_from_cpp = false);
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
  std::pair<int64_t, int64_t> add_task_from_config(const HookConfig& cfg, at::Tensor tensor, int64_t step_id);

  // Internal helpers ----------------------------------------------------
  int64_t deduce_pos_dim(const std::string& name);
  SliceSpec parse_slice_py(py::object obj);
  TaskSpec parse_task_tuple(const py::tuple& task_tuple);
  SliceSpec parse_slice_tuple(const py::tuple& slice_tuple);
  void append_task_entry_and_maybe_seal(int64_t step_id, TaskEntry&& entry);
  int64_t estimate_task_bytes(const TaskSpec& spec) const;
  std::optional<StepWork> maybe_cut_open_step_chunk_locked(int64_t step_id, bool force_tail = false);
  bool maybe_refresh_memory_stats(int device_index);
  int64_t compute_allowed_inflight_bytes(int device_index);
  bool should_capture_request(int64_t request_id) const;
  bool should_capture_step(int64_t step_id, int64_t phase) const;
  void update_capture_enabled(int64_t step_id, int64_t phase);
  bool is_capture_enabled() const {
    return capture_enabled_.load(std::memory_order_acquire);
  }

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

  // Global-callback support ------------------------------------------------
  // Enabled hook names for current step (set from Python per run)
  std::mutex enabled_mutex_;
  std::unordered_set<std::string> enabled_hooks_;
  void set_enabled_hooks(py::object names_iterable);
  bool is_hook_enabled_unlocked(const std::string& name) const {
    return enabled_hooks_.empty() || enabled_hooks_.count(name) > 0;
  }

  // Step-local name->token mapping for collect_step_futures
  std::unordered_map<int64_t, std::vector<std::pair<std::string, std::pair<int64_t, int64_t> >>> step_name_tokens_;
  void record_step_name_token(int64_t step_id, const std::string& name, int64_t token, int64_t task_size);

  // Data ----------------------------------------------------------------
  // D2H offload controls
  bool move_to_cpu_{false};
  bool use_pinned_{true};
  bool auto_cleanup_{true};

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

  // Partial seal + congestion controls
  bool partial_seal_enabled_{true};
  int64_t partial_seal_chunk_bytes_{64 * 1024 * 1024};
  bool congestion_cap_enabled_{false};
  double congestion_cap_ratio_{0.8};
  int64_t driver_guard_bytes_{1024ll * 1024ll * 1024ll};
  std::atomic<int64_t> inflight_bytes_{0};
  int64_t memory_stats_refresh_interval_{8};
  std::atomic<int64_t> memory_stats_refresh_tick_{0};
  std::atomic<int64_t> last_total_mem_bytes_{0};
  std::atomic<int64_t> last_driver_free_bytes_{0};
  std::atomic<int64_t> last_allocated_bytes_{0};
  std::atomic<int64_t> last_reserved_bytes_{0};

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
  std::atomic<int64_t> current_request_id_{0};
  std::atomic<int64_t> current_phase_{static_cast<int64_t>(StepPhase::kUnknown)};
  std::atomic<bool> request_capture_enabled_{true};
  std::atomic<bool> capture_enabled_{true};
  CaptureSchedule schedule_;

  std::thread worker_;
  std::atomic<bool> closed_{false};

  // Stats ---------------------------------------------------------------
  std::atomic<int64_t> stats_total_steps_{0};
  std::atomic<int64_t> stats_total_tasks_{0};
  std::atomic<int64_t> stats_submit_us_{0};
  std::atomic<int64_t> stats_process_us_{0};
  std::atomic<int64_t> stats_callback_us_{0};
  bool stats_step_log_{false};
};

}  // namespace monitoring

#endif  // MONITORING_NATIVE_ENGINE_INTERNAL_H_
