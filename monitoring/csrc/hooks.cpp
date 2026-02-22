// Hook callback + builder path

#include "native_engine_internal.h"
#include "engine_utils.h"
#include "nvtx_shim.h"

namespace monitoring {

int64_t NativeMonitoringEngine::Impl::estimate_task_bytes(const TaskSpec& spec) const {
  return tensor_nbytes(spec.tensor);
}

std::optional<StepWork> NativeMonitoringEngine::Impl::maybe_cut_open_step_chunk_locked(int64_t step_id,
                                                                                        bool force_tail) {
  auto it = open_steps_.find(step_id);
  if (it == open_steps_.end()) {
    return std::nullopt;
  }

  StepWork& open = it->second;
  if (open.tasks.empty()) {
    if (force_tail) {
      open_steps_.erase(it);
    }
    return std::nullopt;
  }

  bool should_cut = force_tail;
  if (!should_cut && partial_seal_enabled_ && partial_seal_chunk_bytes_ > 0) {
    should_cut = open.bytes >= partial_seal_chunk_bytes_;
  }
  if (!should_cut) {
    return std::nullopt;
  }

  StepWork chunk;
  chunk.step_id = step_id;
  chunk.tasks = std::move(open.tasks);
  chunk.bytes = open.bytes;
  chunk.final_chunk = force_tail;
  open.bytes = 0;
  open_steps_.erase(it);
  return chunk;
}

void NativeMonitoringEngine::Impl::append_task_entry_and_maybe_seal(int64_t step_id, TaskEntry&& entry) {
  std::optional<StepWork> ready_chunk;
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    StepWork& open = open_steps_[step_id];
    open.step_id = step_id;
    open.tasks.emplace_back(std::move(entry));
    open.bytes += estimate_task_bytes(open.tasks.back().spec);
    ready_chunk = maybe_cut_open_step_chunk_locked(step_id, /*force_tail=*/false);
  }

  if (!ready_chunk.has_value()) {
    return;
  }

  // Mid-forward chunk: record producer-stream event so processing on cache stream
  // observes correct ordering before touching GPU tensors.
  int device_index = -1;
  for (const auto& task : ready_chunk->tasks) {
    if (task.spec.tensor.defined() && task.spec.tensor.is_cuda()) {
      device_index = task.spec.tensor.get_device();
      break;
    }
  }
  if (device_index >= 0) {
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    C10_CUDA_CHECK(cudaEventRecord(event, stream.stream()));
    ready_chunk->event = event;
  }

  dispatch_step(std::move(*ready_chunk));
}

HookConfig* NativeMonitoringEngine::Impl::upsert_hook_config(const std::string& hook_name,
                                                             bool remove_batch_dim,
                                                             py::object pos_slice,
                                                             py::object target_device) {
  mon_nvtx_push("MonEng::upsert_hook_config");
  auto config = std::make_unique<HookConfig>();
  config->name = hook_name;
  config->pos_dim = deduce_pos_dim(hook_name);
  config->remove_batch_dim = remove_batch_dim;
  config->slice = parse_slice_py(std::move(pos_slice));
  if (!target_device.is_none()) {
    config->target_device = target_device.cast<c10::Device>();
  } else {
    config->target_device.reset();
  }

  HookConfig* cfg_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(hook_config_mutex_);
    auto& entry = hook_configs_[hook_name];
    if (!entry) entry = std::make_unique<HookConfig>();
    *entry = std::move(*config);
    cfg_ptr = entry.get();
  }
  mon_nvtx_pop();
  return cfg_ptr;
}

HookConfig* NativeMonitoringEngine::Impl::upsert_hook_config_tuple(const std::string& hook_name,
                                                                   bool remove_batch_dim,
                                                                   py::tuple slice_tuple,
                                                                   py::object target_device) {
  mon_nvtx_push("MonEng::upsert_hook_config");
  auto config = std::make_unique<HookConfig>();
  config->name = hook_name;
  config->pos_dim = deduce_pos_dim(hook_name);
  config->remove_batch_dim = remove_batch_dim;
  config->slice = parse_slice_tuple(slice_tuple);
  if (!target_device.is_none()) {
    config->target_device = target_device.cast<c10::Device>();
  } else {
    config->target_device.reset();
  }

  HookConfig* cfg_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(hook_config_mutex_);
    auto& entry = hook_configs_[hook_name];
    if (!entry) entry = std::make_unique<HookConfig>();
    *entry = std::move(*config);
    cfg_ptr = entry.get();
  }
  mon_nvtx_pop();
  return cfg_ptr;
}

void NativeMonitoringEngine::Impl::append_hook_current_step(const HookConfig& cfg, at::Tensor tensor) {
  mon_nvtx_push("MonEng::append_hook_current_step");
  if (!is_capture_enabled()) {
    mon_nvtx_pop();
    return;
  }
  TaskSpec spec;
  spec.tensor = std::move(tensor);
  spec.slice_dim = cfg.pos_dim;
  spec.remove_batch_dim = cfg.remove_batch_dim;
  spec.slice = cfg.slice;
  spec.target_device = cfg.target_device;

  int64_t dim = spec.slice_dim;
  int64_t tensor_dims = spec.tensor.dim();
  bool can_slice = true;
  if (dim >= 0) {
    can_slice = tensor_dims > dim;
  } else {
    can_slice = tensor_dims >= -dim;
  }
  spec.can_slice = can_slice;

  int64_t step_id = current_step_id_.load(std::memory_order_acquire);
  TaskEntry entry;
  entry.spec = std::move(spec);
  entry.token = 0;  // token assigned at process time
  pending_tasks_.fetch_add(1, std::memory_order_relaxed);
  stats_total_tasks_.fetch_add(1, std::memory_order_relaxed);
  append_task_entry_and_maybe_seal(step_id, std::move(entry));
  mon_nvtx_pop();
}

int64_t NativeMonitoringEngine::Impl::add_task_from_config(const HookConfig& cfg,
                                                           at::Tensor tensor,
                                                           int64_t step_id) {
  mon_nvtx_push("MonEng::add_task_from_config");
  if (!is_capture_enabled()) {
    mon_nvtx_pop();
    return 0;
  }
  TaskSpec spec;
  spec.tensor = std::move(tensor);
  spec.slice_dim = cfg.pos_dim;
  spec.remove_batch_dim = cfg.remove_batch_dim;
  spec.slice = cfg.slice;
  spec.target_device = cfg.target_device;

  int64_t dim = spec.slice_dim;
  int64_t tensor_dims = spec.tensor.dim();
  spec.can_slice = (dim >= 0) ? (tensor_dims > dim) : (tensor_dims >= -dim);

  // Allocate token + result slot eagerly
  int64_t token = next_token_++;
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slots_.emplace(token, std::make_shared<ResultSlot>());
  }

  TaskEntry entry;
  entry.spec = std::move(spec);
  entry.token = token;
  pending_tasks_.fetch_add(1, std::memory_order_relaxed);
  stats_total_tasks_.fetch_add(1, std::memory_order_relaxed);
  append_task_entry_and_maybe_seal(step_id, std::move(entry));
  mon_nvtx_pop();
  return token;
}

void NativeMonitoringEngine::Impl::set_enabled_hooks(py::object names_iterable) {
  std::lock_guard<std::mutex> lock(enabled_mutex_);
  enabled_hooks_.clear();
  if (names_iterable.is_none()) return;
  for (auto item : names_iterable) {
    try {
      std::string name = py::cast<std::string>(item);
      enabled_hooks_.insert(std::move(name));
    } catch (...) {
      // ignore non-string items
    }
  }
}

void NativeMonitoringEngine::Impl::record_step_name_token(int64_t step_id, const std::string& name, int64_t token) {
  std::lock_guard<std::mutex> lock(staging_mutex_);
  step_name_tokens_[step_id].emplace_back(name, token);
}

void NativeMonitoringEngine::Impl::append_hook(int64_t step_id,
                                               const std::string& hook_name,
                                               at::Tensor tensor,
                                               bool remove_batch_dim,
                                               py::object pos_slice,
                                               py::object target_device) {
  mon_nvtx_push("MonEng::append_hook");
  if (!is_capture_enabled()) {
    mon_nvtx_pop();
    return;
  }
  TaskSpec spec;
  spec.tensor = std::move(tensor);
  spec.remove_batch_dim = remove_batch_dim;
  spec.slice_dim = deduce_pos_dim(hook_name);
  spec.can_slice = (spec.slice_dim >= 0) ? (spec.tensor.dim() > spec.slice_dim)
                                         : (spec.tensor.dim() >= -spec.slice_dim);
  spec.slice = parse_slice_py(pos_slice);
  if (!target_device.is_none()) {
    spec.target_device = target_device.cast<c10::Device>();
  }

  TaskEntry entry;
  entry.spec = std::move(spec);
  entry.token = 0; // token assigned at process/dispatch
  pending_tasks_.fetch_add(1, std::memory_order_relaxed);
  stats_total_tasks_.fetch_add(1, std::memory_order_relaxed);
  append_task_entry_and_maybe_seal(step_id, std::move(entry));
  mon_nvtx_pop();
}

int64_t NativeMonitoringEngine::Impl::deduce_pos_dim(const std::string& name) {
  auto ends_with = [](const std::string& s, const char* suf) -> bool {
    size_t n = s.size(); size_t m = std::char_traits<char>::length(suf);
    return n >= m && s.compare(n - m, m, suf) == 0;
  };
  if (ends_with(name, "hook_q") || ends_with(name, "hook_k") || ends_with(name, "hook_v") ||
      ends_with(name, "hook_z") || ends_with(name, "hook_result")) {
    return -3;
  }
  return -2;
}

}  // namespace monitoring
