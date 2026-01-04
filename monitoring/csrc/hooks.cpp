// Hook callback + builder path

#include "native_engine_internal.h"
#include "nvtx_shim.h"

namespace monitoring {

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

  std::lock_guard<std::mutex> lock(staging_mutex_);
  StepWork& work = open_steps_[step_id];
  work.step_id = step_id;
  TaskEntry entry;
  entry.spec = std::move(spec);
  entry.token = 0;  // token assigned at process time
  work.tasks.emplace_back(std::move(entry));
  pending_tasks_.fetch_add(1, std::memory_order_relaxed);
  stats_total_tasks_.fetch_add(1, std::memory_order_relaxed);
  mon_nvtx_pop();
}

int64_t NativeMonitoringEngine::Impl::add_task_from_config(const HookConfig& cfg, at::Tensor tensor) {
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

  int64_t step_id = current_step_id_.load(std::memory_order_acquire);

  // Allocate token + result slot eagerly
  int64_t token = next_token_++;
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slots_.emplace(token, std::make_shared<ResultSlot>());
  }

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
  stats_total_tasks_.fetch_add(1, std::memory_order_relaxed);
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

  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    StepWork& work = open_steps_[step_id];
    work.step_id = step_id;
    TaskEntry entry;
    entry.spec = std::move(spec);
    entry.token = 0; // token assigned at process/dispatch
    work.tasks.emplace_back(std::move(entry));
  }
  pending_tasks_.fetch_add(1, std::memory_order_relaxed);
  stats_total_tasks_.fetch_add(1, std::memory_order_relaxed);
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
