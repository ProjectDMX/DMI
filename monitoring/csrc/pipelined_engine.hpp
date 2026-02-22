// pipelined_engine.hpp
#ifndef DMX_HOST_PIPELINED_ENGINE_HPP_
#define DMX_HOST_PIPELINED_ENGINE_HPP_

#include "batching_queue.hpp"

#include <any>
#include <array>
#include <iostream>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>   // std::promise/std::shared_future for start gate
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dmx_host {

// ============================================================
// Compile-time queue options bundle (separates queue knobs)
// ============================================================

template <bool NotifyAllConsumers_ = false,
          bool CheckMinSizeProducers_ = false,
          bool EnableQueueProfiling_ = false>
struct QueueOptions {
  static constexpr bool kNotifyAllConsumers = NotifyAllConsumers_;
  static constexpr bool kCheckMinSizeProducers = CheckMinSizeProducers_;
  static constexpr bool kEnableQueueProfiling = EnableQueueProfiling_;

  template <typename ItemT, typename SizeT>
  using Queue = WatermarkBatchingQueue<ItemT,
                                       SizeT,
                                       NotifyAllConsumers_,
                                       CheckMinSizeProducers_,
                                       EnableQueueProfiling_>;
};

// ============================================================
// Policies / configs (runtime)
// ============================================================

enum class ThreadExceptionPolicy : std::uint8_t {
  STOP_ENGINE = 0,
  CONTINUE = 1,
};

enum class OnFullPolicy : std::uint8_t {
  RAISE = 0,
  DROP = 1,
  RETRY = 2,
  ABORT = 3,
};

enum class OnClosedPolicy : std::uint8_t {
  RAISE = 0,
  DROP = 1,
};

template <typename DurationT>
struct EnqueuePolicyT {
  bool block = true;
  std::optional<DurationT> timeout = std::nullopt;

  OnFullPolicy on_full = OnFullPolicy::RAISE;
  int max_retries = 0;
  DurationT retry_backoff = DurationT{0};

  OnClosedPolicy on_closed = OnClosedPolicy::RAISE;
  bool drop_if_stopping = true;
};

template <typename SizeT, typename DurationT>
struct QueueConfigT {
  std::optional<int> min_batch_items = 1;
  std::optional<SizeT> min_batch_size = std::nullopt;
  std::optional<DurationT> max_linger = std::nullopt;

  std::optional<int> max_batch_items = std::nullopt;
  std::optional<SizeT> max_batch_size = std::nullopt;

  std::optional<int> high_watermark_items = std::nullopt;
  std::optional<SizeT> high_watermark_size = std::nullopt;

  template <typename QueueT>
  std::unique_ptr<QueueT> build_queue(std::string name,
                                      std::optional<QueueProfilingConfig> profiling = std::nullopt,
                                      typename QueueT::WarnFn warn = nullptr) const {
    return std::make_unique<QueueT>(
        min_batch_items,
        min_batch_size,
        max_linger,
        max_batch_items,
        max_batch_size,
        high_watermark_items,
        high_watermark_size,
        std::move(name),
        profiling,
        std::move(warn));
  }
};

template <typename ItemT>
struct NoOutputHandler {
  void operator()(std::vector<ItemT>&&) const noexcept {
    // drop
  }
};

template <typename DurationT>
struct EngineConfigT {
  std::string name = "pipelined-engine";
  ThreadExceptionPolicy exception_policy = ThreadExceptionPolicy::STOP_ENGINE;

  bool close_queues_on_abort = true;
  bool suppress_closed_queue_errors_during_shutdown = true;

  // If close_queues_on_abort == false, must be set to avoid workers blocking forever in dequeue_batch().
  // Must be > 0 when set.
  std::optional<DurationT> worker_dequeue_timeout = std::nullopt;

  // Engine profiling runtime switches (compiled out if EnableEngineProfiling == false).
  bool enable_stats = false;
  bool enable_timing = false;

  bool timing_profile_submit = true;
  bool timing_profile_dequeue = true;
  bool timing_profile_process = true;
  bool timing_profile_enqueue = true;
  bool timing_profile_output = true;

  // Queue profiling config passed into each stage input queue (queue still may compile profiling out).
  std::optional<QueueProfilingConfig> queue_profiling = std::nullopt;

  // Optional warning hook (also forwarded into queues).
  std::function<void(const std::string&)> warn = nullptr;
};

// ============================================================
// PipelinedEngine:
// - submit_items(items) only (no input handler in C++).
// - Stage process signature:
//     process(batch, next_queue_ptr_or_null) -> optional<vector<ItemT>>
//   Return nullopt => stage pushed outputs itself (engine doesn't push).
// - Output handler is compile-time templated; default drops.
// - Engine profiling is compile-time optional (EnableEngineProfiling) like the queue.
//
// ---------------------------
// Thread-safety contract
// ---------------------------
// - submit_items(...) may be called concurrently by multiple producer threads.
// - For each stage S, StageConfig::process_fn is invoked concurrently by `parallelism` worker threads.
//   It must be thread-safe with respect to any shared state it touches (or use TLS / per-thread resources).
// - OutputHandlerT (last-stage consumer) is invoked concurrently by the last stage worker threads.
//   If your output handler is NOT thread-safe, set the LAST stage parallelism = 1, or wrap your handler
//   with a mutex externally (the engine will not serialize output by default).
// ============================================================

template <typename ItemT,
          typename SizeT,
          std::size_t NumStages,
          typename QueueOptionsT = QueueOptions<>,
          bool EnableEngineProfiling = false,
          typename OutputHandlerT = NoOutputHandler<ItemT>>
class PipelinedEngine
    : private ebo_storage<OutputHandlerT, 0>,
      private ebo_storage<std::conditional_t<EnableEngineProfiling, std::optional<struct EngineProfilingConfig_>, empty_profiling_storage>, 1>,
      private ebo_storage<std::conditional_t<EnableEngineProfiling, std::optional<struct EngineProfile_>, empty_profiling_storage>, 2> {
 public:
  static_assert(NumStages > 0, "NumStages must be > 0");

  using Duration = std::chrono::duration<double>;
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  using QueueT = typename QueueOptionsT::template Queue<ItemT, SizeT>;

  using EnqueuePolicy = EnqueuePolicyT<Duration>;
  using QueueConfig = QueueConfigT<SizeT, Duration>;
  using EngineConfig = EngineConfigT<Duration>;

  // next_q == nullptr for the last stage.
  // Return:
  //   - nullopt        => stage pushed outputs itself (engine does not enqueue)
  //   - vector<ItemT>  => engine enqueues (or calls output handler if last)
  using StageProcessFn =
      std::function<std::optional<std::vector<ItemT>>(std::vector<ItemT> batch, QueueT* next_q)>;

  using StageThreadInitFn =
      std::function<void(int /*thread_idx*/, const std::any& /*thread_config*/)>;
  using StageThreadCleanupFn = std::function<void()>;

  struct StageConfig {
    std::string name;
    int parallelism = 1;

    StageProcessFn process_fn;

    std::any thread_init_config;
    StageThreadInitFn thread_init = nullptr;
    StageThreadCleanupFn thread_cleanup = nullptr;

    QueueConfig input_queue{};
    EnqueuePolicy ingress_policy{};

    std::optional<std::string> thread_name_prefix = std::nullopt;
  };

  struct ThreadFailure {
    std::string stage;
    std::string thread_name;
    std::string where;
    std::string exc_type;
    std::string exc_what;
  };

  // -------- Engine profiling (compile-time optional) --------
  struct EngineProfilingConfig_ {
    bool enable_stats = false;
    bool enable_timing = false;

    bool timing_profile_submit = true;
    bool timing_profile_dequeue = true;
    bool timing_profile_process = true;
    bool timing_profile_enqueue = true;
    bool timing_profile_output = true;
  };

  struct IngestStats {
    std::uint64_t submit_calls = 0;
    std::uint64_t items_submitted = 0;

    std::uint64_t submit_enqueue_calls = 0;
    double submit_enqueue_s = 0.0;
  };

  struct QueueStats {
    std::uint64_t enqueued = 0;
    std::uint64_t dropped = 0;
    std::uint64_t full_errors = 0;
    std::uint64_t closed_errors = 0;
    std::uint64_t too_large_errors = 0;
    std::uint64_t retries = 0;
  };

  struct StageStats {
    std::uint64_t batches = 0;
    std::uint64_t items_in = 0;
    std::uint64_t items_out = 0;

    std::uint64_t dequeue_calls = 0;
    std::uint64_t dequeue_timeouts = 0;
    std::uint64_t process_calls = 0;
    std::uint64_t enqueue_calls = 0;
    std::uint64_t output_calls = 0;
    std::uint64_t output_items = 0;

    double dequeue_s = 0.0;
    double dequeue_idle_s = 0.0;
    double process_s = 0.0;
    double enqueue_s = 0.0;
    double output_s = 0.0;
  };

  struct StatsSnapshot {
    IngestStats ingest{};
    std::array<QueueStats, NumStages> queue_by_stage{};
    std::array<StageStats, NumStages> stage_by_stage{};
  };

  using QueueProfilingSnapshotArray = std::array<std::optional<ProfilingSnapshot>, NumStages>;

 private:
  struct EngineProfile_ {
    IngestStats ingest{};
    std::array<QueueStats, NumStages> queue_by_stage{};
    std::array<StageStats, NumStages> stage_by_stage{};
  };

  using OutBase = ebo_storage<OutputHandlerT, 0>;
  using ProfCfgStorage = std::conditional_t<EnableEngineProfiling, std::optional<EngineProfilingConfig_>, empty_profiling_storage>;
  using ProfStorage = std::conditional_t<EnableEngineProfiling, std::optional<EngineProfile_>, empty_profiling_storage>;
  using ProfCfgBase = ebo_storage<ProfCfgStorage, 1>;
  using ProfBase = ebo_storage<ProfStorage, 2>;

  OutputHandlerT& output_handler_() noexcept { return OutBase::get(); }
  const OutputHandlerT& output_handler_() const noexcept { return OutBase::get(); }

  ProfCfgStorage& prof_cfg_storage_() { return ProfCfgBase::get(); }
  const ProfCfgStorage& prof_cfg_storage_() const { return ProfCfgBase::get(); }

  ProfStorage& prof_storage_() { return ProfBase::get(); }
  const ProfStorage& prof_storage_() const { return ProfBase::get(); }

  const EngineProfilingConfig_* prof_cfg_ptr_() const noexcept {
    if constexpr (!EnableEngineProfiling) {
      return nullptr;
    } else {
      const auto& cfg_opt = prof_cfg_storage_();
      return cfg_opt ? &(*cfg_opt) : nullptr;
    }
  }

  void init_engine_profiling_() {
    if constexpr (!EnableEngineProfiling) {
      return;
    } else {
      bool stats = cfg_.enable_stats;
      bool timing = cfg_.enable_timing;

      // Like the queue: timing requires counts.
      if (timing && !stats) {
        stats = true;
        if (cfg_.warn) {
          cfg_.warn("EngineConfig: enable_stats set to true because enable_timing is true (timing needs counts).");
        }
      }

      if (!(stats || timing)) return;

      auto& cfg_opt = prof_cfg_storage_();
      auto& prof_opt = prof_storage_();

      cfg_opt.emplace();
      cfg_opt->enable_stats = stats;
      cfg_opt->enable_timing = timing;
      cfg_opt->timing_profile_submit = cfg_.timing_profile_submit;
      cfg_opt->timing_profile_dequeue = cfg_.timing_profile_dequeue;
      cfg_opt->timing_profile_process = cfg_.timing_profile_process;
      cfg_opt->timing_profile_enqueue = cfg_.timing_profile_enqueue;
      cfg_opt->timing_profile_output = cfg_.timing_profile_output;

      prof_opt.emplace();  // zero-init
    }
  }

 public:
  explicit PipelinedEngine(std::array<StageConfig, NumStages> stages,
                          EngineConfig config = EngineConfig{},
                          OutputHandlerT output_handler = OutputHandlerT{})
      : OutBase(std::move(output_handler)),
        stages_(std::move(stages)),
        cfg_(std::move(config)) {
    validate_config_();
    build_queues_();
    init_engine_profiling_();
  }

  ~PipelinedEngine() noexcept {
    // best effort; never throw
    try {
      request_abort();
      (void)join_all_threads_(std::nullopt);
    } catch (...) {
    }
  }

  PipelinedEngine(const PipelinedEngine&) = delete;
  PipelinedEngine& operator=(const PipelinedEngine&) = delete;

  // ------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------

  void start() {
    std::lock_guard<std::mutex> lk(state_mu_);
    if (started_) return;
    if (stop_event_.load(std::memory_order_acquire)) {
      throw std::runtime_error("Engine is stopped/aborted and cannot be restarted");
    }

    // Prepare thread containers.
    for (std::size_t s = 0; s < NumStages; ++s) {
      threads_[s].clear();
      threads_[s].reserve(static_cast<std::size_t>(stages_[s].parallelism));
    }

    // Reset join bookkeeping to match threads actually created.
    {
      std::lock_guard<std::mutex> dlk(done_mu_);
      stage_running_.fill(0);
      total_running_ = 0;
    }

    // Start gate: threads wait here until start() finishes construction.
    // This prevents worker threads from decrementing stage_running_/total_running_
    // before we have recorded that they exist (and also prevents half-started races).
    auto start_promise = std::make_shared<std::promise<void>>();
    std::shared_future<void> start_gate = start_promise->get_future().share();

    try {
      for (std::size_t stage_idx = 0; stage_idx < NumStages; ++stage_idx) {
        const int par = stages_[stage_idx].parallelism;
        for (int t = 0; t < par; ++t) {
          threads_[stage_idx].emplace_back([this, stage_idx, t, start_gate] {
            // Never let exceptions escape std::thread entrypoint.
            try {
              start_gate.wait();
              stage_worker_(stage_idx, t);
            } catch (const std::exception& e) {
              record_failure_noexcept_(ThreadFailure{
                  stages_[stage_idx].name,
                  thread_name_(stage_idx, t),
                  "thread_entry",
                  typeid(e).name(),
                  e.what(),
              });
              maybe_abort_on_failure_();
            } catch (...) {
              record_failure_noexcept_(ThreadFailure{
                  stages_[stage_idx].name,
                  thread_name_(stage_idx, t),
                  "thread_entry",
                  "unknown",
                  "unknown",
              });
              maybe_abort_on_failure_();
            }
          });

          // Record that this thread exists (safe because it cannot run past the gate yet).
          {
            std::lock_guard<std::mutex> dlk(done_mu_);
            stage_running_[stage_idx] += 1;
            total_running_ += 1;
          }
        }
      }

      // Fully started.
      started_ = true;

      // Release workers.
      start_promise->set_value();

    } catch (...) {
      // If thread creation throws (e.g. std::system_error), we must not leave joinable
      // threads behind (std::thread::~thread would std::terminate).
      //
      // Cleanup strategy:
      // 1) mark stopping + close queues to unblock dequeue_batch()
      // 2) release start gate so created threads can observe stop and exit
      // 3) join any created threads
      stop_event_.store(true, std::memory_order_release);
      input_closed_ = true;

      for (auto& q : queues_) {
        try { if (q) q->close(); } catch (...) {}
      }

      try { start_promise->set_value(); } catch (...) {}

      for (std::size_t s = 0; s < NumStages; ++s) {
        for (auto& th : threads_[s]) {
          if (th.joinable()) {
            try { th.join(); } catch (...) {}
          }
        }
        threads_[s].clear();
      }

      {
        std::lock_guard<std::mutex> dlk(done_mu_);
        stage_running_.fill(0);
        total_running_ = 0;
      }

      // start() failed: this instance is now in "aborted" state (stop_event_ set).
      throw;
    }
  }

  void close_input() {
    std::lock_guard<std::mutex> lk(state_mu_);
    if (input_closed_) return;
    input_closed_ = true;
    queues_[0]->close();
  }

  void request_abort() {
    {
      std::lock_guard<std::mutex> lk(state_mu_);
      if (stop_event_.load(std::memory_order_acquire)) {
        input_closed_ = true;
        return;
      }
      stop_event_.store(true, std::memory_order_release);
      input_closed_ = true;
    }

    if (cfg_.close_queues_on_abort) {
      for (auto& q : queues_) {
        try {
          q->close();
        } catch (...) {
        }
      }
    }
  }

  bool abort(std::optional<Duration> timeout = std::nullopt) {
    request_abort();
    return join_all_threads_(timeout);
  }

  bool stop(bool graceful = true, std::optional<Duration> timeout = std::nullopt) {
    if (!graceful) return abort(timeout);
    close_input();
    const bool ok = join(timeout);
    if (ok) {
      stop_event_.store(true, std::memory_order_release);
      return true;
    }

    // timed out -> escalate
    return abort(timeout);
  }

  // Join stage-by-stage (closing next stage queue as each stage finishes).
  bool join(std::optional<Duration> timeout = std::nullopt) {
    if (!started_) return true;

    TimePoint end_at;
    const bool have_deadline = timeout.has_value();
    if (have_deadline) {
      if (timeout->count() < 0.0) throw std::invalid_argument("timeout must be >= 0");
      end_at = Clock::now() + std::chrono::duration_cast<Clock::duration>(*timeout);
    }

    for (std::size_t stage_idx = 0; stage_idx < NumStages; ++stage_idx) {
      {
        std::unique_lock<std::mutex> lk(done_mu_);
        auto pred = [&] { return stage_running_[stage_idx] == 0; };
        if (!have_deadline) {
          done_cv_.wait(lk, pred);
        } else {
          if (!done_cv_.wait_until(lk, end_at, pred)) return false;
        }
      }

      if (stage_idx + 1 < NumStages) {
        queues_[stage_idx + 1]->close();
      }

      if (have_deadline && Clock::now() >= end_at) {
        std::lock_guard<std::mutex> lk(done_mu_);
        if (total_running_ == 0) break;
        return false;
      }
    }

    return join_all_threads_(timeout);
  }

  bool started() const {
    std::lock_guard<std::mutex> lk(state_mu_);
    return started_;
  }

  bool input_closed() const {
    std::lock_guard<std::mutex> lk(state_mu_);
    return input_closed_;
  }

  bool stopping() const { return stop_event_.load(std::memory_order_acquire); }

  // ------------------------------------------------------------
  // Submitting work
  // ------------------------------------------------------------

  void submit_items(std::vector<ItemT> items) {
    {
      std::lock_guard<std::mutex> lk(state_mu_);
      if (!started_) throw std::runtime_error("Engine not started. Call start() first.");
      if (stop_event_.load(std::memory_order_acquire)) throw std::runtime_error("Engine is stopping/aborted");
      if (input_closed_) throw std::runtime_error("Input is closed");
    }

    const auto* pcfg = prof_cfg_ptr_();
    const bool want_stats = pcfg && pcfg->enable_stats;
    const bool want_timing = pcfg && pcfg->enable_timing && pcfg->timing_profile_submit &&
                             pcfg->timing_profile_enqueue;

    if (want_stats) {
      prof_ingest_add_submit_(1, static_cast<std::uint64_t>(items.size()));
    }
    if (items.empty()) return;

    const TimePoint t0 = want_timing ? Clock::now() : TimePoint{};
    enqueue_many_(*queues_[0], std::move(items), stages_[0].ingress_policy, /*target_stage_idx=*/0);

    if (want_stats) {
      prof_ingest_add_submit_enqueue_(want_timing ? std::chrono::duration<double>(Clock::now() - t0).count() : 0.0);
    }
  }

  // ------------------------------------------------------------
  // Failures
  // ------------------------------------------------------------

  std::vector<ThreadFailure> failures() const {
    std::lock_guard<std::mutex> lk(fail_mu_);
    return failures_;
  }

  void raise_if_failed() const {
    std::lock_guard<std::mutex> lk(fail_mu_);
    if (failures_.empty()) return;
    const auto& f = failures_.back();
    throw std::runtime_error("Engine recorded worker failure: [" + f.stage + "::" + f.thread_name +
                             "::" + f.where + "] " + f.exc_type + ": " + f.exc_what);
  }

  // ------------------------------------------------------------
  // Profiling snapshots
  // ------------------------------------------------------------

  // Returns nullopt if engine profiling is compile-time disabled OR runtime disabled.
  std::optional<StatsSnapshot> profiling() const {
    if constexpr (!EnableEngineProfiling) {
      return std::nullopt;
    } else {
      const auto& cfg_opt = prof_cfg_storage_();
      const auto& prof_opt = prof_storage_();
      if (!cfg_opt || !prof_opt) return std::nullopt;

      // NOTE: We always store this mutex as a direct mutable member (not via ebo_storage):
      // - Locking a std::mutex mutates it; profiling() is logically-const but must lock.
      // - `mutable` cannot be applied to EBO base subobjects; and our generic ebo_storage returns
      //   a `const T&` in const methods, which would force `const_cast` to lock (UB for truly-const engines).
      // - Paying one mutex in the object is the simplest, const-correct, UB-free solution in C++17.
      std::lock_guard<std::mutex> lk(prof_mu_);

      StatsSnapshot snap;
      snap.ingest = prof_opt->ingest;
      snap.queue_by_stage = prof_opt->queue_by_stage;
      snap.stage_by_stage = prof_opt->stage_by_stage;
      return snap;
    }
  }

  QueueProfilingSnapshotArray queue_profiling() const {
    QueueProfilingSnapshotArray out;
    for (std::size_t i = 0; i < NumStages; ++i) {
      out[i] = queues_[i]->profiling();
    }
    return out;
  }

  void reset_metrics() {
    if constexpr (EnableEngineProfiling) {
      auto& cfg_opt = prof_cfg_storage_();
      auto& prof_opt = prof_storage_();
      if (cfg_opt && prof_opt) {
        std::lock_guard<std::mutex> lk(prof_mu_);
        prof_opt.emplace();
      }
    }
    if (cfg_.queue_profiling.has_value()) {
      for (auto& q : queues_) q->reset_profiling();
    }
  }

  // Access queues (advanced / for stage functions that want to push themselves).
  QueueT& stage_queue(std::size_t stage_idx) { return *queues_.at(stage_idx); }
  const QueueT& stage_queue(std::size_t stage_idx) const { return *queues_.at(stage_idx); }

 private:
  // ------------------------------------------------------------
  // Validation / construction
  // ------------------------------------------------------------

  void validate_config_() const {
    if (cfg_.worker_dequeue_timeout && cfg_.worker_dequeue_timeout->count() <= 0.0) {
      throw std::invalid_argument("worker_dequeue_timeout must be > 0 or nullopt");
    }
    if (!cfg_.close_queues_on_abort && !cfg_.worker_dequeue_timeout) {
      throw std::invalid_argument(
          "close_queues_on_abort=false requires worker_dequeue_timeout; otherwise workers can block forever.");
    }

    std::unordered_set<std::string> seen;
    for (std::size_t i = 0; i < NumStages; ++i) {
      const auto& s = stages_[i];
      if (s.name.empty()) {
        throw std::invalid_argument("Stage at index " + std::to_string(i) + " must have a non-empty name");
      }
      if (seen.count(s.name)) {
        throw std::invalid_argument("Duplicate stage name: " + s.name);
      }
      seen.insert(s.name);

      if (s.parallelism <= 0) {
        throw std::invalid_argument("Stage '" + s.name + "' parallelism must be > 0");
      }
      if (!s.process_fn) {
        throw std::invalid_argument("Stage '" + s.name + "' must set process_fn");
      }

      if (s.ingress_policy.max_retries < 0) {
        throw std::invalid_argument("Stage '" + s.name + "': ingress_policy.max_retries must be >= 0");
      }
      if (s.ingress_policy.timeout && s.ingress_policy.timeout->count() < 0.0) {
        throw std::invalid_argument("Stage '" + s.name + "': ingress_policy.timeout must be >= 0 or nullopt");
      }
      if (s.ingress_policy.retry_backoff.count() < 0.0) {
        throw std::invalid_argument("Stage '" + s.name + "': ingress_policy.retry_backoff must be >= 0");
      }

      if (!cfg_.close_queues_on_abort) {
        if (s.ingress_policy.block && !s.ingress_policy.timeout) {
          throw std::invalid_argument(
              "close_queues_on_abort=false requires every stage ingress_policy to have timeout set (or block=false).");
        }
      }
    }
  }

  void build_queues_() {
    for (std::size_t i = 0; i < NumStages; ++i) {
      const auto& stage = stages_[i];
      const std::string qname = cfg_.name + "." + stage.name + ".in";
      queues_[i] = stage.input_queue.template build_queue<QueueT>(qname, cfg_.queue_profiling, cfg_.warn);
    }
  }

  std::string thread_name_(std::size_t stage_idx, int thread_idx) const {
    const auto& s = stages_[stage_idx];
    const std::string prefix = s.thread_name_prefix.value_or(cfg_.name + "." + s.name);
    return prefix + ".t" + std::to_string(thread_idx);
  }

  void maybe_abort_on_failure_() {
    if (cfg_.exception_policy == ThreadExceptionPolicy::STOP_ENGINE) request_abort();
  }

  void record_failure_noexcept_(ThreadFailure f) noexcept {
    try {
      std::lock_guard<std::mutex> lk(fail_mu_);
      failures_.push_back(std::move(f));
    } catch (...) {
      // swallow
    }
  }

  void warn_once_no_output_handler_(const std::string& stage_name, std::size_t n_items) {
    if constexpr (!std::is_same<OutputHandlerT, NoOutputHandler<ItemT>>::value) {
      return;
    }
    if (!cfg_.warn) return;

    bool first = false;
    {
      std::lock_guard<std::mutex> lk(warn_mu_);
      if (!warned_no_output_.count(stage_name)) {
        warned_no_output_.insert(stage_name);
        first = true;
      }
    }
    if (first) {
      cfg_.warn("Stage '" + stage_name + "' produced " + std::to_string(n_items) +
                " output item(s) but no output handler is configured; dropping outputs.");
    }
  }

  // ------------------------------------------------------------
  // Worker loop
  // ------------------------------------------------------------

  void stage_worker_(std::size_t stage_idx, int thread_idx) {
    const StageConfig& stage = stages_[stage_idx];
    const std::string stage_name = stage.name;
    const std::string tname = thread_name_(stage_idx, thread_idx);

    struct DoneGuard {
      PipelinedEngine* self;
      std::size_t stage_idx;
      ~DoneGuard() {
        {
          std::lock_guard<std::mutex> lk(self->done_mu_);
          self->stage_running_[stage_idx] -= 1;
          self->total_running_ -= 1;
        }
        self->done_cv_.notify_all();
      }
    } done_guard{this, stage_idx};

    struct CleanupGuard {
      PipelinedEngine* self;
      const StageConfig& stage;
      std::string stage_name;
      std::string thread_name;
      ~CleanupGuard() {
        if (!stage.thread_cleanup) return;
        try {
          stage.thread_cleanup();
        } catch (const std::exception& e) {
          self->record_failure_noexcept_(
              ThreadFailure{stage_name, thread_name, "thread_cleanup", typeid(e).name(), e.what()});
          self->maybe_abort_on_failure_();
        } catch (...) {
          self->record_failure_noexcept_(
              ThreadFailure{stage_name, thread_name, "thread_cleanup", "unknown", "unknown"});
          self->maybe_abort_on_failure_();
        }
      }
    } cleanup_guard{this, stage, stage_name, tname};

    const auto* pcfg = prof_cfg_ptr_();
    const bool want_stats = pcfg && pcfg->enable_stats;
    const bool want_timing = pcfg && pcfg->enable_timing;

    try {
      if (stage.thread_init) stage.thread_init(thread_idx, stage.thread_init_config);

      QueueT& in_q = *queues_[stage_idx];
      const bool is_last = (stage_idx + 1 == NumStages);

      QueueT* next_q = nullptr;
      const StageConfig* next_stage = nullptr;
      if (!is_last) {
        next_q = queues_[stage_idx + 1].get();
        next_stage = &stages_[stage_idx + 1];
      }

      while (true) {
        if (stop_event_.load(std::memory_order_acquire)) return;

        // ---- dequeue
        const bool time_deq = want_timing && pcfg->timing_profile_dequeue;
        const TimePoint t_deq0 = time_deq ? Clock::now() : TimePoint{};

        std::vector<ItemT> batch;
        try {
          batch = in_q.dequeue_batch(true, cfg_.worker_dequeue_timeout);
        } catch (const QueueEmptyError&) {
          if (want_stats || time_deq) {
            const double idle_dt =
                time_deq ? std::chrono::duration<double>(Clock::now() - t_deq0).count() : 0.0;
            prof_stage_add_(stage_idx, /*deq_calls=*/0, /*deq_timeouts=*/1, /*deq_s=*/0.0,
                            /*deq_idle_s=*/idle_dt);
          }
          continue;
        }

        if (time_deq || want_stats) {
          const double dt = time_deq ? std::chrono::duration<double>(Clock::now() - t_deq0).count() : 0.0;
          prof_stage_add_(stage_idx, /*deq_calls=*/1, /*deq_timeouts=*/0, /*deq_s=*/dt, /*deq_idle_s=*/0.0);
        }

        if (stop_event_.load(std::memory_order_acquire)) return;
        if (batch.empty()) return;  // closed and empty

        if (want_stats) {
          prof_stage_batched_io_(stage_idx, /*batches=*/1,
                                 /*items_in=*/static_cast<std::uint64_t>(batch.size()),
                                 /*items_out=*/0);
        }

        // ---- process
        const bool time_proc = want_timing && pcfg->timing_profile_process;
        const TimePoint t_p0 = time_proc ? Clock::now() : TimePoint{};

        std::optional<std::vector<ItemT>> outs_opt = stage.process_fn(std::move(batch), next_q);

        if (time_proc || want_stats) {
          const double dt = time_proc ? std::chrono::duration<double>(Clock::now() - t_p0).count() : 0.0;
          prof_stage_proc_(stage_idx, /*proc_calls=*/1, /*proc_s=*/dt);
        }

        if (!outs_opt.has_value()) {
          // stage pushed outputs itself
          continue;
        }

        std::vector<ItemT> outs = std::move(*outs_opt);
        if (want_stats) {
          prof_stage_batched_io_(stage_idx, /*batches=*/0, /*items_in=*/0,
                                 /*items_out=*/static_cast<std::uint64_t>(outs.size()));
        }
        if (outs.empty()) continue;

        // ---- last stage => output handler (may run concurrently; see contract above)
        if (is_last) {
          if constexpr (std::is_same<OutputHandlerT, NoOutputHandler<ItemT>>::value) {
            warn_once_no_output_handler_(stage_name, outs.size());
          }

          const bool time_out = want_timing && pcfg->timing_profile_output;
          const TimePoint t_o0 = time_out ? Clock::now() : TimePoint{};

          output_handler_()(std::move(outs));

          if (time_out || want_stats) {
            const double dt = time_out ? std::chrono::duration<double>(Clock::now() - t_o0).count() : 0.0;
            prof_stage_out_(stage_idx, /*out_calls=*/1,
                            /*out_items=*/static_cast<std::uint64_t>(outs.size()),
                            /*out_s=*/dt);
          }
          continue;
        }

        // ---- enqueue to next stage (if engine is responsible)
        if (!next_q || !next_stage) {
          record_failure_noexcept_(ThreadFailure{stage_name, tname, "enqueue_outputs", "logic_error", "missing next stage"});
          maybe_abort_on_failure_();
          return;
        }

        const bool time_enq = want_timing && pcfg->timing_profile_enqueue;
        const TimePoint t_e0 = time_enq ? Clock::now() : TimePoint{};

        enqueue_many_(*next_q, std::move(outs), next_stage->ingress_policy, stage_idx + 1);

        if (time_enq || want_stats) {
          const double dt = time_enq ? std::chrono::duration<double>(Clock::now() - t_e0).count() : 0.0;
          prof_stage_enq_(stage_idx, /*enq_calls=*/1, /*enq_s=*/dt);
        }
      }

    } catch (const std::exception& e) {
      record_failure_noexcept_(ThreadFailure{stage_name, tname, "worker", typeid(e).name(), e.what()});
      maybe_abort_on_failure_();
      return;
    } catch (...) {
      record_failure_noexcept_(ThreadFailure{stage_name, tname, "worker", "unknown", "unknown"});
      maybe_abort_on_failure_();
      return;
    }
  }

  // ------------------------------------------------------------
  // Enqueue helpers (policy + engine stats)
  // ------------------------------------------------------------

  void enqueue_many_(QueueT& q,
                     std::vector<ItemT> items,
                     const EnqueuePolicy& policy,
                     std::size_t target_stage_idx) {
    if (items.empty()) return;

    if (policy.drop_if_stopping && stop_event_.load(std::memory_order_acquire)) {
      prof_queue_add_(target_stage_idx, /*enq=*/0, /*drop=*/static_cast<std::uint64_t>(items.size()));
      return;
    }

    for (auto& item : items) {
      enqueue_one_(q, std::move(item), policy, target_stage_idx);
    }
  }

  void enqueue_one_(QueueT& q,
                    ItemT item,
                    const EnqueuePolicy& policy,
                    std::size_t target_stage_idx) {
    if (policy.drop_if_stopping && stop_event_.load(std::memory_order_acquire)) {
      prof_queue_add_(target_stage_idx, /*enq=*/0, /*drop=*/1);
      return;
    }

    int attempts = 0;
    while (true) {
      try {
        q.enqueue(std::move(item), policy.block, policy.timeout);
        prof_queue_add_(target_stage_idx, /*enq=*/1, /*drop=*/0);
        return;

      } catch (const ItemTooLargeError&) {
        prof_queue_add_(target_stage_idx, 0, 0, 0, 0, 1, 0);
        if (policy.on_full == OnFullPolicy::DROP) {
          prof_queue_add_(target_stage_idx, 0, 1);
          return;
        }
        if (policy.on_full == OnFullPolicy::ABORT) request_abort();
        throw;

      } catch (const QueueFullError&) {
        prof_queue_add_(target_stage_idx, 0, 0, 1, 0, 0, 0);

        if (policy.on_full == OnFullPolicy::DROP) {
          prof_queue_add_(target_stage_idx, 0, 1);
          return;
        }
        if (policy.on_full == OnFullPolicy::ABORT) {
          request_abort();
          throw;
        }
        if (policy.on_full == OnFullPolicy::RETRY) {
          if (attempts >= policy.max_retries) throw;
          attempts += 1;
          prof_queue_add_(target_stage_idx, 0, 0, 0, 0, 0, 1);
          if (policy.retry_backoff.count() > 0.0) std::this_thread::sleep_for(policy.retry_backoff);
          if (policy.drop_if_stopping && stop_event_.load(std::memory_order_acquire)) {
            prof_queue_add_(target_stage_idx, 0, 1);
            return;
          }
          continue;
        }
        throw;

      } catch (const QueueClosedError&) {
        prof_queue_add_(target_stage_idx, 0, 0, 0, 1, 0, 0);
        if (policy.on_closed == OnClosedPolicy::DROP ||
            (cfg_.suppress_closed_queue_errors_during_shutdown &&
             stop_event_.load(std::memory_order_acquire))) {
          prof_queue_add_(target_stage_idx, 0, 1);
          return;
        }
        throw;
      }
    }
  }

  // ------------------------------------------------------------
  // Join helpers (portable timeouts)
  // ------------------------------------------------------------

  bool join_all_threads_(std::optional<Duration> timeout) {
    if (!started_) return true;

    TimePoint end_at;
    const bool have_deadline = timeout.has_value();
    if (have_deadline) {
      if (timeout->count() < 0.0) throw std::invalid_argument("timeout must be >= 0");
      end_at = Clock::now() + std::chrono::duration_cast<Clock::duration>(*timeout);
    }

    {
      std::unique_lock<std::mutex> lk(done_mu_);
      auto pred = [&] { return total_running_ == 0; };
      if (!have_deadline) {
        done_cv_.wait(lk, pred);
      } else {
        if (!done_cv_.wait_until(lk, end_at, pred)) return false;
      }
    }

    for (std::size_t s = 0; s < NumStages; ++s) {
      for (auto& th : threads_[s]) {
        if (th.joinable()) {
          try { th.join(); } catch (...) {}
        }
      }
    }
    return true;
  }

  // ------------------------------------------------------------
  // Engine profiling updates (compiled out when EnableEngineProfiling == false)
  // ------------------------------------------------------------

  void prof_ingest_add_submit_(std::uint64_t submit_calls, std::uint64_t items_submitted) {
    if constexpr (!EnableEngineProfiling) {
      (void)submit_calls; (void)items_submitted;
      return;
    } else {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      prof_opt->ingest.submit_calls += submit_calls;
      prof_opt->ingest.items_submitted += items_submitted;
    }
  }

  void prof_ingest_add_submit_enqueue_(double enqueue_s) {
    if constexpr (!EnableEngineProfiling) {
      (void)enqueue_s;
      return;
    } else {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      prof_opt->ingest.submit_enqueue_calls += 1;
      prof_opt->ingest.submit_enqueue_s += enqueue_s;
    }
  }

  void prof_queue_add_(std::size_t stage_idx,
                       std::uint64_t enq,
                       std::uint64_t drop,
                       std::uint64_t full = 0,
                       std::uint64_t closed = 0,
                       std::uint64_t too_large = 0,
                       std::uint64_t retries = 0) {
    if constexpr (!EnableEngineProfiling) {
      (void)stage_idx; (void)enq; (void)drop; (void)full; (void)closed; (void)too_large; (void)retries;
      return;
    } else {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      auto& qs = prof_opt->queue_by_stage[stage_idx];
      qs.enqueued += enq;
      qs.dropped += drop;
      qs.full_errors += full;
      qs.closed_errors += closed;
      qs.too_large_errors += too_large;
      qs.retries += retries;
    }
  }

  void prof_stage_batched_io_(std::size_t stage_idx,
                              std::uint64_t batches,
                              std::uint64_t items_in,
                              std::uint64_t items_out) {
    if constexpr (EnableEngineProfiling) {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      auto& ss = prof_opt->stage_by_stage[stage_idx];
      ss.batches += batches;
      ss.items_in += items_in;
      ss.items_out += items_out;
    }
  }

  void prof_stage_add_(std::size_t stage_idx,
                       std::uint64_t deq_calls,
                       std::uint64_t deq_timeouts,
                       double deq_s,
                       double deq_idle_s) {
    if constexpr (EnableEngineProfiling) {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      auto& ss = prof_opt->stage_by_stage[stage_idx];
      ss.dequeue_calls += deq_calls;
      ss.dequeue_timeouts += deq_timeouts;
      ss.dequeue_s += deq_s;
      ss.dequeue_idle_s += deq_idle_s;
    }
  }

  void prof_stage_proc_(std::size_t stage_idx, std::uint64_t proc_calls, double proc_s) {
    if constexpr (EnableEngineProfiling) {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      auto& ss = prof_opt->stage_by_stage[stage_idx];
      ss.process_calls += proc_calls;
      ss.process_s += proc_s;
    }
  }

  void prof_stage_enq_(std::size_t stage_idx, std::uint64_t enq_calls, double enq_s) {
    if constexpr (EnableEngineProfiling) {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      auto& ss = prof_opt->stage_by_stage[stage_idx];
      ss.enqueue_calls += enq_calls;
      ss.enqueue_s += enq_s;
    }
  }

  void prof_stage_out_(std::size_t stage_idx,
                       std::uint64_t out_calls,
                       std::uint64_t out_items,
                       double out_s) {
    if constexpr (EnableEngineProfiling) {
      const auto* pcfg = prof_cfg_ptr_();
      auto& prof_opt = prof_storage_();
      if (!pcfg || !pcfg->enable_stats || !prof_opt) return;
      std::lock_guard<std::mutex> lk(prof_mu_);
      auto& ss = prof_opt->stage_by_stage[stage_idx];
      ss.output_calls += out_calls;
      ss.output_items += out_items;
      ss.output_s += out_s;
    }
  }

 private:
  std::array<StageConfig, NumStages> stages_;
  EngineConfig cfg_;

  std::array<std::unique_ptr<QueueT>, NumStages> queues_{};
  std::array<std::vector<std::thread>, NumStages> threads_{};

  // state
  mutable std::mutex state_mu_;
  bool started_ = false;
  bool input_closed_ = false;
  std::atomic<bool> stop_event_{false};

  // join bookkeeping
  mutable std::mutex done_mu_;
  std::condition_variable done_cv_;
  std::array<int, NumStages> stage_running_{};
  int total_running_ = 0;

  // failures
  mutable std::mutex fail_mu_;
  std::vector<ThreadFailure> failures_;

  // warnings
  mutable std::mutex warn_mu_;
  std::unordered_set<std::string> warned_no_output_;

  // profiling mutex (always present; see comment in profiling())
  mutable std::mutex prof_mu_;
};

}  // namespace dmx_host

#endif  // DMX_HOST_PIPELINED_ENGINE_HPP_
