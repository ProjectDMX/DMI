// batching_queue.hpp
#ifndef DMX_HOST_BATCHING_QUEUE_HPP_
#define DMX_HOST_BATCHING_QUEUE_HPP_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
// TODO: Clock overhead. Remove clock when no timeout, and reuse clock number without calling frequently.
// TODO: Fairness(Starvation): Use a per-CV way for policy control.

namespace dmx_host {

// ---------------------------
// Exceptions
// ---------------------------

struct ItemTooLargeError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct QueueClosedError : public std::runtime_error {
  // Subclasses runtime_error for compatibility with code catching std::runtime_error.
  using std::runtime_error::runtime_error;
};

struct QueueFullError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct QueueEmptyError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------
// Queue profiling (optional)
// ---------------------------

struct QueueProfilingConfig {
  // If both are false, profiling is disabled.
  bool enable_stats = false;
  bool enable_timing = false;

  // Granular timing switches
  bool timing_profile_enqueue = true;
  bool timing_profile_dequeue = true;
  bool timing_profile_wait = true;
  bool timing_profile_batch_build = true;
};

struct QueueProfile {
  // counts
  std::uint64_t enqueue_calls = 0;
  std::uint64_t enqueue_full_errors = 0;
  std::uint64_t enqueue_closed_errors = 0;

  std::uint64_t dequeue_calls = 0;
  std::uint64_t dequeue_empty_errors = 0;  // includes non-blocking and user-timeout empties
  std::uint64_t dequeued_batches = 0;
  std::uint64_t dequeued_items = 0;
  std::uint64_t dequeue_returned_empty_on_close = 0;

  std::uint64_t close_calls = 0;

  // Per-condition-variable notification counts.
  // Producer CV = cv_can_enqueue_ (producers waiting for capacity).
  std::uint64_t producer_notify_one_calls = 0;
  std::uint64_t producer_notify_all_calls = 0;

  // Consumer CV = cv_can_dequeue_ (consumers waiting for batch availability).
  std::uint64_t consumer_notify_one_calls = 0;
  std::uint64_t consumer_notify_all_calls = 0;

  // timing totals (seconds) -- still part of public interface, but no longer used internally
  double enqueue_total_s = 0.0;
  double enqueue_wait_s = 0.0;
  std::uint64_t enqueue_wait_calls = 0;

  double dequeue_total_s = 0.0;
  double dequeue_wait_s = 0.0;
  std::uint64_t dequeue_wait_calls = 0;
  double dequeue_batch_build_s = 0.0;
};

struct ProfilingSnapshot {
  std::string name;

  struct Counts {
    std::uint64_t enqueue_calls = 0;
    std::uint64_t enqueue_full_errors = 0;
    std::uint64_t enqueue_closed_errors = 0;

    std::uint64_t dequeue_calls = 0;
    std::uint64_t dequeue_empty_errors = 0;
    std::uint64_t dequeued_batches = 0;
    std::uint64_t dequeued_items = 0;
    std::uint64_t dequeue_returned_empty_on_close = 0;

    std::uint64_t close_calls = 0;
    // Producer CV (cv_can_enqueue_)
    std::uint64_t producer_notify_one_calls = 0;
    std::uint64_t producer_notify_all_calls = 0;

    // Consumer CV (cv_can_dequeue_)
    std::uint64_t consumer_notify_one_calls = 0;
    std::uint64_t consumer_notify_all_calls = 0;

    std::uint64_t enqueue_wait_calls = 0;
    std::uint64_t dequeue_wait_calls = 0;
  } counts;

  struct Timing {
    double enqueue_total_s = 0.0;
    double enqueue_avg_s = 0.0;

    double dequeue_total_s = 0.0;
    double dequeue_avg_s = 0.0;

    double enqueue_wait_s = 0.0;
    double enqueue_wait_avg_s = 0.0;

    double dequeue_wait_s = 0.0;
    double dequeue_wait_avg_s = 0.0;

    double dequeue_batch_build_s = 0.0;
    double dequeue_batch_build_avg_s = 0.0;
  };

  std::optional<Timing> timing;
};

// ---------------------------
// C++17 EBO helpers (no extra namespaces; all in dmx_host)
// ---------------------------

struct empty_profiling_storage {};

// Internal integer timing accumulators (ns)
struct QueueTimingProfile {
  std::uint64_t enqueue_total_ns = 0;
  std::uint64_t enqueue_wait_ns = 0;
  std::uint64_t dequeue_total_ns = 0;
  std::uint64_t dequeue_wait_ns = 0;
  std::uint64_t dequeue_batch_build_ns = 0;
};

// EBO helper for C++17: stores T either as an empty base (when possible) or as a member.
template <typename T,
          int Index,
          bool UseEbo = (std::is_empty<T>::value && !std::is_final<T>::value)>
struct ebo_storage;

template <typename T, int Index>
struct ebo_storage<T, Index, true> : private T {
  ebo_storage() = default;

  template <typename... Args,
            typename = std::enable_if_t<std::is_constructible<T, Args...>::value>>
  explicit ebo_storage(Args&&... args) : T(std::forward<Args>(args)...) {}

  T& get() noexcept { return *this; }
  const T& get() const noexcept { return *this; }
};

template <typename T, int Index>
struct ebo_storage<T, Index, false> {
  T value;

  ebo_storage() = default;

  template <typename... Args,
            typename = std::enable_if_t<std::is_constructible<T, Args...>::value>>
  explicit ebo_storage(Args&&... args) : value(std::forward<Args>(args)...) {}

  T& get() noexcept { return value; }
  const T& get() const noexcept { return value; }
};

template <bool Enable>
using prof_cfg_storage_t =
    std::conditional_t<Enable, std::optional<QueueProfilingConfig>, empty_profiling_storage>;

template <bool Enable>
using prof_storage_t =
    std::conditional_t<Enable, std::optional<QueueProfile>, empty_profiling_storage>;

template <bool Enable>
using prof_time_storage_t =
    std::conditional_t<Enable, std::optional<QueueTimingProfile>, empty_profiling_storage>;

// ---------------------------
// Producer wait-size tracking (compile-time optional)
// ---------------------------

template <typename SizeT, bool Enable>
struct ProducerWaitSizeTracker;

template <typename SizeT>
struct ProducerWaitSizeTracker<SizeT, false> {
  // empty
};

template <typename SizeT>
struct ProducerWaitSizeTracker<SizeT, true> {
  std::multiset<SizeT> sizes;
};

// ---------------------------
// WatermarkBatchingQueue
// ---------------------------

template <typename ItemT,
          typename SizeT,
          bool NotifyAllConsumers = false,
          bool CheckMinSizeProducers = false,
          bool EnableProfiling = false>
class WatermarkBatchingQueue
    : private ebo_storage<prof_cfg_storage_t<EnableProfiling>, 0>,
      private ebo_storage<prof_storage_t<EnableProfiling>, 1>,
      private ebo_storage<ProducerWaitSizeTracker<SizeT, CheckMinSizeProducers>, 2>,
      private ebo_storage<prof_time_storage_t<EnableProfiling>, 3> {
 private:
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;
  using Count = std::uint64_t;

  // Internal time representation: integer nanoseconds.
  using Tick = std::int64_t;        // signed for diffs
  using TickU = std::uint64_t;      // non-negative accumulator
  
  // Size arithmetic contract:
  // SizeT is expected to be a non-negative size type (typically uint64_t).
  // The queue performs additions like (buffered_size_ + item_size) and (batch_size + item_size)
  // when comparing against watermark / batch-size limits.
  // Callers/configuration must ensure these additions do not overflow SizeT.
  // If overflow occurs, enqueue/dequeue decisions may become incorrect (wrap-around for unsigned).

  static_assert(
      std::is_convertible_v<decltype(std::declval<const ItemT&>().size()), SizeT>,
      "ItemT must have size() const convertible to SizeT");

  static Tick now_ticks_() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
  }

  static Tick duration_to_ticks_(const Duration& d) noexcept {
    using NsFloat = std::chrono::duration<long double, std::nano>;
    const long double ns_ld = std::chrono::duration_cast<NsFloat>(d).count();
    if (ns_ld <= 0.0L) return Tick{0};
    const long double max_ld = static_cast<long double>(std::numeric_limits<Tick>::max());
    if (ns_ld >= max_ld) return std::numeric_limits<Tick>::max();
    return static_cast<Tick>(ns_ld);  // trunc
  }

  static Tick add_ticks_sat_(Tick a, Tick b) noexcept {
    if (b <= 0) return a;
    const Tick maxv = std::numeric_limits<Tick>::max();
    if (a >= maxv - b) return maxv;
    return a + b;
  }

  static TickU clamp_to_u_(Tick v) noexcept {
    return (v <= 0) ? TickU{0} : static_cast<TickU>(v);
  }

  static std::chrono::nanoseconds to_ns_wait_(Tick v) noexcept {
    if (v <= 0) return std::chrono::nanoseconds{0};
    return std::chrono::nanoseconds{v};
  }

  static double ns_to_s_(TickU ns) noexcept {
    return static_cast<double>(ns) * 1e-9;
  }

  struct Entry {
    ItemT item;
    SizeT size;      // must not change after enqueue
    Tick enq_at;     // integer monotonic time (ns)
  };

 public:
  using WarnFn = std::function<void(const std::string&)>;

  WatermarkBatchingQueue(std::optional<int> min_batch_items = std::nullopt,
                         std::optional<SizeT> min_batch_size = std::nullopt,
                         std::optional<Duration> max_linger = std::nullopt,
                         std::optional<int> max_batch_items = std::nullopt,
                         std::optional<SizeT> max_batch_size = std::nullopt,
                         std::optional<int> high_watermark_items = std::nullopt,
                         std::optional<SizeT> high_watermark_size = std::nullopt,
                         std::string name = "",
                         std::optional<QueueProfilingConfig> profiling = std::nullopt,
                         WarnFn warn_logger = nullptr)
      : min_batch_items_(min_batch_items),
        min_batch_size_(min_batch_size),
        max_linger_ns_(max_linger ? std::optional<Tick>(duration_to_ticks_(*max_linger))
                                 : std::nullopt),
        max_batch_items_(max_batch_items),
        max_batch_size_(max_batch_size),
        high_watermark_items_(high_watermark_items),
        high_watermark_size_(high_watermark_size),
        name_(std::move(name)),
        warn_(std::move(warn_logger)) {
    if (!min_batch_items_ && !min_batch_size_ && !max_linger_ns_) {
      throw std::invalid_argument(
          "At least one of min_batch_items/min_batch_size/max_linger must be set");
    }

    if (min_batch_items_ && *min_batch_items_ <= 0) {
      throw std::invalid_argument("min_batch_items must be > 0 or nullopt");
    }
    if (min_batch_size_ && *min_batch_size_ < SizeT{0}) {
      throw std::invalid_argument("min_batch_size must be >= 0 or nullopt");
    }
    if (max_linger && max_linger->count() < 0.0) {
      throw std::invalid_argument("max_linger must be >= 0 or nullopt");
    }

    if (max_batch_items_ && *max_batch_items_ <= 0) {
      throw std::invalid_argument("max_batch_items must be > 0 or nullopt");
    }
    if (max_batch_size_ && *max_batch_size_ <= SizeT{0}) {
      throw std::invalid_argument("max_batch_size must be > 0 or nullopt");
    }

    if (high_watermark_items_ && *high_watermark_items_ <= 0) {
      throw std::invalid_argument("high_watermark_items must be > 0 or nullopt");
    }
    if (high_watermark_size_ && *high_watermark_size_ <= SizeT{0}) {
      throw std::invalid_argument("high_watermark_size must be > 0 or nullopt");
    }

    if (min_batch_items_ && high_watermark_items_){
        if (*min_batch_items_ > *high_watermark_items_){
            throw std::invalid_argument("min_batch_items must be <= high_watermark_items if both set");
        }
    }

    if (min_batch_size_ && high_watermark_size_){
        if(*min_batch_size_ > *high_watermark_size_){
            throw std::invalid_argument("min_batch_size must be <= high_watermark_size if both set");
        }
    }

    if (min_batch_items_ && max_batch_items_) {
        if (*min_batch_items_ > *max_batch_items_) {
            throw std::invalid_argument(
                "min_batch_items must be <= max_batch_items; otherwise dequeue may leave a tail "
                "smaller than min_batch_items that blocks until more items arrive or the queue is closed");
        }
    }

    if (min_batch_size_ && max_batch_size_) {
        if (*min_batch_size_ > *max_batch_size_) {
            throw std::invalid_argument(
                "min_batch_size must be <= max_batch_size; otherwise dequeue may leave a tail "
                "smaller than min_batch_size that blocks until more items arrive or the queue is closed");
        }
    }


    if (min_batch_size_ && *min_batch_size_ == SizeT{0}) {
      if (!min_batch_items_) {
        throw std::invalid_argument(
            "min_batch_size=0 is only allowed when min_batch_items=1 (equivalent on non-empty "
            "queues). Set min_batch_items=1 and omit min_batch_size.");
      }
      if (*min_batch_items_ == 1) {
        if (warn_) {
          warn_("WatermarkBatchingQueue(" + name_ +
                "): min_batch_size=0 is redundant with min_batch_items=1; treating min_batch_size "
                "as nullopt.");
        }
        min_batch_size_.reset();
      } else {
        throw std::invalid_argument("min_batch_size=0 requires min_batch_items=1");
      }
    }

    if constexpr (EnableProfiling) {
      auto& cfg_opt = prof_cfg_storage_();
      auto& prof_opt = prof_storage_();
      auto& time_opt = prof_time_storage_();

      cfg_opt = profiling;
      if (cfg_opt && (cfg_opt->enable_stats || cfg_opt->enable_timing)) {
        prof_opt.emplace();
        if (cfg_opt->enable_timing && !cfg_opt->enable_stats) {
          cfg_opt->enable_stats = true;
          if (warn_) warn_("profiling.enable_stats set to true because enable_timing is true");
        }
        if (cfg_opt->enable_timing) time_opt.emplace();
      }
    } else {
      (void)profiling;
    }
  }

  const std::string& name() const { return name_; }

  void close() {
    std::unique_lock<std::mutex> lk(mu_);
    closed_ = true;
    if constexpr (EnableProfiling) {
      auto& prof_opt = prof_storage_();
      if (prof_opt) {
        prof_opt->close_calls += 1;
        prof_opt->producer_notify_all_calls += 1;
        prof_opt->consumer_notify_all_calls += 1;
      }
    }
    cv_can_enqueue_.notify_all();
    cv_can_dequeue_.notify_all();
  }

  bool closed() const {
    std::unique_lock<std::mutex> lk(mu_);
    return closed_;
  }

  std::size_t size() const {
    std::unique_lock<std::mutex> lk(mu_);
    return static_cast<std::size_t>(buffered_items_);
  }

  SizeT buffered_size() const {
    std::unique_lock<std::mutex> lk(mu_);
    return buffered_size_;
  }

  std::optional<ProfilingSnapshot> profiling() const {
    if constexpr (!EnableProfiling) {
      return std::nullopt;
    } else {
      std::unique_lock<std::mutex> lk(mu_);
      const auto& prof_opt = prof_storage_();
      const auto& cfg_opt = prof_cfg_storage_();
      const auto& time_opt = prof_time_storage_();
      if (!prof_opt || !cfg_opt) return std::nullopt;

      ProfilingSnapshot snap;
      snap.name = name_;

      snap.counts.enqueue_calls = prof_opt->enqueue_calls;
      snap.counts.enqueue_full_errors = prof_opt->enqueue_full_errors;
      snap.counts.enqueue_closed_errors = prof_opt->enqueue_closed_errors;

      snap.counts.dequeue_calls = prof_opt->dequeue_calls;
      snap.counts.dequeue_empty_errors = prof_opt->dequeue_empty_errors;
      snap.counts.dequeued_batches = prof_opt->dequeued_batches;
      snap.counts.dequeued_items = prof_opt->dequeued_items;
      snap.counts.dequeue_returned_empty_on_close = prof_opt->dequeue_returned_empty_on_close;

      snap.counts.close_calls = prof_opt->close_calls;

      snap.counts.producer_notify_one_calls = prof_opt->producer_notify_one_calls;
      snap.counts.producer_notify_all_calls = prof_opt->producer_notify_all_calls;
      snap.counts.consumer_notify_one_calls = prof_opt->consumer_notify_one_calls;
      snap.counts.consumer_notify_all_calls = prof_opt->consumer_notify_all_calls;

      snap.counts.enqueue_wait_calls = prof_opt->enqueue_wait_calls;
      snap.counts.dequeue_wait_calls = prof_opt->dequeue_wait_calls;

      if (cfg_opt->enable_timing && time_opt) {
        ProfilingSnapshot::Timing t{};

        if (cfg_opt->timing_profile_enqueue) {
          t.enqueue_total_s = ns_to_s_(time_opt->enqueue_total_ns);
          if (prof_opt->enqueue_calls > 0) {
            t.enqueue_avg_s = t.enqueue_total_s / static_cast<double>(prof_opt->enqueue_calls);
          }
        }

        if (cfg_opt->timing_profile_dequeue) {
          t.dequeue_total_s = ns_to_s_(time_opt->dequeue_total_ns);
          if (prof_opt->dequeue_calls > 0) {
            t.dequeue_avg_s = t.dequeue_total_s / static_cast<double>(prof_opt->dequeue_calls);
          }
        }

        if (cfg_opt->timing_profile_wait) {
          t.enqueue_wait_s = ns_to_s_(time_opt->enqueue_wait_ns);
          if (prof_opt->enqueue_wait_calls > 0) {
            t.enqueue_wait_avg_s =
                t.enqueue_wait_s / static_cast<double>(prof_opt->enqueue_wait_calls);
          }

          t.dequeue_wait_s = ns_to_s_(time_opt->dequeue_wait_ns);
          if (prof_opt->dequeue_wait_calls > 0) {
            t.dequeue_wait_avg_s =
                t.dequeue_wait_s / static_cast<double>(prof_opt->dequeue_wait_calls);
          }
        }

        if (cfg_opt->timing_profile_batch_build) {
          t.dequeue_batch_build_s = ns_to_s_(time_opt->dequeue_batch_build_ns);
          if (prof_opt->dequeued_batches > 0) {
            t.dequeue_batch_build_avg_s =
                t.dequeue_batch_build_s / static_cast<double>(prof_opt->dequeued_batches);
          }
        }

        snap.timing = t;
      }

      return snap;
    }
  }

  void reset_profiling() {
    if constexpr (EnableProfiling) {
      std::unique_lock<std::mutex> lk(mu_);
      auto& prof_opt = prof_storage_();
      auto& time_opt = prof_time_storage_();
      if (!prof_opt) return;
      prof_opt.emplace();
      if (time_opt) time_opt.emplace();
    }
  }

  // timeout: seconds (Duration). If block=false, timeout is ignored.
  void enqueue(ItemT item, bool block = true, std::optional<Duration> timeout = std::nullopt) {
    const SizeT sz = static_cast<SizeT>(item.size());
    if (sz <= SizeT{0}) throw std::invalid_argument("item.size() must be > 0");
    if (max_batch_size_ && sz > *max_batch_size_) throw ItemTooLargeError("item.size() exceeds max_batch_size");
    if (timeout && timeout->count() < 0.0) throw std::invalid_argument("timeout must be >= 0 or nullopt");

    std::optional<Tick> deadline_ns;
    if (timeout) {
      deadline_ns = add_ticks_sat_(now_ticks_(), duration_to_ticks_(*timeout));
    }

    std::unique_lock<std::mutex> lk(mu_);

    [[maybe_unused]] bool do_time_total = false;
    [[maybe_unused]] bool do_time_wait = false;
    [[maybe_unused]] Tick t0_ns = 0;

    if constexpr (EnableProfiling) {
      auto& prof_opt = prof_storage_();
      auto& cfg_opt = prof_cfg_storage_();
      auto& time_opt = prof_time_storage_();

      do_time_total = static_cast<bool>(time_opt && cfg_opt && cfg_opt->enable_timing &&
                                        cfg_opt->timing_profile_enqueue);
      do_time_wait = static_cast<bool>(time_opt && cfg_opt && cfg_opt->enable_timing &&
                                       cfg_opt->timing_profile_wait);

      if (do_time_total) t0_ns = now_ticks_();
      if (prof_opt) prof_opt->enqueue_calls += 1;
    }

    auto finish_total = [&] {
      if constexpr (EnableProfiling) {
        auto& time_opt = prof_time_storage_();
        if (do_time_total && time_opt) {
          const Tick dt = now_ticks_() - t0_ns;
          time_opt->enqueue_total_ns += clamp_to_u_(dt);
        }
      }
    };

    auto wait_can_enqueue_dispatch = [&](std::optional<Tick> dur_ns) {
      if constexpr (CheckMinSizeProducers) {
        wait_can_enqueue_(lk, dur_ns, sz);
      } else {
        wait_can_enqueue_(lk, dur_ns);
      }
    };

    try {
      if (closed_) {
        if constexpr (EnableProfiling) {
          auto& prof_opt = prof_storage_();
          if (prof_opt) prof_opt->enqueue_closed_errors += 1;
        }
        throw QueueClosedError("Queue is closed");
      }

      while (!can_enqueue_(sz)) {
        if (closed_) {
          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            if (prof_opt) prof_opt->enqueue_closed_errors += 1;
          }
          throw QueueClosedError("Queue is closed");
        }
        if (!block) {
          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            if (prof_opt) prof_opt->enqueue_full_errors += 1;
          }
          throw QueueFullError("enqueue would block (high watermark)");
        }

        if (!deadline_ns) {
          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            auto& time_opt = prof_time_storage_();
            if (do_time_wait) {
              const Tick w0 = now_ticks_();
              wait_can_enqueue_dispatch(std::nullopt);
              const Tick wd = now_ticks_() - w0;
              if (prof_opt) prof_opt->enqueue_wait_calls += 1;
              if (time_opt) time_opt->enqueue_wait_ns += clamp_to_u_(wd);
            } else {
              wait_can_enqueue_dispatch(std::nullopt);
              if (prof_opt) prof_opt->enqueue_wait_calls += 1;
            }
          } else {
            wait_can_enqueue_dispatch(std::nullopt);
          }
        } else {
          const Tick now = now_ticks_();
          if (now >= *deadline_ns) {
            if constexpr (EnableProfiling) {
              auto& prof_opt = prof_storage_();
              if (prof_opt) prof_opt->enqueue_full_errors += 1;
            }
            throw QueueFullError("enqueue timeout");
          }
          const Tick remaining = *deadline_ns - now;

          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            auto& time_opt = prof_time_storage_();
            if (do_time_wait) {
              const Tick w0 = now_ticks_();
              wait_can_enqueue_dispatch(std::optional<Tick>{remaining});
              const Tick wd = now_ticks_() - w0;
              if (prof_opt) prof_opt->enqueue_wait_calls += 1;
              if (time_opt) time_opt->enqueue_wait_ns += clamp_to_u_(wd);
            } else {
              wait_can_enqueue_dispatch(std::optional<Tick>{remaining});
              if (prof_opt) prof_opt->enqueue_wait_calls += 1;
            }
          } else {
            wait_can_enqueue_dispatch(std::optional<Tick>{remaining});
          }
        }
      }

      if (closed_) {
        if constexpr (EnableProfiling) {
          auto& prof_opt = prof_storage_();
          if (prof_opt) prof_opt->enqueue_closed_errors += 1;
        }
        throw QueueClosedError("Queue is closed");
      }

      const Tick now = now_ticks_();
      const bool was_empty = q_.empty();
      const bool allowed_before =
          was_empty ? false : dequeue_allowed_(now, buffered_items_, buffered_size_);

      do_enqueue_(std::move(item), sz, now);

      std::uint64_t consumer_notify_one_calls = 0;
      std::uint64_t consumer_notify_all_calls = 0;
      if (waiting_dequeuers_ > 0) {
        const bool allowed_after = dequeue_allowed_(now, buffered_items_, buffered_size_);
        if (was_empty || (!allowed_before && allowed_after)) {
          if constexpr (NotifyAllConsumers) {
            cv_can_dequeue_.notify_all();
            consumer_notify_all_calls += 1;
          } else {
            cv_can_dequeue_.notify_one();
            consumer_notify_one_calls += 1;
          }
        }
      }

      if constexpr (EnableProfiling) {
        auto& prof_opt = prof_storage_();
        if (prof_opt) {
          prof_opt->consumer_notify_one_calls += consumer_notify_one_calls;
          prof_opt->consumer_notify_all_calls += consumer_notify_all_calls;
        }
      }

      finish_total();
    } catch (...) {
      finish_total();
      throw;
    }
  }


  // Dequeue a batch of items.
  //
  // Contract:
  // - Returns a NON-empty batch when items are available and batching triggers are met
  //   (min_batch_items/min_batch_size/max_linger) or high-watermark forces a flush.
  // - Returns an empty vector ONLY when the queue is closed AND empty (drained). This is
  //   intended as a shutdown sentinel for worker loops.
  // - If block=false and a batch is not ready, throws QueueEmptyError.
  // - If timeout is provided and expires, throws QueueEmptyError.
  std::vector<ItemT> dequeue_batch(bool block = true, std::optional<Duration> timeout = std::nullopt) {
    if (timeout && timeout->count() < 0.0) {
      throw std::invalid_argument("timeout must be >= 0 or nullopt");
    }

    std::optional<Tick> deadline_ns;
    if (timeout) {
      deadline_ns = add_ticks_sat_(now_ticks_(), duration_to_ticks_(*timeout));
    }

    std::unique_lock<std::mutex> lk(mu_);

    [[maybe_unused]] bool do_time_total = false;
    [[maybe_unused]] bool do_time_wait = false;
    [[maybe_unused]] bool do_time_build = false;
    [[maybe_unused]] Tick t0_ns = 0;

    if constexpr (EnableProfiling) {
      auto& prof_opt = prof_storage_();
      auto& cfg_opt = prof_cfg_storage_();
      auto& time_opt = prof_time_storage_();

      do_time_total = static_cast<bool>(time_opt && cfg_opt && cfg_opt->enable_timing &&
                                        cfg_opt->timing_profile_dequeue);
      do_time_wait = static_cast<bool>(time_opt && cfg_opt && cfg_opt->enable_timing &&
                                       cfg_opt->timing_profile_wait);
      do_time_build = static_cast<bool>(time_opt && cfg_opt && cfg_opt->enable_timing &&
                                        cfg_opt->timing_profile_batch_build);

      if (do_time_total) t0_ns = now_ticks_();
      if (prof_opt) prof_opt->dequeue_calls += 1;
    }

    auto finish_total = [&] {
      if constexpr (EnableProfiling) {
        auto& time_opt = prof_time_storage_();
        if (do_time_total && time_opt) {
          const Tick dt = now_ticks_() - t0_ns;
          time_opt->dequeue_total_ns += clamp_to_u_(dt);
        }
      }
    };

    try {
      while (true) {
        if (!q_.empty()) {
          const Tick now = now_ticks_();
          if (dequeue_allowed_(now, buffered_items_, buffered_size_)) break;

          if (!block) {
            if constexpr (EnableProfiling) {
              auto& prof_opt = prof_storage_();
              if (prof_opt) prof_opt->dequeue_empty_errors += 1;
            }
            throw QueueEmptyError("dequeue would block (triggers not met)");
          }

          const auto wait_for_ns = compute_wait_for_dequeue_(now, deadline_ns);

          if (wait_for_ns && *wait_for_ns <= 0) {
            if (deadline_ns && now >= *deadline_ns) {
              if constexpr (EnableProfiling) {
                auto& prof_opt = prof_storage_();
                if (prof_opt) prof_opt->dequeue_empty_errors += 1;
              }
              throw QueueEmptyError("dequeue timeout");
            }
            continue;
          }

          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            auto& time_opt = prof_time_storage_();
            if (do_time_wait) {
              const Tick w0 = now_ticks_();
              wait_can_dequeue_(lk, wait_for_ns);
              const Tick wd = now_ticks_() - w0;
              if (prof_opt) prof_opt->dequeue_wait_calls += 1;
              if (time_opt) time_opt->dequeue_wait_ns += clamp_to_u_(wd);
            } else {
              wait_can_dequeue_(lk, wait_for_ns);
              if (prof_opt) prof_opt->dequeue_wait_calls += 1;
            }
          } else {
            wait_can_dequeue_(lk, wait_for_ns);
          }
          continue;
        }

        if (closed_) {
          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            if (prof_opt) prof_opt->dequeue_returned_empty_on_close += 1;
          }
          finish_total();
          return {};
        }
        if (!block) {
          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            if (prof_opt) prof_opt->dequeue_empty_errors += 1;
          }
          throw QueueEmptyError("dequeue would block (empty)");
        }

        const Tick now = now_ticks_();
        const auto wait_for_ns = compute_wait_for_dequeue_(now, deadline_ns);
        if (wait_for_ns && *wait_for_ns <= 0) {
          if constexpr (EnableProfiling) {
            auto& prof_opt = prof_storage_();
            if (prof_opt) prof_opt->dequeue_empty_errors += 1;
          }
          throw QueueEmptyError("dequeue timeout");
        }

        if constexpr (EnableProfiling) {
          auto& prof_opt = prof_storage_();
          auto& time_opt = prof_time_storage_();
          if (do_time_wait) {
            const Tick w0 = now_ticks_();
            wait_can_dequeue_(lk, wait_for_ns);
            const Tick wd = now_ticks_() - w0;
            if (prof_opt) prof_opt->dequeue_wait_calls += 1;
            if (time_opt) time_opt->dequeue_wait_ns += clamp_to_u_(wd);
          } else {
            wait_can_dequeue_(lk, wait_for_ns);
            if (prof_opt) prof_opt->dequeue_wait_calls += 1;
          }
        } else {
          wait_can_dequeue_(lk, wait_for_ns);
        }
      }

      [[maybe_unused]] Tick b0_ns = 0;
      if constexpr (EnableProfiling) {
        if (do_time_build) b0_ns = now_ticks_();
      }

      std::vector<ItemT> batch;
      SizeT batch_size = SizeT{0};

      while (!q_.empty()) {
        if (max_batch_items_ && batch.size() >= static_cast<std::size_t>(*max_batch_items_)) break;

        const Entry& head = q_.front();

        if (max_batch_size_) {
          const SizeT max_sz = *max_batch_size_;
          /*
          //Now larger than max_batch_size is rejected.
          if (batch.empty() && head.size > max_sz) {
            Entry e = std::move(q_.front());
            q_.pop_front();
            do_dequeue_(e);
            batch.emplace_back(std::move(e.item));
            break;
          }
          */
          // Invariant: enqueue() rejects items larger than max_batch_size_.
          // If this ever triggers, something violated the queue contract; fail fast
          // instead of returning an empty batch and confusing consumers.
          if (head.size > max_sz) {
              throw std::logic_error("WatermarkBatchingQueue(" + name_ +
                                  "): invariant violated: item.size() > max_batch_size");
          }
          // PRECONDITION: batch_size + head.size does not overflow SizeT.
          // (SizeT is typically uint64_t and sizes/watermarks are configured far below 2^64.)
          if ((batch_size + head.size) > max_sz) break;
        }

        Entry e = std::move(q_.front());
        q_.pop_front();
        do_dequeue_(e);
        batch.emplace_back(std::move(e.item));
        batch_size = batch_size + e.size;
      }

      if constexpr (EnableProfiling) {
        auto& prof_opt = prof_storage_();
        auto& time_opt = prof_time_storage_();
        if (do_time_build && time_opt) {
          const Tick bd = now_ticks_() - b0_ns;
          time_opt->dequeue_batch_build_ns += clamp_to_u_(bd);
        }
        if (prof_opt) {
          prof_opt->dequeued_batches += 1;
          prof_opt->dequeued_items += static_cast<std::uint64_t>(batch.size());
        }
      }

      std::uint64_t producer_notify_all_calls = 0;
      std::uint64_t consumer_notify_one_calls = 0;

      if ((high_watermark_items_ || high_watermark_size_) && waiting_enqueuers_ > 0) {
        const Tick now_n = now_ticks_();
        const bool dq_allowed_now = dequeue_allowed_(now_n, buffered_items_, buffered_size_);

        bool should_notify = false;
        if (!dq_allowed_now) {
          // This is flip from flase to true.
          should_notify = true;
        } else {
          bool count_ok = true;
          if (high_watermark_items_) {
            count_ok = buffered_items_ < static_cast<Count>(*high_watermark_items_);
          }

          bool size_ok = true;
          if (high_watermark_size_) {
            if constexpr (CheckMinSizeProducers) {
              const auto& sizes = producer_wait_sizes_().sizes;
              if (!sizes.empty()) {
                const SizeT min_sz = *sizes.begin();
                size_ok = (buffered_size_ + min_sz) <= *high_watermark_size_;
              }
            }
          }

          should_notify = count_ok && size_ok;
        }

        if (should_notify) {
          cv_can_enqueue_.notify_all();
          producer_notify_all_calls += 1;
        }
      }

      if constexpr (!NotifyAllConsumers) {
        if (waiting_dequeuers_ > 0 && !q_.empty()) {
          const Tick now2 = now_ticks_();
          if (dequeue_allowed_(now2, buffered_items_, buffered_size_)) {
            cv_can_dequeue_.notify_one();
            consumer_notify_one_calls += 1;
          }
        }
      }

      if constexpr (EnableProfiling) {
        auto& prof_opt = prof_storage_();
        if (prof_opt) {
          prof_opt->producer_notify_all_calls += producer_notify_all_calls;
          prof_opt->consumer_notify_one_calls += consumer_notify_one_calls;
        }
      }

      finish_total();
      return batch;
    } catch (...) {
      finish_total();
      throw;
    }
  }

 private:
  using ProfCfgStorage = prof_cfg_storage_t<EnableProfiling>;
  using ProfStorage = prof_storage_t<EnableProfiling>;
  using ProfTimeStorage = prof_time_storage_t<EnableProfiling>;

  using ProfCfgBase = ebo_storage<ProfCfgStorage, 0>;
  using ProfBase = ebo_storage<ProfStorage, 1>;
  using ProducerWaitSizes = ProducerWaitSizeTracker<SizeT, CheckMinSizeProducers>;
  using ProducerWaitSizesBase = ebo_storage<ProducerWaitSizes, 2>;
  using ProfTimeBase = ebo_storage<ProfTimeStorage, 3>;

  ProfCfgStorage& prof_cfg_storage_() { return ProfCfgBase::get(); }
  const ProfCfgStorage& prof_cfg_storage_() const { return ProfCfgBase::get(); }

  ProfStorage& prof_storage_() { return ProfBase::get(); }
  const ProfStorage& prof_storage_() const { return ProfBase::get(); }

  ProducerWaitSizes& producer_wait_sizes_() { return ProducerWaitSizesBase::get(); }
  const ProducerWaitSizes& producer_wait_sizes_() const { return ProducerWaitSizesBase::get(); }

  ProfTimeStorage& prof_time_storage_() { return ProfTimeBase::get(); }
  const ProfTimeStorage& prof_time_storage_() const { return ProfTimeBase::get(); }

  void do_enqueue_(ItemT item, const SizeT& sz, Tick enq_at) {
    q_.push_back(Entry{std::move(item), sz, enq_at});
    buffered_items_ += 1;
    buffered_size_ = buffered_size_ + sz;
  }

  void do_dequeue_(const Entry& e) {
    buffered_items_ -= 1;
    buffered_size_ = buffered_size_ - e.size;
  }

  bool high_watermark_reached_(Count cnt, const SizeT& size) const {
    if (high_watermark_items_ && cnt >= static_cast<Count>(*high_watermark_items_)) return true;
    if (high_watermark_size_ && size >= *high_watermark_size_) return true;
    return false;
  }

  bool flush_triggers_met_(Tick now, Count cnt, const SizeT& size, std::optional<Tick> oldest) const {
    if (min_batch_items_ && cnt >= static_cast<Count>(*min_batch_items_)) return true;
    if (min_batch_size_ && size >= *min_batch_size_) return true;
    if (max_linger_ns_ && oldest) {
      if (add_ticks_sat_(*oldest, *max_linger_ns_) <= now) return true;
    }
    return false;
  }

  bool dequeue_allowed_(Tick now, Count cnt, const SizeT& size) const {
    if (closed_) return true;
    std::optional<Tick> oldest = q_.empty() ? std::nullopt : std::optional<Tick>(q_.front().enq_at);
    if (flush_triggers_met_(now, cnt, size, oldest)) return true;
    if (high_watermark_reached_(cnt, size)) return true;
    return false;
  }

  bool can_enqueue_(const SizeT& sz) const {
    if (!high_watermark_items_ && !high_watermark_size_) return true;

    bool exceed_cnt = false;
    bool exceed_size = false;

    if (high_watermark_items_ && (buffered_items_ + 1) > static_cast<Count>(*high_watermark_items_))
      exceed_cnt = true;
    // PRECONDITION: buffered_size_ + sz does not overflow SizeT.
    // (SizeT is typically uint64_t and sizes/watermarks are configured far below 2^64.)
    if (high_watermark_size_ && (buffered_size_ + sz) > *high_watermark_size_) exceed_size = true;

    if (!exceed_cnt && !exceed_size) return true;

    const Tick now = now_ticks_();
    const bool allowed_before = dequeue_allowed_(now, buffered_items_, buffered_size_);

    const Count after_cnt = buffered_items_ + 1;
    const SizeT after_size = buffered_size_ + sz;
    const Tick oldest_after = q_.empty() ? now : q_.front().enq_at;

    const bool triggers_after = flush_triggers_met_(now, after_cnt, after_size, oldest_after);
    const bool max_after = high_watermark_reached_(after_cnt, after_size);
    const bool allowed_after = closed_ || triggers_after || max_after;

    if ((!allowed_before) && allowed_after) return true;  // nudge
    return false;
  }

  std::optional<Tick> compute_wait_for_dequeue_(Tick now, std::optional<Tick> deadline) const {
    bool have = false;
    Tick best = 0;

    auto consider = [&](Tick d) {
      if (!have || d < best) {
        best = d;
        have = true;
      }
    };

    if (deadline) consider(*deadline - now);
    if (max_linger_ns_ && !q_.empty()) {
      consider(add_ticks_sat_(q_.front().enq_at, *max_linger_ns_) - now);
    }

    if (!have) return std::nullopt;
    if (best < 0) return Tick{0};
    return best;
  }

  struct WaiterGuard {
    int& c;
    explicit WaiterGuard(int& c_) : c(c_) { ++c; }
    ~WaiterGuard() { --c; }
  };

  struct ProducerSizeGuard {
    std::multiset<SizeT>& sizes;
    typename std::multiset<SizeT>::iterator it;

    ProducerSizeGuard(std::multiset<SizeT>& s, const SizeT& val) 
        : sizes(s) {
        it = sizes.insert(val);
    }

    ~ProducerSizeGuard() {
        sizes.erase(it);
    }

    // Prevent copying or moving to avoid double-erasure
    ProducerSizeGuard(const ProducerSizeGuard&) = delete;
    ProducerSizeGuard& operator=(const ProducerSizeGuard&) = delete;
    ProducerSizeGuard(ProducerSizeGuard&&) = delete;
    ProducerSizeGuard& operator=(ProducerSizeGuard&&) = delete;

  };

  template <bool Enable = CheckMinSizeProducers, std::enable_if_t<!Enable, int> = 0>
  void wait_can_enqueue_(std::unique_lock<std::mutex>& lk, std::optional<Tick> dur_ns) {
    WaiterGuard g(waiting_enqueuers_);
    if (dur_ns) cv_can_enqueue_.wait_for(lk, to_ns_wait_(*dur_ns));
    else cv_can_enqueue_.wait(lk);
  }

  template <bool Enable = CheckMinSizeProducers, std::enable_if_t<Enable, int> = 0>
  void wait_can_enqueue_(std::unique_lock<std::mutex>& lk,
                      std::optional<Tick> dur_ns,
                      const SizeT& attempted_size) {
    // 1. Increment waiting counter (using your existing RAII WaiterGuard)
    WaiterGuard g(waiting_enqueuers_);

    // 2. Insert into multiset via RAII
    ProducerSizeGuard size_guard(producer_wait_sizes_().sizes, attempted_size);

    // 3. Wait
    if (dur_ns) {
        cv_can_enqueue_.wait_for(lk, to_ns_wait_(*dur_ns));
    } else {
        cv_can_enqueue_.wait(lk);
    }  
    // 4. size_guard goes out of scope and erases 'it' automatically
  }

  void wait_can_dequeue_(std::unique_lock<std::mutex>& lk, std::optional<Tick> dur_ns) {
    WaiterGuard g(waiting_dequeuers_);
    if (dur_ns) cv_can_dequeue_.wait_for(lk, to_ns_wait_(*dur_ns));
    else cv_can_dequeue_.wait(lk);
  }

 private:
  std::optional<int> min_batch_items_;
  std::optional<SizeT> min_batch_size_;
  std::optional<Tick> max_linger_ns_;

  std::optional<int> max_batch_items_;
  std::optional<SizeT> max_batch_size_;

  std::optional<int> high_watermark_items_;
  std::optional<SizeT> high_watermark_size_;

  std::string name_;
  WarnFn warn_;

  mutable std::mutex mu_;
  std::condition_variable cv_can_enqueue_;
  std::condition_variable cv_can_dequeue_;

  int waiting_enqueuers_ = 0;
  int waiting_dequeuers_ = 0;

  std::deque<Entry> q_;
  Count buffered_items_ = 0;
  SizeT buffered_size_ = SizeT{0};
  bool closed_ = false;
};

}  // namespace dmx_host

#endif  // DMX_HOST_BATCHING_QUEUE_HPP_