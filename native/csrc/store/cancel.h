// Cancellation for object-store work: a flag set once to stop for good, and
// a deadline armed for one bounded stretch.
//
// The storage service owns two. One cuts its uploads: it goes to the S3
// client and the uploader that do them, and a flush arms it at its
// deadline. The other cuts its index reads and the reconcile's requests:
// it goes to the S3 client those read through, and a flush arms it one
// catalog request timeout past its deadline. stop() cancels both for good,
// so a stalled PUT or GET or a retry backoff cannot hold it. What a cancel
// cuts short is left where it was: a pack whose upload was cancelled stays
// in the spool, which is where a pack waits for an object store; one
// uploaded but not yet indexed, whose read was cut, stays owed to the
// catalog (in memory, or in the bucket for a start's reconcile once stop()
// drops it).
//
// Header-only, so every target that builds the S3 client gets it without a
// source list to keep in step.

#ifndef DMI_STORE_CANCEL_H_
#define DMI_STORE_CANCEL_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace dmi_store {

class Cancellation {
 public:
  Cancellation() = default;
  Cancellation(const Cancellation&) = delete;
  Cancellation& operator=(const Cancellation&) = delete;

  // Cancels everything from now on, until Reset(), and wakes every sleeper.
  void Cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_.store(true, std::memory_order_release);
    ++generation_;
    cv_.notify_all();
  }

  // Clears Cancel() and the deadline.
  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_.store(false, std::memory_order_release);
    deadline_ns_.store(0, std::memory_order_release);
    ++generation_;
    cv_.notify_all();
  }

  // Arms a deadline on the steady clock, in ns (NowNs()), or disarms it
  // with 0: past it, cancelled() is true.
  void set_deadline(uint64_t steady_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    deadline_ns_.store(steady_ns, std::memory_order_release);
    ++generation_;
    cv_.notify_all();
  }

  // Cancel()ed, or past the armed deadline. Cheap: libcurl's progress
  // callback asks it while a transfer runs.
  bool cancelled() const {
    if (cancelled_.load(std::memory_order_acquire)) return true;
    const uint64_t deadline = deadline_ns_.load(std::memory_order_acquire);
    return deadline != 0 && NowNs() >= deadline;
  }

  // Cancel()ed, whatever the deadline.
  bool cancelled_for_good() const {
    return cancelled_.load(std::memory_order_acquire);
  }

  // Sleeps for `duration`, or until cancelled, whichever comes first -- a
  // deadline armed while it sleeps included. True if it slept it out.
  bool SleepFor(std::chrono::nanoseconds duration) const {
    const uint64_t end =
        NowNs() + static_cast<uint64_t>(std::max<int64_t>(0, duration.count()));
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      if (cancelled()) return false;
      const uint64_t now = NowNs();
      if (now >= end) return true;
      uint64_t wake = end;
      const uint64_t deadline = deadline_ns_.load(std::memory_order_acquire);
      if (deadline != 0 && deadline < wake) wake = deadline;
      const uint64_t generation = generation_;
      cv_.wait_for(lock, std::chrono::nanoseconds(wake - now),
                   [&] { return generation_ != generation; });
    }
  }

  static uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::atomic<bool> cancelled_{false};
  std::atomic<uint64_t> deadline_ns_{0};
  uint64_t generation_ = 0;  // guarded by mutex_; moves on every change
};

}  // namespace dmi_store

#endif  // DMI_STORE_CANCEL_H_
