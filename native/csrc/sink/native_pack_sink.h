// Native capture sink: a ring::RecordSink that packs envelopes directly.
//
// The reference path (record_adapter.py + ReferencePythonCaptureSink) crosses
// the GIL once per envelope and copies every payload through Python. This
// sink implements the same wire contract — layout "capture_pack_reference_v1"
// rows of (metadata JSON, payload slice) — entirely in C++: envelopes arrive
// on the ring's record worker, rows validate into RowInputs, and PackSink
// admits them. No Python runs on the capture path.

#ifndef DMI_SINK_NATIVE_PACK_SINK_H_
#define DMI_SINK_NATIVE_PACK_SINK_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "../pack/pack_builder.h"
#include "pack_sink.h"
#include "record_row.h"
#include "ring/record_sink.h"

namespace dmi_sink {

// ATen scalar type value -> capture dtype name. Empty when unsupported.
std::string AtenDtypeName(int32_t scalar_type);

class NativePackSink final : public ring::RecordSink {
 public:
  // How long the release backstop (on_engine_release) waits for the open
  // pack to reach the spool, unless the constructor is given another bound.
  static constexpr std::chrono::milliseconds kDefaultReleaseFlushTimeout{
      30'000};

  // Takes ownership of a started-or-unstarted PackSink; Start()s it here so
  // construction failure (bad spool dir) throws instead of wedging attach.
  // release_flush_timeout bounds the release backstop; zero turns it off.
  NativePackSink(std::unique_ptr<PackSink> sink, std::string layout,
                 Duration release_flush_timeout = kDefaultReleaseFlushTimeout);
  ~NativePackSink() override;

  NativePackSink(const NativePackSink&) = delete;
  NativePackSink& operator=(const NativePackSink&) = delete;

  // Admits the envelope's rows against one admission deadline
  // (EnvelopeAdmission). Refuses it first if the pipeline has latched a
  // failure or lost an admitted record (see rethrow_if_failed).
  void submit(ring::RecordEnvelope envelope) override;
  // Also fails, after a completed flush, when a record admitted since
  // construction was lost instead of persisted -- the reference adapter's
  // loss check (record_adapter.py _LOSS_COUNTERS) -- or when fewer records
  // were persisted than admitted.
  bool flush_and_wait(Duration timeout) override;
  // Throws on a latched pipeline failure, and on any loss counter
  // (dropped, timed out, oversized, duplicate, rejected-closed, failures)
  // that has moved since construction.
  void rethrow_if_failed() const override;
  // kDropNewest: zero (a full queue refuses at once). kBlock: the
  // admission timeout, now one per envelope; none when it waits forever.
  std::optional<Duration> admission_bound() const override;

  const PackSink& sink() const { return *sink_; }
  // Test seam: the PackSink itself, so a test can park its stages
  // (PackSink::SpoolForTesting) and wedge the pipeline.
  PackSink& sink_for_testing() { return *sink_; }
  const std::string& layout() const { return layout_; }
  Duration release_flush_timeout() const { return release_flush_timeout_; }
  // Whether the sink can no longer write its spool: its engine released it
  // and the release backstop's flush went through. A released sink admits
  // nothing, so once that flush has persisted everything admitted, no stage
  // is left to come -- not from its stagers, its linger, or its destructor.
  // False while attached (and from a new engine's acquire on), after a
  // release whose flush failed or timed out, and when the backstop is off
  // (release_flush_timeout zero): a stage may then still be on its way.
  // The engine lets go of the spool directory's owner lock only on true.
  bool sealed_on_release() const {
    return sealed_on_release_.load(std::memory_order_acquire);
  }

 protected:
  void on_engine_acquire() override {
    sealed_on_release_.store(false, std::memory_order_release);
  }
  // The backstop for a ring that stops without a flush: RingEngine::stop
  // drains its record worker into submit() and then releases the sink, and
  // until a flush seals it the open pack is only in memory -- for up to
  // max_linger_ns, or until this object dies. So the release flushes,
  // bounded by release_flush_timeout -- a flush_and_wait in flight on
  // another thread included, which it waits for only that long -- and the
  // open pack reaches the spool as one .ready file. It cannot throw (it
  // runs in RingEngine::stop and its destructor): a flush that fails or
  // times out writes one line to stderr, and that line is the only report
  // of a timeout. A pipeline failure also counts in snapshot()["failures"];
  // rethrow_if_failed reports it only once the sink is attached again,
  // since a released sink refuses the call as not attached. Whether the
  // flush went through is sealed_on_release().
  void on_engine_release() noexcept override;

 private:
  // "NativePackSink: pipeline reported ..." for the loss counters that
  // moved since baseline_; empty when none did.
  std::string LossError() const;

  std::unique_ptr<PackSink> sink_;
  const std::string layout_;
  const Duration release_flush_timeout_;
  std::atomic<bool> sealed_on_release_{false};
  // Counters at construction. Losses are judged against it, as the
  // reference adapter judges them against its pipeline's baseline.
  SinkSnapshot baseline_;
};

}  // namespace dmi_sink

#endif  // DMI_SINK_NATIVE_PACK_SINK_H_
