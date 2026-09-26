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
  // Takes ownership of a started-or-unstarted PackSink; Start()s it here so
  // construction failure (bad spool dir) throws instead of wedging attach.
  NativePackSink(std::unique_ptr<PackSink> sink, std::string layout);
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
  const std::string& layout() const { return layout_; }

 protected:
  void on_engine_acquire() override {}
  void on_engine_release() noexcept override {}

 private:
  // "NativePackSink: pipeline reported ..." for the loss counters that
  // moved since baseline_; empty when none did.
  std::string LossError() const;

  std::unique_ptr<PackSink> sink_;
  const std::string layout_;
  // Counters at construction. Losses are judged against it, as the
  // reference adapter judges them against its pipeline's baseline.
  SinkSnapshot baseline_;
};

}  // namespace dmi_sink

#endif  // DMI_SINK_NATIVE_PACK_SINK_H_
