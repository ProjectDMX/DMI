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

  void submit(ring::RecordEnvelope envelope) override;
  bool flush_and_wait(Duration timeout) override;
  void rethrow_if_failed() const override;

  const PackSink& sink() const { return *sink_; }
  const std::string& layout() const { return layout_; }

 protected:
  void on_engine_acquire() override {}
  void on_engine_release() noexcept override {}

 private:
  std::unique_ptr<PackSink> sink_;
  const std::string layout_;
};

}  // namespace dmi_sink

#endif  // DMI_SINK_NATIVE_PACK_SINK_H_
