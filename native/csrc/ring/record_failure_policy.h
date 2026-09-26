// What a latched record failure does to the rest of a record runtime.
//
// Plain C++ (no ATen/CUDA) so ring_engine_py.h can name it.

#pragma once

namespace ring {

// kRaiseAtProducer: the failure reaches the forward that publishes the next
// record.  kDisableCapture: capture stops and the forward keeps running --
// descriptors pushed and payloads delivered after the latch are discarded
// and counted.  Under both, the failure still surfaces at every checked
// completion (flush_records_and_wait) and in the capture status.
enum class RecordFailurePolicy {
    kRaiseAtProducer,
    kDisableCapture,
};

}  // namespace ring
