#pragma once
#include "ring_engine_py.h"

// Called from Python activate()/deactivate() to register the engine pointer.
// Only accessed during CUDA graph CAPTURE (when the C++ impl body runs).
// During graph REPLAY, only the captured cudaLaunchKernel args are re-used --
// this pointer is never read.
void ring_set_active_engine(ring_py::RingEnginePy* e);

// Node-toggle capture flag read by the legacy producer ops.  While true,
// each producer launched under stream capture records its kernel node in the
// active engine's toggle registry.  Process-global like the engine pointer;
// cleared whenever the engine pointer is cleared.
void ring_set_toggle_capture(bool enabled);
// Process-wide count of producer kernels the ring producer ops launched.
uint64_t ring_producer_launch_count();
void ring_diag_reset_host_counters();
void ring_diag_print_host_counters();

// The additive ring::record_producer* torch ops use the same explicitly
// activated engine pointer.  They are registered separately from the released
// inference producer ops; no legacy schema is widened or reinterpreted.
