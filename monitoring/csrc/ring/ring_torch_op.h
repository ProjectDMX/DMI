#pragma once
#include "ring_engine_py.h"

// Called from Python activate()/deactivate() to register the engine pointer.
// Only accessed during CUDA graph CAPTURE (when the C++ impl body runs).
// During graph REPLAY, only the captured cudaLaunchKernel args are re-used --
// this pointer is never read.
void ring_set_active_engine(ring_py::RingEnginePy* e);
void ring_diag_reset_host_counters();
void ring_diag_print_host_counters();
