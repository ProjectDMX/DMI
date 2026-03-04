// ring/producer.cu — nvcc compilation unit for the producer kernel.
//
// This file exists solely to compile producer.cuh with nvcc so that
// producer_kernel and launch_producer are available as device/host symbols
// linked into the monitoring shared library.
//
// No torch headers are included here; the kernel depends only on CUDA runtime
// and the ring headers.

#include "producer.cuh"
