// ring/task_entry.h — TaskEntry layout for the GPU task/control ring.
//
// This header is plain C++ (no CUDA required) so it can be included from both
// host-only and device-compiled translation units.
//
// One tensor = one task entry.  No chunking.
// Large tensor bypass: device_src_ptr != nullptr.
// Hook identity is derived from forward-pass order via the TensorMetaFifo.
// Padded allocation size is derived: align_up(tensor_total_bytes, 16).

#pragma once
#include <cstddef>
#include <cstdint>

namespace ring {

// ---------------------------------------------------------------------------
// Sentinel value for ready_seq — indicates slot has not been published yet.
//
// Publish protocol:
//   producer: write all data fields → __threadfence() → write ready_seq = seq_no
//   consumer: poll until __atomic_load_n(ready_seq) == expected → read fields
// ---------------------------------------------------------------------------
static constexpr uint64_t READY_SEQ_SENTINEL = ~uint64_t(0);

// ---------------------------------------------------------------------------
// TaskEntry — one slot in the fixed-size task/control ring.
//
// alignas(64) = one CPU cache line.  Entries are in CPU-preferred managed
// memory so the drain thread polls ready_seq via fast local DRAM reads.
// GPU producer writes via PCIe posted writes (fire-and-forget).
//
// Large tensor bypass: device_src_ptr != nullptr → payload fields are zero,
// drain thread issues D2H directly from device_src_ptr.
// ---------------------------------------------------------------------------
struct alignas(64) TaskEntry {
    // -- sequence guard (written LAST by producer, read FIRST by consumer) --
    uint64_t ready_seq;                //  8 B  offset   0

    // -- tensor metadata --
    uint64_t tensor_total_bytes;       //  8 B  offset   8

    // -- two-span payload descriptor (zero for large tensor bypass) --
    uint64_t payload_off1;             //  8 B  offset  16
    uint64_t payload_len1;             //  8 B  offset  24
    uint64_t payload_off2;             //  8 B  offset  32
    uint64_t payload_len2;             //  8 B  offset  40

    // -- large tensor bypass: device source pointer (null for normal) --
    const uint8_t* device_src_ptr;     //  8 B  offset  48

    // -- padding --
    uint8_t  _padding[8];              //  8 B  offset  56
                                       //       total   64 B
};

static_assert(sizeof(TaskEntry)  == 64,
    "TaskEntry must be exactly 64 bytes; adjust _padding if fields change");
static_assert(alignof(TaskEntry) == 64,
    "TaskEntry must be 64-byte aligned (one CPU cache line)");
static_assert(offsetof(TaskEntry, ready_seq)          ==  0);
static_assert(offsetof(TaskEntry, tensor_total_bytes) ==  8);
static_assert(offsetof(TaskEntry, payload_off1)       == 16);
static_assert(offsetof(TaskEntry, payload_len1)       == 24);
static_assert(offsetof(TaskEntry, payload_off2)       == 32);
static_assert(offsetof(TaskEntry, payload_len2)       == 40);
static_assert(offsetof(TaskEntry, device_src_ptr)     == 48);

}  // namespace ring
