// ring/task_entry.h — TaskEntry layout for the GPU task/control ring.
//
// This header is plain C++ (no CUDA required) so it can be included from both
// host-only and device-compiled translation units.

#pragma once
#include <cstddef>
#include <cstdint>

namespace ring {

// ---------------------------------------------------------------------------
// Flag bits — TaskEntry::flags
// ---------------------------------------------------------------------------
enum : uint32_t {
    TASK_FLAG_IS_FIRST = 1u << 0,  // first chunk of a logical task
    TASK_FLAG_IS_LAST  = 1u << 1,  // last chunk (may also be FIRST on single-chunk tasks)
    TASK_FLAG_IS_DROP  = 1u << 2,  // drop marker: no payload, consumer should discard
};

// ---------------------------------------------------------------------------
// Reason codes for DROP entries (TaskEntry::reason)
// ---------------------------------------------------------------------------
enum : uint32_t {
    DROP_REASON_NONE                = 0,
    DROP_REASON_TIMEOUT_NO_PROGRESS = 1,  // producer timed out waiting for consumer progress
};

// ---------------------------------------------------------------------------
// Sentinel value for ready_seq — indicates slot has not been published yet.
//
// Publish protocol:
//   producer: write all data fields → __threadfence() → write ready_seq = seq_no
//   consumer: spin until volatile_read(ready_seq) == expected_seq_no → __threadfence()
//
// NOTE: seq_no is a monotonically increasing 64-bit counter. Collision with
// SENTINEL (2^64-1) would require ~5.8×10^11 years at 1 billion ops/sec.
// ---------------------------------------------------------------------------
static constexpr uint64_t READY_SEQ_SENTINEL = ~uint64_t(0);

// ---------------------------------------------------------------------------
// TaskEntry — one slot in the fixed-size task/control ring.
//
// alignas(128) places each entry on its own cache-line pair, which prevents
// false sharing between adjacent slots and keeps the ready_seq field isolated.
//
// Field order: heavy 64-bit fields first, then 32-bit fields, then padding.
// ready_seq is the only ordering signal; all other fields are data.
// ---------------------------------------------------------------------------
struct alignas(128) TaskEntry {
    // -- sequence guard (written LAST by producer, read FIRST by consumer) --
    uint64_t ready_seq;           //  8 B  offset   0

    // -- identity --
    uint64_t seq_no;              //  8 B  offset   8  slot index at publish time
    uint64_t logical_task_id;     //  8 B  offset  16  packed {hook_id, seq_no, tensor_idx}

    // -- chunking --
    uint64_t chunk_offset_bytes;  //  8 B  offset  24  byte offset of chunk within logical tensor
    uint64_t tensor_total_bytes;  //  8 B  offset  32  total logical tensor bytes (valid on IS_FIRST)

    // -- two-span payload descriptor (off2/len2 == 0 means single span) --
    uint64_t payload_off1;        //  8 B  offset  40
    uint64_t payload_len1;        //  8 B  offset  48
    uint64_t payload_off2;        //  8 B  offset  56
    uint64_t payload_len2;        //  8 B  offset  64

    // -- 32-bit fields --
    uint32_t chunk_idx;           //  4 B  offset  72
    uint32_t hook_type;           //  4 B  offset  76
    uint32_t hook_id;             //  4 B  offset  80
    uint32_t flags;               //  4 B  offset  84  TASK_FLAG_*
    uint32_t reason;              //  4 B  offset  88  DROP_REASON_* (drop entries only)
    uint32_t _pad0;               //  4 B  offset  92

    // -- explicit padding to reach 128 bytes --
    uint8_t  _padding[32];        // 32 B  offset  96
                                  //       total  128 B
};

static_assert(sizeof(TaskEntry)  == 128,
    "TaskEntry must be exactly 128 bytes; adjust _padding if fields change");
static_assert(alignof(TaskEntry) == 128,
    "TaskEntry must be 128-byte aligned");
static_assert(offsetof(TaskEntry, ready_seq)          ==  0);
static_assert(offsetof(TaskEntry, seq_no)             ==  8);
static_assert(offsetof(TaskEntry, logical_task_id)    == 16);
static_assert(offsetof(TaskEntry, chunk_offset_bytes) == 24);
static_assert(offsetof(TaskEntry, tensor_total_bytes) == 32);
static_assert(offsetof(TaskEntry, payload_off1)       == 40);
static_assert(offsetof(TaskEntry, payload_len1)       == 48);
static_assert(offsetof(TaskEntry, payload_off2)       == 56);
static_assert(offsetof(TaskEntry, payload_len2)       == 64);
static_assert(offsetof(TaskEntry, chunk_idx)          == 72);
static_assert(offsetof(TaskEntry, hook_type)          == 76);
static_assert(offsetof(TaskEntry, hook_id)            == 80);
static_assert(offsetof(TaskEntry, flags)              == 84);
static_assert(offsetof(TaskEntry, reason)             == 88);

}  // namespace ring
