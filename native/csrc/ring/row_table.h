// Producer row tables: the segment-count clamp shared by the gather, head and
// fused producers.  Plain C++ (no CUDA dependency) so host code and CPU unit
// tests use the exact device arithmetic.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define DMX_ROW_TABLE_HD __host__ __device__
#else
#define DMX_ROW_TABLE_HD
#endif

namespace ring {

// Upper bound on the number of row segments in a gather table
// (shared-memory sizing).  Segments are merged runs of selected rows, so the
// count is bounded by the number of concurrently scheduled requests; a host
// with more must fall back to a full-row table (over-capture is safe).
static constexpr uint32_t GATHER_MAX_SEGS = 256;

// Segments a table of `table_numel` int64 words can physically hold: word 0
// is the count, segment k occupies words 1 + 2k and 2 + 2k.
DMX_ROW_TABLE_HD inline int64_t row_table_capacity_segs(int64_t table_numel) {
    return table_numel > 0 ? (table_numel - 1) / 2 : 0;
}

// The number of segments a producer reads from a table whose first word is
// `n_raw`: a negative count reads none, and the count is bounded by both the
// shared-memory limit and the table's own storage, so the segment reads
// table[1 + 2k] and table[2 + 2k] stay below table_numel for every k read.
// The count stays a device-side value: callers pass the table's element
// count (known on the host at launch/capture time), never a host read of
// table[0], so graph replays still re-read the count.
DMX_ROW_TABLE_HD inline uint32_t row_table_segment_count(int64_t n_raw,
                                                         int64_t table_numel) {
    if (n_raw < 0) n_raw = 0;
    if (n_raw > static_cast<int64_t>(GATHER_MAX_SEGS))
        n_raw = static_cast<int64_t>(GATHER_MAX_SEGS);
    const int64_t capacity = row_table_capacity_segs(table_numel);
    if (n_raw > capacity) n_raw = capacity;
    return static_cast<uint32_t>(n_raw);
}

}  // namespace ring

#undef DMX_ROW_TABLE_HD
