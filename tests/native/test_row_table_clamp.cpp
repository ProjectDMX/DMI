// Row-table segment-count clamp (native/csrc/ring/row_table.h): the gather,
// head and fused producers read table[1 + 2k] and table[2 + 2k] for every
// k < row_table_segment_count(table[0], table_numel).  Those reads must stay
// below table_numel whatever the count word holds, and the count must also
// respect GATHER_MAX_SEGS.  Compiled and run on the CPU by
// tests/test_row_table_clamp.py; plain C++17, no CUDA.
#include "ring/row_table.h"

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

int failures = 0;

void expect(int64_t n_raw, int64_t numel, uint32_t want) {
    const uint32_t got = ring::row_table_segment_count(n_raw, numel);
    if (got != want) {
        std::printf("FAIL count(n_raw=%lld, numel=%lld) = %u, want %u\n",
                    static_cast<long long>(n_raw), static_cast<long long>(numel),
                    got, want);
        ++failures;
    }
    // Every word the loader reads is inside the table.
    for (uint32_t k = 0; k < got; ++k) {
        const int64_t last = 2 + 2 * static_cast<int64_t>(k);
        if (last >= numel) {
            std::printf("FAIL read table[%lld] past numel=%lld (n_raw=%lld)\n",
                        static_cast<long long>(last), static_cast<long long>(numel),
                        static_cast<long long>(n_raw));
            ++failures;
            break;
        }
    }
}

}  // namespace

int main() {
    static_assert(ring::GATHER_MAX_SEGS == 256, "row-table capacity");
    const int64_t big = std::numeric_limits<int64_t>::max();

    // One-word tables hold only the count: no segment is ever read.
    expect(0, 1, 0);
    expect(1, 1, 0);
    expect(256, 1, 0);
    expect(big, 1, 0);
    // Three words hold one segment.
    expect(2, 3, 1);
    expect(1, 3, 1);
    // Even lengths: the trailing odd word is not a whole segment.
    expect(2, 2, 0);
    expect(2, 4, 1);
    expect(3, 6, 2);
    // Odd lengths.
    expect(2, 5, 2);
    expect(3, 7, 3);
    expect(9, 7, 3);
    // Negative counts (including the -1 disable sentinel) read nothing.
    expect(-1, 9, 0);
    expect(-7, 9, 0);
    expect(std::numeric_limits<int64_t>::min(), 9, 0);
    // GATHER_MAX_SEGS limits: a full-width table.
    expect(255, 1 + 2 * 256, 255);
    expect(256, 1 + 2 * 256, 256);
    expect(257, 1 + 2 * 256, 256);
    expect(257, 1 + 2 * 257, 256);
    expect(big, 1 + 2 * 300, 256);
    // One word short of 256 segments.
    expect(256, 2 * 256, 255);
    // Degenerate storage (rejected by the host before launch) reads nothing.
    expect(1, 0, 0);
    expect(1, -3, 0);

    if (ring::row_table_capacity_segs(1) != 0 || ring::row_table_capacity_segs(3) != 1 ||
        ring::row_table_capacity_segs(4) != 1 || ring::row_table_capacity_segs(513) != 256 ||
        ring::row_table_capacity_segs(0) != 0) {
        std::printf("FAIL row_table_capacity_segs\n");
        ++failures;
    }
    if (failures == 0) std::printf("ok\n");
    return failures == 0 ? 0 : 1;
}
