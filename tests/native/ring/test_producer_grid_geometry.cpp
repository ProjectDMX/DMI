// Host-only checks of the producer grid geometry helpers in producer.cuh.
// Built with a plain C++ compiler by tests/test_producer_grid_geometry.py;
// no CUDA toolkit or device is involved.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "ring/producer.cuh"

static int failures = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                           \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

int main() {
    constexpr uint64_t per_block = ring::PRODUCER_BYTES_PER_BLOCK;
    constexpr uint32_t cap = ring::PRODUCER_MAX_BLOCKS;

    EXPECT(ring::producer_grid_blocks(0) == 1);
    EXPECT(ring::producer_grid_blocks(per_block) == 1);
    EXPECT(ring::producer_grid_blocks(per_block + 1) == 2);
    EXPECT(ring::producer_grid_blocks(cap * per_block) == cap);
    EXPECT(ring::producer_grid_blocks(cap * per_block + 1) == cap);
    EXPECT(ring::producer_grid_blocks(UINT64_MAX) == cap);

    // The audit's example: a capped 264-block grid for 64 chunks used to be
    // rounded up to 320; it now rounds down to 256, within the cap.
    EXPECT(ring::producer_chunked_grid_blocks(cap * per_block, 64) == 256);
    EXPECT(ring::producer_chunked_grid_blocks(UINT64_MAX, 64) == 256);
    // Rounding up is kept while it stays within the cap.
    EXPECT(ring::producer_chunked_grid_blocks(70 * per_block, 64) == 128);
    EXPECT(ring::producer_chunked_grid_blocks(200 * per_block, 64) == 256);
    EXPECT(ring::producer_chunked_grid_blocks(10 * per_block, 3) == 12);
    EXPECT(ring::producer_chunked_grid_blocks(UINT64_MAX, 3) == 264);
    EXPECT(ring::producer_chunked_grid_blocks(UINT64_MAX, 5) == 260);
    EXPECT(ring::producer_chunked_grid_blocks(UINT64_MAX, 7) == 259);
    // Small transfers still get one block per chunk.
    EXPECT(ring::producer_chunked_grid_blocks(0, 64) == 64);
    EXPECT(ring::producer_chunked_grid_blocks(per_block, 3) == 3);
    // A single chunk is the unchunked grid.
    EXPECT(ring::producer_chunked_grid_blocks(UINT64_MAX, 1) == cap);
    EXPECT(ring::producer_chunked_grid_blocks(5 * per_block, 1) == 5);
    // Only a chunk count above the cap can exceed it (one block per chunk).
    EXPECT(ring::producer_chunked_grid_blocks(UINT64_MAX, cap) == cap);
    EXPECT(ring::producer_chunked_grid_blocks(UINT64_MAX, cap + 1) == cap + 1);
    EXPECT(ring::producer_chunked_grid_blocks(0, 300) == 300);

    for (uint32_t k = 1; k <= cap; ++k) {
        for (uint64_t bytes : {uint64_t(0), uint64_t(1), per_block,
                               5 * per_block + 1, 63 * per_block,
                               uint64_t(cap) * per_block,
                               uint64_t(cap) * per_block + 1,
                               uint64_t(1) << 40, UINT64_MAX}) {
            const uint32_t raw = ring::producer_grid_blocks(bytes);
            const uint32_t blocks = ring::producer_chunked_grid_blocks(bytes, k);
            EXPECT(blocks % k == 0);
            EXPECT(blocks >= k);
            EXPECT(blocks <= cap);
            // Never more than one partition step away from the proportional
            // grid: the nearest multiple of k at or above it, or the largest
            // multiple at or below the cap.
            if (raw <= k) {
                EXPECT(blocks == k);
            } else {
                const uint32_t up = ((raw - 1) / k + 1) * k;
                EXPECT(blocks == (up <= cap ? up : (cap / k) * k));
            }
        }
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("producer grid geometry: ok\n");
    return 0;
}
