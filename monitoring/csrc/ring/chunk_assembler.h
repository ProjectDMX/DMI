// ring/chunk_assembler.h — Reassembles multi-chunk tensors from completed D2H
// chunks and delivers complete tensors to an output callback.
//
// Chunks for the same logical_task_id arrive in chunk_idx order (drain is
// sequential).  When the chunk with TASK_FLAG_IS_LAST is received, all chunks
// for that tensor are concatenated into a single pageable host buffer and
// delivered via the TensorCallback.
//
// Drop markers (TASK_FLAG_IS_DROP) are forwarded immediately as complete
// tensors with is_drop=true and empty data.
//
// Thread safety: ChunkAssembler is not thread-safe.  Call push() only from
// the drain thread.

#pragma once
#include "drain_thread.h"
#include "pinned_pool.h"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace ring {

struct AssembledTensor {
    std::vector<uint8_t> data;  // pageable host copy; empty for drops
    uint64_t logical_task_id;
    uint64_t tensor_total_bytes;
    uint32_t hook_type;
    uint32_t hook_id;
    bool     is_drop;
};

using TensorCallback = std::function<void(AssembledTensor&&)>;

class ChunkAssembler {
public:
    ChunkAssembler(PinnedPool& pool, TensorCallback cb)
        : pool_(pool), cb_(std::move(cb)) {}

    // Called by the drain thread for each completed DrainedChunk.
    void push(DrainedChunk&& chunk);

private:
    struct Pending {
        std::vector<std::vector<uint8_t>> chunks;  // in arrival (chunk_idx) order
        uint64_t tensor_total_bytes = 0;
        uint32_t hook_type          = 0;
        uint32_t hook_id            = 0;
    };

    PinnedPool&    pool_;
    TensorCallback cb_;
    std::unordered_map<uint64_t, Pending> pending_;
};

}  // namespace ring
