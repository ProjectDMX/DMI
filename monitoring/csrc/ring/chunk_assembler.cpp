// ring/chunk_assembler.cpp
#include "chunk_assembler.h"
#include "task_entry.h"   // TASK_FLAG_IS_FIRST, TASK_FLAG_IS_LAST, TASK_FLAG_IS_DROP

#include <cstring>
#include <stdexcept>

namespace ring {

void ChunkAssembler::push(DrainedChunk&& chunk) {
    // --- Drop marker: no data, forward immediately. ---
    if (chunk.flags & TASK_FLAG_IS_DROP) {
        if (chunk.alloc_bytes > 0) pool_.free(chunk.alloc_bytes);
        AssembledTensor t{};
        t.logical_task_id   = chunk.logical_task_id;
        t.tensor_total_bytes = chunk.tensor_total_bytes;
        t.hook_type          = chunk.hook_type;
        t.hook_id            = chunk.hook_id;
        t.is_drop            = true;
        cb_(std::move(t));
        return;
    }

    Pending& p = pending_[chunk.logical_task_id];

    if (chunk.flags & TASK_FLAG_IS_FIRST) {
        p.tensor_total_bytes = chunk.tensor_total_bytes;
        p.hook_type          = chunk.hook_type;
        p.hook_id            = chunk.hook_id;
    }

    // Copy pinned → pageable, then free pinned ring space immediately.
    std::vector<uint8_t> buf(chunk.len);
    if (chunk.len > 0) {
        std::memcpy(buf.data(), chunk.pinned, chunk.len);
    }
    if (chunk.alloc_bytes > 0) pool_.free(chunk.alloc_bytes);

    p.chunks.push_back(std::move(buf));

    if (chunk.flags & TASK_FLAG_IS_LAST) {
        AssembledTensor t;
        t.logical_task_id    = chunk.logical_task_id;
        t.tensor_total_bytes = p.tensor_total_bytes;
        t.hook_type          = p.hook_type;
        t.hook_id            = p.hook_id;
        t.is_drop            = false;

        // Concatenate all chunks in order into a single pageable buffer.
        t.data.resize(p.tensor_total_bytes);
        uint64_t off = 0;
        for (auto& c : p.chunks) {
            std::memcpy(t.data.data() + off, c.data(), c.size());
            off += c.size();
        }

        pending_.erase(chunk.logical_task_id);
        cb_(std::move(t));
    }
}

}  // namespace ring
