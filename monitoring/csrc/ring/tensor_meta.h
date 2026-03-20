// ring/tensor_meta.h -- Tensor metadata FIFO for the ring transport.
//
// Per-step StepContext (heap-allocated, owned by p2p after pop) holds shared
// data (model_id, shard_rank, requests).  TensorMeta holds per-hook data
// (hook_name, shape, dtype).  last_in_step signals p2p to free the context.
//
// No ATen dependency -- safe to include from both g++ and nvcc translation units.

#pragma once
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace ring_py {

struct RequestMeta {
    std::string req_id;
    int32_t     start_token = 0;
    int32_t     end_token   = 0;
};

// Per-step shared context.  Heap-allocated by push_step caller.
// Ownership transfers to p2p thread via pop_context; p2p deletes it.
struct StepContext {
    std::string              model_id;
    int32_t                  shard_rank = 0;
    std::vector<RequestMeta> requests;
};

// Per-hook metadata.
struct TensorMeta {
    std::string              hook_name;
    std::vector<int64_t>     shape;
    int                      dtype        = 0;  // at::ScalarType integer value
    bool                     last_in_step = false;
};

// Thread-safe dual FIFO for TensorMeta + StepContext.
class TensorMetaFifo {
public:
    // Push context + all hook metas in one lock.
    // ctx is heap-allocated by caller; ownership transfers here.
    void push_step(StepContext* ctx, std::vector<TensorMeta>& metas) {
        std::lock_guard<std::mutex> lk(mu_);
        ctx_q_.push_back(ctx);
        for (auto& m : metas)
            q_.push_back(std::move(m));
    }

    bool pop(TensorMeta& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // Pop front step context.  Caller takes ownership (must delete).
    StepContext* pop_context() {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx_q_.empty()) return nullptr;
        StepContext* p = ctx_q_.front();
        ctx_q_.pop_front();
        return p;
    }

private:
    std::mutex                mu_;
    std::deque<TensorMeta>    q_;
    std::deque<StepContext*>   ctx_q_;
};

}  // namespace ring_py
