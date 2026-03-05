// ring/tensor_meta.h — Tensor metadata FIFO for the ring transport.
//
// Pushed by the Python forward-pass thread (one entry per tensor offloaded),
// popped by the C++ callback thread (drain side) in arrival order.
//
// No ATen dependency — safe to include from both g++ and nvcc translation units.

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

struct TensorMeta {
    std::string              hook_name;
    std::string              model_id;
    int32_t                  shard_rank = 0;
    std::vector<int64_t>     shape;
    int                      dtype      = 0;  // at::ScalarType integer value
    std::vector<RequestMeta> requests;
};

// Thread-safe FIFO for TensorMeta.
// push() is called from the Python forward-pass thread.
// pop() is called from the C++ callback thread.
class TensorMetaFifo {
public:
    void push(TensorMeta meta) {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(std::move(meta));
    }

    // Pop the front entry. Returns false if the queue is empty.
    bool pop(TensorMeta& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // Pop the most-recently pushed entry (used to undo a push when hook() fails).
    bool pop_last(TensorMeta& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return false;
        out = std::move(q_.back());
        q_.pop_back();
        return true;
    }

private:
    std::mutex              mu_;
    std::deque<TensorMeta>  q_;
};

}  // namespace ring_py
