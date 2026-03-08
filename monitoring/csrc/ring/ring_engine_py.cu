// ring/ring_engine_py.cu — Pimpl implementation of RingEnginePy.
// Compiled with nvcc so it can instantiate ring::RingEngine (needs CUDA).
// Includes ATen for tensor reconstruction and slicing (GIL-free).

#include "ring_engine_py.h"
#include "ring/ring_engine.h"
#include "ring/drain_thread.h"
#include "ring/ring_state.h"
#include "ring/tensor_meta.h"

#include <ATen/ATen.h>
#include <cstring>

// Forward-declare symbols from producer.cu so we don't pull in producer.cuh
// (which defines producer_kernel as a __global__ function, causing
// duplicate-symbol errors when both producer.cu and this TU are linked into
// the same .so).
namespace ring {
void launch_producer_with_notify(
    const RingState& ring, const uint8_t* d_src, uint64_t src_bytes,
    uint64_t logical_task_id, uint32_t hook_type, uint32_t hook_id,
    cudaHostFn_t notify_fn, void* notify_arg, cudaStream_t stream);
void set_ring_null_mode(bool enabled);
}  // namespace ring

namespace ring_py {

// ---------------------------------------------------------------------------
// ATen helpers (no GIL required for CPU tensors)
// ---------------------------------------------------------------------------

static std::pair<int32_t, std::string> parse_internal_id(const std::string& act_name) {
    auto dot = act_name.find('.');
    if (dot == std::string::npos) return {-1, act_name};
    std::string prefix = act_name.substr(0, dot);
    if (prefix != "blocks" && prefix != "layers") return {-1, act_name};
    auto dot2 = act_name.find('.', dot + 1);
    if (dot2 == std::string::npos) return {-1, act_name};
    try {
        int32_t layer = std::stoi(act_name.substr(dot + 1, dot2 - dot - 1));
        return {layer, prefix + "." + act_name.substr(dot2 + 1)};
    } catch (...) {
        return {-1, act_name};
    }
}

static bool is_attn_hook(const std::string& act_name) {
    auto ends_with = [&](const std::string& suffix) {
        return act_name.size() >= suffix.size() &&
               act_name.compare(act_name.size() - suffix.size(),
                                suffix.size(), suffix) == 0;
    };
    return ends_with("attn.hook_attn_scores") || ends_with("attn.hook_pattern");
}

// Slice tensor for a single request.  Returns an undefined tensor if
// start_token >= end_token.  The returned tensor is always contiguous and
// owns its storage (safe to submit to a background queue).
static at::Tensor slice_for_request(
    const at::Tensor& tensor,
    int    batch_idx,
    int32_t start_token,
    int32_t end_token,
    bool    is_attn)
{
    if (start_token >= end_token) return at::Tensor{};
    int32_t eff = end_token - start_token;

    at::Tensor s = tensor.select(0, batch_idx);

    // Token-dimension narrowing (strip left padding; "always suffix" convention)
    int token_dim = (is_attn && s.dim() >= 2) ? (s.dim() - 2) : 0;
    if (s.dim() > token_dim) {
        int64_t delta = s.size(token_dim);
        if (delta > eff) {
            int64_t skip = delta - eff;
            s = s.narrow(token_dim, skip, eff);
        }
    }

    // Attention key-dimension narrowing
    if (is_attn && s.dim() >= 2) {
        int key_dim = s.dim() - 1;
        int64_t want_k = static_cast<int64_t>(end_token);
        if (want_k >= 0 && s.size(key_dim) > want_k) {
            int64_t skip = s.size(key_dim) - want_k;
            s = s.narrow(key_dim, skip, want_k);
        }
    }

    // Always return an owning, contiguous tensor so the caller can safely
    // move it into a background queue after on_tensor returns.
    return s.contiguous().clone();
}

// ---------------------------------------------------------------------------
struct RingEnginePy::Impl {
    ring::RingEngine engine;
    TensorMetaFifo   fifo;
    SubmitFn         submit_fn;

    Impl(ring::RingConfig cfg, SubmitFn sf)
        : engine(std::move(cfg),
                 [this](ring::AssembledTensor&& t) { on_tensor(std::move(t)); })
        , submit_fn(std::move(sf))
    {}

    ~Impl() = default;

    void on_tensor(ring::AssembledTensor&& t) {
        // Always pop so FIFO stays in sync even for drops.
        TensorMeta meta;
        if (!fifo.pop(meta)) return;

        if (t.is_drop || t.data.empty() || meta.shape.empty()) return;
        if (!submit_fn) return;

        // Reconstruct CPU tensor from pageable assembled data.
        // Use at::empty + memcpy so the tensor owns its storage; slices
        // (views of this tensor) remain valid after the loop via ref-counting.
        at::Tensor tensor = at::empty(
            std::vector<int64_t>(meta.shape.begin(), meta.shape.end()),
            at::TensorOptions().dtype(static_cast<at::ScalarType>(meta.dtype)));
        // Guard against shape/dtype mismatch that would overflow the allocation.
        if (static_cast<size_t>(tensor.nbytes()) != t.data.size()) return;
        std::memcpy(tensor.data_ptr(), t.data.data(), t.data.size());

        bool is_attn = is_attn_hook(meta.hook_name);
        auto [layer_no, act_name] = parse_internal_id(meta.hook_name);

        for (size_t j = 0; j < meta.requests.size(); ++j) {
            if (static_cast<int64_t>(j) >= tensor.size(0)) break;
            const RequestMeta& req = meta.requests[j];
            at::Tensor slice = slice_for_request(
                tensor, static_cast<int>(j),
                req.start_token, req.end_token, is_attn);
            if (!slice.defined()) continue;
            try {
                submit_fn(meta.model_id, meta.shard_rank,
                          req.req_id, act_name, layer_no,
                          req.start_token, req.end_token,
                          std::move(slice));
            } catch (...) {
                // Swallow: cannot propagate from callback thread.
            }
        }
    }
};

// ---------------------------------------------------------------------------
static ring::RingConfig convert(const RingConfig& c) {
    ring::RingConfig r{};
    r.task_ring_entries           = c.task_ring_entries;
    r.payload_ring_bytes          = c.payload_ring_bytes;
    r.chunk_bytes                 = c.chunk_bytes;
    r.pinned_pool_bytes           = c.pinned_pool_bytes;
    r.wait_policy                 = static_cast<ring::WaitPolicy>(c.wait_policy);
    r.no_progress_timeout_cycles  = c.no_progress_timeout_cycles;
    r.drop_reporting              = static_cast<ring::DropReporting>(c.drop_reporting);
    return r;
}

// ---------------------------------------------------------------------------
RingEnginePy::RingEnginePy(RingConfig cfg, SubmitFn submit_fn) {
    impl_ = std::make_unique<Impl>(convert(cfg), std::move(submit_fn));
}

RingEnginePy::~RingEnginePy() = default;

void RingEnginePy::init(uint64_t stream_handle) {
    impl_->engine.init(reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::start() { impl_->engine.start(); }
void RingEnginePy::stop()  { impl_->engine.stop();  }

void RingEnginePy::flush(uint64_t stream_handle) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    // Phase 1: wait for all producer kernels on this stream (and their
    // cudaLaunchHostFunc callbacks) to complete.  After this, task_head
    // reflects all tasks that will ever be submitted from this stream.
    cudaStreamSynchronize(stream);

    // Phase 2: read the drain target.
    const uint64_t target = __atomic_load_n(
        impl_->engine.ring_state().task_head, __ATOMIC_ACQUIRE);

    if (target == 0) return;  // nothing was ever submitted

    // Wake the drain thread in case it is sleeping (it may not have seen the
    // last hostfunc notification yet due to scheduling).
    impl_->engine.drain_thread().notify();

    // Phase 3: wait for all D2H copies + assembler calls up to `target`.
    impl_->engine.drain_thread().wait_until_completed(target);

    // Phase 4: wait for the callback thread to finish all on_tensor calls
    // that were enqueued as a result of the drained chunks.
    impl_->engine.wait_callbacks_empty();
}

void RingEnginePy::set_null_mode(bool enabled) {
    ring::set_ring_null_mode(enabled);
}

void RingEnginePy::push_meta(TensorMeta meta) {
    impl_->fifo.push(std::move(meta));
}

void RingEnginePy::pop_last_meta() {
    TensorMeta discard;
    impl_->fifo.pop_last(discard);
}

void RingEnginePy::hook(uint64_t d_ptr, uint64_t nbytes,
                        uint64_t logical_task_id,
                        uint32_t hook_type, uint32_t hook_id,
                        uint64_t stream_handle)
{
    launch_producer_with_notify(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes, logical_task_id, hook_type, hook_id,
        ring::DrainThread::hostfunc_cb,
        &impl_->engine.drain_thread(),
        reinterpret_cast<cudaStream_t>(stream_handle));
}

}  // namespace ring_py
