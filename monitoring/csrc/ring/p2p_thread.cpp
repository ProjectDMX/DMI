// ring/p2p_thread.cpp — Pinned-to-pageable thread implementation.
// Compiled with g++ (not nvcc); uses ATen for tensor operations.

#include "p2p_thread.h"
#include "task_entry.h"
#include "pinned_staging.h"

#include <ATen/ATen.h>
#include <cstring>

namespace ring {

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

// Slice tensor for a single request.
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

    return s;
}

// ---------------------------------------------------------------------------
P2PThread::P2PThread(DrainThread& drain, ring_py::TensorMetaFifo& fifo,
                     const RingConfig& cfg, SubmitFn submit_fn)
    : drain_(drain), fifo_(fifo), cfg_(cfg), submit_fn_(std::move(submit_fn))
{}

P2PThread::~P2PThread() noexcept {
    stop();
}

void P2PThread::start() {
    thread_ = std::thread([this] { loop(); });
}

void P2PThread::stop() {
    // drain_.stop() must be called before p2p stop to signal shutdown.
    if (thread_.joinable()) thread_.join();
}

// ---------------------------------------------------------------------------
void P2PThread::loop() {
    while (true) {
        uint64_t n = drain_.wait_for_tasks();
        if (n == 0) break;  // only returns 0 when p2p_stop_requested (after drain joined)

        std::vector<DrainTask> local;
        drain_.pop_tasks(n, local);
        process(local);
    }
}

// ---------------------------------------------------------------------------
void P2PThread::process(std::vector<DrainTask>& tasks) {
    size_t i = 0;
    while (i < tasks.size()) {
        DrainTask& task = tasks[i];

        if (task.flags & TASK_FLAG_IS_DROP) {
            // Drop marker — pop metadata to keep FIFO in sync, then skip
            ring_py::TensorMeta discard;
            fifo_.pop(discard);
            i++;
            continue;
        }

        if (task.flags & TASK_FLAG_LARGE_TENSOR) {
            // Large tensor — already in pageable memory
            at::Tensor tensor = std::move(task.large_tensor);
            do_post_processing(tensor, task);

            // Release bypass budget
            {
                std::lock_guard<std::mutex> lk(drain_.bypass_mu());
                drain_.in_flight_bypass_bytes() -= task.tensor_total_bytes;
            }
            drain_.notify_bypass_freed();
            i++;
            continue;
        }

        // Normal tensor: IS_FIRST must be set
        uint64_t total_bytes = task.tensor_total_bytes;
        const DrainTask& first_task = task;

        // ATen allocate flat pageable tensor
        auto tensor = at::empty({static_cast<int64_t>(total_bytes)},
                                at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        uint8_t* dst = tensor.data_ptr<uint8_t>();

        // Copy all entries for this tensor
        uint64_t staging_release_total = 0;
        while (true) {
            DrainTask& t = tasks[i];

            // Copy from pinned staging (1-2 memcpy for ring wrap)
            if (t.data_len1 > 0) {
                std::memcpy(dst + t.chunk_offset_bytes, t.data_ptr1, t.data_len1);
            }
            if (t.data_len2 > 0) {
                std::memcpy(dst + t.chunk_offset_bytes + t.data_len1,
                            t.data_ptr2, t.data_len2);
            }

            staging_release_total += t.alloc_bytes;
            bool is_last = (t.flags & TASK_FLAG_IS_LAST) != 0;
            i++;
            if (is_last) break;
        }

        // Release staging space for this tensor (advances tail under lock, notifies drain)
        if (staging_release_total > 0) {
            drain_.notify_staging_freed_bytes(staging_release_total);
        }

        // Post-processing (no locks held, staging freed)
        // Use first_task (captured before the loop advanced i)
        do_post_processing(tensor, first_task);
    }
}

// ---------------------------------------------------------------------------
void P2PThread::do_post_processing(at::Tensor& tensor, const DrainTask& first_task) {
    // Pop metadata from TensorMetaFifo (C++ FIFO, GIL-free)
    ring_py::TensorMeta meta;
    if (!fifo_.pop(meta)) return;

    if (meta.shape.empty() || first_task.tensor_total_bytes == 0) return;
    if (!submit_fn_) return;

    // Reshape flat byte tensor to typed tensor
    auto dtype = static_cast<at::ScalarType>(meta.dtype);
    int64_t elem_size = at::elementSize(dtype);
    int64_t expected_elems = 1;
    for (auto d : meta.shape) expected_elems *= d;
    int64_t expected_bytes = expected_elems * elem_size;

    // Guard against shape/dtype mismatch (before reshape to avoid crash)
    if (static_cast<uint64_t>(expected_bytes) != first_task.tensor_total_bytes) {
        fprintf(stderr, "[p2p] WARN: shape/bytes mismatch: expected=%ld actual=%lu hook=%s\n",
                (long)expected_bytes, (unsigned long)first_task.tensor_total_bytes,
                meta.hook_name.c_str());
        return;
    }

    tensor = tensor.view(dtype).reshape(
        std::vector<int64_t>(meta.shape.begin(), meta.shape.end()));

    bool is_attn = is_attn_hook(meta.hook_name);
    auto [layer_no, act_name] = parse_internal_id(meta.hook_name);

    bool should_clone = cfg_.clone_slices && meta.requests.size() > 1;

    for (size_t j = 0; j < meta.requests.size(); ++j) {
        if (static_cast<int64_t>(j) >= tensor.size(0)) break;
        const ring_py::RequestMeta& req = meta.requests[j];
        at::Tensor slice = slice_for_request(
            tensor, static_cast<int>(j),
            req.start_token, req.end_token, is_attn);
        if (!slice.defined()) continue;

        // Always make contiguous: narrow() may return non-contiguous
        slice = slice.contiguous();
        if (should_clone) {
            slice = slice.clone();
        }

        try {
            submit_fn_(meta.model_id, meta.shard_rank,
                       req.req_id, act_name, layer_no,
                       req.start_token, req.end_token,
                       std::move(slice));
        } catch (...) {
            // Swallow: cannot propagate from background thread.
        }
    }
}

}  // namespace ring
