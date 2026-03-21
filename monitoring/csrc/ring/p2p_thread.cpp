// ring/p2p_thread.cpp -- Pinned-to-pageable thread implementation.
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

// Build ClickHouse act_name from hook_type + layer_no.
// layer_no >= 0: "layers.<hook_type_name>"  (layer_no passed separately)
// layer_no < 0:  "<hook_type_name>"         (global hook like embed, final_ln)
static std::string make_act_name(int hook_type, int layer_no) {
    const char* name = ring_py::hook_type_name(hook_type);
    if (layer_no >= 0) {
        return std::string("layers.") + name;
    }
    return std::string(name);
}

// Resolve shard_rank from hook_type and step context ranks.
static int32_t resolve_shard_rank(int hook_type, const ring_py::StepContext& ctx) {
    if (ring_py::is_tp_sharded(hook_type)) {
        return ctx.tp_rank;
    }
    // For MoE models, MLP hooks are EP-sharded.  But is_tp_sharded already
    // returns true for MLP_IN/MLP_OUT (dense case).  A future MoE flag on
    // StepContext would override this.  For now, TP takes precedence.
    return 0;
}

// HF batched slicing: tensor is [batch, q_len, ...].
static at::Tensor slice_for_request(
    const at::Tensor& tensor,
    int64_t batch_idx,
    int32_t start_token,
    int32_t end_token,
    bool    is_attn)
{
    if (start_token >= end_token) return at::Tensor{};
    int32_t eff = end_token - start_token;

    at::Tensor s = tensor.select(0, batch_idx);

    int token_dim = (is_attn && s.dim() >= 2) ? (s.dim() - 2) : 0;
    if (s.dim() > token_dim) {
        int64_t delta = s.size(token_dim);
        if (delta > eff) {
            int64_t skip = delta - eff;
            s = s.narrow(token_dim, skip, eff);
        }
    }

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

// vLLM flattened slicing: tensor is [total_tokens, ...], right-padded.
static at::Tensor slice_flattened(
    const at::Tensor& tensor,
    int64_t dim0_offset,
    int32_t start_token,
    int32_t end_token)
{
    int64_t num_tokens = static_cast<int64_t>(end_token - start_token);
    if (num_tokens <= 0) return at::Tensor{};
    if (dim0_offset + num_tokens > tensor.size(0)) return at::Tensor{};
    return tensor.narrow(0, dim0_offset, num_tokens);
}

// ---------------------------------------------------------------------------
P2PThread::P2PThread(DrainThread& drain, ring_py::TensorMetaFifo& fifo,
                     const RingConfig& cfg, SubmitFn submit_fn)
    : drain_(drain), fifo_(fifo), cfg_(cfg), submit_fn_(std::move(submit_fn))
{}

P2PThread::~P2PThread() noexcept {
    stop();
    delete current_ctx_;
}

void P2PThread::start() {
    thread_ = std::thread([this] { loop(); });
}

void P2PThread::stop() {
    if (thread_.joinable()) thread_.join();
}

// ---------------------------------------------------------------------------
void P2PThread::loop() {
    while (true) {
        uint64_t n = drain_.wait_for_tasks();
        if (n == 0) break;

        std::vector<DrainTask> local;
        drain_.pop_tasks(n, local);
        process(local);
    }
}

// ---------------------------------------------------------------------------
void P2PThread::process(std::vector<DrainTask>& tasks) {
    for (size_t i = 0; i < tasks.size(); ++i) {
        DrainTask& task = tasks[i];

        if (task.cpu_paged_tensor.defined()) {
            // CPU-direct tensor -- already in pageable memory, skip staging copy
            at::Tensor tensor = std::move(task.cpu_paged_tensor);
            do_post_processing(tensor, task);
            continue;
        }

        // Normal tensor: copy from pinned staging to pageable
        uint64_t total_bytes = task.tensor_total_bytes;

        auto tensor = at::empty({static_cast<int64_t>(total_bytes)},
                                at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        uint8_t* dst = tensor.data_ptr<uint8_t>();

        if (task.data_len1 > 0) {
            std::memcpy(dst, task.data_ptr1, task.data_len1);
        }
        if (task.data_len2 > 0) {
            std::memcpy(dst + task.data_len1, task.data_ptr2, task.data_len2);
        }

        // Release staging space
        if (task.alloc_bytes > 0) {
            drain_.notify_staging_freed_bytes(task.alloc_bytes);
        }

        do_post_processing(tensor, task);
    }
}

// ---------------------------------------------------------------------------
void P2PThread::do_post_processing(at::Tensor& tensor, const DrainTask& first_task) {
    ring_py::TensorMeta meta;
    if (!fifo_.pop(meta)) return;

    // Get step context -- pop from context queue if this is the first
    // hook in a new step (current_ctx_ is null).
    if (!current_ctx_) {
        current_ctx_ = fifo_.pop_context();
        if (!current_ctx_) return;  // no context available
    }

    if (meta.shape.empty() || first_task.tensor_total_bytes == 0) {
        if (meta.last_in_step) { delete current_ctx_; current_ctx_ = nullptr; }
        return;
    }
    if (!submit_fn_) {
        if (meta.last_in_step) { delete current_ctx_; current_ctx_ = nullptr; }
        return;
    }

    auto dtype = static_cast<at::ScalarType>(meta.dtype);
    int64_t elem_size = at::elementSize(dtype);
    int64_t expected_elems = 1;
    for (auto d : meta.shape) expected_elems *= d;
    int64_t expected_bytes = expected_elems * elem_size;

    if (static_cast<uint64_t>(expected_bytes) != first_task.tensor_total_bytes) {
        fprintf(stderr, "[p2p] WARN: shape/bytes mismatch: expected=%ld actual=%lu hook=%s\n",
                (long)expected_bytes, (unsigned long)first_task.tensor_total_bytes,
                ring_py::hook_type_name(meta.hook_type));
        if (meta.last_in_step) { delete current_ctx_; current_ctx_ = nullptr; }
        return;
    }

    tensor = tensor.view(dtype).reshape(
        std::vector<int64_t>(meta.shape.begin(), meta.shape.end()));

    // Build act_name and resolve shard_rank from hook_type
    std::string act_name = make_act_name(meta.hook_type, meta.layer_no);
    int32_t shard_rank = resolve_shard_rank(meta.hook_type, *current_ctx_);
    bool is_attn = ring_py::is_attn_hook(meta.hook_type);

    const auto& requests = current_ctx_->requests;
    bool should_clone = cfg_.clone_slices && requests.size() > 1;

    for (size_t j = 0; j < requests.size(); ++j) {
        const ring_py::RequestMeta& req = requests[j];
        at::Tensor slice;

        if (current_ctx_->flattened) {
            // vLLM: packed [total_tokens, ...], right-padded
            if (is_attn) continue;
            slice = slice_flattened(tensor, req.dim0_offset,
                                    req.start_token, req.end_token);
        } else {
            // HF: [batch, q_len, ...], left-padded tokens
            if (req.dim0_offset >= tensor.size(0)) break;
            slice = slice_for_request(
                tensor, req.dim0_offset,
                req.start_token, req.end_token, is_attn);
        }
        if (!slice.defined()) continue;

        slice = slice.contiguous();
        if (should_clone) {
            slice = slice.clone();
        }

        try {
            submit_fn_(current_ctx_->model_id, shard_rank,
                       req.req_id, act_name, meta.layer_no,
                       req.start_token, req.end_token,
                       std::move(slice));
        } catch (...) {
        }
    }

    // Last hook in step -- free context
    if (meta.last_in_step) {
        delete current_ctx_;
        current_ctx_ = nullptr;
    }
}

}  // namespace ring
