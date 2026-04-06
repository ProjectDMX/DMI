// ring/tensor_meta.h -- Tensor metadata FIFO for the ring transport.
//
// Per-step StepContext (heap-allocated, owned by p2p after pop) holds shared
// data (model_id, ranks, requests).  TensorMeta holds per-hook data
// (hook_type, layer_no, shape, dtype).  last_in_step signals p2p to free
// the context.
//
// No ATen dependency -- safe to include from both g++ and nvcc translation units.

#pragma once
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace ring_py {

// Hook type integer constants (must match ring_transport.py HOOK_TYPE_* values).
//
// Removed (gaps in numbering are intentional):
//   10 (RESULT):  removed, attn_out captures the same tensor (o_proj output).
//   resid_post:   removed (was per-layer).  Replaced by RESID_FINAL (global).
//                 resid_post[i] == resid_pre[i+1] for all i < N-1.
//
// Kept despite overlap:
//   LN2 vs MLP_IN: identical for dense models, differs for MoE (post-router).
enum HookType : int {
    HOOK_TYPE_RESID_PRE    = 0,
    HOOK_TYPE_LN1          = 1,
    HOOK_TYPE_ATTN_OUT     = 2,
    HOOK_TYPE_RESID_MID    = 3,
    HOOK_TYPE_ATTN_SCORES  = 4,
    HOOK_TYPE_PATTERN      = 5,
    HOOK_TYPE_Q            = 6,
    HOOK_TYPE_K            = 7,
    HOOK_TYPE_V            = 8,
    HOOK_TYPE_Z            = 9,
    // 10 removed (result == attn_out)
    HOOK_TYPE_LN2          = 11,
    HOOK_TYPE_MLP_IN       = 12,  // == LN2 for dense; differs for MoE (post-router)
    HOOK_TYPE_MLP_OUT      = 13,
    HOOK_TYPE_RESID_FINAL  = 14,  // global: last layer's residual before final norm
    HOOK_TYPE_EMBED        = 15,
    HOOK_TYPE_POS_EMBED    = 16,
    HOOK_TYPE_FINAL_LN     = 17,
    HOOK_TYPE_TOKEN_IDS    = 18,
    HOOK_TYPE_FINAL_LOGITS = 19,
    HOOK_TYPE_MLP_POST     = 20,  // after activation, before down_proj (TransformerLens hook_post)
    HOOK_TYPE_COUNT        = 21,
};

// Hook type -> display name for ClickHouse act_name column.
// Uses old-style paths so segment_merger and test code work unchanged.
// Per-layer hooks use "blocks." prefix; make_act_name prepends "blocks.<L>.".
inline const char* hook_type_name(int hook_type) {
    static const char* NAMES[] = {
        "hook_resid_pre", "hook_ln1", "hook_attn_out", "hook_resid_mid",
        "attn.hook_attn_scores", "attn.hook_pattern",
        "attn.hook_q", "attn.hook_k", "attn.hook_v",
        "attn.hook_z", nullptr, "hook_ln2", "hook_mlp_in", "hook_mlp_out",
        "hook_resid_final", "hook_embed", "hook_pos_embed", "hook_final_ln",
        "token_ids", "final_logits", "hook_mlp_post",
    };
    if (hook_type >= 0 && hook_type < HOOK_TYPE_COUNT) return NAMES[hook_type];
    return "unknown";
}

// True if this hook type produces TP-sharded tensors.
inline bool is_tp_sharded(int hook_type) {
    switch (hook_type) {
        case HOOK_TYPE_Q: case HOOK_TYPE_K: case HOOK_TYPE_V: case HOOK_TYPE_Z:
        case HOOK_TYPE_ATTN_OUT:
        case HOOK_TYPE_MLP_OUT:
        case HOOK_TYPE_FINAL_LOGITS:
            return true;
        default:
            return false;
    }
}

// True if this hook type produces EP-sharded tensors (MoE expert computation).
// For dense models, MLP hooks are TP-sharded; for MoE, they are EP-sharded.
// Caller must check model type to decide; this is the EP candidate set.
inline bool is_ep_sharded(int hook_type) {
    return hook_type == HOOK_TYPE_MLP_IN || hook_type == HOOK_TYPE_MLP_OUT;
}

// True if this hook type is an attention weight matrix (scores/pattern).
inline bool is_attn_hook(int hook_type) {
    return hook_type == HOOK_TYPE_ATTN_SCORES || hook_type == HOOK_TYPE_PATTERN;
}

struct RequestMeta {
    std::string req_id;
    int32_t     start_token = 0;
    int32_t     end_token   = 0;
    int64_t     dim0_offset = 0;   // HF: batch_idx, vLLM: token offset
    int32_t     kv_offset   = 0;   // attn kv-dim start (HF dynamic cache: pad_len, else 0)
};

// Per-step shared context.  Heap-allocated by push_step caller.
// Ownership transfers to p2p thread via pop_context; p2p deletes it.
struct StepContext {
    std::string              model_id;
    int32_t                  tp_rank   = 0;
    int32_t                  dp_rank   = 0;
    int32_t                  ep_rank   = 0;
    int32_t                  pp_rank   = 0;
    std::vector<RequestMeta> requests;
    bool                     flattened = false;  // false=HF batched, true=vLLM packed
};

// Per-hook metadata.  No strings -- hook_type + layer_no replace hook_name.
struct TensorMeta {
    int                      hook_type    = 0;   // HookType enum value
    int                      layer_no     = -1;  // -1 for global hooks
    std::vector<int64_t>     shape;
    int                      dtype        = 0;   // at::ScalarType integer value
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
