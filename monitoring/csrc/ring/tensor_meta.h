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

// ---------------------------------------------------------------------------
// Hook type definitions — single source of truth.
//
// The enum provides compile-time constants for switch/case.
// HOOK_DEFS[] associates each enum value with its act_name (for ClickHouse),
// short_name (for Python selection presets), and per_layer flag.
// Python imports this table via pybind11 and auto-derives all mappings.
//
// To add a new hook type: add one enum value + one HOOK_DEFS row.
// ---------------------------------------------------------------------------

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
    HOOK_TYPE_MLP_IN       = 12,
    HOOK_TYPE_MLP_OUT      = 13,
    HOOK_TYPE_RESID_FINAL  = 14,
    HOOK_TYPE_EMBED        = 15,
    HOOK_TYPE_POS_EMBED    = 16,
    HOOK_TYPE_FINAL_LN     = 17,
    HOOK_TYPE_TOKEN_IDS    = 18,
    HOOK_TYPE_FINAL_LOGITS = 19,
    HOOK_TYPE_MLP_POST     = 20,
    HOOK_TYPE_COUNT        = 21,
};

struct HookDef {
    int         id;          // enum value
    const char* act_name;    // ClickHouse act_name (p2p_thread uses this)
    const char* short_name;  // Python selection preset name
    bool        per_layer;   // true = "blocks.<L>.<act_name>", false = "<act_name>"
    const char* group;       // "attn", "mlp", or "other"
    bool        tp_sharded;  // true = tensor is TP-sharded (pre-all-reduce)
};

//  id                      act_name                    short_name      per_layer  group    tp_sharded
static constexpr HookDef HOOK_DEFS[] = {
    {HOOK_TYPE_RESID_PRE,   "hook_resid_pre",           "resid_pre",    true,  "other", false},
    {HOOK_TYPE_LN1,         "hook_ln1",                 "ln1",          true,  "other", false},
    {HOOK_TYPE_ATTN_OUT,    "hook_attn_out",            "attn_out",     true,  "attn",  false},
    {HOOK_TYPE_RESID_MID,   "hook_resid_mid",           "resid_mid",    true,  "other", false},
    {HOOK_TYPE_ATTN_SCORES, "attn.hook_attn_scores",    "attn_scores",  true,  "attn",  true },
    {HOOK_TYPE_PATTERN,     "attn.hook_pattern",         "pattern",      true,  "attn",  true },
    {HOOK_TYPE_Q,           "attn.hook_q",              "q",            true,  "attn",  true },
    {HOOK_TYPE_K,           "attn.hook_k",              "k",            true,  "attn",  true },
    {HOOK_TYPE_V,           "attn.hook_v",              "v",            true,  "attn",  true },
    {HOOK_TYPE_Z,           "attn.hook_z",              "z",            true,  "attn",  true },
    {HOOK_TYPE_LN2,         "hook_ln2",                 "ln2",          true,  "other", false},
    {HOOK_TYPE_MLP_IN,      "hook_mlp_in",              "mlp_in",       true,  "mlp",   false},
    {HOOK_TYPE_MLP_OUT,     "hook_mlp_out",             "mlp_out",      true,  "mlp",   false},
    {HOOK_TYPE_MLP_POST,    "hook_mlp_post",            "mlp_post",     true,  "mlp",   true },
    {HOOK_TYPE_RESID_FINAL, "hook_resid_final",         "resid_final",  false, "other", false},
    {HOOK_TYPE_EMBED,       "hook_embed",               "embed",        false, "other", false},
    {HOOK_TYPE_POS_EMBED,   "hook_pos_embed",           "pos_embed",    false, "other", false},
    {HOOK_TYPE_FINAL_LN,    "hook_final_ln",            "final_ln",     false, "other", false},
    {HOOK_TYPE_TOKEN_IDS,   "token_ids",                "token_ids",    false, "other", false},
    {HOOK_TYPE_FINAL_LOGITS,"final_logits",             "final_logits", false, "other", false},
};
static constexpr int HOOK_DEFS_COUNT = sizeof(HOOK_DEFS) / sizeof(HOOK_DEFS[0]);

// Auto-derived: int → act_name lookup (O(1) via indexed array).
inline const char* hook_type_name(int hook_type) {
    static const char* NAMES[HOOK_TYPE_COUNT] = {};
    static bool init = false;
    if (!init) {
        for (int i = 0; i < HOOK_TYPE_COUNT; i++) NAMES[i] = nullptr;
        for (int i = 0; i < HOOK_DEFS_COUNT; i++) NAMES[HOOK_DEFS[i].id] = HOOK_DEFS[i].act_name;
        init = true;
    }
    if (hook_type >= 0 && hook_type < HOOK_TYPE_COUNT && NAMES[hook_type])
        return NAMES[hook_type];
    return "unknown";
}

// Auto-derived from HOOK_DEFS tp_sharded column.
inline bool is_tp_sharded(int hook_type) {
    static bool FLAGS[HOOK_TYPE_COUNT] = {};
    static bool init = false;
    if (!init) {
        for (int i = 0; i < HOOK_TYPE_COUNT; i++) FLAGS[i] = false;
        for (int i = 0; i < HOOK_DEFS_COUNT; i++) FLAGS[HOOK_DEFS[i].id] = HOOK_DEFS[i].tp_sharded;
        init = true;
    }
    return hook_type >= 0 && hook_type < HOOK_TYPE_COUNT && FLAGS[hook_type];
}

// True if this hook type produces EP-sharded tensors (MoE expert computation).
// For dense models, MLP hooks are TP-sharded; for MoE, they are EP-sharded.
// Caller must check model type to decide; this is the EP candidate set.
inline bool is_ep_sharded(int hook_type) {
    return hook_type == HOOK_TYPE_MLP_IN || hook_type == HOOK_TYPE_MLP_OUT;
}

// True if this hook type is an attention weight matrix (attn_scores or pattern).
// These have shape [batch, heads, q_len, kv_len] — token dim is dim[-2], not dim[0].
inline bool is_attn_weight_matrix(int hook_type) {
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
