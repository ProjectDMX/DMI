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
// HOOK_DEFS[] associates each enum value with all its properties.
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

// Hook group — which sub-block produces this tensor.
enum HookGroup : int { GROUP_ATTN = 0, GROUP_MLP = 1, GROUP_OTHER = 2 };

// Shape class — determines the shape formula in _compute_hook_shape:
//   SHAPE_HIDDEN   : [batch, q_len, hidden_dim]
//   SHAPE_QKV_Q    : [batch, q_len, num_heads/tp, head_dim]
//   SHAPE_QKV_KV   : [batch, q_len, kv_heads/tp, head_dim]
//   SHAPE_QKV_Z    : [batch, q_len, num_heads/tp, head_dim]
//   SHAPE_ATTN_WT  : [batch, num_heads/tp, q_len, kv_len]  (attention weight matrix)
//   SHAPE_MLP_POST : [batch, q_len, intermediate_dim/tp]
//   SHAPE_TOKEN_IDS: [batch, q_len]  (int64, no feature dim)
//   SHAPE_LOGITS   : [batch, logits_to_keep, vocab_size]
enum ShapeClass : int {
    SHAPE_HIDDEN = 0, SHAPE_QKV_Q = 1, SHAPE_QKV_KV = 2, SHAPE_QKV_Z = 3,
    SHAPE_ATTN_WT = 4, SHAPE_MLP_POST = 5, SHAPE_TOKEN_IDS = 6, SHAPE_LOGITS = 7,
};

// Pipeline-parallel stage placement.
enum PpStage : int { PP_ANY = 0, PP_FIRST = 1, PP_LAST = 2 };

struct HookDef {
    int         id;          // HookType enum value
    const char* act_name;    // ClickHouse act_name (p2p_thread uses this)
    const char* short_name;  // Python selection preset name
    bool        per_layer;   // true = "blocks.<L>.<act_name>", false = global
    int         group;       // HookGroup: GROUP_ATTN / GROUP_MLP / GROUP_OTHER
    bool        tp_sharded;  // true = tensor is TP-sharded (pre-all-reduce)
    int         shape_class; // ShapeClass: determines shape formula
    int         pp_stage;    // PpStage: PP_ANY / PP_FIRST / PP_LAST
};

//  id                      act_name                    short       perlyr group        tp_sh  shape           pp_stage
static constexpr HookDef HOOK_DEFS[] = {
    {HOOK_TYPE_RESID_PRE,   "hook_resid_pre",           "resid_pre",    true,  GROUP_OTHER, false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_LN1,         "hook_ln1",                 "ln1",          true,  GROUP_OTHER, false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_ATTN_OUT,    "hook_attn_out",            "attn_out",     true,  GROUP_ATTN,  false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_RESID_MID,   "hook_resid_mid",           "resid_mid",    true,  GROUP_OTHER, false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_ATTN_SCORES, "attn.hook_attn_scores",    "attn_scores",  true,  GROUP_ATTN,  true,  SHAPE_ATTN_WT,  PP_ANY  },
    {HOOK_TYPE_PATTERN,     "attn.hook_pattern",         "pattern",      true,  GROUP_ATTN,  true,  SHAPE_ATTN_WT,  PP_ANY  },
    {HOOK_TYPE_Q,           "attn.hook_q",              "q",            true,  GROUP_ATTN,  true,  SHAPE_QKV_Q,    PP_ANY  },
    {HOOK_TYPE_K,           "attn.hook_k",              "k",            true,  GROUP_ATTN,  true,  SHAPE_QKV_KV,   PP_ANY  },
    {HOOK_TYPE_V,           "attn.hook_v",              "v",            true,  GROUP_ATTN,  true,  SHAPE_QKV_KV,   PP_ANY  },
    {HOOK_TYPE_Z,           "attn.hook_z",              "z",            true,  GROUP_ATTN,  true,  SHAPE_QKV_Z,    PP_ANY  },
    {HOOK_TYPE_LN2,         "hook_ln2",                 "ln2",          true,  GROUP_OTHER, false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_MLP_IN,      "hook_mlp_in",              "mlp_in",       true,  GROUP_MLP,   false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_MLP_OUT,     "hook_mlp_out",             "mlp_out",      true,  GROUP_MLP,   false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_MLP_POST,    "hook_mlp_post",            "mlp_post",     true,  GROUP_MLP,   true,  SHAPE_MLP_POST, PP_ANY  },
    {HOOK_TYPE_RESID_FINAL, "hook_resid_final",         "resid_final",  false, GROUP_OTHER, false, SHAPE_HIDDEN,   PP_ANY  },
    {HOOK_TYPE_EMBED,       "hook_embed",               "embed",        false, GROUP_OTHER, false, SHAPE_HIDDEN,   PP_FIRST},
    {HOOK_TYPE_POS_EMBED,   "hook_pos_embed",           "pos_embed",    false, GROUP_OTHER, false, SHAPE_HIDDEN,   PP_FIRST},
    {HOOK_TYPE_FINAL_LN,    "hook_final_ln",            "final_ln",     false, GROUP_OTHER, false, SHAPE_HIDDEN,   PP_LAST },
    {HOOK_TYPE_TOKEN_IDS,   "token_ids",                "token_ids",    false, GROUP_OTHER, false, SHAPE_TOKEN_IDS,PP_FIRST},
    {HOOK_TYPE_FINAL_LOGITS,"final_logits",             "final_logits", false, GROUP_OTHER, false, SHAPE_LOGITS,   PP_LAST },
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

// Auto-derived: EP-sharding candidates (MoE expert computation).
// For dense models these are TP-sharded; for MoE they are EP-sharded.
// Derivation: group == GROUP_MLP && !tp_sharded.
inline bool is_ep_sharded(int hook_type) {
    static bool FLAGS[HOOK_TYPE_COUNT] = {};
    static bool init = false;
    if (!init) {
        for (int i = 0; i < HOOK_TYPE_COUNT; i++) FLAGS[i] = false;
        for (int i = 0; i < HOOK_DEFS_COUNT; i++)
            FLAGS[HOOK_DEFS[i].id] = (HOOK_DEFS[i].group == GROUP_MLP && !HOOK_DEFS[i].tp_sharded);
        init = true;
    }
    return hook_type >= 0 && hook_type < HOOK_TYPE_COUNT && FLAGS[hook_type];
}

// Auto-derived: attention weight matrix (shape_class == SHAPE_ATTN_WT).
// These have shape [batch, heads, q_len, kv_len] — token dim is dim[-2], not dim[0].
inline bool is_attn_weight_matrix(int hook_type) {
    static bool FLAGS[HOOK_TYPE_COUNT] = {};
    static bool init = false;
    if (!init) {
        for (int i = 0; i < HOOK_TYPE_COUNT; i++) FLAGS[i] = false;
        for (int i = 0; i < HOOK_DEFS_COUNT; i++)
            FLAGS[HOOK_DEFS[i].id] = (HOOK_DEFS[i].shape_class == SHAPE_ATTN_WT);
        init = true;
    }
    return hook_type >= 0 && hook_type < HOOK_TYPE_COUNT && FLAGS[hook_type];
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

// Per-hook metadata flag bits.  Packed into TensorMeta::flags.
//   META_FLAG_ALLOW_MISMATCH -- consumer recomputes dim-0 from the
//     actual bytes the producer wrote rather than asserting that the
//     received byte count equals shape's product * elem_size.  Set
//     when the source HookSpec has allow_token_cnt_mismatch=True
//     (dynamic-shape EP hooks).
static constexpr uint8_t META_FLAG_ALLOW_MISMATCH = 1u << 0;

// Per-hook metadata.  No strings -- hook_type + layer_no replace hook_name.
// flags occupies a padding byte; struct size is unchanged.
struct TensorMeta {
    int                      hook_type    = 0;   // HookType enum value
    int                      layer_no     = -1;  // -1 for global hooks
    std::vector<int64_t>     shape;
    int                      dtype        = 0;   // at::ScalarType integer value
    bool                     last_in_step = false;
    uint8_t                  flags        = 0;   // META_FLAG_* bitmask
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
